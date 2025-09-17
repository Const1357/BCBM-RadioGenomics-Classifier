import torch
import torch.nn.functional as F
from torch import nn
import os

# Why U-Net?
# U-Net is popular for segmentation tasks due to its encoder-decoder structure with skip connections.
# The encoder produces hierarchical features, while the decoder reconstructs the segmentation map.
# Skip connections help retain spatial information lost during downsampling.

# The shared encoder extracts features useful for both segmentation and classification.
# Our main task is binary classification of 3 classes, but we perform segmentation as a secondary task.
# The segmentation task helps the model learn better spatial features, improving classification performance.

# U-Net is also able to learn from limited (and augmented) data, which is beneficial in medical imaging contexts.
# GAP (Global Average Pooling) is used in the classifier head to reduce overfitting and model size.

class ConvBlock3D(nn.Module):
    # halves the spatial dimensions
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else None

    def forward(self, x):

        x = self.block(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class UNetEncoder(torch.nn.Module):

    def __init__(self, depth=3, base_filters=16):
        super(UNetEncoder, self).__init__()

        self.depth = depth
        self.base_filters = base_filters

        self.enc_blocks = nn.ModuleList()
        self.maxpool_layers = nn.ModuleList()

        last_in_channels = 1    # input has 1 channel (grayscale)
        last_out_channels = base_filters
        
        # Input: [B, C, D, H, W] = [B, 1->base_filters(=16), 64, 128, 128]
        for i in range(depth):
            
            # each block HALVES the spatial dimensions and DOUBLES the number of channels
            enc = ConvBlock3D(last_in_channels, last_out_channels)
            self.enc_blocks.append(enc)

            if i < depth - 1:   # no maxpool after the last block
                maxpool = nn.MaxPool3d(2)
                self.maxpool_layers.append(maxpool)

            last_in_channels = last_out_channels
            last_out_channels = last_out_channels * 2

            # Out = [B, 2*Channels, D/2, H/2, W/2]

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x) # store result for skip connection

            if i < self.depth - 1:  # apply maxpool except on the last block (output is the bottleneck)
                x = self.maxpool_layers[i](x)

        return x, skips  # x is the bottleneck output
    

class UNetDecoder(nn.Module):
    def __init__(self, depth=3, base_filters=16):
        super(UNetDecoder, self).__init__()

        self.depth = depth
        self.base_filters = base_filters

        self.upconv_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Start with bottleneck channels
        last_in_channels = base_filters * (2 ** (depth - 1))
        last_out_channels = last_in_channels // 2

        for i in range(depth - 1):
            # 1. Up-convolution (upsampling step)
            upconv = nn.ConvTranspose3d(
                in_channels=last_in_channels,
                out_channels=last_out_channels,
                kernel_size=2,
                stride=2
            )
            self.upconv_layers.append(upconv)

            # 2. Decoder block takes [skip + upsampled feature map]
            dec_in_channels = last_out_channels * 2  # because of concatenation with skip connection
            dec = ConvBlock3D(dec_in_channels, last_out_channels)
            self.dec_blocks.append(dec)

            # Prepare for next iteration
            last_in_channels = last_out_channels
            last_out_channels = last_out_channels // 2

        # Final 1x1x1 conv to match desired output channels
        out_channels = 1  # binary segmentation
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x, skips):

        for i in range(self.depth - 1):
            # 1. Upsample
            x = self.upconv_layers[i](x)

            # 2. Get corresponding skip connection
            skip_connection = skips[self.depth - 2 - i]  # reverse order

            # 3. Concatenate skip and upsampled
            x = torch.cat((skip_connection, x), dim=1)

            # 4. Process with ConvBlock3D
            x = self.dec_blocks[i](x)

        # 5. Final output
        x = self.final_conv(x)
        return x
    
class ClassifierHead(nn.Module):
    def __init__(self, in_channels, hidden_size=64, num_classes=3):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global Average Pooling (GAP) instead of flattening. 
            # for output size [B, 128, 8, 16, 16] => [B, 128, 1, 1, 1] instead of [B, 128*8*16*16=16384]
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    

class UNet3D(nn.Module):
    def __init__(self, depth=3, base_filters=16, clf_threshold=0.5, seg_threshold=0.5):
        super(UNet3D, self).__init__()

        self.name = "UNet3D"

        self.depth = depth
        self.base_filters = base_filters
        self.clf_threshold = clf_threshold
        self.seg_threshold = seg_threshold

        self.encoder = UNetEncoder(depth, base_filters)
        self.decoder = UNetDecoder(depth, base_filters)
        self.classifier = ClassifierHead(in_channels=base_filters * (2 ** (depth - 1)), hidden_size=64, num_classes=3)

    def forward(self, x):

        bottleneck, skips = self.encoder(x)
        classifier_out = self.classifier(bottleneck)

        if self.training:
            segmentation_out = self.decoder(bottleneck, skips)
            return classifier_out, segmentation_out
        else:
            return classifier_out

    def predict(self, x, return_raw=False):
        self.eval()

        with torch.no_grad():
            classifier_out = self.forward(x)
            probs = torch.sigmoid(classifier_out)

            preds = (probs > self.clf_threshold).float()

        if return_raw:
            return preds, probs, classifier_out
        
        return preds, probs

    def store(self, filepath):
        state = {
            'model_state_dict': self.state_dict(),
            'depth': self.depth,
            'base_filters': self.base_filters,
            'clf_threshold': self.clf_threshold,
            'seg_threshold': self.seg_threshold
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath, map_location=None):
        checkpoint = torch.load(filepath, map_location=map_location)
        model = cls(
            depth=checkpoint['depth'],
            base_filters=checkpoint['base_filters'],
            clf_threshold=checkpoint.get('clf_threshold', 0.5),
            seg_threshold=checkpoint.get('seg_threshold', 0.5)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_config(self):
        return {
            'depth': self.depth,
            'base_filters': self.base_filters,
            'clf_threshold': self.clf_threshold,
            'seg_threshold': self.seg_threshold,
        }

# TODO:
# - Load a test sample and pass it through the model to verify dimensions
# - Implement training loop with combined loss (DiceLoss + BCE for segmentation, CrossEntropy for classification)
# - Add evaluation metrics (e.g., IoU for segmentation, accuracy/precision/recall/f1 for classification)
# - Experiment with hyperparameters (depth, base_filters, learning rate, etc.)
