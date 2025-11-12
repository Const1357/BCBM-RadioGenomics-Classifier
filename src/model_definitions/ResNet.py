import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import DEVICE

# Basic 3D Residual Block
class ResBlock3D(nn.Module):
    """
    A simple 3D residual block: Conv3D -> BN -> ReLU -> Conv3D -> BN + residual
    Input shape: [B, C_in, D, H, W]
    Output shape: [B, C_out, D, H, W]
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels)

        # Residual projection if in/out channels differ
        self.proj = None
        if in_channels != out_channels or stride != 1:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj:
            identity = self.proj(identity)

        out += identity
        out = self.relu(out)
        return out

# ResNet3D Encoder
class ResNet3DEncoder(nn.Module):
    """
    3D ResNet Encoder with residual blocks.
    - depth: number of residual blocks
    - base_filters: number of channels in first block (doubles at each subsequent block)
    Outputs:
    - bottleneck: final feature map
    - skips: list of intermediate features for classifier aggregation
    """
    def __init__(self, depth=3, base_filters=16):
        super(ResNet3DEncoder, self).__init__()
        self.depth = depth
        self.base_filters = base_filters

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        last_channels = 1  # input is 1 channel (grayscale)
        out_channels = base_filters

        for i in range(depth):
            self.blocks.append(ResBlock3D(last_channels, out_channels))
            if i < depth - 1:
                # Downsample spatial dims by 2
                self.downsamples.append(nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2))
            else:
                self.downsamples.append(None)
            last_channels = out_channels
            out_channels *= 2  # double channels each block

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)  # store features for classifier
            if i < self.depth - 1:
                x = self.downsamples[i](x)
        return x, skips  # x is bottleneck

#   Classifier Head with Skip Weights
class ClassifierHead3D(nn.Module):
    """
    Combines multi-scale skip features via learnable per-skip weights,
    then performs classification via fully connected layers.
    """
    def __init__(self, skip_channels, num_classes=3, dropout=0.0):
        super().__init__()
        self.num_skips = len(skip_channels)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skips))  # learnable scalar weights

        total_channels = sum(skip_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # GAP → [B, C, 1, 1, 1]
            nn.Flatten(),   # for a 5 layer resnet with 16 starting features = 16 + 32 + 64 + 128 + 256 = 496 features
            nn.Linear(total_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, skips: list[torch.Tensor]):
        """
        skips: list of [B, C_i, D_i, H_i, W_i] feature maps from encoder blocks
        self.skip_weights: tensor of shape [num_skips], scalar weight per skip
        """

        # spatial size of bottleneck (last skip)
        target_shape = skips[-1].shape[2:]  # (D_b, H_b, W_b)

        # resize each skip to bottleneck size if needed
        resized_skips = [
            F.adaptive_avg_pool3d(f, target_shape) if f.shape[2:] != target_shape else f
            for f in skips
        ]

        # element-wise multiplication of each skip by its scalar weight
        weighted_skips = [
            f * w.to(f.device).view(1, 1, 1, 1, 1)  # broadcast scalar to [B, C_i, D, H, W]
            for f, w in zip(resized_skips, self.skip_weights)
        ]

        # concatenate along channel dimension
        concat = torch.cat(weighted_skips, dim=1)  # [B, ΣC_i, D, H, W]

        # forward through classifier
        out = self.classifier(concat)
        return out

# Full ResNet3D Classifier
class ResNet3D(nn.Module):
    """
    ResNet for 3D Image MultiLabel Classification.
    - No decoder, residual blocks in encoder
    - Classifier head receives multi-scale skip features
    - Supports dynamic depth
    """
    def __init__(self, depth=4, base_filters=16, clf_threshold=[0.5,0.5,0.5], dropout=0.0):
        super(ResNet3D, self).__init__()
        self.name = "ResNet3D"
        self.depth = depth
        self.base_filters = base_filters
        self.clf_threshold = torch.tensor(clf_threshold).to(DEVICE)
        self.dropout = dropout

        self.encoder = ResNet3DEncoder(depth, base_filters)

        # Compute channels at each skip for classifier head
        skip_channels = []
        c = base_filters
        for i in range(depth):
            skip_channels.append(c)
            c *= 2
        self.classifier = ClassifierHead3D(skip_channels, dropout=dropout)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        classifier_out = self.classifier(skips)
        return classifier_out

    def predict(self, x, return_raw=False, saliency=False):
        self.eval()

        if not saliency:
            with torch.no_grad():
                classifier_out = self.forward(x)
                probs = torch.sigmoid(classifier_out)
                preds = (probs > self.clf_threshold).float()
            if return_raw:
                return preds, probs, classifier_out
            return preds, probs

        # Vanilla Saliency maps
        num_labels = self.classifier.classifier[-1].out_features
        saliency_maps = {}

        # ensure input requires grad
        x_req = x.clone().detach().requires_grad_(True)

        # forward pass
        classifier_out = self.forward(x_req)

        for label_idx in range(num_labels):
            self.zero_grad()
            score = classifier_out[0, label_idx]  # assuming batch size 1
            score.backward(retain_graph=True)
            grad = x_req.grad.detach().cpu()[0, 0]  # [D, H, W], assuming 1 input channel
            saliency_maps[label_idx] = grad.abs().numpy()  # take absolute value

            # reset gradients for next label
            x_req.grad.zero_()

        # standard prediction
        with torch.no_grad():
            probs = torch.sigmoid(classifier_out)
            preds = (probs > self.clf_threshold).float()

        if return_raw:
            return preds, probs, classifier_out, saliency_maps
        return preds, probs, saliency_maps

    def store(self, filepath):
        state = {
            'model_state_dict': self.state_dict(),
            'depth': self.depth,
            'base_filters': self.base_filters,
            'clf_threshold': self.clf_threshold,
            'dropout': self.dropout,
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath, map_location=None):
        checkpoint = torch.load(filepath, map_location=map_location)
        model = cls(
            depth=checkpoint['depth'],
            base_filters=checkpoint['base_filters'],
            clf_threshold=checkpoint.get('clf_threshold', [0.5,0.5,0.5])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_config(self):
        return {
            'depth': self.depth,
            'base_filters': self.base_filters,
            'clf_threshold': self.clf_threshold,
            'dropout': self.dropout,
        }
