import torch
import torch.nn.functional as F
from constants import DEVICE, DTYPE

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, groundtruth):
        # Apply sigmoid to get probabilities
        predicted = torch.sigmoid(predicted)

        # Flatten the tensors
        predicted = predicted.view(-1)
        groundtruth = groundtruth.view(-1)

        intersection = (predicted * groundtruth).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predicted.sum() + groundtruth.sum() + self.smooth)
        return 1 - dice_coeff  # Dice loss is 1 - Dice coefficient

class MixedLoss(torch.nn.Module):

    def __init__(self, mix_coeff = 0.5):
        super(MixedLoss, self).__init__()

        # pre-computed class weights: neg/pos (nan ignored)
        self.pos_weight = torch.tensor([0.934426, 0.516340, 0.624113], device=DEVICE)

        self.mix_coeff = mix_coeff

        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLoss()

    def forward(self, clf_logits, seg_logits, labels, mask, only_clf=False):


        # Classification loss (BCE) with handling of possible asymmetric NaN labels
        num_classes = labels.shape[1]

        clf_loss = 0.0
        valid_classes = 0
        for i in range(num_classes):
            nanmask = ~torch.isnan(labels[:, i])  # [batch], True where label is valid
            if nanmask.sum() > 0:
                class_logits = clf_logits[nanmask, i]
                class_labels = labels[nanmask, i].float()
                clf_loss += self.bce(class_logits, class_labels)
                valid_classes += 1

        if valid_classes > 0:
            clf_loss = clf_loss / valid_classes
        else:
            clf_loss = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

        if only_clf:
            return clf_loss
        # Segmentation loss
        
        seg_loss = self.dice(seg_logits, mask.float())

        # Combined loss
        total_loss = clf_loss + self.mix_coeff * seg_loss
        return total_loss, clf_loss, seg_loss