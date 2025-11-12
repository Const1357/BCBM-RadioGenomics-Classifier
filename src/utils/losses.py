import torch
import torch.nn.functional as F
from utils.constants import DEVICE, DTYPE

class BCEwithNanHandling(torch.nn.Module):
    """
    BCEWithLogitsLoss that ignores NaN labels and computes per-class mean.\
    Returns per-class losses instead of a single scalar.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super(BCEwithNanHandling, self).__init__()
        self.pos_weight = pos_weight.to(DEVICE)
        self.bce_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')  # [B, C]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        labels: [B, C] with possible NaNs
        returns: per-class BCE loss [C]
        """
        valid_mask = labels != -1  # [B, C]
        labels_clamped = torch.where(valid_mask, labels, torch.zeros_like(labels))  # [B, C]

        bce_raw = self.bce_fn(logits, labels_clamped.to(dtype=DTYPE))  # [B, C]
        bce_masked = bce_raw * valid_mask.to(dtype=DTYPE)  # zero out NaNs, [B, C]

        # Tensorized per-class mean
        sum_per_class = bce_masked.sum(dim=0)  # [C]
        count_per_class = valid_mask.sum(dim=0)  # [C]

        per_class_loss = torch.zeros_like(sum_per_class, device=DEVICE, dtype=DTYPE)
        mask_nonzero = count_per_class > 0
        per_class_loss[mask_nonzero] = sum_per_class[mask_nonzero] / count_per_class[mask_nonzero]  # [C]

        return per_class_loss
    

class AsymmetricFocalLossWithNanHandling(torch.nn.Module):
    """
    ASL with per-class gamma and optional positive & negative shifts.
    Ignores NaN labels (-1) and returns per-class losses.
    """

    def __init__(self, 
                 gamma_pos: torch.Tensor,
                 gamma_neg: torch.Tensor,
                 shift_m_pos: torch.Tensor,
                 shift_m_neg: torch.Tensor):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.shift_m_pos = shift_m_pos
        self.shift_m_neg = shift_m_neg

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        labels: [B, C] with possible -1
        returns: per-class loss [C]
        """
        valid_mask = labels != -1
        labels_clamped = torch.where(valid_mask, labels, torch.zeros_like(labels)).to(dtype=DTYPE)

        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Shifted probabilities
        # Positive shift: move probs down slightly to reduce easy positive contribution
        p_pos = (probs - self.shift_m_pos).clamp(min=0.0, max=1.0)
        # Negative shift: move probs down as before
        p_neg = (probs - self.shift_m_neg).clamp(min=0.0, max=1.0)

        # Positive loss
        loss_pos = -labels_clamped * ((1 - p_pos) ** self.gamma_pos) * torch.log(p_pos.clamp(min=1e-8))

        # Negative loss
        loss_neg = -(1 - labels_clamped) * (p_neg ** self.gamma_neg) * torch.log((1 - p_neg).clamp(min=1e-8))

        # Combine and mask NaNs
        loss_raw = (loss_pos + loss_neg) * valid_mask.to(dtype=DTYPE)

        # Per-class mean
        sum_per_class = loss_raw.sum(dim=0)
        count_per_class = valid_mask.sum(dim=0)
        per_class_loss = torch.zeros_like(sum_per_class, device=DEVICE, dtype=DTYPE)
        mask_nonzero = count_per_class > 0
        per_class_loss[mask_nonzero] = sum_per_class[mask_nonzero] / count_per_class[mask_nonzero]

        return per_class_loss


# Was used for auxiliary segmentation task using Unet; kept for possible future use
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, groundtruth):
        """
        predicted: [B, 1, D, H, W]
        groundtruth: [B, 1, D, H, W]
        """
        predicted = torch.sigmoid(predicted)  # [B,1,D,H,W]
        predicted = predicted.view(predicted.size(0), -1)  # [B, D*H*W]
        groundtruth = groundtruth.view(groundtruth.size(0), -1)  # [B, D*H*W]

        intersection = (predicted * groundtruth).sum(dim=1)  # [B]
        dice_coeff = (2.0 * intersection + self.smooth) / (predicted.sum(dim=1) + groundtruth.sum(dim=1) + self.smooth)  # [B]

        return 1 - dice_coeff.mean()  # scalar
