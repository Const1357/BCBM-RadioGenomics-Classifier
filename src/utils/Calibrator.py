import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.constants import DEVICE
import numpy as np


class TemperatureCalibrator(nn.Module):
    """Single-label temperature scaling calibrator."""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(torch.log(torch.tensor(init_T)), dtype=torch.float32))

    @property
    def T(self):
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply temperature scaling."""
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).to(DEVICE).float()
        return logits / self.T


class LogisticCalibrator(nn.Module):
    """Single-label logistic regression calibrator (Platt scaling)."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits * self.scale + self.shift)

    def nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(self.forward(logits), labels.float())


class MixedCalibrator(nn.Module):
    """
    Mixed calibrator:
      - ER: temperature scaling
      - PR, HER2: logistic regression (Platt scaling)
    """
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.ER_calibrator = TemperatureCalibrator(init_T)
        self.PR_calibrator = LogisticCalibrator()
        self.HER2_calibrator = LogisticCalibrator()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr=1e-2, max_iter=500, l2_reg=0.0):
        """
        Fit the calibrators independently.
        logits: [N, 3] raw logits
        labels: [N, 3] binary labels
        """
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        logits = logits.detach()
        labels = labels.detach().float()

        def closure():
            optimizer.zero_grad()

            # ER: temperature scaling -> logits
            er_scaled = self.ER_calibrator(logits[:, 0])
            # PR & HER2: logistic regression -> probabilities
            pr_probs = self.PR_calibrator(logits[:, 1])
            her2_probs = self.HER2_calibrator(logits[:, 2])

            # NLL loss
            loss = F.binary_cross_entropy_with_logits(er_scaled, labels[:, 0]) \
                 + F.binary_cross_entropy(pr_probs, labels[:, 1]) \
                 + F.binary_cross_entropy(her2_probs, labels[:, 2])

            # Optional L2 regularization on parameters
            if l2_reg > 0.0:
                loss += l2_reg * (
                    self.ER_calibrator.log_T ** 2 +
                    self.PR_calibrator.scale ** 2 + self.PR_calibrator.shift ** 2 +
                    self.HER2_calibrator.scale ** 2 + self.HER2_calibrator.shift ** 2
                )

            loss.backward()
            return loss

        optimizer.step(closure)

        print("Fitted calibrators:")
        print(f"ER (T) = {self.ER_calibrator.T.item():.3f}")
        print(f"PR (scale, shift) = ({self.PR_calibrator.scale.item():.3f}, {self.PR_calibrator.shift.item():.3f})")
        print(f"HER2 (scale, shift) = ({self.HER2_calibrator.scale.item():.3f}, {self.HER2_calibrator.shift.item():.3f})")

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities.
        ER: apply sigmoid after temperature scaling
        PR/HER2: already probabilities from logistic regression
        """
        with torch.no_grad():
            er_prob = torch.sigmoid(self.ER_calibrator(logits[:, 0]))
            pr_prob = self.PR_calibrator(logits[:, 1])
            her2_prob = self.HER2_calibrator(logits[:, 2])
            return torch.stack([er_prob, pr_prob, her2_prob], dim=1)
