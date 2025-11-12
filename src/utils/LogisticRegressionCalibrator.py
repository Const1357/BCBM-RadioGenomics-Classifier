import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.constants import DEVICE

import numpy as np


class LogisticCalibrator(nn.Module):
    """Single-label logistic regression calibrator (Platt scaling)."""
    def __init__(self):
        super().__init__()
        # Linear transform: scale * logit + shift
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Convert logits to calibrated probability
        return torch.sigmoid(logits * self.scale + self.shift)

    def nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood (binary cross-entropy) for calibration.
        logits: raw model logits [N]
        labels: binary targets [N]
        """
        probs = self.forward(logits)
        return F.binary_cross_entropy(probs, labels.float())


class MultiLabelLogisticCalibrator(nn.Module):
    """Multilabel logistic regression calibrator (Platt scaling) for ER, PR, HER2."""
    def __init__(self):
        super().__init__()
        self.ER_calibrator = LogisticCalibrator()
        self.PR_calibrator = LogisticCalibrator()
        self.HER2_calibrator = LogisticCalibrator()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr=1e-2, max_iter=500):
        """
        Fit three logistic calibrators independently using binary cross-entropy.
        logits: [N, 3] raw logits
        labels: [N, 3] binary labels
        """
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        logits = logits.detach()
        labels = labels.detach().float()

        def closure():
            optimizer.zero_grad()
            loss = self.ER_calibrator.nll_loss(logits[:, 0], labels[:, 0]) \
                + self.PR_calibrator.nll_loss(logits[:, 1], labels[:, 1]) \
                + self.HER2_calibrator.nll_loss(logits[:, 2], labels[:, 2])

            # --- L2 regularization ---

            l2 = 0
            for calibrator in [self.ER_calibrator, self.PR_calibrator, self.HER2_calibrator]:
                loss += l2 * (calibrator.scale ** 2 + calibrator.shift ** 2)

            loss.backward()
            return loss

        optimizer.step(closure)

        print("Fitted logistic calibrators (scale, shift):",
              f"ER=({self.ER_calibrator.scale.item():.3f}, {self.ER_calibrator.shift.item():.3f}), "
              f"PR=({self.PR_calibrator.scale.item():.3f}, {self.PR_calibrator.shift.item():.3f}), "
              f"HER2=({self.HER2_calibrator.scale.item():.3f}, {self.HER2_calibrator.shift.item():.3f})")

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities after fitting.
        """
        with torch.no_grad():
            probs = torch.stack([
                self.ER_calibrator(logits[:, 0]),
                self.PR_calibrator(logits[:, 1]),
                self.HER2_calibrator(logits[:, 2]),
            ], dim=1)
            return probs