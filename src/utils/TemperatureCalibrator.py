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


class MultiLabelTemperatureScaler(nn.Module):
    """Three independent temperature scalers: ER, PR, HER2."""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.ER_calibrator = TemperatureCalibrator(init_T)
        self.PR_calibrator = TemperatureCalibrator(init_T)
        self.HER2_calibrator = TemperatureCalibrator(init_T)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr=1e-2, max_iter=1000):
        """
        Fit the three temperatures by minimizing NLL on calibration data.
        logits: [N, 3] tensor of raw logits from model
        labels: [N, 3] tensor of {0,1} targets
        """
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        logits, labels = logits.detach(), labels.detach()

        # ------ diagnostics ------
        pr_logits = logits[:, 1]
        print(min(pr_logits).item(), max(pr_logits).item(), torch.mean(pr_logits).item(), torch.std(pr_logits).item())
        print(pr_logits)

        def closure():
            optimizer.zero_grad()
            scaled_logits = torch.stack([
                self.ER_calibrator(logits[:, 0]),
                self.PR_calibrator(logits[:, 1]),
                self.HER2_calibrator(logits[:, 2]),
            ], dim=1)

            log_Ts = torch.stack([
                self.ER_calibrator.log_T,
                self.PR_calibrator.log_T,
                self.HER2_calibrator.log_T,
            ])


            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            # loss += 0.01 * log_Ts.sum()**2  # small L2 penalty
            loss.backward()
            return loss

        optimizer.step(closure)

        print("Fitted temperatures:",
              f"ER={self.ER_calibrator.T.item():.3f}, "
              f"PR={self.PR_calibrator.T.item():.3f}, "
              f"HER2={self.HER2_calibrator.T.item():.3f}")

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities."""
        with torch.no_grad():
            scaled_logits = torch.stack([
                self.ER_calibrator(logits[:, 0]),
                self.PR_calibrator(logits[:, 1]),
                self.HER2_calibrator(logits[:, 2]),
            ], dim=1)
            return torch.sigmoid(scaled_logits)