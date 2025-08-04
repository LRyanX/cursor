from typing import Dict
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


def platt_scaling(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling for a single task."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels)
    return lr


def calibrate(tasks_logits: Dict[str, torch.Tensor], tasks_labels: Dict[str, torch.Tensor]):
    calibrators: Dict[str, LogisticRegression] = {}
    for task, logit in tasks_logits.items():
        calibrators[task] = platt_scaling(logit.detach().cpu().numpy(), tasks_labels[task].cpu().numpy())
    return calibrators


def apply_calibration(calibrators: Dict[str, LogisticRegression], logits: Dict[str, torch.Tensor]):
    calibrated = {}
    for task, logit in logits.items():
        probs = calibrators[task].predict_proba(logit.detach().cpu().numpy().reshape(-1, 1))[:, 1]
        calibrated[task] = torch.from_numpy(probs).to(logit.device)
    return calibrated