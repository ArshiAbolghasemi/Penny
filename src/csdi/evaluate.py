"""Test-set evaluation for the CSDI forecaster (spec section 9).

Reports accuracy, macro F1, confusion matrix, trend-ratio Pearson correlation,
and mid-price MAE.  No spread Wasserstein (this approach forecasts the mid, not
the full book).
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from . import labels as lab


@torch.no_grad()
def run_test(model, dataset, config, alpha, device) -> dict:
    model.eval()
    t_past, k = config["T_past"], config["label_k"]
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_true, y_pred, l_true, l_pred, maes = [], [], [], [], []
    for batch in loader:
        true_mid = batch["true_mid"].to(device).float()
        boundary = true_mid[:, t_past - 1]
        pred_ret = model.forecast(batch, device)
        fut_mid = boundary.view(-1, 1) * (1.0 + pred_ret)
        bwd = batch["bwd_smoothed"].to(device).float()
        l_vals = ((fut_mid[:, :k].mean(dim=1) - bwd) / (bwd + 1e-12)).cpu().numpy()
        mean_mid = fut_mid.cpu().numpy()
        true_future = true_mid[:, t_past : t_past + k].cpu().numpy()
        for j in range(len(l_vals)):
            y_true.append(int(batch["label"][j]))
            y_pred.append(lab.label_from_l(float(l_vals[j]), alpha))
            l_true.append(float(batch["l"][j]))
            l_pred.append(float(l_vals[j]))
            maes.append(float(np.mean(np.abs(mean_mid[j, :k] - true_future[j]))))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    corr = float(pearsonr(l_true, l_pred)[0]) if np.std(l_pred) > 0 else 0.0
    mae = float(np.mean(maes))
    logger.info("TEST accuracy={:.4f} macro_f1={:.4f}", acc, f1)
    logger.info("TEST confusion (rows=true down/stat/up):\n{}", cm)
    logger.info("TEST trend-ratio Pearson r={:.4f}", corr)
    logger.info("TEST mid MAE (first {} steps, IRT)={:.2f}", k, mae)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion": cm,
        "trend_corr": corr,
        "mid_mae": mae,
    }
