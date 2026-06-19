"""Test-set evaluation for the painting approach (spec section 9).

Runs the DDIM+RePaint sampler over every test window (``n_samples`` each) and
reports accuracy, macro F1, confusion, trend-ratio correlation, mid MAE, and —
for ``lob`` mode — the bid-ask spread Wasserstein distance.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import confusion_matrix, f1_score

from . import labels as lab


@torch.no_grad()
def _spread(model, painted, dataset, idx, config):
    n, t_past = config["n_levels"], config["T_past"]
    ls = model.level_starts
    bb = painted[:, 0, ls[n - 1] : ls[n], t_past:].mean(dim=1).mean(dim=1)
    ba = painted[:, 0, ls[n] : ls[n + 1], t_past:].mean(dim=1).mean(dim=1)
    bb = model.norm.denorm_channel0(bb.cpu().numpy(), n - 1)
    ba = model.norm.denorm_channel0(ba.cpu().numpy(), n)
    s = int(dataset.starts[idx])
    fut = slice(s + t_past, s + config["T_total"])
    rbb = model.norm.denorm_channel0(dataset.rows[fut, n - 1, 0], n - 1)
    rba = model.norm.denorm_channel0(dataset.rows[fut, n, 0], n)
    return (ba - bb).mean(), (rba - rbb).mean()


@torch.no_grad()
def run_test(model, diffusion, dataset, config, gamma, alpha, device) -> dict:
    model.eval()
    k, t_past = config["label_k"], config["T_past"]
    is_lob = config["feature_mode"] == "lob"
    y_true, y_pred, l_true, l_pred, maes = [], [], [], [], []
    p_spreads, r_spreads = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        ns = config["n_samples"]
        x0_known = s["image"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        m = s["mask"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        painted = diffusion.sample(model, x0_known, m, config["ddim_steps"], device)
        batch = {"mid_ref": torch.full((ns,), float(s["mid_ref"]))}
        fut_mid = model.future_mid(painted, batch, gamma)
        mean_mid = fut_mid.mean(dim=0).cpu().numpy()
        fwd = fut_mid[:, :k].mean(dim=1).cpu().numpy()
        bwd = float(s["bwd_smoothed"])
        l_vals = (fwd - bwd) / (bwd + 1e-12)
        modal = int(
            np.bincount(
                [lab.label_from_l(float(x), alpha) for x in l_vals], minlength=3
            ).argmax()
        )
        y_true.append(s["label"])
        y_pred.append(modal)
        l_true.append(s["l"])
        l_pred.append(float(l_vals.mean()))
        true_future = s["true_mid"].numpy()[t_past : t_past + k]
        maes.append(float(np.mean(np.abs(mean_mid[:k] - true_future))))
        if is_lob:
            ps, rs = _spread(model, painted, dataset, i, config)
            p_spreads.append(ps)
            r_spreads.append(rs)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    corr = float(pearsonr(l_true, l_pred)[0]) if np.std(l_pred) > 0 else 0.0
    mae = float(np.mean(maes))
    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion": cm,
        "trend_corr": corr,
        "mid_mae": mae,
    }
    logger.info("TEST accuracy={:.4f} macro_f1={:.4f}", acc, f1)
    logger.info("TEST confusion (rows=true down/stat/up):\n{}", cm)
    logger.info("TEST trend-ratio Pearson r={:.4f}", corr)
    logger.info("TEST mid MAE (first {} steps, IRT)={:.2f}", k, mae)
    if is_lob and p_spreads:
        w = float(wasserstein_distance(np.array(p_spreads), np.array(r_spreads)))
        metrics["spread_wasserstein"] = w
        logger.info("TEST spread Wasserstein={:.4f}", w)
    else:
        metrics["spread_wasserstein"] = None
        logger.info("TEST spread Wasserstein: N/A (ofi mode)")
    return metrics
