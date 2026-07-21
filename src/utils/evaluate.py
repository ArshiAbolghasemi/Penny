"""Shared test-set evaluation for all crypto model families.

Every model in this repo exposes ``model.predict(batch, device) → (B, 3) logits``
so the metrics (accuracy, macro-F1, confusion, mean probs) are identical across
DeepLOB, JointDiffusion, LOBTransformer, etc.  Import and use ``run_test`` from here
rather than defining it per-family.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def run_test(model, dataset, config, device) -> dict:
    """Evaluate *model* on *dataset* and log accuracy / F1 / confusion.

    Args:
        model:   Any model with ``predict(batch, device) → (B, 3) logits``.
        dataset: PyTorch Dataset; items must include a ``"label"`` key.
        config:  Config dict; must contain ``"batch_size"``.
        device:  ``torch.device``.

    Returns:
        ``{"accuracy": float, "macro_f1": float, "confusion": ndarray (3,3)}``.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    y_true, y_pred, probs_all = [], [], []
    for batch in loader:
        logits = model.predict(batch, device)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(batch["label"].tolist())
        y_pred.extend(preds.tolist())
        probs_all.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs_all = np.concatenate(probs_all, axis=0)

    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    logger.info("TEST  accuracy={:.4f}  macro_f1={:.4f}", acc, f1)
    logger.info("TEST  confusion (rows=true down/stat/up):\n{}", cm)
    logger.info(
        "TEST  mean probs  down={:.3f}  stat={:.3f}  up={:.3f}",
        probs_all[:, 0].mean(),
        probs_all[:, 1].mean(),
        probs_all[:, 2].mean(),
    )
    return {"accuracy": acc, "macro_f1": f1, "confusion": cm}
