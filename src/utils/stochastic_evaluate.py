"""Evaluation and regime diagnostics for inference-time stochastic forecasts."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


def _safe_detection_metrics(errors: np.ndarray, uncertainty: np.ndarray) -> dict:
    if np.unique(errors).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(errors, uncertainty)),
        "auprc": float(average_precision_score(errors, uncertainty)),
    }


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += (
                mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
            )
    return float(ece)


def risk_coverage(errors: np.ndarray, uncertainty: np.ndarray) -> dict:
    """Risk after retaining the least-uncertain fraction of predictions."""
    order = np.argsort(uncertainty)
    ranked = errors[order]
    n = len(ranked)
    coverage = np.linspace(0.05, 1.0, 20)
    risk = [float(ranked[: max(1, int(n * c))].mean()) for c in coverage]
    return {"coverage": coverage.tolist(), "risk": risk}


def _window_statistics(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream = x.squeeze(1).mean(dim=-1)
    delta = stream[:, 1:] - stream[:, :-1]
    variance = delta.var(dim=1).cpu().numpy()
    maximum = delta.abs().max(dim=1).values.cpu().numpy()
    stress = maximum / np.sqrt(variance + 1e-12)
    return stress, variance, maximum


def _jump_residual(variance: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    xv, ym = np.log(variance + 1e-12), np.log(maximum + 1e-12)
    if len(xv) < 3 or np.std(xv) < 1e-12:
        return ym - ym.mean()
    slope, intercept = np.polyfit(xv, ym, 1)
    return ym - (slope * xv + intercept)


def _regime_bins(
    score: np.ndarray, alpha_route: np.ndarray, jump_route: np.ndarray, rate: np.ndarray
) -> list[dict]:
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, 11))
    rows = []
    for index in range(10):
        mask = (score >= quantiles[index]) & (
            score <= quantiles[index + 1]
            if index == 9
            else score < quantiles[index + 1]
        )
        rows.append(
            {
                "bin": index + 1,
                "count": int(mask.sum()),
                "pi_alpha": float(alpha_route[mask].mean())
                if mask.any()
                else float("nan"),
                "pi_jump": float(jump_route[mask].mean())
                if mask.any()
                else float("nan"),
                "jump_intensity": float(rate[mask].mean())
                if mask.any()
                else float("nan"),
            }
        )
    return rows


@torch.no_grad()
def _mc_dropout_disagreement(
    model, batch: dict, device: torch.device, samples: int
) -> torch.Tensor:
    """Pure dropout disagreement with latent process noise temporarily disabled."""
    if samples < 2:
        return torch.zeros(len(batch["label"]))
    was_training = model.training
    dynamics_mode = model.dynamics.mode
    model.train()
    model.dynamics.mode = "deterministic"
    predictions = []
    try:
        x = batch["x"].to(device).float()
        horizon = batch.get("horizon", model.default_horizon)
        for _ in range(samples):
            predictions.append(
                model(x, horizon, trajectories=1).probabilities.squeeze(0)
            )
    finally:
        model.dynamics.mode = dynamics_mode
        model.train(was_training)
    p = torch.stack(predictions)
    return ((p - p.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean(dim=0).cpu()


@torch.no_grad()
def evaluate_stochastic(
    model,
    dataset,
    config: dict,
    device: torch.device,
    trajectories: int | None = None,
    include_mc_dropout: bool = True,
) -> dict:
    """Evaluate accuracy, calibration, disagreement, routing and market regimes."""
    model.eval()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    labels, probabilities, entropies, disagreements = [], [], [], []
    routes, rates, stress, variance, maximum, horizons = [], [], [], [], [], []
    dropout_disagreements = []

    for batch in loader:
        out = model.predict_distribution(batch, device, trajectories)
        probs = out.mean_probability.cpu()
        probabilities.append(probs)
        labels.append(batch["label"].cpu())
        entropies.append(out.predictive_entropy.cpu())
        disagreements.append(out.trajectory_disagreement.cpu())
        routes.append(out.routing.mean(dim=(0, 1)).cpu())
        rates.append(out.jump_intensity.mean(dim=(0, 1, 3)).cpu())
        s, v, m = _window_statistics(batch["x"])
        stress.append(s)
        variance.append(v)
        maximum.append(m)
        horizon = batch.get("horizon")
        if horizon is None:
            horizon = torch.full_like(batch["label"], model.default_horizon)
        horizons.append(horizon.cpu())
        if include_mc_dropout:
            dropout_disagreements.append(
                _mc_dropout_disagreement(
                    model, batch, device, int(config.get("mc_dropout_samples", 20))
                )
            )

    y = torch.cat(labels).numpy()
    p = torch.cat(probabilities).numpy()
    entropy = torch.cat(entropies).numpy()
    disagreement = torch.cat(disagreements).numpy()
    route = torch.cat(routes).numpy()
    rate = torch.cat(rates).numpy()
    horizon = torch.cat(horizons).numpy()
    stress_np = np.concatenate(stress)
    variance_np = np.concatenate(variance)
    maximum_np = np.concatenate(maximum)
    jump_np = _jump_residual(variance_np, maximum_np)
    pred = p.argmax(axis=1)
    errors = (pred != y).astype(np.int64)
    dropout_disagreement = (
        torch.cat(dropout_disagreements).numpy() if dropout_disagreements else None
    )

    result = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "ece": expected_calibration_error(p, y),
        "brier": float(((p - np.eye(3)[y]) ** 2).sum(axis=1).mean()),
        "mean_predictive_entropy": float(entropy.mean()),
        "mean_trajectory_disagreement": float(disagreement.mean()),
        "mean_pi_alpha": float(route[:, 0].mean()),
        "mean_pi_jump": float(route[:, 1].mean()),
        "mean_jump_intensity": float(rate.mean()),
        "error_detection": {
            "one_minus_max_softmax": _safe_detection_metrics(
                errors, 1.0 - p.max(axis=1)
            ),
            "predictive_entropy": _safe_detection_metrics(errors, entropy),
            "trajectory_disagreement": _safe_detection_metrics(errors, disagreement),
        },
        "risk_coverage": {
            "predictive_entropy": risk_coverage(errors, entropy),
            "trajectory_disagreement": risk_coverage(errors, disagreement),
        },
        "regimes": {
            "stress": _regime_bins(stress_np, route[:, 0], route[:, 1], rate),
            "realized_variance": _regime_bins(
                variance_np, route[:, 0], route[:, 1], rate
            ),
            "jump_residual": _regime_bins(jump_np, route[:, 0], route[:, 1], rate),
        },
        "by_horizon": {},
    }
    if dropout_disagreement is not None:
        result["mean_mc_dropout_disagreement"] = float(dropout_disagreement.mean())
        result["error_detection"]["mc_dropout_disagreement"] = _safe_detection_metrics(
            errors, dropout_disagreement
        )
        result["risk_coverage"]["mc_dropout_disagreement"] = risk_coverage(
            errors, dropout_disagreement
        )
    for k in sorted(np.unique(horizon)):
        mask = horizon == k
        result["by_horizon"][str(int(k))] = {
            "count": int(mask.sum()),
            "accuracy": float((pred[mask] == y[mask]).mean()),
            "macro_f1": float(
                f1_score(
                    y[mask],
                    pred[mask],
                    average="macro",
                    labels=[0, 1, 2],
                    zero_division=0,
                )
            ),
            "trajectory_disagreement": float(disagreement[mask].mean()),
            "predictive_entropy": float(entropy[mask].mean()),
            "pi_alpha": float(route[mask, 0].mean()),
            "pi_jump": float(route[mask, 1].mean()),
            "jump_intensity": float(rate[mask].mean()),
            "disagreement_error_detection": _safe_detection_metrics(
                errors[mask], disagreement[mask]
            ),
        }
    return result
