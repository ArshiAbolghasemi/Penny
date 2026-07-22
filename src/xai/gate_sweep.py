"""How does JumpGateLOB's trunk re-weight itself as the noise level rises?

Model-specific by construction: this probe exists only because the architecture
has adaLN-Zero gates conditioned on a diffusion timestep ``t``.  CTABL and DLA
have no analogue, so this is not part of the three-way comparison — it is the
reading that the diffusion trunk affords and the discriminative baselines do not.

Two sweeps, which answer different questions and must not be conflated:

* **gate** — feed a *clean* window and vary only the ``t`` conditioning.  The
  ``adaLN-Zero`` gates (``gate_attn``/``gate_mlp`` in ``TemporalAttnBlock``, and
  ``ga``/``gm`` scaling each residual write) are produced by ``ada(c)`` from the
  timestep embedding alone, so their magnitude *is* how much the block writes
  into the residual stream at that noise level.  This is a statement about the
  trunk's learned noise schedule, not about accuracy.

* **robust** — noise the window at ``t`` with the real Lévy jump-diffusion
  forward process, then classify.  This reproduces the deployment path, and so
  measures whether the ``L_robust`` term actually bought noise tolerance.

The ``t=0`` conditioning subtlety matters and is why the two are separate.
``train_jumpgatelob`` classifies *even noised* windows at ``t = 0`` — "deployment
never knows the noise level" — so ``classify(x, t=...)`` with ``t > 0`` is an
**analysis-only** counterfactual that never occurs in training or inference.  The
robust sweep therefore reports both: ``t=0`` conditioning (the honest deployment
path) and oracle ``t`` conditioning (what the trunk *could* do if it were told
the noise level).  The gap between them is the value of information the deployed
model deliberately forgoes.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from crypto.train_jumpgatelob import _diffusion_cfg
from levy.diffusion.forward import ForwardProcess


def build_forward_process(config: dict, device: torch.device) -> ForwardProcess:
    """The same jump-diffusion forward process the model was trained against.

    Built from the checkpoint's own config, so the swept ``t`` indexes the
    schedule the trunk actually saw rather than a re-invented one.
    """
    d = 1 * config["T_past"] * config["n_features"]
    return ForwardProcess(_diffusion_cfg(config), d=d, device=device)


def sweep_timesteps(t_max: int, n_points: int) -> np.ndarray:
    """Timesteps to sweep, always including 0 and ``t_max - 1``."""
    return np.unique(np.linspace(0, t_max - 1, n_points).round().astype(int))


@torch.no_grad()
def gate_sweep(
    model,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    timesteps: np.ndarray,
    batch_size: int | None = None,
) -> dict:
    """Gate magnitudes vs ``t`` on clean windows.

    Returns ``{"t", "gate_attn", "gate_mlp"}`` — mean |gate| per timestep, i.e.
    how strongly the attention and MLP branches write into the residual stream at
    each noise level.
    """
    model.eval()
    bs = batch_size or config.get("batch_size", 64)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=bs, shuffle=False)
    ga_rows, gm_rows = [], []
    for t_val in timesteps:
        ga_sum = gm_sum = 0.0
        n = 0
        for batch in loader:
            x = batch["x"].to(device).float()
            t = torch.full((x.shape[0],), int(t_val), dtype=torch.long, device=device)
            _, attn = model.classify(x, return_attn=True, t=t)
            ga_sum += float(attn["gate_attn"].abs().mean()) * x.shape[0]
            gm_sum += float(attn["gate_mlp"].abs().mean()) * x.shape[0]
            n += x.shape[0]
        ga_rows.append(ga_sum / max(n, 1))
        gm_rows.append(gm_sum / max(n, 1))
    return {
        "t": timesteps.tolist(),
        "gate_attn": ga_rows,
        "gate_mlp": gm_rows,
    }


@torch.no_grad()
def robustness_sweep(
    model,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    timesteps: np.ndarray,
    fp: ForwardProcess,
    batch_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Accuracy on jump-noised windows vs ``t``, at deployment and oracle conditioning.

    Returns ``{"t", "accuracy_t0", "accuracy_oracle"}`` — the first is the real
    inference path (always ``t=0`` conditioning, as trained); the second tells the
    trunk the true noise level, which deployment never knows.
    """
    model.eval()
    bs = batch_size or config.get("batch_size", 64)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=bs, shuffle=False)
    acc0, acc_or = [], []
    for t_val in timesteps:
        c0 = cor = n = 0
        torch.manual_seed(seed)  # same noise draw across timesteps and conditionings
        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["label"].to(device)
            t = torch.full((x.shape[0],), int(t_val), dtype=torch.long, device=device)
            x_t, _ = fp.add_noise(x, t)
            c0 += int((model.classify(x_t).argmax(1) == y).sum())
            cor += int((model.classify(x_t, t=t).argmax(1) == y).sum())
            n += int(y.numel())
        acc0.append(c0 / max(n, 1))
        acc_or.append(cor / max(n, 1))
    return {
        "t": timesteps.tolist(),
        "accuracy_t0": acc0,
        "accuracy_oracle": acc_or,
    }


def low_t_boundary(fp: ForwardProcess) -> int:
    """Largest ``t`` in the SNR>=1 region the robust loss trained on (VP: abar_t >= 0.5).

    Accuracy inside this band is what ``L_robust`` optimised; beyond it the model
    is extrapolating, so the two regimes should not be read as one curve.
    """
    if fp.schedule.kind == "vp":
        mask = (fp.schedule.a**2) >= 0.5
    else:
        mask = fp.schedule.sigma < 1.0
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    return int(idx.max()) if len(idx) else 0


def format_table(gates: dict, robust: dict, boundary: int) -> str:
    """Render both sweeps as one table; ``*`` marks the trained low-t region."""
    head = (
        f"{'t':>6} {'':>1} {'gate_attn':>10} {'gate_mlp':>10} "
        f"{'acc(t=0)':>9} {'acc(oracle)':>12}"
    )
    lines = [head, "-" * len(head)]
    for i, t in enumerate(gates["t"]):
        mark = "*" if t <= boundary else " "
        lines.append(
            f"{t:>6} {mark:>1} {gates['gate_attn'][i]:>10.4f} "
            f"{gates['gate_mlp'][i]:>10.4f} {robust['accuracy_t0'][i]:>9.4f} "
            f"{robust['accuracy_oracle'][i]:>12.4f}"
        )
    lines.append(f"(* = SNR>=1 region the robust loss trained on, t <= {boundary})")
    return "\n".join(lines)


def log_sweep(gates: dict, robust: dict, boundary: int) -> None:
    logger.info("JumpGateLOB t-sweep\n{}", format_table(gates, robust, boundary))
