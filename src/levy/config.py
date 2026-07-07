"""Configuration for JointDiffusion-Lévy.

A single nested dataclass drives every component so the pipeline is reproducible
from one JSON file.  Load with :func:`load_config` (JSON overrides the defaults);
each sub-config is consumed by exactly one subpackage.

TODO(user): several fields below encode *design choices* that affect the physics
of the forward process and the label definition.  They are marked ``# TODO`` and
given defensible defaults so the pipeline runs, but you should confirm them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    # source: "synthetic" for smoke tests, or a real exchange adapter
    source: str = "synthetic"  # "synthetic" | "binance" | "nobitex"
    symbol: str = "BTCUSDT"  # e.g. BTCUSDT (binance) / BTCIRT (nobitex)
    data_dir: str = "data/resampled"  # parquet root for real adapters
    n_levels: int = 10  # LOB price levels per side
    seq_len: int = 100  # timesteps per window
    # feature dim per timestep is derived by the data module; written back into
    # the config at build time as ``n_features``.
    n_features: int = 0
    horizons: tuple[int, ...] = (10, 20, 50, 100)  # trend-label lookaheads
    # TODO(user): trend threshold. label = up/flat/down by comparing the smoothed
    # future mid return against +/- alpha.  ``None`` -> calibrate per horizon on
    # the training split so the three classes are roughly balanced.
    label_alpha: float | None = None
    stride: int = 1
    train_frac: float = 0.7
    val_frac: float = 0.15
    # synthetic generator knobs (used only when source == "synthetic")
    n_samples: int = 20000
    synth_jump_rate: float = 0.02  # mid-price jump probability per step
    synth_seed: int = 0


@dataclass
class DiffusionConfig:
    process: str = "levy"  # ablation toggle: "gaussian" | "levy"
    schedule: str = "vp"  # "vp" (DDPM) | "ve"
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    # VE schedule bounds (used when schedule == "ve")
    sigma_min: float = 1e-2
    sigma_max: float = 50.0
    # --- Lévy jump-diffusion parameters (used when process == "levy") ---
    # Compound-Poisson rate.  TODO(user): interpretation of lambda as jumps per
    # unit diffusion-time; here Lambda_t = jump_rate * (t+1)/num_timesteps.
    jump_rate: float = 1.0
    # Generalized-Laplace (subordinated-Gaussian) jump amplitude:
    #   z = sqrt(S) * xi,  xi ~ N(0, I_d),  S ~ Gamma(shape, scale)
    # finite variance E[|z_i|^2] = shape*scale.  TODO(user): tune tail heaviness.
    jump_gamma_shape: float = 1.0
    jump_gamma_scale: float = 1.0
    # generalized-score table resolution (precomputed offline)
    table_num_r: int = 512
    table_mc_samples: int = 20000  # MC draws of W per table entry
    table_seed: int = 0


@dataclass
class BackboneConfig:
    base_channels: int = 32
    depth: int = 3
    time_emb_dim: int = 128
    conditioning: str = "adaln"  # "adaln" | "film"
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 500
    # loss balancing
    use_uncertainty_weighting: bool = True  # Kendall-Gal
    use_pcgrad: bool = True  # gradient surgery
    lambda_trend: float = 1.0  # fallback fixed weight if uncertainty off
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "checkpoints/levy"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, overrides: dict[str, Any]) -> Any:
    """Recursively overlay a plain dict onto a dataclass instance."""
    if not is_dataclass(dc):
        return overrides
    valid = {f.name: f for f in fields(dc)}
    for key, val in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config key '{key}' for {type(dc).__name__}")
        cur = getattr(dc, key)
        if is_dataclass(cur) and isinstance(val, dict):
            setattr(dc, key, _merge(cur, val))
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(dc, key, tuple(val))
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config`, overlaying a JSON file and/or keyword overrides."""
    cfg = Config()
    if path is not None:
        data = json.loads(Path(path).read_text())
        cfg = _merge(cfg, data)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
