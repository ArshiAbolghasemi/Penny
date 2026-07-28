"""Calibrate AlphaStableLOB/JumpGateLOB corruption-kernel hyperparameters per altcoin.

Both models currently train with a single global kernel shared across every
Coinbase market: ``astable_alpha=1.5`` (AlphaStableLOB) and ``levy_jump_rate=1.0``
(JumpGateLOB) — untouched defaults, never fit to any symbol's actual tail/jump
behaviour (see ``src/levy/config.py``'s own TODO on ``jump_rate``). This script
estimates two symbol-specific, data-derived replacements for BNBIRT, ETHIRT,
XRPIRT, and USDTIRT (BTCIRT is left as the untouched reference point):

  astable_alpha  — the Hill tail-index alpha-hat of that symbol's own OFI
                   distribution: absolute per-level Cont-OFI, pooled across all
                   20 levels and every day, normalised by the symbol's own
                   median |OFI| (top-5% tail, ``hill_at``) — this reproduces the
                   exact procedure behind the paper's headline "OFI inputs are
                   heavy-tailed" claim, so the estimate targets the same
                   quantity AlphaStableLOB's forward process actually corrupts
                   (the OFI block of the feature vector), not a price-return
                   series it never touches. Clamped to AlphaStableLOB's valid
                   range (0, 2]. The alpha-stable stability index is
                   scale-invariant, so this transfers to the z-scored feature
                   space with no further unit-conversion.

  levy_jump_rate — set so the forward process's *terminal* (highest-noise)
                   expected jump count, Lambda_{t=T_max} = jump_rate (see
                   ``src/levy/diffusion/generalized_score.py::jump_intensity``),
                   matches the empirically observed number of Lee-Mykland-
                   flagged jumps within a real T_past-length window of that
                   symbol's own MID-PRICE RETURN history: jump_rate = T_past *
                   lm_jump_fraction. Lee-Mykland/BNS are derived for price
                   (semimartingale) processes, not order-flow imbalance, so
                   jump timing is estimated on mid-returns even though tail
                   shape is estimated on OFI — each statistic uses the series
                   its underlying theory actually applies to. The resulting
                   rate is a frequency (dimensionless), unit-safe to transfer.

Deliberately NOT recalibrated here: ``levy_gamma_shape``/``levy_gamma_scale``
(jump *size*, an absolute magnitude in raw log-return units — transferring that
to the z-scored/clipped feature space needs an explicit rescaling this script
does not attempt, so those two stay at their existing defaults).

Reuses the exact Hill/Lee-Mykland/BNS implementations already validated in
``notebooks/increment_diagnostics.ipynb`` (same functions, copied verbatim) so
these numbers are consistent with the tail estimates already reported.

Usage::

    uv run python scripts/calibrate_altcoin_kernels.py            # dry run, prints only
    uv run python scripts/calibrate_altcoin_kernels.py --apply     # also writes configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from crypto.features import extract_features  # noqa: E402

CRYPTO_DIR = REPO / "data" / "resampled" / "coinbase"
ALPHASTABLE_CFG_DIR = REPO / "configs" / "crypto" / "coinbase" / "alphastablelob"
JUMPGATE_CFG_DIR = REPO / "configs" / "crypto" / "coinbase" / "jumpgatelob"

ALTCOINS = [
    "BNBIRT",
    "ETHIRT",
    "XRPIRT",
    "USDTIRT",
    "BTCIRT",
]  # BTCIRT left as reference
HORIZONS = [10, 20, 50, 100]
T_PAST = (
    60  # confirmed identical across every altcoin jumpgatelob/alphastablelob config
)
LM_SIG = (
    0.01  # Lee-Mykland significance level (matches increment_diagnostics.ipynb default)
)

ALPHA_MIN, ALPHA_MAX = (
    1.1,
    1.9,
)  # stay clear of the (0,2] boundary; model hard-errors outside
JUMP_RATE_MIN, JUMP_RATE_MAX = (
    0.05,
    20.0,
)  # sane guardrail against a pathological estimate

# ── diagnostic toolkit, copied verbatim from notebooks/increment_diagnostics.ipynb ──
MU1 = np.sqrt(2.0 / np.pi)


def hill_estimator(x, ks=None):
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 50:
        return np.array([]), np.array([]), np.array([])
    if ks is None:
        ks = np.unique(np.linspace(10, int(0.15 * n), 80).astype(int))
        ks = ks[ks < n]
    la = np.log(a)
    alpha, se = [], []
    for k in ks:
        Hk = la[:k].mean() - la[k]
        al = 1.0 / Hk if Hk > 0 else np.nan
        alpha.append(al)
        se.append(al / np.sqrt(k) if np.isfinite(al) else np.nan)
    return np.asarray(ks), np.asarray(alpha), np.asarray(se)


def hill_at(x, frac=0.05):
    ks, al, se = hill_estimator(x)
    if len(ks) == 0:
        return np.nan, np.nan, np.nan
    n = np.sum(np.isfinite(x) & (np.asarray(x, float) != 0))
    ktarget = max(10, int(frac * n))
    j = int(np.argmin(np.abs(ks - ktarget)))
    return float(al[j]), float(al[j] - 1.96 * se[j]), float(al[j] + 1.96 * se[j])


def lee_mykland(r, K=None, sig=0.01):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if K is None:
        K = int(np.clip(np.sqrt(n) * 3, 15, 270))
    if n < K + 5:
        return 0, 0
    ar = np.abs(r)
    bp = ar[1:] * ar[:-1]
    csum = np.concatenate([[0.0], np.cumsum(bp)])
    L = np.full(n, np.nan)
    for i in range(K, n):
        sig2 = (csum[i] - csum[i - (K - 1)]) / (K - 2)
        if sig2 > 0:
            L[i] = r[i] / (MU1 * np.sqrt(sig2))
    tested = np.isfinite(L)
    m = int(tested.sum())
    if m < 5:
        return 0, 0
    c = np.sqrt(2.0 * np.log(m))
    Cn = c - (np.log(np.pi) + np.log(np.log(m))) / (2.0 * c)
    Sn = 1.0 / c
    thresh = Cn - Sn * np.log(-np.log(1.0 - sig))
    return int(np.sum(np.abs(L[tested]) > thresh)), m


def crypto_increments(symbol: str):
    """Mid-price log-return increments, segmented by calendar day (matches
    increment_diagnostics.ipynb's crypto_increments)."""
    p = CRYPTO_DIR / f"{symbol}.parquet.gz"
    df = pd.read_parquet(p, columns=["timestamp_utc", "bids[0].price", "asks[0].price"])
    mid = (df["bids[0].price"] + df["asks[0].price"]).to_numpy(float) / 2.0
    day = pd.to_datetime(df["timestamp_utc"]).dt.normalize().to_numpy()
    ok = np.isfinite(mid) & (mid > 0)
    logm, day = np.log(mid[ok]), day[ok]
    segments, pooled = [], []
    for d in np.unique(day):
        s = logm[day == d]
        if len(s) > 2:
            r = np.diff(s)
            segments.append(r)
            pooled.append(r)
    return (np.concatenate(pooled) if pooled else np.array([])), segments


OFI_FEATURE_CONFIG = {
    "n_lob_levels": 20,
    "feature_mode": "ofi",
}  # matches training configs


def ofi_pooled(symbol: str) -> np.ndarray:
    """Absolute per-level Cont-OFI, pooled across levels and days, normalised by
    that symbol's own median |OFI| — reproduces the exact procedure behind the
    paper's headline tail claim ("OFI inputs are heavy-tailed"), so the
    astable_alpha calibration targets the same quantity AlphaStableLOB's forward
    process actually corrupts (the OFI block of the feature vector), not a
    price-return series it never touches directly."""
    p = CRYPTO_DIR / f"{symbol}.parquet.gz"
    n = OFI_FEATURE_CONFIG["n_lob_levels"]
    cols = (
        ["timestamp_utc"]
        + [f"bids[{i}].price" for i in range(n)]
        + [f"bids[{i}].amount" for i in range(n)]
        + [f"asks[{i}].price" for i in range(n)]
        + [f"asks[{i}].amount" for i in range(n)]
        + ["mid", "spread", "trade_count", "buy_vol", "sell_vol", "vwap"]
    )
    df = pd.read_parquet(p, columns=cols)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    day = pd.to_datetime(df["timestamp_utc"]).dt.date
    blocks = []
    for _, day_df in df.groupby(day, sort=True):
        day_df = day_df.reset_index(drop=True)
        feat = extract_features(day_df, OFI_FEATURE_CONFIG)  # (N_day, n+11)
        blocks.append(np.abs(feat[:, :n]))  # raw per-level OFI columns only
    ofi_abs = np.concatenate(blocks, axis=0).reshape(-1)
    ofi_abs = ofi_abs[np.isfinite(ofi_abs)]
    median = np.median(ofi_abs[ofi_abs > 0]) if np.any(ofi_abs > 0) else 1.0
    return ofi_abs / median


def calibrate(symbol: str, sig: float = LM_SIG) -> dict:
    ofi_abs = ofi_pooled(symbol)
    alpha_raw, alpha_lo, alpha_hi = hill_at(ofi_abs, frac=0.05)
    alpha = float(np.clip(alpha_raw, ALPHA_MIN, ALPHA_MAX))

    pooled, segments = crypto_increments(symbol)  # mid-return series, for jump timing
    lm_j = lm_t = 0
    for seg in segments:
        j, t = lee_mykland(seg, sig=sig)
        lm_j += j
        lm_t += t
    lm_frac = lm_j / lm_t if lm_t else np.nan
    jump_rate_raw = T_PAST * lm_frac if np.isfinite(lm_frac) else np.nan
    jump_rate = float(np.clip(jump_rate_raw, JUMP_RATE_MIN, JUMP_RATE_MAX))

    return {
        "symbol": symbol,
        "n_increments": len(pooled),
        "n_ofi_obs": len(ofi_abs),
        "n_sessions": len(segments),
        "hill_alpha_raw": alpha_raw,
        "hill_alpha_ci": (alpha_lo, alpha_hi),
        "astable_alpha": round(alpha, 3),
        "lm_jumps": lm_j,
        "lm_tested": lm_t,
        "lm_jump_frac_pct": 100 * lm_frac if np.isfinite(lm_frac) else np.nan,
        "jump_rate_raw": jump_rate_raw,
        "levy_jump_rate": round(jump_rate, 3),
    }


def patch_config(path: Path, updates: dict) -> bool:
    cfg = json.loads(path.read_text())
    changed = {k: (cfg.get(k), v) for k, v in updates.items() if cfg.get(k) != v}
    if not changed:
        return False
    cfg.update(updates)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply", action="store_true", help="write the config files (default: dry run)"
    )
    ap.add_argument(
        "--sig", type=float, default=LM_SIG, help="Lee-Mykland significance level"
    )
    args = ap.parse_args()

    results = [calibrate(sym, sig=args.sig) for sym in ALTCOINS]

    print(
        f"\n{'symbol':<9}{'n_incr':>10}{'sessions':>9}  "
        f"{'hill_a':>7}{'[lo,hi]':>16}  ->  {'astable_alpha':>13}   "
        f"{'lm_jump%':>9}  ->  {'levy_jump_rate':>14}"
    )
    for r in results:
        lo, hi = r["hill_alpha_ci"]
        print(
            f"{r['symbol']:<9}{r['n_increments']:>10,}{r['n_sessions']:>9}  "
            f"{r['hill_alpha_raw']:>7.3f}  [{lo:>5.2f},{hi:>5.2f}]  ->  "
            f"{r['astable_alpha']:>13.3f}   "
            f"{r['lm_jump_frac_pct']:>8.4f}%  ->  {r['levy_jump_rate']:>14.3f}"
        )
    print("\n(reference, unchanged) BTCIRT: astable_alpha=1.5  levy_jump_rate=1.0")
    print(
        f"\nsig={args.sig}  T_past={T_PAST}  alpha clamp=[{ALPHA_MIN},{ALPHA_MAX}]  "
        f"jump_rate clamp=[{JUMP_RATE_MIN},{JUMP_RATE_MAX}]"
    )

    if not args.apply:
        print(
            "\n(dry run — pass --apply to write configs/crypto/coinbase/{alphastablelob,jumpgatelob}/*.json)"
        )
        return

    n_astable = n_jump = 0
    for r in results:
        sym_lower = r["symbol"].lower()
        for k in HORIZONS:
            astable_path = ALPHASTABLE_CFG_DIR / f"{sym_lower}_ofi_k{k}.json"
            if astable_path.exists() and patch_config(
                astable_path, {"astable_alpha": r["astable_alpha"]}
            ):
                n_astable += 1
            jumpgate_path = JUMPGATE_CFG_DIR / f"{sym_lower}_ofi_k{k}.json"
            if jumpgate_path.exists() and patch_config(
                jumpgate_path, {"levy_jump_rate": r["levy_jump_rate"]}
            ):
                n_jump += 1
    print(f"\npatched {n_astable} alphastablelob configs, {n_jump} jumpgatelob configs")


if __name__ == "__main__":
    main()
