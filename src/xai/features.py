"""Names and economic groupings for the model input features.

Attribution over a raw ``(T, F)`` grid is unreadable; the same numbers summed
into named blocks — best-level OFI, deep OFI, microstructure, trades — are a
claim someone can argue with.  The layout mirrors ``crypto/features.py`` exactly
(see ``docs/data/features.md``); ``feature_names`` is derived from the config
rather than hardcoded, so it stays correct for either ``feature_mode`` and any
``n_lob_levels``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 11-feature tail both modes append, in order.
_TAIL = [
    "spread/mid",
    "depth_imbalance",
    "log_return",
    "log_buy_vol",
    "log_sell_vol",
    "trade_imbalance",
    "log_trade_count",
    "vwap_dev",
    "activity",
    "spread_norm",
    "abs_log_return",
]


def feature_names(config: dict) -> list[str]:
    """Per-column names for the ``(T, F)`` input, in feature order."""
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    if mode == "ofi":
        head = [f"OFI_L{i + 1}" for i in range(n)]
    elif mode == "lob":
        head = (
            [f"bid_price_L{i + 1}" for i in range(n)]
            + [f"ask_price_L{i + 1}" for i in range(n)]
            + [f"bid_vol_L{i + 1}" for i in range(n)]
            + [f"ask_vol_L{i + 1}" for i in range(n)]
        )
    else:
        raise ValueError(f"feature_mode must be ofi|lob, got {mode!r}")
    return head + list(_TAIL)


@dataclass(frozen=True)
class FeatureGroups:
    """Named column blocks over the feature axis.

    ``groups`` maps a group name to the column indices it owns; every column
    belongs to exactly one group.
    """

    groups: dict[str, list[int]]

    @property
    def names(self) -> list[str]:
        return list(self.groups)

    def __iter__(self):
        return iter(self.groups.items())


def feature_groups(config: dict, best_levels: int = 3) -> FeatureGroups:
    """Group the feature columns into economically meaningful blocks.

    The LOB block is split at ``best_levels`` because Cont's order-flow-imbalance
    theory predicts price impact concentrates at the top of the book: keeping
    "best" and "deep" apart is what makes that prediction testable rather than
    decorative.

    Args:
        config:      Model config; reads ``n_lob_levels`` and ``feature_mode``.
        best_levels: How many top levels count as "best" (default 3).

    Returns:
        A :class:`FeatureGroups` whose indices tile ``range(n_features)``.
    """
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    b = min(best_levels, n)
    groups: dict[str, list[int]] = {}

    if mode == "ofi":
        groups["OFI_best"] = list(range(0, b))
        if b < n:
            groups["OFI_deep"] = list(range(b, n))
        off = n
    else:
        # prices then volumes, each split best/deep across bid+ask
        groups["price_best"] = list(range(0, b)) + list(range(n, n + b))
        if b < n:
            groups["price_deep"] = list(range(b, n)) + list(range(n + b, 2 * n))
        groups["volume_best"] = list(range(2 * n, 2 * n + b)) + list(
            range(3 * n, 3 * n + b)
        )
        if b < n:
            groups["volume_deep"] = list(range(2 * n + b, 3 * n)) + list(
                range(3 * n + b, 4 * n)
            )
        off = 4 * n

    groups["microstructure"] = list(range(off, off + 3))
    groups["trade"] = list(range(off + 3, off + 8))
    groups["quote_activity"] = list(range(off + 8, off + 11))
    return FeatureGroups(groups)


def group_attribution(
    per_feature: np.ndarray, groups: FeatureGroups, normalize: bool = True
) -> dict[str, float]:
    """Sum a per-feature attribution vector into group totals.

    Args:
        per_feature: ``(F,)`` per-column scores — typically ``|attribution|``
                     already reduced over batch and time.
        groups:      Blocks from :func:`feature_groups`.
        normalize:   Return each group's share of the total (the default) rather
                     than a raw sum, so numbers compare across models.

    Returns:
        ``{group_name: score}`` in the group order of ``groups``.
    """
    if per_feature.ndim != 1:
        raise ValueError(f"expected a (F,) vector, got shape {per_feature.shape}")
    out = {g: float(per_feature[idx].sum()) for g, idx in groups}
    if normalize:
        total = sum(out.values())
        if total > 0:
            out = {g: v / total for g, v in out.items()}
    return out
