"""In-RAM windowed dataset for Feishu equity data.

Mirrors the crypto ``LOBDataset``: the per-(asset, day) feature matrix is held
**once** as a single ``(N_rows, NF)`` array in RAM, and each T-day window is
sliced on demand in ``__getitem__``.  Storage therefore scales with
``N_rows × NF`` (the compact day-matrix) — not with the number of windows
(which would duplicate every day's features ~T times).

Each ``start`` index points at the first row of a window inside one asset's
contiguous block of rows; windows never straddle assets (the builder only
emits starts that stay within a single asset).

Batch format (matches every crypto model's ``predict``)::

    {"x": FloatTensor (1, T, NF), "label": LongTensor scalar}

The causal label paired with a window starting at row ``s`` is the row label
at ``s + T_past`` (the trade entered on the day *after* the window ends).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LOBDataset(Dataset):
    def __init__(
        self,
        feat: np.ndarray,
        starts: np.ndarray,
        labels: np.ndarray,
        t_past: int,
    ) -> None:
        """Args:
        feat:   ``(N_rows, NF)`` float32 array of pre-normalised features (in RAM).
        starts: ``(N_windows,)`` int64 window start rows (within one asset).
        labels: ``(N_rows,)`` int64 per-row causal labels (-1 = invalid).
        t_past: Window length in trading days.
        """
        self.feat = feat
        self.starts = starts
        self.labels = labels
        self.t_past = t_past

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        window = self.feat[s : s + self.t_past].astype(np.float32)  # (T, NF)
        x = torch.from_numpy(window.copy()).unsqueeze(0)  # (1, T, NF)
        return {
            "x": x,
            "label": torch.tensor(int(self.labels[s + self.t_past]), dtype=torch.long),
        }
