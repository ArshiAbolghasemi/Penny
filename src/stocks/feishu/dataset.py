"""HDF5-backed dataset for Feishu equity data.

WindowWriter writes sliding windows once at cache-build time; DiskLOBDataset
reads them lazily so the full dataset never resides in RAM.

The batch format matches the crypto dataset:
    {"x": FloatTensor (1, T, NF), "label": LongTensor scalar}

HDF5 layout
-----------
  /X      (N, 1, T, NF)  float32
  /y      (N,)           int64     — class labels {0, 1, 2}
  /asset  (N,)           int64     — integer asset index (stored for offline analysis)
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowWriter:
    """Append-mode HDF5 writer for (X, y, asset_idx) windows.

    Usage::

        writer = WindowWriter("path/to/train.h5", T=50, NF=259)
        writer.write(X, labels, asset_idx)   # one call per asset
        writer.close()
    """

    def __init__(self, path: str, T: int = 50, NF: int = 259, chunk: int = 512) -> None:
        self._f = h5py.File(path, "w")
        self._X = self._f.create_dataset(
            "X",
            shape=(0, 1, T, NF),
            maxshape=(None, 1, T, NF),
            dtype="float32",
            chunks=(chunk, 1, T, NF),
        )
        self._y = self._f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype="int64",
            chunks=(chunk * 8,),
        )
        self._asset = self._f.create_dataset(
            "asset",
            shape=(0,),
            maxshape=(None,),
            dtype="int64",
            chunks=(chunk * 8,),
        )

    def write(self, X: np.ndarray, labels: np.ndarray, asset_idx: int) -> None:
        """Append windows for one asset.

        Args:
            X:         ``(N, 1, T, NF)`` float32 feature windows.
            labels:    ``(N,)`` int64 class labels.
            asset_idx: Integer index for this asset (stored for analysis).
        """
        n = len(X)
        if n == 0:
            return
        asset_col = np.full(n, asset_idx, dtype=np.int64)
        for ds, data in [
            (self._X, X),
            (self._y, labels),
            (self._asset, asset_col),
        ]:
            old = ds.shape[0]
            ds.resize(old + n, axis=0)
            ds[old:] = data

    def close(self) -> None:
        self._f.close()


class DiskLOBDataset(Dataset):
    """Lazy HDF5 reader — each DataLoader worker opens its own file handle.

    Returns batches compatible with all crypto model ``predict()`` methods::

        {"x": FloatTensor (1, T, NF), "label": LongTensor scalar}
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with h5py.File(path, "r") as f:
            self._len = int(f["y"].shape[0])
        self._file: Any = None  # h5py.File opened lazily per worker

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._handle()
        return {
            "x": torch.from_numpy(f["X"][idx]),
            "label": torch.tensor(int(f["y"][idx]), dtype=torch.long),
        }

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
