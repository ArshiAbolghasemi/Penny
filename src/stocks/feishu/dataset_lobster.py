"""Custom PyTorch Dataset for LOBSTER financial sequences within the Penny workspace."""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np


class LOBSTERDataset(Dataset):
    """Memory-efficient rolling window loader returning a standardized channel-first dictionary."""
    def __init__(
        self, 
        feat: np.ndarray, 
        starts: np.ndarray, 
        row_labels: np.ndarray, 
        T: int, 
        row_asset: np.ndarray | None = None
    ):
        self.feat = torch.from_numpy(feat).float()
        self.starts = starts
        self.row_labels = torch.from_numpy(row_labels).long()
        self.T = T
        self.row_asset = torch.from_numpy(row_asset).long() if row_asset is not None else None

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_idx = self.starts[idx]
        end_idx = start_idx + self.T
        
        # Extract sequence and add the singleton channel dimension: (1, T_past, F)
        x = self.feat[start_idx:end_idx].unsqueeze(0)
        label = self.row_labels[end_idx - 1]
        
        batch = {
            "x": x,
            "label": label
        }
        
        if self.row_asset is not None:
            batch["asset"] = self.row_asset[end_idx - 1]
            
        return batch