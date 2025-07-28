import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

class ADDataset(Dataset):
    def __init__(self, data_dir: str, task: str = "binary", pair_mode: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.pair_mode = pair_mode

        feat_dir = os.path.join(data_dir, "Feature")
        self.files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".npy")])
        self.X = [os.path.join(feat_dir, f) for f in self.files]

        labels = np.load(os.path.join(data_dir, "Label", "label.npy"))
        if labels.ndim != 2 or labels.shape[0] != len(self.X) or labels.shape[1] < 2:
            raise ValueError(f"Unexpected label shape: {labels.shape}; expected (N,2) matching features")
        self.labels_raw = labels

        if task == "binary":
            self.y = labels[:,0].astype(int)
            self.num_classes = 2
        elif task == "id":
            ids = labels[:,1].astype(int)
            unique_ids = sorted(np.unique(ids).tolist())
            id_to_idx = {idv:i for i,idv in enumerate(unique_ids)}
            self.y = np.array([id_to_idx[i] for i in ids], dtype=int)
            self.num_classes = len(unique_ids)
        else:
            raise ValueError("task must be 'binary' or 'id'")

        self.cls_to_indices = {}
        for idx, c in enumerate(self.y):
            self.cls_to_indices.setdefault(c, []).append(idx)

        self.indices = np.arange(len(self.X))

    def __len__(self):
        return len(self.X)

    def _load_x(self, idx: int) -> torch.Tensor:
        x = np.load(self.X[idx])
        if x.ndim != 3:
            raise ValueError(f"Feature at {self.X[idx]} has shape {x.shape}; expected (C, L, F)")
        C, L, F = x.shape
        x = x.astype(np.float32)
        x = x.reshape(C*F, L)
        return torch.from_numpy(x)

    def _sample_positive(self, cls: int, exclude_idx: int) -> int:
        choices = self.cls_to_indices[cls]
        if len(choices) == 1:
            return exclude_idx
        while True:
            j = np.random.choice(choices)
            if j != exclude_idx:
                return j

    def _sample_negative(self, cls: int) -> int:
        other_classes = [c for c in self.cls_to_indices.keys() if c != cls]
        c2 = np.random.choice(other_classes)
        return np.random.choice(self.cls_to_indices[c2])

    def __getitem__(self, idx: int):
        if not self.pair_mode:
            x = self._load_x(idx)
            y = int(self.y[idx])
            return x, y

        y = int(self.y[idx])
        if np.random.rand() < 0.5 and len(self.cls_to_indices[y]) > 0:
            j = self._sample_positive(y, idx)
            label = 1
        else:
            j = self._sample_negative(y)
            label = 0

        xi = self._load_x(idx)
        xj = self._load_x(j)
        return xi, xj, int(label), y
