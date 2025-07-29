import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ADDataset(Dataset):
    """
    AD dataset:
      data_dir/
        Feature/*.npy    each (C, L, F)
        Label/label.npy  shape (N, 2): [:,0] binary, [:,1] ID
    """

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

        if task == "binary":
            self.y = labels[:, 0].astype(int)
            self.num_classes = 2
        elif task == "id":
            ids = labels[:, 1].astype(int)
            uniq = sorted(np.unique(ids).tolist())
            idx_map = {v: i for i, v in enumerate(uniq)}
            self.y = np.array([idx_map[v] for v in ids], dtype=int)
            self.num_classes = len(uniq)
        else:
            raise ValueError("task must be 'binary' or 'id'")

        # Precompute indices per class
        self.cls_to_indices = {}
        for i, c in enumerate(self.y):
            self.cls_to_indices.setdefault(int(c), []).append(i)

    def __len__(self):
        return len(self.X)

    def _load_x(self, idx: int) -> torch.Tensor:
        x = np.load(self.X[idx])  # (C, L, F)
        if x.ndim != 3:
            raise ValueError(f"{self.X[idx]} has shape {x.shape}; expected (C, L, F)")
        C, L, F = x.shape
        # --- IMPORTANT: ensure contiguous OWNED memory before from_numpy ---
        x = np.ascontiguousarray(x.reshape(C * F, L).astype(np.float32)).copy()
        t = torch.from_numpy(x)  # (C*F, L)
        # in rare cases, enforce contiguous torch memory too
        return t.contiguous()

    def _sample_positive(self, cls: int, exclude_idx: int) -> int:
        pool = self.cls_to_indices[cls]
        if len(pool) == 1:
            return exclude_idx
        while True:
            j = np.random.choice(pool)
            if j != exclude_idx:
                return j

    def _sample_negative(self, cls: int) -> int:
        other = [c for c in self.cls_to_indices.keys() if c != cls]
        c2 = int(np.random.choice(other))
        return int(np.random.choice(self.cls_to_indices[c2]))

    def __getitem__(self, idx: int):
        if not self.pair_mode:
            x = self._load_x(idx)                  # (C*F, L)
            y = int(self.y[idx])
            # --- return tensors with owned storage/dtypes set ---
            return x.float(), torch.tensor(y, dtype=torch.long)

        # Pair mode
        y_anchor = int(self.y[idx])
        if np.random.rand() < 0.5 and len(self.cls_to_indices[y_anchor]) > 0:
            j = self._sample_positive(y_anchor, idx)
            y_sim = 1
        else:
            j = self._sample_negative(y_anchor)
            y_sim = 0

        xi = self._load_x(idx).float()
        xj = self._load_x(j).float()

        # --- make labels tensors now to avoid dtype surprises in collate ---
        y_sim = torch.tensor(y_sim, dtype=torch.float32)  # contrastive target
        y_anchor = torch.tensor(y_anchor, dtype=torch.long)
        return xi, xj, y_sim, y_anchor
