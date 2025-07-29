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

        # Global class -> indices (full dataset)
        self.cls_to_indices = {}
        for i, c in enumerate(self.y):
            self.cls_to_indices.setdefault(int(c), []).append(i)

        # ---- Sampling pool (train-only) ----
        # If set via set_pair_sampling_pool(), pair-mode sampling will be restricted to this pool.
        self._pool_indices = None
        self._cls_to_pool = None

    def set_pair_sampling_pool(self, indices):
        """
        Restrict positive/negative sampling to these dataset indices (e.g., train_idx).
        `indices` can be list/ndarray of ints (global dataset indices).
        """
        if indices is None:
            self._pool_indices = None
            self._cls_to_pool = None
            return

        pool = np.array(indices, dtype=int)
        self._pool_indices = pool
        cls_to_pool = {}
        for gid in pool:
            c = int(self.y[gid])
            cls_to_pool.setdefault(c, []).append(int(gid))
        self._cls_to_pool = cls_to_pool

    def __len__(self):
        return len(self.X)

    def _load_x(self, idx: int) -> torch.Tensor:
        x = np.load(self.X[idx])  # (C, L, F)
        if x.ndim != 3:
            raise ValueError(f"{self.X[idx]} has shape {x.shape}; expected (C, L, F)")
        C, L, F = x.shape
        x = np.ascontiguousarray(x.reshape(C * F, L).astype(np.float32)).copy()
        t = torch.from_numpy(x)
        return t.contiguous()

    # ------- helpers that respect the sampling pool if set -------
    def _sample_positive(self, cls: int, exclude_idx: int) -> int:
        # Prefer pool-restricted indices
        pool = None
        if self._cls_to_pool is not None and cls in self._cls_to_pool:
            pool = self._cls_to_pool[cls]
        else:
            pool = self.cls_to_indices[cls]

        if len(pool) == 1:
            return exclude_idx
        while True:
            j = int(np.random.choice(pool))
            if j != exclude_idx:
                return j

    def _sample_negative(self, cls: int) -> int:
        # Choose a different class
        all_classes = list(self.cls_to_indices.keys())
        other = [c for c in all_classes if c != cls]
        c2 = int(np.random.choice(other))

        if self._cls_to_pool is not None and c2 in self._cls_to_pool and len(self._cls_to_pool[c2]) > 0:
            return int(np.random.choice(self._cls_to_pool[c2]))
        else:
            return int(np.random.choice(self.cls_to_indices[c2]))

    def __getitem__(self, idx: int):
        if not self.pair_mode:
            x = self._load_x(idx).float()
            y = int(self.y[idx])
            return x.float(), torch.tensor(y, dtype=torch.long)

        # Pair mode
        y_anchor = int(self.y[idx])
        # Sample within pool if defined
        if np.random.rand() < 0.5:
            j = self._sample_positive(y_anchor, idx)
            y_sim = 1
        else:
            j = self._sample_negative(y_anchor)
            y_sim = 0

        xi = self._load_x(idx).float()
        xj = self._load_x(j).float()
        y_sim = torch.tensor(y_sim, dtype=torch.float32)
        y_anchor = torch.tensor(y_anchor, dtype=torch.long)
        return xi.float(), xj.float(), y_sim, y_anchor
