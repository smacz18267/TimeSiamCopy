import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class ADWindowDataset(Dataset):
    """
    Windowed AD dataset

    Folder layout:
      data_dir/
        Feature/*.npy   (each: (C,L,F) or (C,L) or (L,C))
        Label/label.npy (shape (N,2) or (N,))

    Behavior:
      - converts every file to (C,L); if (C,L,F), flattens F into channels (C*F,L)
      - slides fixed-length windows (target_len) with stride
      - pads/truncates channels to target_ch (auto: max across files)
      - expands per-file label to all its windows
      - exposes `groups` = file index per window (for grouped splits)
    """

    def __init__(self,
                 data_dir: str,
                 task: str = "binary",         # "binary" or "id" (multi-class)
                 target_len: int = 256,
                 stride: int = 128,
                 target_ch: int | None = None,
                 pair_mode: bool = True):
        super().__init__()
        self.dir = data_dir
        self.task = task
        self.target_len = target_len
        self.stride = stride
        self.target_ch = target_ch
        self.pair_mode = pair_mode

        feat_dir = os.path.join(data_dir, "Feature")
        files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".npy")])
        if not files:
            raise ValueError(f"No .npy in {feat_dir}")
        self.files = [os.path.join(feat_dir, f) for f in files]

        # --- labels ---
        lab_path = os.path.join(data_dir, "Label", "label.npy")
        labels = np.load(lab_path, mmap_mode='r')
        if labels.ndim == 2 and labels.shape[1] >= 2:
            if task == "binary":
                y_file = labels[:, 0].astype(int)  # binary col
                self.num_classes = 2
            else:
                # treat 2nd col as ID category; remap to 0..K-1
                ids = labels[:, 1].astype(int)
                uniq = sorted(np.unique(ids).tolist())
                remap = {v:i for i,v in enumerate(uniq)}
                y_file = np.array([remap[v] for v in ids], dtype=int)
                self.num_classes = len(uniq)
        elif labels.ndim == 1:
            y_file = labels.astype(int)
            self.num_classes = len(np.unique(y_file))
        else:
            raise ValueError("Unexpected label format in Label/label.npy")

        if len(y_file) != len(self.files):
            raise ValueError(f"Label length {len(y_file)} != number of feature files {len(self.files)}")

        # --- pre-scan to decide target_ch (max channels) and max length ---
        maxC = 0
        for fp in self.files:
            arr = np.load(fp, mmap_mode='r')
            shp = arr.shape
            if arr.ndim == 3:  # (C,L,F)
                C, L, F = shp
                C = int(C * F)
            elif arr.ndim == 2:
                if shp[0] < shp[1]:
                    C = shp[0]     # (C,L)
                else:
                    C = shp[1]     # (L,C) -> will transpose on load
            else:
                raise ValueError(f"Unsupported shape {shp} in {fp}")
            maxC = max(maxC, int(C))
        if self.target_ch is None:
            self.target_ch = maxC

        # --- make windows ---
        self.windows: List[Tuple[int,int,int]] = []   # (file_idx, start, end)
        self.file_windows: List[Tuple[int,int]] = []  # (file_idx, num_windows)
        self.groups: List[int] = []
        for fi, fp in enumerate(self.files):
            arr = np.load(fp, mmap_mode='r')
            if arr.ndim == 3:
                _, L, _ = arr.shape
            else:
                L = arr.shape[1] if arr.shape[0] < arr.shape[1] else arr.shape[0]

            if L <= self.target_len:
                starts = [0]
            else:
                starts = list(range(0, L - self.target_len + 1, self.stride))
                if (L - self.target_len) % self.stride != 0:
                    starts.append(L - self.target_len)

            n_win = len(starts)
            self.file_windows.append((fi, n_win))
            for s in starts:
                e = s + self.target_len
                self.windows.append((fi, s, e))
                self.groups.append(fi)

        # expand labels per window (same label for all windows from a file)
        self.y_file = y_file
        self.y = np.repeat(self.y_file, [nw for _, nw in self.file_windows])

        # per-class global indices for pair sampling
        self.cls_to_indices: Dict[int, List[int]] = {}
        for gi, cls in enumerate(self.y):
            self.cls_to_indices.setdefault(int(cls), []).append(gi)

    def __len__(self):
        return len(self.windows)

    def _load_clip(self, gi: int) -> torch.Tensor:
        fi, s, e = self.windows[gi]
        arr = np.load(self.files[fi], mmap_mode='r')  # (C,L,F) or (C,L) or (L,C)
        if arr.ndim == 3:
            C, L, F = arr.shape
            x = arr[:, s:e, :].astype(np.float32)   # (C,T,F)
            x = x.reshape(C * F, e - s)             # -> (C*F, T)
        elif arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:         # (C,L)
                x = arr[:, s:e].astype(np.float32)  # (C,T)
            else:                                    # (L,C)
                x = arr[s:e, :].astype(np.float32).T  # -> (C,T)
        else:
            raise ValueError

        Ccur, T = x.shape
        if Ccur >= self.target_ch:
            x = x[:self.target_ch, :]
        else:
            pad = np.zeros((self.target_ch - Ccur, T), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        return torch.from_numpy(np.ascontiguousarray(x))

    def __getitem__(self, gi: int):
        if not self.pair_mode:
            x = self._load_clip(gi)
            y = torch.tensor(int(self.y[gi]), dtype=torch.long)
            return x, y

        # pair sampling for contrastive pretraining
        y_anchor = int(self.y[gi])
        if np.random.rand() < 0.5 and len(self.cls_to_indices[y_anchor]) > 1:
            pool = self.cls_to_indices[y_anchor]
            while True:
                gj = int(np.random.choice(pool))
                if gj != gi: break
            y_sim = 1
        else:
            other = [c for c in self.cls_to_indices if c != y_anchor]
            c2 = int(np.random.choice(other))
            gj = int(np.random.choice(self.cls_to_indices[c2]))
            y_sim = 0

        xi = self._load_clip(gi)
        xj = self._load_clip(gj)
        return xi, xj, torch.tensor(y_sim, dtype=torch.float32), torch.tensor(y_anchor, dtype=torch.long)
