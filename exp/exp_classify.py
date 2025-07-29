import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import dataclass

from data_provider.ad_dataset import ADDataset
from models.siamese1d import SiameseClassifier, contrastive_loss

@dataclass
class TrainConfig:
    data_dir: str
    task: str = "binary"           # "binary" or "id"
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    emb_dim: int = 256
    contrastive_weight: float = 0.5
    ce_weight: float = 1.0
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_split: float = 0.2
    pair_mode: bool = True
    seed: int = 42


class Exp_Classify:
    def __init__(self, args):
        # Map run.py args to our config (with sensible defaults)
        self.args = args
        self.cfg = TrainConfig(
            data_dir = args.data_root if hasattr(args, 'data_root') else './AD',
            task = getattr(args, 'cls_task', 'binary'),
            batch_size = getattr(args, 'batch_size', 8),
            epochs = getattr(args, 'train_epochs', 30),
            lr = getattr(args, 'learning_rate', 1e-3),
            weight_decay = getattr(args, 'weight_decay', 1e-4),
            emb_dim = getattr(args, 'emb_dim', 256),
            contrastive_weight = getattr(args, 'contrastive_weight', 0.5),
            ce_weight = getattr(args, 'ce_weight', 1.0),
            num_workers = getattr(args, 'num_workers', 4),
            device = 'cuda' if (getattr(args,'use_gpu',False) and torch.cuda.is_available()) else 'cpu',
            val_split = getattr(args, 'val_split', 0.2),
            pair_mode = bool(getattr(args, 'pair_mode', 1)),
            seed = getattr(args, 'seed', 42),
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Load dataset (pair-mode for contrastive; we will create a no-pair clone for CE)
        base_ds = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=self.cfg.pair_mode)

        # --- IMPORTANT: determine fixed channel & time sizes from the WHOLE dataset ---
        # We only read shapes via mmap (fast) — no full loads.
        max_C, max_L = 0, 0
        for p in base_ds.X:
            shp = np.load(p, mmap_mode='r').shape  # (C, L, F)
            C, L, F = int(shp[0]), int(shp[1]), int(shp[2])
            max_C = max(max_C, C * F)
            max_L = max(max_L, L)
        self.fixed_C = max_C
        self.fixed_L = max_L

        # Build model with fixed in_channels (max across dataset)
        self.num_classes = base_ds.num_classes
        self.model = SiameseClassifier(self.fixed_C, self.num_classes, emb_dim=self.cfg.emb_dim).to(self.cfg.device)

        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt:
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location='cpu')
                self.model.load_state_dict(state)
                print(f"Loaded checkpoint: {ckpt}")
            else:
                print(f"[Warn] --load_checkpoints given but file not found: {ckpt}")

        # Build splits & mirror datasets for CE (single) mode
        full_len = len(base_ds)
        val_len = max(1, int(full_len * self.cfg.val_split))
        train_len = full_len - val_len

        # Pair-mode dataset for contrastive training
        self.train_pair, self.val_pair = random_split(base_ds, [train_len, val_len])

        # Single-sample dataset (no pairing) for cross-entropy head training/val
        ds_ce = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        self.train_ce, self.val_ce = random_split(ds_ce, [train_len, val_len])

        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    # ===== Collate helpers that PAD/TRUNCATE to (fixed_C, fixed_L) =====
    def _pad_to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, L)
        C, L = x.shape
        C_new = min(self.fixed_C, C)
        L_new = min(self.fixed_L, L)
        out = torch.zeros(self.fixed_C, self.fixed_L, dtype=x.dtype)
        out[:C_new, :L_new] = x[:C_new, :L_new]
        return out

    def _collate_single(self, batch):
        xs, ys = zip(*batch)  # xs: list[(C,L)], ys: list[tensor scalar]
        xs = [self._pad_to_fixed(t.contiguous()) for t in xs]
        X = torch.stack(xs, dim=0)  # (B, fixed_C, fixed_L)
        Y = torch.stack(list(ys), dim=0).long()  # (B,)
        return X, Y

    def _collate_pair(self, batch):
        xi, xj, ysim, ycls = zip(*batch)
        xi = [self._pad_to_fixed(t.contiguous()) for t in xi]
        xj = [self._pad_to_fixed(t.contiguous()) for t in xj]
        Xi = torch.stack(xi, dim=0)         # (B, fixed_C, fixed_L)
        Xj = torch.stack(xj, dim=0)         # (B, fixed_C, fixed_L)
        ysim = torch.stack(list(ysim), dim=0).float()   # (B,)
        ycls = torch.stack(list(ycls), dim=0).long()    # (B,)
        return Xi, Xj, ysim, ycls

    # ===== Steps =====
    def _step_pair(self, batch):
        xi, xj, y_sim, _y_anchor = batch
        xi = xi.to(self.cfg.device).float()
        xj = xj.to(self.cfg.device).float()
        y_sim = y_sim.to(self.cfg.device).float()
        z1 = self.model.embed(xi)
        z2 = self.model.embed(xj)
        loss_c = contrastive_loss(z1, z2, y_sim, margin=1.0)
        return loss_c

    def _step_ce(self, batch):
        x, y = batch
        x = x.to(self.cfg.device).float()
        y = y.to(self.cfg.device).long()
        logits = self.model(x)
        return nn.CrossEntropyLoss()(logits, y)

    def _evaluate(self, loader, task='binary'):
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.cfg.device).float()
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                y = y.numpy()
                ys.append(y)
                ps.append(prob)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)

        if task == 'binary' and ps.shape[1] >= 2:
            acc = accuracy_score(ys, ps.argmax(axis=1))
            try:
                auc = roc_auc_score(ys, ps[:, 1])
            except Exception:
                auc = float('nan')
            return {'acc': acc, 'auc': auc}
        else:
            acc = accuracy_score(ys, ps.argmax(axis=1))
            return {'acc': acc}

    # ===== Public API =====
    def train(self, setting=None):
        print("Starting classification training with config:", self.cfg)

        train_pair_loader = DataLoader(
            self.train_pair,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_pair,      # <— use custom collate
        )
        train_ce_loader = DataLoader(
            self.train_ce,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_single,    # <— use custom collate
        )
        val_loader = DataLoader(
            self.val_ce,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_single,    # <— use custom collate
        )

        best_metric = -np.inf
        best_state = None

        steps = min(len(train_pair_loader), len(train_ce_loader)) if self.cfg.pair_mode else len(train_ce_loader)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_c, total_ce = 0.0, 0.0

            if self.cfg.pair_mode:
                iter_pair = iter(train_pair_loader)
            iter_ce = iter(train_ce_loader)

            for _ in range(steps):
                self.opt.zero_grad()

                if self.cfg.pair_mode:
                    loss_c = self._step_pair(next(iter_pair)) * self.cfg.contrastive_weight
                else:
                    loss_c = torch.tensor(0.0, device=self.cfg.device)

                loss_ce = self._step_ce(next(iter_ce)) * self.cfg.ce_weight
                loss = loss_c + loss_ce
                loss.backward()
                self.opt.step()

                total_c += float(loss_c.detach().cpu())
                total_ce += float(loss_ce.detach().cpu())

            # Validation
            metrics = self._evaluate(val_loader, task=self.cfg.task)
            score = metrics.get('auc', metrics.get('acc', 0.0)) if self.cfg.task == 'binary' else metrics.get('acc', 0.0)
            print(f"Epoch {epoch+1}/{self.cfg.epochs} | contrastive: {total_c/max(1,steps):.4f} | ce: {total_ce/max(1,steps):.4f} | val: {metrics}")

            if score > best_metric:
                best_metric = score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Save best checkpoint
        os.makedirs('./outputs/checkpoints_classify', exist_ok=True)
        ckpt_path = f'./outputs/checkpoints_classify/{(setting or "classify")}_{self.cfg.task}.pt'
        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"Saved best model to {ckpt_path} (best score={best_metric:.4f})")
        else:
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved last model to {ckpt_path}")

    def test(self, setting=None, test=0):
        # --- auto-pick checkpoint if not provided ---
        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt is None:
            ckpt = f'./outputs/checkpoints_classify/{(setting or "classify")}_{self.cfg.task}.pt'
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint for test: {ckpt}")
        else:
            print(f"[Warn] No checkpoint found at {ckpt}; testing current weights.")

        # Evaluate on the full dataset (single-sample)
        full_ds = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        loader = DataLoader(
            full_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_single,
        )
        metrics = self._evaluate(loader, task=self.cfg.task)
        print("Test metrics:", metrics)
        return metrics

