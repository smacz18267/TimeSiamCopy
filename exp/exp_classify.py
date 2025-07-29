import os, json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit

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
    num_workers: int = 0
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu"
    val_split: float = 0.2
    test_split: float = 0.2        # held-out test
    pair_mode: bool = True
    seed: int = 42


class Exp_Classify:
    def __init__(self, args):
        self.args = args  # keep args so test() can see --load_checkpoints, etc.

        # Map CLI -> config
        self.cfg = TrainConfig(
            data_dir = getattr(args, 'data_root', './AD'),
            task = getattr(args, 'cls_task', 'binary'),
            batch_size = getattr(args, 'batch_size', 8),
            epochs = getattr(args, 'train_epochs', 30),
            lr = getattr(args, 'learning_rate', 1e-3),
            weight_decay = getattr(args, 'weight_decay', 1e-4),
            emb_dim = getattr(args, 'emb_dim', 256),
            contrastive_weight = getattr(args, 'contrastive_weight', 0.5),
            ce_weight = getattr(args, 'ce_weight', 1.0),
            num_workers = getattr(args, 'num_workers', 0),
            device = 'cuda' if (getattr(args,'use_gpu',False) and torch.cuda.is_available()) else 'cpu',
            val_split = getattr(args, 'val_split', 0.2),
            test_split = getattr(args, 'test_split', 0.2),
            pair_mode = bool(getattr(args, 'pair_mode', 1)),
            seed = getattr(args, 'seed', 42),
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Base dataset (pair-mode) just to scan shapes & for contrastive
        base_pair = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=True)
        self.num_classes = base_pair.num_classes

        # Determine (fixed_C, fixed_L) = max across dataset (flatten C*F)
        max_C, max_L = 0, 0
        for p in base_pair.X:
            shp = np.load(p, mmap_mode='r').shape  # (C,L,F)
            C, L, F = int(shp[0]), int(shp[1]), int(shp[2])
            max_C = max(max_C, C * F)
            max_L = max(max_L, L)
        self.fixed_C, self.fixed_L = max_C, max_L

        # Build the classifier
        self.model = SiameseClassifier(self.fixed_C, self.num_classes,
                                       emb_dim=self.cfg.emb_dim).to(self.cfg.device)

        # Optional: load checkpoint (for fine-tune or test)
        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt:
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_mode='cpu') if hasattr(torch.load, '__call__') else torch.load(ckpt, map_location='cpu')
                self.model.load_state_dict(state)
                print(f"Loaded checkpoint: {ckpt}")
            else:
                print(f"[Warn] --load_checkpoints given but file not found: {ckpt}")

        # ---------- Reproducible stratified train/val/test split at FILE level ----------
        base_single = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        y_all = np.array([int(base_single.y[i]) for i in range(len(base_single))], dtype=int)

        os.makedirs('./outputs/splits_classify', exist_ok=True)
        split_tag = getattr(args, 'model_id', 'AD')
        self.splits_path = f'./outputs/splits_classify/{split_tag}_{self.cfg.task}_seed{self.cfg.seed}.npz'

        if os.path.exists(self.splits_path):
            pack = np.load(self.splits_path, allow_pickle=True)
            self.train_idx = pack['train_idx']
            self.val_idx   = pack['val_idx']
            self.test_idx  = pack['test_idx']
            print(f"Loaded splits from {self.splits_path} (train {len(self.train_idx)}, val {len(self.val_idx)}, test {len(self.test_idx)})")
        else:
            idx_all = np.arange(len(y_all))
            test_frac = float(self.cfg.test_split)
            val_frac  = float(self.cfg.val_split)

            if test_frac > 0:
                sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=self.cfg.seed)
                (trainval_idx, test_idx), = sss_test.split(idx_all, y_all)
            else:
                trainval_idx, test_idx = idx_all, np.array([], dtype=int)

            if val_frac > 0:
                sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac/(1.0 - test_frac + 1e-12),
                                                 random_state=self.cfg.seed)
                (train_idx, val_idx), = sss_val.split(trainval_idx, y_all[trainval_idx])
            else:
                train_idx, val_idx = trainval_idx, np.array([], dtype=int)

            self.train_idx = train_idx
            self.val_idx   = val_idx
            self.test_idx  = test_idx

            np.savez(self.splits_path, train_idx=self.train_idx, val_idx=self.val_idx, test_idx=self.test_idx)
            print(f"Saved splits to {self.splits_path} (train {len(self.train_idx)}, val {len(self.val_idx)}, test {len(self.test_idx)})")

        # Build Datasets for each split
        self.ds_pair = base_pair
        self.ds_ce   = base_single

        # restrict pair sampling to train only (prevents leakage)
        if hasattr(self.ds_pair, "set_pair_sampling_pool"):
            self.ds_pair.set_pair_sampling_pool(self.train_idx)

        # (Optional) leakage sanity check
        try:
            if hasattr(self.ds_pair, "_pool_indices") and self.ds_pair._pool_indices is not None:
                pool = set(int(i) for i in self.ds_pair._pool_indices.tolist())
                overlap = (set(self.val_idx.tolist()) | set(self.test_idx.tolist())) & pool
                print(f"[DEBUG] Pair-sampling pool size: {len(pool)} | overlap val/test: {len(overlap)} (must be 0)")
        except Exception:
            pass

        self.train_pair = Subset(self.ds_pair, self.train_idx)
        self.val_pair   = Subset(self.ds_pair, self.val_idx)
        self.train_ce   = Subset(self.ds_ce,   self.train_idx)
        self.val_ce     = Subset(self.ds_ce,   self.val_idx)
        self.test_ce    = Subset(self.ds_ce,   self.test_idx if len(self.test_idx) > 0 else self.val_idx)

        # Optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # threshold chosen on validation (used in test)
        self.best_threshold = 0.5

    # ===== Collate helpers that PAD/TRUNCATE to (fixed_C, fixed_L) =====
    def _pad_to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        C, L = x.shape
        C_new = min(self.fixed_C, C)
        L_new = min(self.fixed_L, L)
        out = torch.zeros(self.fixed_C, self.fixed_L, dtype=x.dtype)
        out[:C_new, :L_new] = x[:C_new, :L_new]
        return out

    def _collate_single(self, batch):
        xs, ys = zip(*batch)
        xs = [self._pad_to_fixed(t.contiguous()) for t in xs]
        X = torch.stack(xs, dim=0)               # (B, fixed_C, fixed_L)
        Y = torch.stack(list(ys), dim=0).long()  # (B,)
        return X, Y

    def _collate_pair(self, batch):
        xi, xj, ysim, ycls = zip(*batch)
        xi = [self._pad_to_fixed(t.contiguous()) for t in xi]
        xj = [self._pad_to_fixed(t.contiguous()) for t in xj]
        Xi = torch.stack(xi, dim=0)
        Xj = torch.stack(xj, dim=0)
        ysim = torch.stack(list(ysim), dim=0).float()
        ycls = torch.stack(list(ycls), dim=0).long()
        return Xi, Xj, ysim, ycls

    # ===== One training step for each loss =====
    def _step_pair(self, batch):
        xi, xj, y_sim, _ = batch
        xi = xi.to(self.cfg.device).float()
        xj = xj.to(self.cfg.device).float()
        y_sim = y_sim.to(self.cfg.device).float()
        z1 = self.model.embed(xi)
        z2 = self.model.embed(xj)
        return contrastive_loss(z1, z2, y_sim, margin=1.0)

    def _step_ce(self, batch):
        x, y = batch
        x = x.to(self.cfg.device).float()
        y = y.to(self.cfg.device).long()
        logits = self.model(x)
        return nn.CrossEntropyLoss()(logits, y)

    # ===== Evaluation helpers =====
    def _eval_pass(self, loader):
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.cfg.device).float()
                logits = self.model(x)  # (B, num_classes)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                ys.append(y.numpy())
                ps.append(prob)
        ys = np.concatenate(ys, axis=0)
        ps = np.concatenate(ps, axis=0)
        return ys, ps

    def _compute_metrics_binary(self, y_true, p_pos, threshold=0.5):
        y_pred = (p_pos >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true, p_pos)
        except Exception:
            auc = float('nan')
        try:
            auprc = average_precision_score(y_true, p_pos)
        except Exception:
            auprc = float('nan')
        return {"acc": acc, "precision": precision, "recall": recall,
                "f1": f1, "auc": auc, "auprc": auprc}

    def _pick_threshold(self, y_true, y_score, method="f1", min_precision=None):
        """
        Pick a threshold on validation scores with small-sample stability:
          - Candidate thresholds = score quantiles from 5%..95%
          - Optimize F1 (default) or Youden's J
          - Optionally enforce a minimum precision
        Returns: best_threshold, stats_at_best
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)

        if np.allclose(y_score, y_score[0]):
            return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        qs = np.linspace(0.05, 0.95, 19)
        cands = np.unique(np.quantile(y_score, qs))

        best_t = 0.5
        best_key = -1.0
        best_stats = None

        for t in cands:
            y_pred = (y_score >= t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if min_precision is not None and prec < min_precision:
                continue

            if method == "f1":
                key = f1
            elif method == "youden":
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                tpr = tp / (tp + fn + 1e-9)
                fpr = fp / (fp + tn + 1e-9)
                key = tpr - fpr
            else:
                raise ValueError("Unknown method")

            if key > best_key:
                best_key = key
                best_t = float(t)
                best_stats = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

        if best_stats is None:
            # fall back if all candidates filtered by min_precision
            best_t = float(np.median(y_score))
            y_pred = (y_score >= best_t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            best_stats = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

        return best_t, best_stats

    # ===== Public API =====
    def train(self, setting=None):
        print("Starting classification training with config:", self.cfg)

        train_pair_loader = DataLoader(
            self.train_pair, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, drop_last=False,
            collate_fn=self._collate_pair,
        )
        train_ce_loader = DataLoader(
            self.train_ce, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, drop_last=False,
            collate_fn=self._collate_single,
        )
        val_loader = DataLoader(
            self.val_ce, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, drop_last=False,
            collate_fn=self._collate_single,
        )

        best_metric = -np.inf
        best_state = None
        best_threshold = 0.5

        steps = min(len(train_pair_loader), len(train_ce_loader)) if self.cfg.pair_mode else len(train_ce_loader)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_c, total_ce = 0.0, 0.0

            iter_pair = iter(train_pair_loader) if self.cfg.pair_mode else None
            iter_ce = iter(train_ce_loader)

            for _ in range(steps):
                self.opt.zero_grad()

                if self.cfg.pair_mode and self.cfg.contrastive_weight > 0:
                    loss_c = self._step_pair(next(iter_pair)) * self.cfg.contrastive_weight
                else:
                    loss_c = torch.tensor(0.0, device=self.cfg.device)

                loss_ce = self._step_ce(next(iter_ce)) * self.cfg.ce_weight
                (loss_c + loss_ce).backward()
                self.opt.step()

                total_c += float(loss_c.detach().cpu())
                total_ce += float(loss_ce.detach().cpu())

            # ---- Validation metrics + pick threshold ----
            y_val, p_val = self._eval_pass(val_loader)
            if self.cfg.task == 'binary' and self.num_classes >= 2:
                p_pos = p_val[:, 1]
                cur_thr, thr_stats = self._pick_threshold(y_val, p_pos, method="f1", min_precision=0.6)
                metrics = self._compute_metrics_binary(y_val, p_pos, threshold=cur_thr)
                score = metrics["auc"] if not np.isnan(metrics["auc"]) else metrics["f1"]
            else:
                y_pred = p_val.argmax(axis=1)
                acc = accuracy_score(y_val, y_pred)
                metrics = {"acc": acc}
                score = acc

            print(f"Epoch {epoch+1}/{self.cfg.epochs} | contrastive: {total_c/max(1,steps):.4f} | "
                  f"ce: {total_ce/max(1,steps):.4f} | val: {metrics}")

            if score > best_metric:
                best_metric = score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                best_threshold = cur_thr if self.cfg.task == 'binary' and self.num_classes >= 2 else 0.5

        # Save best checkpoint (+ threshold)
        os.makedirs('./outputs/checkpoints_classify', exist_ok=True)
        tag = (setting or "classify")
        ckpt_path = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}.pt'
        thr_path  = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}_thr.npy'
        if best_state is not None:
            torch.save(best_state, ckpt_path)
            np.save(thr_path, np.array([best_threshold], dtype=np.float32))
            print(f"Saved best model to {ckpt_path} (best score={best_metric:.4f})")
            print(f"Saved validation threshold to {thr_path} (thr={best_threshold:.3f})")
        else:
            torch.save(self.model.state_dict(), ckpt_path)
            np.save(thr_path, np.array([0.5], dtype=np.float32))
            print(f"Saved last model to {ckpt_path}")

    def test(self, setting=None, test=0):
        # auto-pick checkpoint if not provided
        tag = (setting or "classify")
        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt is None:
            ckpt = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}.pt'
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint for test: {ckpt}")
        else:
            print(f"[Warn] No checkpoint found at {ckpt}; testing current weights.")

        # try to load saved threshold; if missing, derive from val
        thr_path  = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}_thr.npy'
        if os.path.exists(thr_path):
            self.best_threshold = float(np.load(thr_path)[0])
        else:
            # compute from val set
            val_loader = DataLoader(
                self.val_ce, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, collate_fn=self._collate_single,
            )
            y_v, p_v = self._eval_pass(val_loader)
            self.best_threshold, _ = self._pick_threshold(y_v, p_v[:, 1], method="f1", min_precision=0.6)

        print(f"Using decision threshold from validation: {self.best_threshold:.2f}")

        # Evaluate on TEST
        test_loader = DataLoader(
            self.test_ce, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=self._collate_single,
        )
        if self.cfg.task == 'binary' and self.num_classes >= 2:
            y_t, p_t = self._eval_pass(test_loader)
            p_pos_t = p_t[:, 1]
            metrics = self._compute_metrics_binary(y_t, p_pos_t, threshold=self.best_threshold)
            y_pred = (p_pos_t >= self.best_threshold).astype(int)
            cm = confusion_matrix(y_t, y_pred)
            print("Confusion matrix (test):\n", cm)
        else:
            y_t, p_t = self._eval_pass(test_loader)
            y_pred = p_t.argmax(axis=1)
            metrics = {"acc": accuracy_score(y_t, y_pred)}

        print("Test metrics:", metrics)

        # save metrics JSON
        os.makedirs('./outputs/results_classify', exist_ok=True)
        out_name = f'{tag}_{self.cfg.task}_seed{self.cfg.seed}.json'
        with open(os.path.join('./outputs/results_classify', out_name), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        print(f"Saved test metrics to ./outputs/results_classify/{out_name}")
        return metrics
