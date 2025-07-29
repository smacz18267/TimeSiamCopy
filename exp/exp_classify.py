import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

from data_provider.ad_window_dataset import ADWindowDataset
from models.siamese1d import SiameseClassifier, contrastive_loss


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    data_dir: str
    task: str = "binary"           # "binary" or "id" (multi-class)
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    emb_dim: int = 256
    contrastive_weight: float = 0.5
    ce_weight: float = 1.0
    num_workers: int = 4
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu"
    val_split: float = 0.2
    test_split: float = 0.2
    pair_mode: bool = True
    seed: int = 42


# ----------------------------
# Experiment
# ----------------------------
class Exp_Classify:
    def __init__(self, args):
        # Map run.py args -> config
        self.args = args
        self.cfg = TrainConfig(
            data_dir=getattr(args, 'data_root', './AD'),
            task=getattr(args, 'cls_task', 'binary'),
            batch_size=getattr(args, 'batch_size', 8),
            epochs=getattr(args, 'train_epochs', 30),
            lr=getattr(args, 'learning_rate', 1e-3),
            weight_decay=getattr(args, 'weight_decay', 1e-4),
            emb_dim=getattr(args, 'emb_dim', 256),
            contrastive_weight=getattr(args, 'contrastive_weight', 0.5),
            ce_weight=getattr(args, 'ce_weight', 1.0),
            num_workers=getattr(args, 'num_workers', 4),
            device='cuda' if (getattr(args, 'use_gpu', False) and torch.cuda.is_available()) else 'cpu',
            val_split=getattr(args, 'val_split', 0.2),
            test_split=getattr(args, 'test_split', 0.2),
            pair_mode=bool(getattr(args, 'pair_mode', 1)),
            seed=getattr(args, 'seed', 42),
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # ----------------------------
        # Dataset: windowed AD
        # ----------------------------
        target_len = getattr(args, 'target_len', 256)
        stride = getattr(args, 'stride', 128)

        # pair-mode dataset (for contrastive)
        base_pair = ADWindowDataset(
            data_dir=self.cfg.data_dir,
            task=self.cfg.task,
            target_len=target_len,
            stride=stride,
            target_ch=None,          # auto
            pair_mode=True
        )
        # single-sample dataset (for CE / evaluation)
        base_single = ADWindowDataset(
            data_dir=self.cfg.data_dir,
            task=self.cfg.task,
            target_len=target_len,
            stride=stride,
            target_ch=base_pair.target_ch,  # keep same channel count
            pair_mode=False
        )

        # Group-aware stratified split (by file index)
        groups = np.array(base_single.groups)
        y_all = np.array([int(base_single.y[i]) for i in range(len(base_single))], dtype=int)
        tr_idx, va_idx, te_idx = self._grouped_stratified_split(
            groups, y_all, self.cfg.val_split, self.cfg.test_split, self.cfg.seed
        )

        # Subsets that share the same window index space
        self.train_pair = Subset(base_pair, tr_idx.tolist())
        self.val_pair   = Subset(base_pair, va_idx.tolist())
        self.train_ce   = Subset(base_single, tr_idx.tolist())
        self.val_ce     = Subset(base_single, va_idx.tolist())
        self.test_ce    = Subset(base_single, te_idx.tolist())

        # Fixed shapes for padding
        self.fixed_C = base_pair.target_ch
        self.fixed_L = target_len
        self.num_classes = base_pair.num_classes

        # ----------------------------
        # Model & (optional) checkpoint load
        # ----------------------------
        self.model = SiameseClassifier(self.fixed_C, self.num_classes, emb_dim=self.cfg.emb_dim).to(self.cfg.device)

        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint: {ckpt}")
        elif ckpt:
            print(f"[Warn] --load_checkpoints not found: {ckpt}")

        # Optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Build class-weighted CE (helps imbalance) — computed on train_ce once
        self._ce_criterion = None
        self._init_ce_weights()

    # ----------------------------
    # Splits
    # ----------------------------
    @staticmethod
    def _grouped_stratified_split(groups, y, val_frac=0.2, test_frac=0.2, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stratify at file-level, then expand to window indices."""
        groups = np.asarray(groups)
        y = np.asarray(y)

        file_ids = np.unique(groups)
        # group label = majority class within group (here constant by design)
        g_labels = []
        for gid in file_ids:
            yy = y[groups == gid]
            g_labels.append(int(np.bincount(yy).argmax()))
        g_labels = np.array(g_labels)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        (trainval_g_idx, test_g_idx), = sss1.split(file_ids, g_labels)
        trainval_files, test_files = file_ids[trainval_g_idx], file_ids[test_g_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
        (train_g_idx, val_g_idx), = sss2.split(trainval_files, g_labels[trainval_g_idx])
        train_files, val_files = trainval_files[train_g_idx], trainval_files[val_g_idx]

        idx = np.arange(len(y))
        train_idx = idx[np.isin(groups, train_files)]
        val_idx   = idx[np.isin(groups, val_files)]
        test_idx  = idx[np.isin(groups, test_files)]
        return train_idx, val_idx, test_idx

    # ----------------------------
    # Collate with padding to (fixed_C, fixed_L)
    # ----------------------------
    def _pad_to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, L)
        C, L = x.shape
        Cn, Ln = min(self.fixed_C, C), min(self.fixed_L, L)
        out = torch.zeros(self.fixed_C, self.fixed_L, dtype=x.dtype)
        out[:Cn, :Ln] = x[:Cn, :Ln]
        return out

    def _collate_single(self, batch):
        xs, ys = zip(*batch)
        X = torch.stack([self._pad_to_fixed(t.contiguous()) for t in xs], dim=0)
        Y = torch.as_tensor(ys, dtype=torch.long)
        return X, Y

    def _collate_pair(self, batch):
        xi, xj, ysim, ycls = zip(*batch)
        Xi = torch.stack([self._pad_to_fixed(t.contiguous()) for t in xi], dim=0)
        Xj = torch.stack([self._pad_to_fixed(t.contiguous()) for t in xj], dim=0)
        ysim = torch.as_tensor(ysim, dtype=torch.float32)
        ycls = torch.as_tensor(ycls, dtype=torch.long)
        return Xi, Xj, ysim, ycls

    # ----------------------------
    # Loss steps
    # ----------------------------
    def _step_pair(self, batch) -> torch.Tensor:
        xi, xj, y_sim, _ = batch
        xi = xi.to(self.cfg.device).float()
        xj = xj.to(self.cfg.device).float()
        y_sim = y_sim.to(self.cfg.device).float()
        z1 = self.model.embed(xi)
        z2 = self.model.embed(xj)
        return contrastive_loss(z1, z2, y_sim, margin=1.0)

    def _init_ce_weights(self):
        """Compute class weights from train set for CE; called once."""
        if self._ce_criterion is not None:
            return
        # Collect train labels once (from subset indices)
        ys = []
        for i in range(len(self.train_ce)):
            _, y = self.train_ce[i]
            ys.append(int(y))
        classes = np.unique(ys)
        counts = np.array([(np.array(ys) == c).sum() for c in classes], dtype=np.float32)
        # inverse-frequency-ish weights
        weights = counts.sum() / (len(classes) * counts + 1e-8)
        w = torch.tensor(weights, dtype=torch.float32, device=self.cfg.device)
        self._ce_criterion = nn.CrossEntropyLoss(weight=w)

    def _step_ce(self, batch) -> torch.Tensor:
        x, y = batch
        x = x.to(self.cfg.device).float()
        y = y.to(self.cfg.device).long()
        logits = self.model(x)
        return self._ce_criterion(logits, y)

    # ----------------------------
    # Metrics
    # ----------------------------
    def _evaluate(self, loader: DataLoader, task: str = 'binary') -> Dict[str, float]:
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.cfg.device).float()
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                ys.append(y.numpy())
                ps.append(prob)
        y_true = np.concatenate(ys, axis=0)
        y_prob = np.concatenate(ps, axis=0)
        y_pred = y_prob.argmax(axis=1)

        metrics: Dict[str, Any] = {}
        metrics['acc'] = float(accuracy_score(y_true, y_pred))

        if task == 'binary' and y_prob.shape[1] >= 2:
            # binary metrics (positive class = 1)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            metrics['precision'] = float(prec)
            metrics['recall'] = float(rec)
            metrics['f1'] = float(f1)
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except Exception:
                metrics['auc'] = float('nan')
            try:
                metrics['auprc'] = float(average_precision_score(y_true, y_prob[:, 1]))
            except Exception:
                metrics['auprc'] = float('nan')
        else:
            # multi-class (macro)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['precision'] = float(prec)
            metrics['recall'] = float(rec)
            metrics['f1'] = float(f1)
            # One-vs-rest AUROC / AUPRC (macro)
            classes = np.unique(y_true)
            y_bin = label_binarize(y_true, classes=classes)
            try:
                metrics['auc'] = float(roc_auc_score(y_bin, y_prob[:, classes], multi_class='ovr', average='macro'))
            except Exception:
                metrics['auc'] = float('nan')
            try:
                # average_precision_score supports multilabel [N,K]
                metrics['auprc'] = float(average_precision_score(y_bin, y_prob[:, classes], average='macro'))
            except Exception:
                metrics['auprc'] = float('nan')

        return metrics

    # ----------------------------
    # Public API
    # ----------------------------
    def train(self, setting: str | None = None):
        print("Starting classification training with config:", self.cfg)

        # Loaders
        train_pair_loader = DataLoader(
            self.train_pair,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_pair,
        )
        train_ce_loader = DataLoader(
            self.train_ce,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_single,
        )
        val_loader = DataLoader(
            self.val_ce,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_single,
        )

        best_metric = -np.inf
        best_state = None

        # training steps per epoch
        steps = min(len(train_pair_loader), len(train_ce_loader)) if self.cfg.pair_mode else len(train_ce_loader)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_c, total_ce = 0.0, 0.0

            it_pair = iter(train_pair_loader)
            it_ce   = iter(train_ce_loader)

            for _ in range(steps):
                self.opt.zero_grad()

                loss_c = torch.tensor(0.0, device=self.cfg.device)
                if self.cfg.contrastive_weight > 0.0 and self.cfg.pair_mode:
                    try:
                        batch_pair = next(it_pair)
                    except StopIteration:
                        break
                    loss_c = self._step_pair(batch_pair) * self.cfg.contrastive_weight

                try:
                    batch_ce = next(it_ce)
                except StopIteration:
                    break
                loss_ce = self._step_ce(batch_ce) * self.cfg.ce_weight

                loss = loss_c + loss_ce
                loss.backward()
                self.opt.step()

                total_c  += float(loss_c.detach().cpu())
                total_ce += float(loss_ce.detach().cpu())

            # Validation
            metrics = self._evaluate(val_loader, task=self.cfg.task)
            # selection metric: AUC (binary) else ACC
            score = metrics.get('auc', metrics.get('acc', 0.0)) if self.cfg.task == 'binary' else metrics.get('acc', 0.0)
            print(f"Epoch {epoch+1}/{self.cfg.epochs} | contrastive: {total_c/max(1,steps):.4f} | "
                  f"ce: {total_ce/max(1,steps):.4f} | val: {metrics}")

            if score > best_metric:
                best_metric = score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Save best checkpoint
        os.makedirs('./outputs/checkpoints_classify', exist_ok=True)
        tag = (setting or f"classify_{getattr(self.args, 'model_id', 'AD')}")
        ckpt_path = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}.pt'
        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"Saved best model to {ckpt_path} (best score={best_metric:.4f})")
        else:
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved last model to {ckpt_path}")

    def test(self, setting: str | None = None, test: int = 0) -> Dict[str, float]:
        # Load checkpoint (explicit or inferred)
        tag = (setting or f"classify_{getattr(self.args, 'model_id', 'AD')}")
        ckpt = getattr(self.args, 'load_checkpoints', None)
        if ckpt is None:
            ckpt = f'./outputs/checkpoints_classify/{tag}_{self.cfg.task}.pt'
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint for test: {ckpt}")
        else:
            print(f"[Warn] No checkpoint found at {ckpt}; testing current weights.")

        # Test loader (held‑out split from __init__)
        loader = DataLoader(
            self.test_ce,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=self._collate_single,
        )
        metrics = self._evaluate(loader, task=self.cfg.task)
        print("Test metrics:", metrics)

        # Dump JSON for aggregation (e.g., 5‑seed mean±std)
        os.makedirs('./outputs/results_classify', exist_ok=True)
        seed = getattr(self.args, 'seed', 0)
        outp = f'./outputs/results_classify/{tag}_{self.cfg.task}_seed{seed}.json'
        with open(outp, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved test metrics to {outp}")
        return metrics
