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
    task: str = "binary"         
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
            pair_mode = getattr(args, 'pair_mode', True),
            seed = getattr(args, 'seed', 42),
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        tmp_ds = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=self.cfg.pair_mode)
        x0, *_ = tmp_ds[0]
        in_channels = x0.shape[0]
        self.num_classes = tmp_ds.num_classes

        self.model = SiameseClassifier(in_channels, self.num_classes, emb_dim=self.cfg.emb_dim).to(self.cfg.device)

        full_len = len(tmp_ds)
        val_len = max(1, int(full_len * self.cfg.val_split))
        train_len = full_len - val_len

        self.train_pair, self.val_pair = random_split(tmp_ds, [train_len, val_len])

        self.train_ce = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        self.val_ce = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        self.train_ce, self.val_ce = random_split(self.train_ce, [train_len, val_len])

        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _step_pair(self, batch):
        xi, xj, y_sim, y_anchor = batch 
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
                auc = roc_auc_score(ys, ps[:,1])
            except Exception:
                auc = float('nan')
            return {'acc': acc, 'auc': auc}
        else:
            acc = accuracy_score(ys, ps.argmax(axis=1))
            return {'acc': acc}

    def train(self, setting=None):
        print("Starting classification training with config:", self.cfg)
        train_pair_loader = DataLoader(self.train_pair, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.num_workers, drop_last=False, collate_fn=collate_pair)
        train_ce_loader = DataLoader(self.train_ce, batch_size=self.cfg.batch_size, shuffle=True,
                                     num_workers=self.cfg.num_workers, drop_last=False, collate_fn=collate_single)
        val_loader = DataLoader(self.val_ce, batch_size=self.cfg.batch_size, shuffle=False,
                                num_workers=self.cfg.num_workers, drop_last=False, collate_fn=collate_single)

        best_metric = -np.inf
        best_state = None

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_c, total_ce = 0.0, 0.0

            steps = min(len(train_pair_loader), len(train_ce_loader))
            iter_pair = iter(train_pair_loader)
            iter_ce = iter(train_ce_loader)

            for _ in range(steps):
                batch_pair = next(iter_pair)
                batch_ce = next(iter_ce)

                self.opt.zero_grad()

                loss_c = self._step_pair(batch_pair) * self.cfg.contrastive_weight
                loss_ce = self._step_ce(batch_ce) * self.cfg.ce_weight
                loss = loss_c + loss_ce
                loss.backward()
                self.opt.step()

                total_c += loss_c.item()
                total_ce += loss_ce.item()

            metrics = self._evaluate(val_loader, task=self.cfg.task)
            if self.cfg.task == 'binary':
                score = metrics.get('auc', metrics.get('acc', 0.0))
            else:
                score = metrics.get('acc', 0.0)

            print(f"Epoch {epoch+1}/{self.cfg.epochs} | contrastive: {total_c/steps:.4f} | ce: {total_ce/steps:.4f} | val: {metrics}")

            if score > best_metric:
                best_metric = score
                best_state = {k:v.cpu() for k,v in self.model.state_dict().items()}

        os.makedirs('./outputs/checkpoints_classify', exist_ok=True)
        ckpt_path = f'./outputs/checkpoints_classify/{(setting or "classify")}_{self.cfg.task}.pt'
        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"Saved best model to {ckpt_path} (best score={best_metric:.4f})")
        else:
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved last model to {ckpt_path}")

    def test(self, setting=None, test=0):
        full_ds = ADDataset(self.cfg.data_dir, task=self.cfg.task, pair_mode=False)
        loader = DataLoader(full_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        metrics = self._evaluate(loader, task=self.cfg.task)
        print("Test metrics:", metrics)
        return metrics

    def collate_pair(batch):
        # batch: list of (xi, xj, y_sim, y_cls)
        xi, xj, y_sim, y_cls = zip(*batch)
        xi = torch.stack([t.contiguous() for t in xi], dim=0)   # (B, C, L)
        xj = torch.stack([t.contiguous() for t in xj], dim=0)   # (B, C, L)
        y_sim = torch.stack(list(y_sim), dim=0)                 # (B,)
        y_cls = torch.stack(list(y_cls), dim=0)                 # (B,)
        return xi, xj, y_sim, y_cls
    
    def collate_single(batch):
        # batch: list of (x, y)
        x, y = zip(*batch)
        x = torch.stack([t.contiguous() for t in x], dim=0)     # (B, C, L)
        y = torch.stack(list(y), dim=0)                         # (B,)
        return x, y
