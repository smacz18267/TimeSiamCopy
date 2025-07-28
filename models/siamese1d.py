import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder1D(nn.Module):
    """
    Simple 1D CNN encoder for multichannel time series.
    Input: (B, C, L)
    Output: embedding vector (B, D)
    """
    def __init__(self, in_channels: int, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, emb_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        h = self.net(x).squeeze(-1) 
        z = self.proj(h)           
        z = F.normalize(z, dim=-1)  
        return z

class SiameseClassifier(nn.Module):
    """
    Siamese encoder + optional classification head
    """
    def __init__(self, in_channels: int, num_classes: int, emb_dim: int = 256):
        super().__init__()
        self.encoder = TemporalEncoder1D(in_channels, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def embed(self, x):
        return self.encoder(x)

    def classify(self, x):
        z = self.embed(x)
        return self.classifier(z)

    def forward(self, x):
        return self.classify(x)

def contrastive_loss(z1, z2, y, margin: float = 1.0):
    d = torch.norm(z1 - z2, dim=1)
    pos = y * (d ** 2)
    neg = (1 - y) * F.relu(margin - d) ** 2
    return (pos + neg).mean()
