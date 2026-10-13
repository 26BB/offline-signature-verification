import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseEmbeddingNet(nn.Module):
    """
    Twin CNN Architecture for learning discriminative feature representations of signatures.
    Maps 1-channel 128x128 images into a 128-dimensional unit hypersphere embedding.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseEmbeddingNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        embedding = self.fc(feat)
        # L2 normalize so embeddings live on a unit sphere (easier Euclidean/cosine distance thresholding)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function:
    L(y, D) = 0.5 * (1 - y) * D^2 + 0.5 * y * max(0, m - D)^2
    where:
      y = 0 for genuine pairs (encourages distance -> 0)
      y = 1 for forged / impostor pairs (pushes distance >= margin)
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance = F.pairwise_distance(out1, out2)
        loss = torch.mean(
            (1.0 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss
