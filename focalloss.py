import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        prob = torch.sigmoid(pred)
        pt = target * prob + (1 - target) * (1 - prob)

        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss

        return loss.mean()