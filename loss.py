import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice 损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 确保是 0~1 概率
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


# Focal 损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return loss.mean()


# Combined 损失（去除调试打印）
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        dice_val = self.dice(inputs, targets)
        focal_val = self.focal(inputs, targets)
        total = self.dice_weight * dice_val + self.focal_weight * focal_val
        return total
