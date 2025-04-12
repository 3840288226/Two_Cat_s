import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_iou(pred_mask, true_mask, num_classes=3):
    ious = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (true_mask == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # 避免除0
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)

def compute_f1(pred_mask, true_mask, num_classes=3):
    pred_flat = pred_mask.view(-1).cpu().numpy()
    true_flat = true_mask.view(-1).cpu().numpy()
    return f1_score(true_flat, pred_flat, average='macro')