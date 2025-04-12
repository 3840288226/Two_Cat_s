import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm
from sklearn.metrics import f1_score
from bc_dataset import BreastCancerDataset
from model import UNet
from loss import CombinedLoss

# IoU 计算函数
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()

# 可视化保存函数：保存预测图像、掩码和原图
def save_visualization(imgs, masks, preds, save_path, num_samples=4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    for i in range(min(num_samples, imgs.size(0))):
        img_np = imgs[i].permute(1, 2, 0).squeeze().numpy()  # CHW -> HWC
        mask_np = masks[i].squeeze().numpy()
        pred_np = preds[i].squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
        axs[0].set_title('Image')
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title('Prediction')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_path}_sample{i}.png")
        plt.close()

# 主训练函数
def train_model(img_root, mask_root, batch_size, lr, num_epochs, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集
    train_dataset = BreastCancerDataset(img_root, mask_root)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = UNet(num_classes=3).to(device)
    criterion_seg = CombinedLoss(dice_weight=0.5, focal_weight=0.5).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_miou = 0.0
    history = {"loss": [], "f1": [], "miou": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        cls_preds_all = []
        cls_labels_all = []
        miou_total = 0

        for imgs, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", ncols=100):
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            seg_out, cls_out = model(imgs)

            masks_bin = (masks > 0.5).float()
            if masks_bin.dim() == 3:
                masks_bin = masks_bin.unsqueeze(1)

            loss_seg = criterion_seg(seg_out, masks_bin)
            loss_cls = criterion_cls(cls_out, labels)
            loss = loss_seg + loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cls_preds = torch.argmax(cls_out, dim=1).detach().cpu().numpy()
            cls_labels = labels.detach().cpu().numpy()
            cls_preds_all.extend(cls_preds)
            cls_labels_all.extend(cls_labels)

            seg_preds = torch.sigmoid(seg_out).detach()
            miou_batch = compute_iou(seg_preds, masks_bin)
            miou_total += miou_batch

        avg_loss = total_loss / len(train_loader)
        avg_miou = miou_total / len(train_loader)
        f1 = f1_score(cls_labels_all, cls_preds_all, average='macro')

        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | F1-score: {f1:.4f} | mIoU: {avg_miou:.4f}")
        history["loss"].append(avg_loss)
        history["f1"].append(f1)
        history["miou"].append(avg_miou)

        # 保存最佳模型
        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        # 每 10 个 epoch 保存预测图像
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                sample_imgs, sample_masks, _ = next(iter(train_loader))
                sample_imgs = sample_imgs.to(device)
                seg_out, _ = model(sample_imgs)
                seg_preds = (torch.sigmoid(seg_out) > 0.5).float()

                save_path = os.path.join(save_dir, "visuals", f"epoch_{epoch+1}")
                save_visualization(sample_imgs, sample_masks, seg_preds, save_path)

    print(f"训练完成，最佳分割 mIoU: {best_miou:.4f}")
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(history, f)

    return model

# 运行主入口（用于 notebook 外部运行）
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(
        img_root='/home/ma-user/work/data/images',
        mask_root='/home/ma-user/work/data/masks',
        batch_size=8,
        lr=1e-3,
        num_epochs=50,
        device=device,
        save_dir='./results'
    )