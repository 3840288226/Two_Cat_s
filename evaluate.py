import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import UNet
from bc_dataset import BreastCancerDataset
from torchvision.transforms import ToPILImage
import numpy as np

def save_visualization(imgs, masks, preds, save_path, num_samples=5):
    os.makedirs(save_path, exist_ok=True)
    to_pil = ToPILImage()
    for i in range(min(num_samples, imgs.size(0))):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(to_pil(imgs[i].cpu()), cmap='gray')
        axs[0].set_title('Image')
        axs[1].imshow(masks[i].squeeze().cpu(), cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(preds[i].squeeze().cpu().numpy() > 0.5, cmap='gray')
        axs[2].set_title('Prediction')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"sample_{i}.png"))
        plt.close()

def evaluate_model_on_trainset(model_path, img_root, mask_root, device='cuda', save_dir='./results/predictions'):
    device = torch.device(device)
    model = UNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = BreastCancerDataset(img_root, mask_root)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (imgs, masks, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            seg_preds, _ = model(imgs)
            seg_preds = torch.sigmoid(seg_preds)

            # 保存图像（每隔10组保存一次）
            if i % 10 == 0:
                save_visualization(imgs, masks, seg_preds, os.path.join(save_dir, f"batch_{i}"))

if __name__ == '__main__':
    evaluate_model_on_trainset(
        model_path='./results/best_model.pth',
        img_root='/home/ma-user/work/data/images',
        mask_root='/home/ma-user/work/data/masks',
        device='cuda',
        save_dir='./results/predictions'
    )
