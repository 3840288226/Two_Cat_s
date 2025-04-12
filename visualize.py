import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid

def visualize_results(images, masks, preds, num_images=3):
    """
    在 notebook 中可视化原图、标注与预测结果。
    """
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))
    
    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 2].imshow(preds[i].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        
    plt.tight_layout()
    plt.show()
