import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class BreastCancerDataset(Dataset):
    def __init__(self, img_root, mask_root):
        self.img_paths = []
        self.mask_paths = []
        self.labels = []

        self.class_map = {
            "benign": 0,
            "malignant": 1,
            "normal": 2
        }

        for class_name, label in self.class_map.items():
            img_dir = os.path.join(img_root, class_name)
            mask_dir = os.path.join(mask_root, class_name + "_mask")

            img_files = glob.glob(os.path.join(img_dir, "*.png"))
            for img_path in img_files:
                base = os.path.basename(img_path).replace(".png", "")
                mask_path = os.path.join(mask_dir, base + "_mask.png")

                if os.path.exists(mask_path):
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(label)

        self.transform_img = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        self.transform_mask = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # 将 mask 二值化，确保只有 0 和 1
        mask = (mask > 0.5).float()

        return img, mask, torch.tensor(self.labels[idx], dtype=torch.long)


