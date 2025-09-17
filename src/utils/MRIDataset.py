import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from utils.transformations import MRIAugmentationPipeline

from utils.constants import DTYPE, DEVICE

class MRIDataset(Dataset):
    def __init__(self, processed_dir, labels_csv, is_train=True, augmentations=None):
        self.processed_dir = processed_dir
        self.labels_df = pd.read_csv(labels_csv).set_index('ID')
        self.samples = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d)) and not d.startswith('.')
        ])
        self.is_train = is_train
        self.augmentations = augmentations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = os.path.join(self.processed_dir, sample)
        img_tensor = torch.load(os.path.join(sample_dir, "image.pt")).to(dtype=DTYPE, device=DEVICE)  # convert back to float32 for processing

        label_row = self.labels_df.loc[sample]
        label_tensor = torch.tensor([label_row["ER"], label_row["PR"], label_row["HER2"]], dtype=torch.uint8, device=DEVICE)

        if self.is_train:
            mask_tensor = torch.load(os.path.join(sample_dir, "mask.pt")).to(dtype=torch.uint8, device=DEVICE)
            if self.augmentations:
                img_tensor, mask_tensor = self.augmentations(img_tensor, mask_tensor)
                # z-score normalization after augmentation
                img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
            return img_tensor, mask_tensor, label_tensor
        else:
            img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8) # z-score normalization
            return img_tensor, label_tensor


    
# Example usage:

# dataset = MRIDataset(
#     processed_dir="data_processed/train_data/images_masks",
#     labels_csv="data_processed/train_data/clinical_labels.csv",
#     is_train=True,
#     augmentations=MRIAugmentationPipeline()
# )