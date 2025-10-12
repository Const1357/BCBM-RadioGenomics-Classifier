import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from utils.constants import DTYPE, DEVICE, PROCESSED_DIR

import os
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, labels_df, processed_dir=PROCESSED_DIR, augmentations=None, return_indices=False):
        """
        Args:
            labels_df (pd.DataFrame): DataFrame with columns ['ID', 'ER', 'PR', 'HER2']
            processed_dir (str): directory containing subfolders for each sample with image.pt
            augmentations (callable, optional): optional augmentation function
        """
        self.processed_dir = processed_dir
        self.labels_df = labels_df.set_index('ID')
        self.samples = self.labels_df.index.tolist()
        self.augmentations = augmentations
        self.return_indices = return_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_dir = os.path.join(self.processed_dir, sample_id)

        # --- Load tensors ---
        img_tensor = torch.load(os.path.join(sample_dir, "image.pt")).float()
        label_row = self.labels_df.loc[sample_id]
        label_tensor = torch.tensor(
            [label_row["ER"], label_row["PR"], label_row["HER2"]],
            dtype=torch.int8,
            # device=DEVICE
        )

        if self.augmentations:
            img_tensor = self.augmentations(img_tensor)

        # z-score normalization after augmentation
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
        
        if self.return_indices:
            return img_tensor, label_tensor, idx
        return img_tensor, label_tensor
        
    def set_augmentations(self, augmentations):
        self.augmentations = augmentations

    
# Example usage:

# dataset = MRIDataset(
#     labels_df = train_df
#     processed_dir="data_processed/train_data/images_masks",
#     augmentations=MRIAugmentationPipeline()
# )