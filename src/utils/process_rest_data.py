import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import random

random.seed(12345)

from transformations import bbox_crop, pad_and_resize
import SimpleITK as sitk

RAW_DATA = "data/images_masks"
CLINICAL_XLSX = "data/clinical.xlsx"

TRAIN_PROCESSED_DATA = "data_processed/train_data/images_masks"
TRAIN_PROCESSED_CSV = "data_processed/train_data/clinical_labels.csv"

VAL_PROCESSED_DATA = "data_processed/val_data/images_masks"
VAL_PROCESSED_CSV = "data_processed/val_data/clinical_labels.csv"

TEST_PROCESSED_DATA = "data_processed/test_data/images_masks"
TEST_PROCESSED_CSV = "data_processed/test_data/clinical_labels.csv"

MASKS_PERCENTAGES_CSV = "mask_outside_percentages_original_size.csv"

from constants import TARGET_SHAPE, DTYPE

# --- Bias field correction utility ---
def bias_field_correction(img_np):

    # Convert numpy to SimpleITK image
    img_sitk = sitk.GetImageFromArray(img_np)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    img_corrected = corrector.Execute(img_sitk)
    img_np_corrected = sitk.GetArrayFromImage(img_corrected)
    # SimpleITK uses z,y,x order, transpose back to H, W, D
    if img_np_corrected.shape != img_np.shape:
        img_np_corrected = np.transpose(img_np_corrected, (2, 1, 0))
    return img_np_corrected

# Load and clean clinical data
clinical_df = pd.read_excel(CLINICAL_XLSX, sheet_name="Clinical+Genetics")
clinical_df.columns = clinical_df.columns.str.strip()
clinical_df = clinical_df.dropna(subset=['ER', 'PR', 'HER2'], how='all')
clinical_df = clinical_df.set_index('ID').sort_index()

masks_to_ignore_df = pd.read_csv(MASKS_PERCENTAGES_CSV)
masks_to_ignore = set(masks_to_ignore_df[masks_to_ignore_df['percent_outside'] > 60.0]['mask_file'].tolist())

os.makedirs(TRAIN_PROCESSED_DATA, exist_ok=True)
os.makedirs(VAL_PROCESSED_DATA, exist_ok=True)
os.makedirs(TEST_PROCESSED_DATA, exist_ok=True)

# Transform text labels into numbers in the dataframe
def transform_labels(label):
    if pd.isna(label):
        return np.nan
    label = str(label).strip().lower()
    if label == '+':
        return 1
    elif label == '-':
        return 0
    else:
        return -1  # unknown / not standard

for col in ['ER', 'PR', 'HER2']:
    clinical_df[col] = clinical_df[col].apply(transform_labels)

def balanced_split(clinical_df: pd.DataFrame, val_size=24, test_size=24, max_iter=5000):
    """
    Perform iterative balanced splitting of dataset.
    Returns: train_ids, val_ids, test_ids
    """

    # Step 1: Identify complete / incomplete samples
    complete_mask = clinical_df[['ER', 'PR', 'HER2']].notna().all(axis=1)
    complete_samples = clinical_df[complete_mask]
    incomplete_samples = clinical_df[~complete_mask]

    # Fix: all incomplete go into train
    train_ids = set(incomplete_samples.index.tolist())

    # Step 2: Initialize val/test with random complete samples
    complete_ids = complete_samples.index.tolist()
    random.shuffle(complete_ids)
    val_ids = set(complete_ids[:val_size])
    test_ids = set(complete_ids[val_size:val_size+test_size])
    train_ids.update(complete_ids[val_size+test_size:])

    # Helper: compute skew metric
    def compute_skew(train_ids, val_ids, test_ids):
        ratios = {}
        for label in ['ER', 'PR', 'HER2']:
            stats = {}
            for fold_name, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
                subset = clinical_df.loc[list(ids)][label]
                pos = (subset == 1).sum()
                neg = (subset == 0).sum()
                total = pos + neg
                ratio = pos / total if total > 0 else 0.0
                stats[fold_name] = ratio
            mean_ratio = np.mean(list(stats.values()))
            for fold_name in stats:
                ratios[(label, fold_name)] = abs(stats[fold_name] - mean_ratio)
        return max(ratios.values())

    # Step 3: Iterative balancing (swap samples)
    best_skew = compute_skew(train_ids, val_ids, test_ids)

    for _ in range(max_iter):
        # Pick two samples from different folds
        fold_a, fold_b = random.choice(['train', 'val', 'test']), random.choice(['train', 'val', 'test'])
        if fold_a == fold_b:
            continue

        ids_a = list(eval(f"{fold_a}_ids"))
        ids_b = list(eval(f"{fold_b}_ids"))
        if not ids_a or not ids_b:
            continue

        sa = random.choice(ids_a)
        sb = random.choice(ids_b)

        # Apply swap (only for complete samples!)
        if sa in incomplete_samples.index or sb in incomplete_samples.index:
            continue

        eval(f"{fold_a}_ids").remove(sa)
        eval(f"{fold_b}_ids").remove(sb)
        eval(f"{fold_a}_ids").add(sb)
        eval(f"{fold_b}_ids").add(sa)

        # Recompute skew
        new_skew = compute_skew(train_ids, val_ids, test_ids)

        if new_skew <= best_skew:
            best_skew = new_skew
        else:
            # revert swap
            eval(f"{fold_a}_ids").remove(sb)
            eval(f"{fold_b}_ids").remove(sa)
            eval(f"{fold_a}_ids").add(sa)
            eval(f"{fold_b}_ids").add(sb)

    # Return as lists in original dataframe order
    train_ids = [idx for idx in clinical_df.index if idx in train_ids]
    val_ids = [idx for idx in clinical_df.index if idx in val_ids]
    test_ids = [idx for idx in clinical_df.index if idx in test_ids]

    return train_ids, val_ids, test_ids

val_ids = os.listdir(VAL_PROCESSED_DATA)
test_ids = os.listdir(TEST_PROCESSED_DATA)
# print(len(val_samples))
# exit()
val_samples = clinical_df.loc[val_ids].index.tolist()
test_samples = clinical_df.loc[test_ids].index.tolist()


# # Helper function to calculate and print class distribution
# def print_class_distribution(split_name, samples, clinical_df):
#     print(f"Class distribution for {split_name}:")
#     for label in ['ER', 'PR', 'HER2']:
#         subset = clinical_df.loc[samples][label]
#         pos = (subset == 1).sum()
#         neg = (subset == 0).sum()
#         nan = subset.isna().sum()
#         print(f"  {label}: Positive: {pos}, Negative: {neg}, NaN: {nan}")


def process_set(samples, PROCESSED_DATA, PROCESSED_CSV, is_train=False):
    os.makedirs(PROCESSED_DATA, exist_ok=True)
    processed_labels = []

    for sample in tqdm(samples):
        out_dir = os.path.join(PROCESSED_DATA, sample)
        os.makedirs(out_dir, exist_ok=True)

        sample_dir = os.path.join(RAW_DATA, sample)
        sample_id = sample.strip()

        row = clinical_df.loc[sample_id]

        # Process image
        image_files = [f for f in os.listdir(sample_dir) if "image" in f]   # single file contains image, others are masks
        img_path = os.path.join(sample_dir, image_files[0])                 # access single file by [0].
        img = nib.load(img_path).get_fdata()

        if is_train:
            # Process and combine masks
            mask_files = [f for f in os.listdir(sample_dir) if "mask" in f and f not in masks_to_ignore]
            if not mask_files:
                print(f"WARNING: {sample} has no mask files, saving empty mask.")
            combined_mask = np.zeros(img.shape, dtype=np.uint8)
            for mfile in mask_files:
                mask = nib.load(os.path.join(sample_dir, mfile)).get_fdata()
                combined_mask = np.logical_or(combined_mask, mask > 0)
            combined_mask = combined_mask.astype(np.uint8)

        if is_train:
            img, mask = bbox_crop(img, combined_mask)
        else:
            img = bbox_crop(img, None)


        # binarize image and overlay mask, correcting out regions outside the image
        if is_train:
            bin_img = img > 0
            bin_img_filled = binary_fill_holes(bin_img)
            mask = mask & bin_img_filled.astype(np.uint8)  # constrains mask to be within image region (>0 voxels, filled)

        img = bias_field_correction(img)    # Bias field correction before resizing (downsampling)

        # pad and resize image and mask
        img = pad_and_resize(img, TARGET_SHAPE)
        img = img.transpose((2, 1, 0))  # [W, H, D] ->  [D, H, W]
        img_tensor = torch.from_numpy(img).to(dtype=DTYPE).unsqueeze(0)  # float32 for images
        torch.save(img_tensor, os.path.join(out_dir, "image.pt"))        # [C=1, D, H, W]

        if is_train:
            mask = pad_and_resize(mask, TARGET_SHAPE, is_mask=True) # nearest neighbor for masks
            mask = mask.astype(np.uint8)
            mask = mask.transpose((2, 1, 0))   # [W, H, D] ->  [D, H, W]
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)           # uint8 for masks
            torch.save(mask_tensor, os.path.join(out_dir, "mask.pt"))   # [C=1, D, H, W]
        # batch dim is added from the dataloader

        # processing sample labels (already processed, sanity check)
        label = [
            row["ER"] if not pd.isna(row["ER"]) else -1,
            row["PR"] if not pd.isna(row["PR"]) else -1,
            row["HER2"] if not pd.isna(row["HER2"]) else -1
        ]
        processed_labels.append({"ID": sample_id, "ER": label[0], "PR": label[1], "HER2": label[2]})

    labels_df = pd.DataFrame(processed_labels)
    labels_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Preprocessing complete. Processed data and labels saved in {PROCESSED_CSV}.")


# # Print class distributions
# print_class_distribution("Train", train_samples, clinical_df)
# print_class_distribution("Validation", val_samples, clinical_df)
# print_class_distribution("Test", test_samples, clinical_df)

# prompt user to confirm before proceeding
print(val_ids)
print(test_ids)
proceed = input("Proceed with data processing? (y/n): ")
if proceed.lower() != 'y':
    print("Data processing aborted.")
    exit(0)

# process_set(train_samples, TRAIN_PROCESSED_DATA, TRAIN_PROCESSED_CSV, is_train=True)
process_set(val_samples, VAL_PROCESSED_DATA, VAL_PROCESSED_CSV, is_train=True)
process_set(test_samples, TEST_PROCESSED_DATA, TEST_PROCESSED_CSV, is_train=True)


# Output of class distributions (after running the script):

# Class distribution for Train:
#   ER: Positive: 98, Negative: 90, NaN: 5
#   PR: Positive: 63, Negative: 121, NaN: 9
#   HER2: Positive: 111, Negative: 70, NaN: 12

# Class distribution for Validation:
#   ER: Positive: 12, Negative: 12, NaN: 0
#   PR: Positive: 8, Negative: 16, NaN: 0
#   HER2: Positive: 15, Negative: 9, NaN: 0

# Class distribution for Test:
#   ER: Positive: 12, Negative: 12, NaN: 0
#   PR: Positive: 8, Negative: 16, NaN: 0
#   HER2: Positive: 15, Negative: 9, NaN: 0