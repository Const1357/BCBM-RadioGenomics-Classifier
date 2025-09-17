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


# Prepare processed labels
train_processed_labels = []
val_processed_labels = []
test_processed_labels = []

os.makedirs(TRAIN_PROCESSED_DATA, exist_ok=True)
os.makedirs(VAL_PROCESSED_DATA, exist_ok=True)
os.makedirs(TEST_PROCESSED_DATA, exist_ok=True)

samples = list(clinical_df.index)

complete_samples = clinical_df.dropna(subset=['ER', 'PR', 'HER2']).index.tolist()
random.shuffle(complete_samples) # seeded with 12345

# split samples. test set is 24, val set is 24, rest (193) is train
# test and val sets each consist of 24 samples that have all 3 markers, no NAs
# test and val sets will not contain masks. Masks will only be used in training set
# as an auxiliary supervision task to guide the model towards recognizing the ROIs (tumors)
test_samples = complete_samples[:24]
val_samples = complete_samples[24:48]
train_samples = [s for s in samples if s not in test_samples and s not in val_samples]

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
        img_tensor = torch.from_numpy(img).to(dtype=DTYPE).unsqueeze(0) # float16 for efficiency
        torch.save(img_tensor, os.path.join(out_dir, "image.pt"))   # [C=1, D, H, W]

        if is_train:
            mask = pad_and_resize(mask, TARGET_SHAPE, is_mask=True) # nearest neighbor for masks
            mask = mask.astype(np.uint8)
            mask = mask.transpose((2, 1, 0))   # [W, H, D] ->  [D, H, W]
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)           # uint8 for masks
            torch.save(mask_tensor, os.path.join(out_dir, "mask.pt"))   # [C=1, D, H, W]
        # batch dim is added from the dataloader

        # processing sample labels
        def to_tensor(str_label):
            if pd.isna(str_label):
                return -1
            if isinstance(str_label, str):
                return 1 if '+' in str_label.strip().lower() else 0
            return int(str_label)
        label = [
            to_tensor(row["ER"]),
            to_tensor(row["PR"]),
            to_tensor(row["HER2"])
        ]
        processed_labels.append({"ID": sample_id, "ER": label[0], "PR": label[1], "HER2": label[2]})

    labels_df = pd.DataFrame(processed_labels)
    labels_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Preprocessing complete. Processed data and labels saved in {PROCESSED_CSV}.")



process_set(train_samples, TRAIN_PROCESSED_DATA, TRAIN_PROCESSED_CSV, is_train=True)
process_set(val_samples, VAL_PROCESSED_DATA, VAL_PROCESSED_CSV, is_train=False)
process_set(test_samples, TEST_PROCESSED_DATA, TEST_PROCESSED_CSV, is_train=False)