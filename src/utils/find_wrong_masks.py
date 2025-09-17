import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

DATA_PATH = "data/images_masks"
# TARGET_SHAPE = (128, 128, 64)
CSV_OUT = "mask_outside_percentages_original_size.csv"

# def resize_volume(volume, target_shape=TARGET_SHAPE, order=1):
#     from scipy.ndimage import zoom
#     factors = [t / s for t, s in zip(target_shape, volume.shape)]
#     return zoom(volume, factors, order=order)

def get_filled_image_mask(img):
    """Returns a binary mask of the filled image volume after binarization and hole filling."""
    img_bin = img > 0
    filled = binary_fill_holes(img_bin)
    return filled.astype(np.uint8)

results = []

sample_dirs = [d for d in os.listdir(DATA_PATH) if not d.startswith('.')]
for sample_dir in tqdm(sample_dirs, desc="Samples"):
    sample_path = os.path.join(DATA_PATH, sample_dir)
    image_files = [f for f in os.listdir(sample_path) if "image" in f]
    mask_files = [f for f in os.listdir(sample_path) if "mask" in f]
    if not image_files or not mask_files:
        continue

    # Load and preprocess image
    img = nib.load(os.path.join(sample_path, image_files[0])).get_fdata()
    # img = resize_volume(img, TARGET_SHAPE, order=1)
    filled_img = get_filled_image_mask(img)

    for mask_file in mask_files:
        mask = nib.load(os.path.join(sample_path, mask_file)).get_fdata()
        # mask = resize_volume(mask, TARGET_SHAPE, order=0)
        mask_bin = mask > 0

        # Compute percentage of mask voxels outside the image volume
        mask_outside = mask_bin & (~filled_img.astype(bool))
        n_mask = np.count_nonzero(mask_bin)
        n_outside = np.count_nonzero(mask_outside)
        percent_outside = (n_outside / n_mask) * 100 if n_mask > 0 else 0.0

        results.append({
            "sample": sample_dir,
            "mask_file": mask_file,
            "percent_outside": percent_outside,
            "n_mask_voxels": n_mask,
            "n_outside_voxels": n_outside
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(CSV_OUT, index=False)
print(f"Saved mask outside percentages to {CSV_OUT}")