import torch

TARGET_SHAPE = (256, 256, 128)  # (W, H, D) after padding and resizing/cropping
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = 'data_processed/images_masks'

TRAIN_IMG_DIR = 'data_processed/train_data/images_masks' 
VAL_IMG_DIR = 'data_processed/val_data/images_masks' 
TEST_IMG_DIR = 'data_processed/test_data/images_masks' 

TRAIN_LABEL_FILE = 'data_processed/train_data/clinical_labels.csv' 
VAL_LABEL_FILE = 'data_processed/val_data/clinical_labels.csv' 
TEST_LABEL_FILE = 'data_processed/test_data/clinical_labels.csv' 

# EXPERIMENT_DIR = 'experiments/UNet3D'
EXPERIMENT_DIR = 'experiments'

BATCH_SIZE = 4

GLOBAL_TOP_MODELS_K = 10    # 10 candidate models
GLOBAL_TOP_MODELS = []

NUM_EPOCHS = 40
N_TRIALS = 40

print(DEVICE, DTYPE)