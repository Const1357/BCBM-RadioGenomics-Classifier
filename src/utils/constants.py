import torch

TARGET_SHAPE = (128, 128, 64)
DTYPE = torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMG_DIR = 'data_processed/train_data/images_masks' 
VAL_IMG_DIR = 'data_processed/val_data/images_masks' 
TEST_IMG_DIR = 'data_processed/test_data/images_masks' 

TRAIN_LABEL_FILE = 'data_processed/train_data/clinical_labels.csv' 
VAL_LABEL_FILE = 'data_processed/val_data/clinical_labels.csv' 
TEST_LABEL_FILE = 'data_processed/test_data/clinical_labels.csv' 