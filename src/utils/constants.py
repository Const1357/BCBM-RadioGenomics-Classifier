import torch

TARGET_SHAPE = (128, 128, 64)
DTYPE = torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
