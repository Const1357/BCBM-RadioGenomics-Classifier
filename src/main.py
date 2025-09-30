import sys

from utils.constants import*

from model_definitions.UNet import UNet3D
from utils.MRIDataset import MRIDataset
from utils.transformations import MRIAugmentationPipeline

from torch.utils.data import DataLoader

from Trainer import Trainer
from utils.losses import MixedLoss


# Dataset instantiation.
TRAIN_DATASET = MRIDataset(
    TRAIN_IMG_DIR,
    TRAIN_LABEL_FILE,
    is_train=True,
    augmentations=MRIAugmentationPipeline(),
)

VAL_DATASET = MRIDataset(
    VAL_IMG_DIR,
    VAL_LABEL_FILE,
    is_train=False,
    augmentations=None,
)

TEST_DATASET = MRIDataset(
    TEST_IMG_DIR,
    TEST_LABEL_FILE,
    is_train=False,
    augmentations=None,
)

# define dataloaders
# TODO: define where to get the arguments from
train_loader_kwargs = {}
val_loader_kwargs = {}
test_loader_kwargs = {}

TRAIN_LOADER = DataLoader(TRAIN_DATASET, **train_loader_kwargs) # TODO: define batch size etc during configuration, assume they exist in constants.
VAL_LOADER = DataLoader(VAL_DATASET, **val_loader_kwargs)
TEST_LOADER = DataLoader(TEST_DATASET, **test_loader_kwargs)

model = UNet3D(depth=3, base_filters=16, clf_threshold=0.5, seg_threshold=0.5).to(DEVICE, DTYPE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # TODO: make lr configurable

trainer = Trainer(
    model=model,
    criterion=MixedLoss(mix_coeff=0.5),
    optimizer=optimizer, # TODO: define optimizer
    num_epochs=100,
    log_train_every=10,
)

def main():

    trainer.train_one_epoch(TRAIN_LOADER, epoch=0)
    pass

if __name__ == "__main__":
    main()