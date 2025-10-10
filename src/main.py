import optuna
import torch
from torch.utils.data import DataLoader

from utils.constants import *
# from model_definitions.UNet import UNet3D
from model_definitions.ResNet import ResNet3D
from utils.MRIDataset import MRIDataset
from utils.transformations import MRIAugmentationPipeline
from Trainer import Trainer
from utils.losses import AsymmetricFocalLossWithNanHandling#, DiceLoss

# Datasets
TRAIN_DATASET = MRIDataset(TRAIN_IMG_DIR, TRAIN_LABEL_FILE, is_train=True, augmentations=MRIAugmentationPipeline())
VAL_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=None)
TEST_DATASET = MRIDataset(TEST_IMG_DIR, TEST_LABEL_FILE, is_train=False, augmentations=None)

batch_size = 2

train_loader_kwargs = {"batch_size": batch_size, "shuffle": True}
val_loader_kwargs = {"batch_size": batch_size, "shuffle": False}
test_loader_kwargs = {"batch_size": batch_size, "shuffle": False}

TRAIN_LOADER = DataLoader(TRAIN_DATASET, **train_loader_kwargs)
VAL_LOADER = DataLoader(VAL_DATASET, **val_loader_kwargs)
TEST_LOADER = DataLoader(TEST_DATASET, **test_loader_kwargs)

# Objective function
def objective(trial: optuna.trial.Trial) -> float:

    print(f"Starting new trial: {trial.number}")

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)    # loguniform 
    mix_coeff = trial.suggest_float("mix_coeff", 0.7, 1.0)  # uniform
    depth = trial.suggest_int("network_depth", 4, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    # Label	Positive	Negative	Unknown	    Positive (%)	Negative (%)	Unknown (%)	    Pos/Neg Ratio
    # ER	98	        90	        5	        50.78%	        46.63%	        2.59%	        1.09
    # PR	63	        121	        9	        32.64%	        62.69%	        4.66%	        0.52
    # HER2  111	        70	        12	        57.51%	        36.27%	        6.22%	        1.59

    # ASL paper suggests γ+ be 0 so that positive samples will incur simple BCE.
    # However, we can set γ+ > 0 if there also exist easy positives (with high confidence: probability > 1 - `shift_m_pos`).
    # γ- in [2.0, 4.0] yields the best results in their experiments (negatives much more common than positives).

    # for an abundance of negatives, γ- > γ+ to downplay the importance of easy negatives (= focus more on positives)
    # for an abundance of positives, γ+ > γ- to downplay the importance of easy positives (= focus more on negatives)

    # ER is balanced => γ+ and γ- can be similar, small shifts
    # HER2 has more positives => γ+ should be larger, shift_m_pos can be larger
    # PR has more negatives => γ- should be larger, shift_m_neg can be larger


    gamma_pos = [
        trial.suggest_float("gamma_pos_ER", 0.2, 1.0),
        trial.suggest_float("gamma_pos_PR", 0.0, 1.0),
        trial.suggest_float("gamma_pos_HER2", 1.5, 3.5),
    ]

    gamma_neg = [
        trial.suggest_float("gamma_neg_ER", 0.0, 1.0),
        trial.suggest_float("gamma_neg_PR", 2.0, 4.0),
        trial.suggest_float("gamma_neg_HER2", 0.0, 1.0),
    ]

    shift_m_pos = [
        trial.suggest_float("shift_m_pos_ER", 0.0, 0.05),
        trial.suggest_float("shift_m_pos_PR", 0.0, 0.0),
        trial.suggest_float("shift_m_pos_HER2", 0.0, 0.2),
    ]

    shift_m_neg = [
        trial.suggest_float("shift_m_neg_ER", 0.0, 0.05),
        trial.suggest_float("shift_m_neg_PR", 0.0, 0.2),
        trial.suggest_float("shift_m_neg_HER2", 0.0, 0.0),
    ]

    # Model
    # model = UNet3D(depth=depth, base_filters=16, clf_threshold=0.5, seg_threshold=0.5, encoder_dropout=dropout).to(DEVICE, DTYPE)
    model = ResNet3D(depth=depth, base_filters=16, clf_threshold=[0.5, 0.5, 0.5], dropout=dropout).to(DEVICE, DTYPE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    clf_loss = AsymmetricFocalLossWithNanHandling(
            gamma_pos=torch.tensor(gamma_pos).to(DEVICE),
            gamma_neg=torch.tensor(gamma_neg).to(DEVICE),
            shift_m_pos=torch.tensor(shift_m_pos).to(DEVICE),
            shift_m_neg=torch.tensor(shift_m_neg).to(DEVICE),
    )
    # seg_loss = DiceLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        clf_loss=clf_loss,
        num_epochs=50,
        log_train_every=5,
    )

    # Train & validate
    trainer.train(TRAIN_LOADER, VAL_LOADER, None, trial=trial)  # Test loader not needed for optuna

    # returning composite objective to ensure that no class has collapsed
    return trainer.best_val_auc_objective  # Return the best validation AUC objective: (average + min) / 2
    


def main():

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
    study.optimize(objective, n_trials=50, timeout=None)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_model_path = f"{EXPERIMENT_DIR}/UNet3D/trial_{trial.number}/model.pt"

    # log best model on a file
    with open("best_model.txt", "w") as f:
        f.write(f"Best trial number: {trial.number}\n")
        f.write(f"Best model path: {best_model_path}\n")


if __name__ == "__main__":
    main()

