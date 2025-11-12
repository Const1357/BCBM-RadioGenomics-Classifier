import os
import optuna
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils.constants import *

from model_definitions.ResNet import ResNet3D
from utils.MRIDataset import MRIDataset
from utils.transformations import MRI_AUGMENTATION_PIPELINE
from Trainer import Trainer
from utils.losses import AsymmetricFocalLossWithNanHandling
from utils.data_split import extract_test_set, split_complete_partial, stratified_multilabel_split

import json

# Datasets
TEST_DF, TRAIN_DF = extract_test_set()


# Objective function
def objective(trial: optuna.trial.Trial) -> float:

    print(f"Starting new trial: {trial.number}")

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)    # loguniform 
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)    # loguniform 
    depth = trial.suggest_int("network_depth", 5, 5)        # fixed depth to reduce search space since dropping trials to 30
    dropout = trial.suggest_float("dropout", 0.0, 0.3)


    gamma_pos = [
        trial.suggest_float("gamma_pos_ER", 0.0, 1.0),   # ER ~ balanced => low (slightly more positives)
        trial.suggest_float("gamma_pos_PR", 0.0, 1.0),   # PR positives abundant => low gamma_pos
        trial.suggest_float("gamma_pos_HER2", 0.2, 1.5), # HER2 slightly more positives => moderate
    ]

    gamma_neg = [
        trial.suggest_float("gamma_neg_ER", 0.0, 1.0),   # ER balanced => low
        trial.suggest_float("gamma_neg_PR", 1.5, 3.5),   # PR negatives few => moderately high gamma_neg to suppress easy negatives
        trial.suggest_float("gamma_neg_HER2", 1.0, 2.5), # HER2 negatives slightly fewer => moderate gamma_neg
    ]

    shift_m_pos = [
        trial.suggest_float("shift_m_pos_ER", 0.0, 0.05),
        trial.suggest_float("shift_m_pos_PR", 0.0, 0.05),
        trial.suggest_float("shift_m_pos_HER2", 0.0, 0.05),
    ]

    shift_m_neg = [
        trial.suggest_float("shift_m_neg_ER", 0.0, 0.05),
        trial.suggest_float("shift_m_neg_PR", 0.0, 0.05),
        trial.suggest_float("shift_m_neg_HER2", 0.0, 0.05),
    ]


    clf_loss = AsymmetricFocalLossWithNanHandling(
            gamma_pos=torch.tensor(gamma_pos).to(DEVICE),
            gamma_neg=torch.tensor(gamma_neg).to(DEVICE),
            shift_m_pos=torch.tensor(shift_m_pos).to(DEVICE),
            shift_m_neg=torch.tensor(shift_m_neg).to(DEVICE),
    )

    optimizer_kwargs = {
        'lr' : lr,
        'weight_decay' : weight_decay
    }

    # --- Initialize model & optimizer ---
    model = ResNet3D(depth=depth, base_filters=16, clf_threshold=[0.5,0.5,0.5], dropout=dropout).to(DEVICE, DTYPE)
    optimizer = torch.optim.AdamW

    trainer = Trainer(
        model=model,
        optimizer=optimizer,    # class
        optimizer_kwargs = optimizer_kwargs,
        clf_loss=clf_loss,
        num_epochs=NUM_EPOCHS,
        log_train_every=5,
    )

    mean_cv_score = trainer.train_cv(TRAIN_DF, trial=trial)    

    return mean_cv_score
    


def main():

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)

    # best fold models (from any trial, from any fold)
    data = [{"score": score, "path": path, "params": params, "val_metrics" : metrics} for score, path, params, metrics in GLOBAL_TOP_MODELS]
    with open(f"{EXPERIMENT_DIR}/ResNet3D/global_top_models.json", "w") as f:
        json.dump(data, f, indent=2)



if __name__ == "__main__":
    main()

