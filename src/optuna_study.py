# Optuna study for hyperparameter tuning
import optuna
from optuna.trial import Trial

def objective(trial: Trial) -> float:
    # TODO: suggest hyperparameters and instantiate a new Trainer class.
    # modify Trainer class to accept the optuna study/trial to report intermediate values
    # and allow pruning/early stopping.
    pass