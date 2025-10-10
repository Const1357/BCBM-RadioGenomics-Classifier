# import optuna
# import torch
# from torch.utils.data import DataLoader, RandomSampler
# import torchmetrics
# from utils.constants import *
# from model_definitions.UNet import UNet3D
# from utils.MRIDataset import MRIDataset
# from utils.transformations import MRIAugmentationPipeline

# VAL_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=MRIAugmentationPipeline())
# VAL_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=None)

# bootstrap_sampler = RandomSampler(VAL_DATASET, replacement=True, num_samples=len(VAL_DATASET))

# val_loader = DataLoader(
#     VAL_DATASET,
#     batch_size=2, sampler=bootstrap_sampler, shuffle=False
# )

# n_bootstraps = 500

# # Top 5 models
# #             ER_auc    PR_auc  HER2_auc   auc_all  composite_score
# # trial_32  0.791667  0.796875  0.748148  0.778897         0.763522 <-- best overall & best for PR: 0.796875
# # trial_46  0.763889  0.777344  0.759259  0.766831         0.763045
# # trial_27  0.760417  0.742188  0.844444  0.782350         0.762269 <-- best for HER2: 0.844444
# # trial_43  0.788194  0.769531  0.748148  0.768625         0.758386
# # trial_42  0.864583  0.753906  0.711111  0.776534         0.743822 <-- best for ER: 0.864583

# models = ['trial_32', 'trial_46', 'trial_27', 'trial_43', 'trial_42']   # best composite and PR, 2nd best composite, best ER, and best HER2
# state_dicts = {m: torch.load(f"{EXPERIMENT_DIR}/UNet3D/{m}/model.pt", map_location=DEVICE) for m in models}

# for model_name,state_dict in zip(models,state_dicts):

#     def objective(trial: optuna.trial.Trial) -> float:

#         ER_threshold = trial.suggest_float("ER_threshold", 0.01, 0.99)
#         PR_threshold = trial.suggest_float("PR_threshold", 0.01, 0.99)
#         HER2_threshold = trial.suggest_float("HER2_threshold", 0.01, 0.99)
#         clf_threshold = [ER_threshold, PR_threshold, HER2_threshold]

#         model = UNet3D(depth=5, base_filters=16, clf_threshold=clf_threshold).to(DEVICE, DTYPE)
#         model.load_state_dict(state_dicts[model_name])  # load best model

#         model.eval()

#         ret_metric = 0.0

#         for i in range(n_bootstraps):
#             all_preds = []
#             all_probs = []
#             all_labels = []

#             with torch.no_grad():
#                 for batch in val_loader:

#                     img, labels = batch

#                     img = img.to(DEVICE)
#                     labels = labels.to(DEVICE)

#                     with torch.amp.autocast(device_type='cuda'):  # mixed precision
#                         preds, probs = model.predict(img)

#                     all_preds.append(preds.cpu())     # [B, 3]
#                     all_probs.append(probs.cpu())     # [B, 3]
#                     all_labels.append(labels.cpu())   # [B, 3]

#             # N = len(eval_loader)
#             all_preds = torch.cat(all_preds)    # [N, 3]
#             all_probs = torch.cat(all_probs)    # [N, 3]
#             all_labels = torch.cat(all_labels)  # [N, 3] ER, PR, HER2, binary classes (multilabel task, num labels=3)

#             # Compute metrics per class
#             class_names = ['ER', 'PR', 'HER2']
#             accs, precs, recs, f1s, aucs = [], [], [], [], []
#             for class_idx, class_name in enumerate(class_names):
#                 preds_class = all_preds[:, class_idx]
#                 labels_class = all_labels[:, class_idx]

#                 acc = torchmetrics.functional.accuracy(preds_class, labels_class, task='binary')
#                 prec = torchmetrics.functional.precision(preds_class, labels_class, task='binary')
#                 rec = torchmetrics.functional.recall(preds_class, labels_class, task='binary')
#                 f1 = torchmetrics.functional.f1_score(preds_class, labels_class, task='binary')
#                 auc = torchmetrics.functional.auroc(all_probs[:, class_idx], labels_class, task='binary')
                
#                 accs.append(acc.item())
#                 precs.append(prec.item())
#                 recs.append(rec.item())
#                 f1s.append(f1.item())
#                 aucs.append(auc.item())

#             # per-class metrics
#             metrics = {}
#             for class_idx, class_name in enumerate(class_names):
#                 metrics[class_name] = {
#                     'accuracy': accs[class_idx],
#                     'precision': precs[class_idx],
#                     'recall': recs[class_idx],
#                     'f1': f1s[class_idx],
#                     'auc': aucs[class_idx],
#                 }

#             current_metric = (metrics['ER']['f1'] + metrics['PR']['f1'] + metrics['HER2']['f1']) / 3.0    # average f1
#             ret_metric += current_metric

#             trial.report(ret_metric / (i + 1), step=i)
#             print(f"  Bootstrap {i+1}/{n_bootstraps}, clf_threshold={clf_threshold}, metric={ret_metric / (i+1):.4f}", end='\r')

#         return ret_metric / n_bootstraps

#     study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100))
#     study.optimize(objective, n_trials=50, timeout=None)

#     # Log the best threshold value for the study
#     with open(f"best_threshold_{model_name}.txt", "w") as f:
#         f.write(f"Best trial number: {study.best_trial.number}\n")
#         f.write(f"Best ER threshold: {study.best_trial.params['ER_threshold']:.4f}\n")
#         f.write(f"Best PR threshold: {study.best_trial.params['PR_threshold']:.4f}\n")
#         f.write(f"Best HER2 threshold: {study.best_trial.params['HER2_threshold']:.4f}\n")
#         f.write(f"Best value: {study.best_trial.value:.4f}\n")
#     print(f"Best threshold for {model_name} logged.")


import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.constants import *
from model_definitions.UNet import UNet3D
from utils.MRIDataset import MRIDataset

# Dataset & DataLoader
VAL_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=None)
val_loader = DataLoader(VAL_DATASET, batch_size=2, shuffle=False)

class_names = ['ER', 'PR', 'HER2']

# -------------------------------
# Helper: collect predicted probabilities & labels
# -------------------------------
def collect_features_labels(model, loader):
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.cpu().numpy()
        with torch.no_grad():
            _, probs = model.predict(imgs, return_raw=True)[1:3]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels)
    return np.vstack(all_probs), np.vstack(all_labels)

# -------------------------------
# Main calibration loop
# -------------------------------
models = ['trial_32', 'trial_46', 'trial_27', 'trial_43', 'trial_42']
state_dicts = {m: torch.load(f"{EXPERIMENT_DIR}/UNet3D/{m}/model.pt", map_location=DEVICE) for m in models}

for model_name in models:
    print(f"Evaluating and calibrating model {model_name}...")
    model = UNet3D(depth=5, base_filters=16).to(DEVICE, DTYPE)
    model.load_state_dict(state_dicts[model_name])
    model.eval()

    # Collect predicted probabilities & labels
    X_val, y_val = collect_features_labels(model, val_loader)

    # -------------------------------
    # Calibrate per class using logistic regression
    # -------------------------------
    calibrated_models = {}
    for cls_idx, cls_name in enumerate(class_names):
        y_cls = y_val[:, cls_idx]
        X_cls = X_val[:, cls_idx].reshape(-1, 1)  # predicted probabilities as features

        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X_cls, y_cls)
        calibrated_models[cls_name] = lr

    print(f"Calibrated models for {model_name} ready.")

    # Example: get calibrated probability for a new sample
    # new_probs = model.predict(new_X_tensor, return_raw=True)[1]
    # calibrated_p_ER = calibrated_models['ER'].predict_proba(new_probs[:,0].reshape(-1,1))[:,1]


# Example: use the validation set again
VAL_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=None)
val_loader = DataLoader(VAL_DATASET, batch_size=2, shuffle=False)

class_names = ['ER', 'PR', 'HER2']

# -------------------------------
# Evaluate calibrated models
# -------------------------------

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def evaluate_calibrated_model(model, calibrated_models, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels_np = labels.cpu().numpy()

        with torch.no_grad():
            _, probs = model.predict(imgs, return_raw=True)[1:3]
        probs = probs.cpu().numpy()

        # Apply per-class calibration
        calibrated_probs = np.zeros_like(probs)
        for i, cls_name in enumerate(class_names):
            lr = calibrated_models[cls_name]
            cls_probs = probs[:, i].reshape(-1,1)
            calibrated_probs[:, i] = lr.predict_proba(cls_probs)[:, 1]

        # Predicted labels using threshold 0.5
        preds = (calibrated_probs >= 0.5).astype(int)

        all_preds.append(preds)
        all_probs.append(calibrated_probs)
        all_labels.append(labels_np)

    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Compute metrics per class
    metrics = {}
    for i, cls_name in enumerate(class_names):
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        metrics[cls_name] = {
            "AUC": auc,
            "F1": f1,
            "ConfusionMatrix": cm
        }
    return metrics

# -------------------------------
# Example usage
# -------------------------------
model_name = 'trial_32'
model = UNet3D(depth=5, base_filters=16).to(DEVICE, DTYPE)
model.load_state_dict(torch.load(f"{EXPERIMENT_DIR}/UNet3D/{model_name}/model.pt", map_location=DEVICE))
model.eval()

# Assume `calibrated_models` was built previously
metrics = evaluate_calibrated_model(model, calibrated_models, val_loader)

for cls_name in class_names:
    print(f"Class {cls_name}: AUC={metrics[cls_name]['AUC']:.4f}, F1={metrics[cls_name]['F1']:.4f}")
    print(f"Confusion matrix:\n{metrics[cls_name]['ConfusionMatrix']}\n")