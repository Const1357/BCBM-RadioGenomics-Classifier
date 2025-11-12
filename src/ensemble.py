import os
import sys
import datetime

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_definitions.ResNet import ResNet3D
from utils.constants import *
from utils.MRIDataset import MRIDataset
from utils.transformations import MRI_AUGMENTATION_PIPELINE

from utils.data_split import extract_test_set, stratified_multilabel_split
from torch.utils.data import DataLoader

import torchmetrics

from utils.TemperatureCalibrator import MultiLabelTemperatureScaler
from utils.LogisticRegressionCalibrator import MultiLabelLogisticCalibrator
from utils.Calibrator import MixedCalibrator

import json


N_BOOTSTRAPS = 10

class_names = ['ER', 'PR', 'HER2']

TEST_DF, TRAIN_DF = extract_test_set('data_processed/clinical_labels.csv')

folds = stratified_multilabel_split(TRAIN_DF)

train_fold_df, val_fold_df = folds[1]   # best performing fold across all candidate models

val_fold_dataset = MRIDataset(val_fold_df, processed_dir='data_processed/images_masks', augmentations=None)
val_fold_dataset_aug = MRIDataset(val_fold_df, processed_dir='data_processed/images_masks', augmentations=MRI_AUGMENTATION_PIPELINE)
sampler = torch.utils.data.RandomSampler(
    val_fold_dataset_aug,
    replacement=True,
    num_samples=len(val_fold_dataset_aug) * N_BOOTSTRAPS
)

val_fold_loader = DataLoader(val_fold_dataset, batch_size=4, shuffle=False)

# bootstrapped OOF val set (+ Augmentations)
val_fold_loader_aug = DataLoader(val_fold_dataset_aug, batch_size=4, sampler=sampler)

# After qualitative analysis of the model's saliency heatmaps on sampled images of their validation set (OOF)
# the selected models for the ensemble are:
ensemble_model_names = [
    'model_trial7_fold1_score0.6431',
    'model_trial7_fold1_score0.6688',
    'model_trial7_fold1_score0.6384',
    ]
# The visualizations can be found in the notebooks/saliency.ipynb (in the local codebase - not uploaded due to large size: multiple full-res animations)


with open(f"{EXPERIMENT_DIR}/ResNet3D/global_top_models.json", "r") as f:
    global_top_models = json.load(f)

ensemble_models = [None]*len(ensemble_model_names)    # list of dicts with keys: 'name', 'model', 'AUCs'={'ER', 'PR', 'HER2'}

for model_num, model_name in enumerate(ensemble_model_names):

    for g in global_top_models:

        if model_name in g['path']:

            state_dict = torch.load(g['path'])
            params = g['params']
            model = ResNet3D(**params).to(DEVICE)
            model.eval()
            model.load_state_dict(state_dict)
            
            AUCs = {
                'ER': g['val_metrics']['ER']['auc'],
                'PR' : g['val_metrics']['PR']['auc'],
                'HER2' : g['val_metrics']['HER2']['auc'],
            }
            ensemble_models[model_num] = {
                'name' : model_name,
                'model' : model,
                'AUCs' : AUCs,
            }
            break


from sklearn.metrics import roc_curve

# calibrated model wrapper for the ensemble
class EnsembleWrapper(torch.nn.Module):
    def __init__(self, name, model: ResNet3D):
        super().__init__()
        self.name = name
        self.model = model.to(DEVICE)
        self.model.eval()

        self.calibrator = MixedCalibrator()
        self.thresholds = {}  # store per-class thresholds

    def calibrate(self, loader: torch.utils.data.DataLoader = val_fold_loader_aug):

        self.oof_logits = []
        self.oof_labels = []

        for img, label in loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            _, _, logits = self.model.predict(img, return_raw=True)
            self.oof_logits.append(logits.detach().cpu())
            self.oof_labels.append(label.detach().cpu())

        self.oof_logits = torch.cat(self.oof_logits, dim=0)  # [N, 3]
        self.oof_labels = torch.cat(self.oof_labels, dim=0)  # [N, 3]

        # fit calibrator
        self.calibrator.fit(self.oof_logits.float(), self.oof_labels.float(), lr=1e-2, max_iter=1000)


    def forward(self, x: torch.Tensor, return_raw=False) -> torch.Tensor:
        x = x.to(DEVICE)
        calibrated_probs = self.calibrator.predict_proba(self.model.predict(x, return_raw=True)[2])
        if return_raw:
            return calibrated_probs, self.model.predict(x, return_raw=True)[2]
        return calibrated_probs
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = x.to(DEVICE)
        calibrated_probs = self(x)  # [B,3]

        preds = torch.zeros_like(calibrated_probs)
        for c_idx, cname in enumerate(['ER', 'PR', 'HER2']):
            thresh = self.thresholds.get(cname, 0.5)  # fallback
            preds[:, c_idx] = (calibrated_probs[:, c_idx] >= thresh).float()

        return preds, calibrated_probs


class EnsembleClassifier(torch.nn.Module):
    def __init__(self, models: list[EnsembleWrapper], meta_C=0.01):
        super().__init__()
        self.models = torch.nn.ModuleList(models)  # register submodules
        self.thresholds = {'ER': 0.5, 'PR': 0.5, 'HER2': 0.5}  # fallback thresholds
        self.to(DEVICE)
        self.eval()
        

    def compute_thresholds(self, loader: torch.utils.data.DataLoader = val_fold_loader):

        class_names = ['ER', 'PR', 'HER2']

        all_model_probs = {c: [] for c in class_names}  # store per-model probabilities
        all_labels = []

        # Collect calibrated probabilities from each model
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            all_labels.append(labels)

            for model in self.models:
                _preds, probs = model.predict(imgs)  # calibrated probabilities [B,3]
                for c_idx, cname in enumerate(class_names):
                    all_model_probs[cname].append(probs[:, c_idx].detach().cpu())

        # Concatenate over batches
        for cname in class_names:
            all_model_probs[cname] = torch.cat(all_model_probs[cname], dim=0)  # [N, num_models]
        all_labels = torch.cat(all_labels, dim=0).cpu()  # [N,3]

        # Compute ensemble-level thresholds per class (F1-max)
        for c_idx, cname in enumerate(class_names):

            # Average per-model probabilities
            ensemble_probs = torch.stack([all_model_probs[cname][i::len(self.models)] for i in range(len(self.models))], dim=1).mean(dim=1)
            true_labels = all_labels[:, c_idx]


            best_thresh = 0.5
            best_f1 = 0.0

            # Search for best threshold (linear scan in [0.1, 0.9])
            for t in torch.linspace(0.1, 0.9, steps=1000):
                preds = (ensemble_probs >= t).float()
                tp = (preds * true_labels).sum()
                fp = (preds * (1 - true_labels)).sum()
                fn = ((1 - preds) * true_labels).sum()
                f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t.item()

            self.thresholds[cname] = best_thresh
            print(f"Ensemble threshold for {cname}: {best_thresh:.3f} (F1={best_f1:.3f})")
        
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate per-model predictions using simple majority voting.
        Returns:
            ensemble_probs: [B,3] fraction of positive votes per class
            preds: [B,3] binary predictions (0/1) using majority vote
        """
        x = x.to(DEVICE)
        class_names = ['ER', 'PR', 'HER2']

        # Collect thresholded predictions from all models
        all_votes = []
        all_probs = []
        for model in self.models:
            model: EnsembleWrapper
            preds, probs = model.predict(x)  # [B,3] thresholded
            all_votes.append(preds)
            all_probs.append(probs)
        all_votes = torch.stack(all_votes, dim=0)  # [num_models, B, 3]
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, B, 3]

        print(all_probs)


        # ------------- Majority Voting ----------------- (Previous Approach, ignore)
        # num_models = all_votes.shape[0]
        # ensemble_probs = all_votes.float().mean(dim=0)  # fraction of models voting positive
        # preds = torch.zeros_like(ensemble_probs)

        # for c_idx, cname in enumerate(class_names):
        #     # Majority threshold: more than half of models vote positive
        #     majority_threshold = num_models / 2
        #     votes_count = all_votes[:, :, c_idx].sum(dim=0)  # sum over models, shape [B]
        #     preds[:, c_idx] = (votes_count > majority_threshold).float()

        #     # Print positive votes per sample
        #     print(f"Class {cname} positive votes per sample: {votes_count.tolist()}")


        # ------------- Probability Averaging ----------------- (Selected Approach, better results)
        ensemble_probs = all_probs.mean(dim=0)  # [B,3]

        # Apply per-class thresholds
        preds = torch.zeros_like(ensemble_probs)
        for c_idx, cname in enumerate(class_names):

            threshold = 0.5  # fallback to 0.5 if not set
            preds[:, c_idx] = (ensemble_probs[:, c_idx] >= threshold).float()

            # Optional: print mean probability per class
            print(f"Class {cname} mean ensemble probability: {ensemble_probs[:, c_idx].mean().item():.4f}")

        return ensemble_probs, preds

    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict:

        all_labels = []
        all_scores = []
        all_preds = []

        for img, label in loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            probs, preds = self.predict(img)

            if isinstance(probs, np.ndarray):
                probs = torch.tensor(probs, dtype=DTYPE)
            if isinstance(preds, np.ndarray):
                preds = torch.tensor(preds, dtype=DTYPE)

            all_labels.append(label)
            all_scores.append(probs)
            all_preds.append(preds)

        # Concatenate batches: [N,3] and move to DEVICE
        all_labels = torch.cat(all_labels, dim=0).to(DEVICE)
        all_scores = torch.cat(all_scores, dim=0).to(DEVICE)
        all_preds = torch.cat(all_preds, dim=0).to(DEVICE)

        metrics = {}
        num_classes = all_labels.shape[1]

        for class_idx in range(num_classes):
            class_labels = all_labels[:, class_idx].int()        # [N] on DEVICE
            class_predictions = all_preds[:, class_idx].float()  # [N] on DEVICE
            class_scores = all_scores[:, class_idx].float()      # [N] on DEVICE

            metrics[class_names[class_idx]] = {
                "accuracy": torchmetrics.functional.accuracy(class_predictions, class_labels, task="binary"),
                "precision": torchmetrics.functional.precision(class_predictions, class_labels, task="binary"),
                "recall": torchmetrics.functional.recall(class_predictions, class_labels, task="binary"),
                "f1": torchmetrics.functional.f1_score(class_predictions, class_labels, task="binary"),
                "auc": torchmetrics.functional.auroc(class_scores, class_labels, task="binary"),
                "roc": torchmetrics.functional.roc(class_scores, class_labels, task="binary"),  # tuple of fpr, tpr, thresholds
                "confusion_matrix": torchmetrics.functional.confusion_matrix(
                    class_predictions, class_labels, task='binary', num_classes=2
                ),
            }

        metrics["micro"] = {
            "accuracy": torchmetrics.functional.accuracy(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="micro"
            ),
            "precision": torchmetrics.functional.precision(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="micro"
            ),
            "recall": torchmetrics.functional.recall(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="micro"
            ),
            "f1": torchmetrics.functional.f1_score(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="micro"
            ),
            "auc": torchmetrics.functional.auroc(
                all_scores, all_labels, task="multilabel", num_labels=num_classes, average="micro"
            ),
            "roc": torchmetrics.functional.roc(
                all_scores, all_labels, task="multilabel", num_labels=num_classes,
            ),
        }

        metrics["macro"] = {
            "accuracy": torchmetrics.functional.accuracy(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="macro"
            ),
            "precision": torchmetrics.functional.precision(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="macro"
            ),
            "recall": torchmetrics.functional.recall(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="macro"
            ),
            "f1": torchmetrics.functional.f1_score(
                all_preds, all_labels, task="multilabel", num_labels=num_classes, average="macro"
            ),
            "auc": torchmetrics.functional.auroc(
                all_scores, all_labels, task="multilabel", num_labels=num_classes, average="macro"
            ),
            "roc": torchmetrics.functional.roc(
                all_scores, all_labels, task="multilabel", num_labels=num_classes,
            ),
        }

        return metrics

# Instantiate EnsembleWrappers for each model
wrapped_models = [
    EnsembleWrapper(
        name=m['name'], 
        model=m['model']) 
    for m in ensemble_models
]

LOADER = val_fold_loader
# LOADER = val_fold_loader_aug  # for val_fold_aug with bootstrapping

for model in wrapped_models:
    model.calibrate(loader=LOADER)       

classifier = EnsembleClassifier(wrapped_models)
classifier.compute_thresholds(loader=LOADER)    # fit ensemble thresholds on OOF val set
print(classifier.thresholds)

test_dataset = MRIDataset(TEST_DF)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

metrics = classifier.evaluate(test_loader)

for classname in ['ER', 'PR', 'HER2']:
    print(metrics[classname]['confusion_matrix'])
    print(metrics[classname]['auc'])

# dump metrics in json
def to_serializable(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

# with open(f"{EXPERIMENT_DIR}/ResNet3D/EnsembleResults_3mixed_{N_BOOTSTRAPS}.json", "w") as f:  # for val_fold_aug with bootstrapping
with open(f"{EXPERIMENT_DIR}/ResNet3D/EnsembleResults_3mixed_no_bootstrapping_2.json", "w") as f:
    json.dump(metrics, f, indent=4, default=to_serializable)

