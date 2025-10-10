import os
import sys
import torch
import numpy as np
import datetime
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import optuna
# enable tensorboard with: tensorboard --logdir experiments/ResNet3D
# then open http://localhost:6006 in browser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import DTYPE, DEVICE, EXPERIMENT_DIR


def composite_score(per_class_auc: torch.Tensor,
                               per_class_j: torch.Tensor,
                               weights=None):
    """
    Compute a composite scalar score combining AUC and Youden J statistics.

    Args:
        per_class_auc: Tensor [C], per-class AUC values in [0,1].
        per_class_j:   Tensor [C], per-class Youden's J values in [0,1].
        weights: dict of weights (optional), same keys as below.

    Returns:
        Scalar torch.Tensor (higher = better).
    """

    if weights is None:
        weights = dict(
            auc_mean=0.40,
            auc_min=0.20,
            j_mean=0.20,
            j_min=0.10,
            std_auc=0.05,
            std_j=0.05
        )

    # Replace NaNs with zeros for safe stats
    per_class_auc = torch.nan_to_num(per_class_auc, nan=0.0)
    per_class_j = torch.nan_to_num(per_class_j, nan=0.0)

    # Clamp to [0,1]
    per_class_auc = per_class_auc.clamp(0, 1)
    per_class_j = per_class_j.clamp(0, 1)

    # Compute stats
    auc_mean = per_class_auc.mean()
    auc_min = per_class_auc.min()
    j_mean = per_class_j.mean()
    j_min = per_class_j.min()
    std_auc = per_class_auc.std(unbiased=False)
    std_j = per_class_j.std(unbiased=False)

    # Weighted combination
    score = (
        weights["auc_mean"] * auc_mean
        + weights["auc_min"] * auc_min
        + weights["j_mean"] * j_mean
        + weights["j_min"] * j_min
        - weights["std_auc"] * std_auc
        - weights["std_j"] * std_j
    )

    return score


class Trainer:
    def __init__(self, 
                 model,
                 optimizer,
                 clf_loss,
                 num_epochs=50,
                 log_train_every=1,
                 ):

        self.model = model.to(DEVICE, dtype=DTYPE)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.log_train_every = log_train_every

        self.scaler = torch.amp.GradScaler(device=DEVICE)  # for mixed precision training

        self.clf_loss = clf_loss
        

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_clf_loss = torch.zeros(3, device=DEVICE, dtype=DTYPE)  # Per-class loss
        total_batches = 0

        for i, (img, mask, labels) in enumerate(train_loader):

            print(f'[{datetime.datetime.now().time().strftime("%H:%M:%S")}] Batch {i+1}/{len(train_loader)} loaded')

            img = img.to(DEVICE, DTYPE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()

            # Compute per-class classification loss
            with torch.amp.autocast(device_type='cuda'):  # mixed precision
                clf_logits = self.model(img)                            # [B, 3]
                per_class_clf_loss = self.clf_loss(clf_logits, labels)  # [C]
                epoch_clf_loss += per_class_clf_loss

                total_loss = per_class_clf_loss.sum()

            self.scaler.scale(total_loss).backward()  # scaled backpropagation

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # inline check for single device
            if int(self.scaler._found_inf_per_device(self.optimizer)[list(self.scaler._found_inf_per_device(self.optimizer).keys())[0]].item()):
                print("Step skipped (nan/inf in gradients)")


            self.scaler.step(self.optimizer)
            self.scaler.update()

            # check_params_for_nan(self.model)
            total_batches += 1

        # Average the losses over all batches
        epoch_clf_loss /= total_batches

        # Log per-class classification loss
        class_names = ['ER', 'PR', 'HER2']
        for class_idx, class_name in enumerate(class_names):
            self.writer.add_scalar(f'Loss/train_clf_{class_name}', epoch_clf_loss[class_idx].item(), epoch)

        return epoch_clf_loss

    def train(self, train_loader, val_loader, test_loader=None, trial: optuna.trial.Trial=None):

        print('train')

        if trial is None:
            self._experiment_dir = os.path.join(EXPERIMENT_DIR, self.model.name, datetime.datetime.now().strftime("experiment_%Y%m%d_%H%M%S"))
        else:
            self._experiment_dir = os.path.join(EXPERIMENT_DIR, self.model.name, f"trial_{trial.number}")

        self._model_dir = os.path.join(self._experiment_dir, "model.pt")
        os.makedirs(self._experiment_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self._experiment_dir)
        self.writer.add_text("config", str(self.model.get_config()), 0)
        self.writer.add_text("optimizer", str(self.optimizer), 0)
        
        # Save Optuna's suggested parameters as a JSON file
        if trial is not None:
            import json
            params_path = os.path.join(self._experiment_dir, "trial_params.json")
            with open(params_path, "w") as f:
                json.dump(trial.params, f, indent=4)

        self.best_val_auc_objective = -float('inf')
        for epoch in range(self.num_epochs):
            train_clf_loss = self.train_one_epoch(train_loader, epoch)

            # Validation every epoch
            val_metrics = self.evaluate(val_loader, epoch, mode='val')

            aucs = torch.tensor([metrics['auc'] for metrics in val_metrics.values()])
            youden_js = torch.tensor([metrics['youden_j'] for metrics in val_metrics.values()]) 

            val_auc_objective = composite_score(aucs, youden_js)
            self.writer.add_scalar(f'val/accuracy_all', val_auc_objective.item(), epoch)


            if val_auc_objective > self.best_val_auc_objective:
                self.best_val_auc_objective = val_auc_objective
                torch.save(self.model.state_dict(), self._model_dir)

            # Log validation metrics
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            for class_name, metrics in val_metrics.items():
                print(f"  {class_name}: ")
                print(f"    Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

            # Train metrics every `log_train_every` epochs
            if epoch % self.log_train_every == 0:
                train_metrics = self.evaluate(train_loader, epoch, mode='train')
                print("  Train Metrics:")
                for class_name, metrics in train_metrics.items():
                    print(f"    {class_name}: Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            
            # Report intermediate objective value to Optuna (to enable pruning)
            if trial is not None:
                trial.report(val_auc_objective, step=epoch)
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.exceptions.TrialPruned()   
                
        # Final evaluation of best performing model on test set
        print('Training complete')

        if test_loader is not None:
            print('Evaluating best model on test set...')
            self.model.load_state_dict(torch.load(self._model_dir))  # best model
            test_metrics = self.evaluate(test_loader, self.num_epochs, mode='test')
            print("Test Metrics:")
            for class_name, metrics in test_metrics.items():
                print(f"  {class_name}: Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        
        return self.best_val_auc_objective

    def evaluate(self, eval_loader, epoch, mode='val'):

        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        if mode == 'val':
            clf_loss = torch.zeros(3, device=DEVICE, dtype=DTYPE)  # Per-class loss

        with torch.no_grad():
            for batch in eval_loader:
                if len(batch) == 3:
                    img, _, labels = batch
                else:
                    img, labels = batch

                img = img.to(DEVICE)
                labels = labels.to(DEVICE)

                if mode == 'val':
                    with torch.amp.autocast(device_type='cuda'):  # mixed precision
                        preds, probs, clf_logits = self.model.predict(img, return_raw=True)
                        per_class_loss = self.clf_loss(clf_logits, labels)  # [C]
                        clf_loss += per_class_loss
                else:
                    with torch.amp.autocast(device_type='cuda'):  # mixed precision
                        preds, probs = self.model.predict(img)

                all_preds.append(preds.cpu())     # [B, 3]
                all_probs.append(probs.cpu())     # [B, 3]
                all_labels.append(labels.cpu())   # [B, 3]

            if mode == 'val':
                clf_loss /= len(eval_loader)
                class_names = ['ER', 'PR', 'HER2']
                for class_idx, class_name in enumerate(class_names):
                    self.writer.add_scalar(f'Loss/val_clf_{class_name}', clf_loss[class_idx].item(), epoch)

        # N = len(eval_loader)
        all_preds = torch.cat(all_preds)    # [N, 3]
        all_probs = torch.cat(all_probs)    # [N, 3], probabilities
        all_labels = torch.cat(all_labels)  # [N, 3] ER, PR, HER2, binary classes (multilabel task, num labels=3)

        # Compute metrics per class
        class_names = ['ER', 'PR', 'HER2']
        accs, precs, recs, f1s, aucs, youden_js = [], [], [], [], [], []
        for class_idx, class_name in enumerate(class_names):
            preds_class = all_preds[:, class_idx]
            probs_class = all_probs[:, class_idx]
            labels_class = all_labels[:, class_idx]

            valid_mask = labels_class != -1 # ignore samples with label -1 (unknown = nan in original dataset, -1 in processed)
            if not valid_mask.any():
                print('[Trainer.evaluate]: should never reach here with current dataset!')
                raise Exception(f"No valid labels for class {class_name}")

            preds_class = preds_class[valid_mask]
            probs_class = probs_class[valid_mask]
            labels_class = labels_class[valid_mask]

            acc = torchmetrics.functional.accuracy(preds_class, labels_class, task='binary')
            prec = torchmetrics.functional.precision(preds_class, labels_class, task='binary')
            rec = torchmetrics.functional.recall(preds_class, labels_class, task='binary')
            f1 = torchmetrics.functional.f1_score(preds_class, labels_class, task='binary')
            auc = torchmetrics.functional.auroc(probs_class, labels_class.int(), task='binary')
            fpr, tpr, thresholds = torchmetrics.functional.roc(probs_class, labels_class.int(), task='binary'); youden_j = torch.max(tpr-fpr)
            conf_matrix = torchmetrics.functional.confusion_matrix(preds_class, labels_class, task='binary')


            self.writer.add_scalar(f'{mode}/{class_name}_accuracy', acc.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_precision', prec.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_recall', rec.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_f1', f1.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_auc', auc.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_youden_j', youden_j)
            self.writer.add_text(f'{mode}/{class_name}_confusion_matrix', str(conf_matrix.cpu().numpy()), epoch)

            accs.append(acc.item())
            precs.append(prec.item())
            recs.append(rec.item())
            f1s.append(f1.item())
            aucs.append(auc.item())
            youden_js.append(youden_j.item())

        # Log averaged metrics
        self.writer.add_scalar(f'{mode}/accuracy_all', np.mean(accs), epoch)
        self.writer.add_scalar(f'{mode}/precision_all', np.mean(precs), epoch)
        self.writer.add_scalar(f'{mode}/recall_all', np.mean(recs), epoch)
        self.writer.add_scalar(f'{mode}/f1_all', np.mean(f1s), epoch)
        self.writer.add_scalar(f'{mode}/auc_all', np.mean(aucs), epoch)
        self.writer.add_scalar(f'{mode}/youden_j_all', np.mean(youden_js), epoch)

        # Return per-class metrics
        metrics = {}
        for class_idx, class_name in enumerate(class_names):
            metrics[class_name] = {
                'accuracy': accs[class_idx],
                'precision': precs[class_idx],
                'recall': recs[class_idx],
                'f1': f1s[class_idx],
                'auc': aucs[class_idx],
                'youden_j': youden_js[class_idx]
            }

        return metrics
    

    def save_checkpoint(self, trial, epoch, checkpoint_path="checkpoint.pth"):
        state = {
            'epoch': epoch,
            'trial_number': trial.number if trial is not None else None,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc_objective': self.best_val_auc_objective
        }
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}, trial {trial.number if trial is not None else 'N/A'} to {checkpoint_path}")

    def __del__(self):
        self.writer.close()
        # self.model.store(self._model_dir)