
import torch
from utils.constants import *
import torchmetrics
import numpy as np


class Evaluator():
    
    def __init__(self, models, model_names):
        self.models = models
        self.model_names = model_names

    def evaluate(self, eval_loader, compute_thresholds = False):
        
        for i in range(len(self.models)):
            self.models[i].eval()

        all_preds = [[] for _ in range(len(self.models))]
        all_probs = [[] for _ in range(len(self.models))]
        all_labels = [[] for _ in range(len(self.models))]

        with torch.no_grad():
            for i,model in enumerate(self.models):

                for batch in eval_loader:
                        
                    img, labels = batch

                    img = img.to(DEVICE)
                    labels = labels.to(DEVICE)

                    with torch.amp.autocast(device_type='cuda'):  # mixed precision
                        preds, probs = model.predict(img)

                    all_preds[i].append(preds.cpu())     # [B, 3]
                    all_probs[i].append(probs.cpu())     # [B, 3]
                    all_labels[i].append(labels.cpu())   # [B, 3]



        # M = # models, N = len(eval_loader)
        # Concatenate batches per model → [N, 3]
        all_preds = [torch.cat(p_list, dim=0) for p_list in all_preds]
        all_probs = [torch.cat(p_list, dim=0) for p_list in all_probs]
        all_labels = [torch.cat(l_list, dim=0) for l_list in all_labels]

        # Stack across models → [M, N, 3]
        all_preds = torch.stack(all_preds, dim=0)
        all_probs = torch.stack(all_probs, dim=0)
        all_labels = torch.stack(all_labels, dim=0)

        # all_preds = torch.cat(all_preds)    # [M, N, 3]
        # all_probs = torch.cat(all_probs)    # [M, N, 3], probabilities
        # all_labels = torch.cat(all_labels)  # [M, N, 3]  ER, PR, HER2, binary classes (multilabel task, num labels=3)

        print(all_preds.shape)

        # Compute metrics per class
        class_names = ['ER', 'PR', 'HER2']
        model_metrics = {}

        for m, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            
            accs, precs, recs, f1s, aucs, youden_js, youden_thresholds, conf_matrices = [], [], [], [], [], [], [], []
            for class_idx, class_name in enumerate(class_names):
                preds_class = all_preds[m][:, class_idx]
                probs_class = all_probs[m][:, class_idx]
                labels_class = all_labels[m][:, class_idx]

                acc = torchmetrics.functional.accuracy(preds_class, labels_class, task='binary')
                prec = torchmetrics.functional.precision(preds_class, labels_class, task='binary')
                rec = torchmetrics.functional.recall(preds_class, labels_class, task='binary')
                f1 = torchmetrics.functional.f1_score(preds_class, labels_class, task='binary')
                auc = torchmetrics.functional.auroc(probs_class, labels_class.int(), task='binary')
                fpr, tpr, thresholds = torchmetrics.functional.roc(probs_class, labels_class.int(), task='binary')
                threshold = thresholds[torch.argmax(tpr-fpr)]
                youden_j = torch.max(tpr-fpr)
                conf_matrix = torchmetrics.functional.confusion_matrix(preds_class, labels_class, task='binary')

                accs.append(acc.item())
                precs.append(prec.item())
                recs.append(rec.item())
                f1s.append(f1.item())
                aucs.append(auc.item())
                youden_thresholds.append(threshold.item())
                youden_js.append(youden_j.item())
                conf_matrices.append(conf_matrix.cpu().numpy().tolist())

            # Return per-class metrics
            metrics = {}
            for class_idx, class_name in enumerate(class_names):
                metrics[class_name] = {
                    'accuracy': accs[class_idx],
                    'precision': precs[class_idx],
                    'recall': recs[class_idx],
                    'f1': f1s[class_idx],
                    'auc': aucs[class_idx],
                    'threshold': youden_thresholds[class_idx],
                    'youden_j': youden_js[class_idx],
                    'conf_matrix' : conf_matrices[class_idx],
                }
            model_metrics[model_name] = metrics

        return model_metrics