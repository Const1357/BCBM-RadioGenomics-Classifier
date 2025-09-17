import os
import sys
import torch
import numpy as np
import datetime
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
# enable tensorboard with: tensorboard --logdir models/
# then open http://localhost:6006 in browser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import DTYPE, DEVICE
# from model_definitions.UNet import UNet3D
# from utils.MRIDataset import MRIDataset
# from utils.transformations import MRIAugmentationPipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models') # store location

class Trainer:
    def __init__(self, 
                 model,
                 criterion,
                 optimizer,
                 num_epochs=100,
                 log_train_every=10,
                 ):

        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.log_train_every = log_train_every

        self._experiment_dir = os.path.join(MODEL_DIR, model.name, datetime.datetime.now().strftime("experiment_%Y%m%d_%H%M%S"))
        self._model_dir = os.path.join(self._experiment_dir, "model.pt")
        os.makedirs(self._experiment_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self._experiment_dir)
        self.writer.add_text("config", str(model.get_config()), 0)
        self.writer.add_text("loss/mix_coeff", str(criterion.mix_coeff), 0)
        self.writer.add_text("loss/class_weights", str(criterion.pos_weight.cpu().numpy()), 0)
        self.writer.add_text("optimizer", str(optimizer), 0)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_clf_loss = 0.0
        epoch_seg_loss = 0.0

        for img, mask, labels in train_loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()

            clf_logits, seg_logits = self.model(img)

            loss, clf_loss, seg_loss = self.criterion(clf_logits, seg_logits, labels, mask)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_clf_loss += clf_loss.item()
            epoch_seg_loss += seg_loss.item()

        epoch_loss /= len(train_loader)
        epoch_clf_loss /= len(train_loader)
        epoch_seg_loss /= len(train_loader)

        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Loss/train_clf', epoch_clf_loss, epoch)
        self.writer.add_scalar('Loss/train_seg', epoch_seg_loss, epoch)

        return epoch_loss
    
    def train(self, train_loader, val_loader, test_loader):
        
        best_val_auc = -1.0
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validation every epoch
            val_metrics= self.evaluate(val_loader, epoch, mode='val')
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(self.model.state_dict(), self._model_dir)

            # Train metrics every `log_train_every` epochs
            if epoch % self.log_train_every == 0:
                self.evaluate(train_loader, epoch, mode='train')

        # Final evaluation of best performing model on test set

        self.model.load_state_dict(torch.load(self._model_dir)['model_state_dict']) # best model
        self.evaluate(test_loader, self.num_epochs, mode='test')

    def evaluate(self, eval_loader, epoch, mode='val'):

        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        if mode == 'val':
            clf_loss = 0.0

        with torch.no_grad():

            for batch in eval_loader:
                if len(batch) == 3:
                    img, _, labels = batch
                else:
                    img, labels = batch

                img = img.to(DEVICE)
                labels = labels.to(DEVICE)

                if mode == 'val':
                    preds, probs, clf_logits = self.model.predict(img, return_raw=True)
                    clf_loss += self.criterion(clf_logits, None, labels, None, only_clf=True).item()
                else:
                    preds, probs = self.model.predict(img)

                all_preds.append(preds)
                all_probs.append(probs)
                all_labels.append(labels)

            if mode == 'val':
                clf_loss /= len(eval_loader)
                self.writer.add_scalar('Loss/val_clf', clf_loss, epoch)


        all_preds = torch.cat(all_preds)    # [N, 3]
        all_probs = torch.cat(all_probs)    # [N, 3], probabilities
        all_labels = torch.cat(all_labels)  # [N, 3] ER, PR, HER2, binary classes

        # Compute metrics per class
        class_names = ['ER', 'PR', 'HER2']
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        for class_idx, class_name in enumerate(class_names):
            preds_class = all_preds[:, class_idx]
            probs_class = all_probs[:, class_idx]
            labels_class = all_labels[:, class_idx]

            valid_mask = ~torch.isnan(labels_class)
            if not valid_mask.any():
                print('[Trainer.evaluate]: should never reach here with current dataset!')
                raise Exception(f"No valid labels for class {class_name}")

            preds_class = preds_class[valid_mask]
            probs_class = probs_class[valid_mask]
            labels_class = labels_class[valid_mask]

            acc = torchmetrics.functional.accuracy(preds_class, labels_class)
            prec = torchmetrics.functional.precision(preds_class, labels_class)
            rec = torchmetrics.functional.recall(preds_class, labels_class)
            f1 = torchmetrics.functional.f1_score(preds_class, labels_class)
            auc = torchmetrics.functional.auroc(probs_class, labels_class.int(), num_classes=2)   # binary classification for each class
            conf_matrix = torchmetrics.functional.confusion_matrix(preds_class, labels_class, num_classes=2)

            self.writer.add_scalar(f'{mode}/{class_name}_accuracy', acc.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_precision', prec.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_recall', rec.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_f1', f1.item(), epoch)
            self.writer.add_scalar(f'{mode}/{class_name}_auc', auc.item(), epoch)
            # Optionally log confusion matrix as image or text
            self.writer.add_text(f'{mode}/{class_name}_confusion_matrix', str(conf_matrix.cpu().numpy()), epoch)

            accs.append(acc.item())
            precs.append(prec.item())
            recs.append(rec.item())
            f1s.append(f1.item())
            aucs.append(auc.item())

        # Log averaged metrics
        self.writer.add_scalar(f'{mode}/accuracy_all', np.mean(accs), epoch)
        self.writer.add_scalar(f'{mode}/precision_all', np.mean(precs), epoch)
        self.writer.add_scalar(f'{mode}/recall_all', np.mean(recs), epoch)
        self.writer.add_scalar(f'{mode}/f1_all', np.mean(f1s), epoch)
        self.writer.add_scalar(f'{mode}/auc_all', np.mean(aucs), epoch)

        return {
            'accuracy': np.mean(accs),
            'precision': np.mean(precs),
            'recall': np.mean(recs),
            'f1': np.mean(f1s),
            'auc': np.mean(aucs),
        }

    def __del__(self):
        self.writer.close()
        # self.model.store(self._model_dir)