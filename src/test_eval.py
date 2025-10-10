import torch
import torchmetrics
from torch.utils.data import DataLoader
from utils.constants import *
from model_definitions.UNet import UNet3D
from utils.MRIDataset import MRIDataset
from utils.transformations import MRIAugmentationPipeline

# --------------------------
# Load test dataset
# --------------------------
TEST_DATASET = MRIDataset(TEST_IMG_DIR, TEST_LABEL_FILE, is_train=False, augmentations=None)
# TEST_DATASET = MRIDataset(VAL_IMG_DIR, VAL_LABEL_FILE, is_train=False, augmentations=None)
test_loader = DataLoader(TEST_DATASET, batch_size=2, shuffle=False)

# --------------------------
# Define models to evaluate
# --------------------------
models = ['trial_32', 'trial_46', 'trial_27', 'trial_43', 'trial_42']   # same 5 top models
state_dicts = {m: torch.load(f"{EXPERIMENT_DIR}/UNet3D/{m}/model.pt", map_location=DEVICE) for m in models}
clf_thresholds = [
    [0.367025, 0.466898, 0.301082],
    [0.204312, 0.618743, 0.384851],
    [0.562326, 0.186415, 0.165311],
    [0.118519, 0.486719, 0.438465],
    [0.462181, 0.288870, 0.416913]
]

clf_thresholds = [
    [0.531 , 0.62,   0.4897],
    [0.5435, 0.661,  0.4604],
    [0.5444, 0.633,  0.4417],
    [0.5596, 0.6987, 0.4639],
    [0.5654, 0.6777, 0.4485],
]


# --------------------------
# Evaluation loop
# --------------------------
for model_name, clf_thresh in zip(models, clf_thresholds):
    print(f"\nEvaluating model: {model_name}")

    model = UNet3D(depth=5, base_filters=16, clf_threshold=clf_thresh).to(DEVICE, DTYPE)
    model.load_state_dict(state_dicts[model_name])
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for img, labels in test_loader:
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast(device_type='cuda'):
                preds, probs = model.predict(img)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate across all batches
    all_preds = torch.cat(all_preds)    # [N, 3]
    all_probs = torch.cat(all_probs)
    # print(all_probs)
    all_labels = torch.cat(all_labels)

    # --------------------------
    # Compute metrics per class
    # --------------------------
    class_names = ['ER', 'PR', 'HER2']
    metrics = {}

    for class_idx, class_name in enumerate(class_names):
        preds_class = all_preds[:, class_idx]
        probs_class = all_probs[:, class_idx]
        labels_class = all_labels[:, class_idx]

        acc = torchmetrics.functional.accuracy(preds_class, labels_class, task='binary')
        prec = torchmetrics.functional.precision(preds_class, labels_class, task='binary')
        rec = torchmetrics.functional.recall(preds_class, labels_class, task='binary')
        f1 = torchmetrics.functional.f1_score(preds_class, labels_class, task='binary')
        auc = torchmetrics.functional.auroc(probs_class, labels_class, task='binary')
        roc = torchmetrics.functional.roc(probs_class, labels_class, task='binary')
        conf_matrix = torchmetrics.functional.confusion_matrix(probs_class, labels_class, task='binary')

        print(roc)

        metrics[class_name] = {
            'accuracy': acc.item(),
            'precision': prec.item(),
            'recall': rec.item(),
            'f1': f1.item(),
            'auc': auc.item(),
            'roc' : roc,
            'conf_matrix' : conf_matrix
        }

    # --------------------------
    # Compute overall (macro-average) metrics
    # --------------------------
    mean_f1 = sum(m['f1'] for m in metrics.values()) / len(class_names)
    mean_auc = sum(m['auc'] for m in metrics.values()) / len(class_names)

    # --------------------------
    # Log results
    # --------------------------
    with open(f"test_metrics_{model_name}.txt", "w") as f:
        for i,class_name in enumerate(class_names):
            m = metrics[class_name]
            f.write(f"{class_name} threshold: {clf_thresh[i]}\n")
            f.write(
                f"{class_name} — Acc: {m['accuracy']:.4f}, "
                f"Prec: {m['precision']:.4f}, Rec: {m['recall']:.4f}, "
                f"F1: {m['f1']:.4f}, AUC: {m['auc']:.4f}\n"
            )
            f.write(f"{m['conf_matrix']}\n\n")
        f.write(f"\nMean F1: {mean_f1:.4f}\nMean AUC: {mean_auc:.4f}\n")

    print(f"Results saved to test_metrics_{model_name}.txt")


# Class distribution for Validation:
#   ER: Positive: 12, Negative: 12, NaN: 0
#   PR: Positive: 8, Negative: 16, NaN: 0
#   HER2: Positive: 15, Negative: 9, NaN: 0

# Class distribution for Test:
#   ER: Positive: 12, Negative: 12, NaN: 0
#   PR: Positive: 8, Negative: 16, NaN: 0
#   HER2: Positive: 15, Negative: 9, NaN: 0