import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

from utils.constants import EXPERIMENT_DIR

dict_path = f"{EXPERIMENT_DIR}/ResNet3D/EnsembleResults_3mixed_no_bootstrapping_2.json"

# Global normalization across all matrices
norm = Normalize(vmin=0, vmax=18)

def plot_confusion_matrices(cm_dict, class_names, experiment_dir):
    for class_name in class_names:
        cm = cm_dict[class_name]['confusion_matrix']
        
        # Create display
        disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
        
        # Plot without colorbar
        disp.plot(ax=plt.gca(), cmap='viridis', colorbar=False)
        
        # Apply global normalization and sharp edges
        disp.im_.set_norm(norm)
        disp.im_.set_interpolation('none')
        
        ax = disp.ax_
        
        # Minor ticks for gridlines
        n_classes = cm.shape[0]
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        
        # Add colorbar only for HER2
        if class_name == 'HER2':
            plt.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04)
        
        # ax.set_title(f'Confusion Matrix: {class_name}')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{experiment_dir}/ResNet3D/Confusion_Matrix_3_{class_name}.png", dpi=300)
        plt.close()

from matplotlib.patches import Patch
import seaborn as sns
sns.set_style('darkgrid')

def plot_roc_curves(roc_dict, class_names, experiment_dir):
    for class_name in class_names:
        roc_data = roc_dict[class_name]['roc']
        
        # Handle both old and new formats
        if isinstance(roc_data[0][0], list):
            # New format: nested lists for micro/macro
            fpr = np.array(roc_data[0][0])
            tpr = np.array(roc_data[1][0])
        else:
            # Old format: flat lists
            fpr = np.array(roc_data[0])
            tpr = np.array(roc_data[1])


        # Compute AUC
        roc_auc = np.trapezoid(tpr, fpr)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4))
        # Plot ROC curve
        ax.step(fpr, tpr, color='darkorange', lw=1, label="ROC Curve")
        # Shade area under curve
        ax.fill_between(fpr, 0, tpr, color='darkorange', alpha=0.2)

        # Diagonal baseline
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.01])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        # ax.set_title(f"ROC Curve - {class_name}")

        # Custom legend: shaded rectangle for AUROC
        auc_patch = Patch(facecolor='darkorange', alpha=0.2, label=f"AUROC = {roc_auc:.4f}")
        ax.legend(handles=[auc_patch, ax.lines[0]], loc="lower right")

        plt.tight_layout()
        plt.savefig(f"{experiment_dir}/ResNet3D/ROC_Curve_3_{class_name}.png", dpi=300)
        plt.close(fig)


def generate_metrics_table(metrics_dict, class_names):
    """
    Build and print the LaTeX table (booktabs style).
    """
    rows = []
    for cls in class_names:
        d = metrics_dict[cls]
        cm = np.array(d['confusion_matrix'])
        sup_neg, sup_pos = cm[0].sum(), cm[1].sum()
        tot = sup_neg + sup_pos
        rows.append([
            cls,
            f"{d['accuracy']:.4f}",
            f"{d['precision']:.4f}",
            f"{d['recall']:.4f}",
            f"{d['f1']:.4f}",
            f"{d['auc']:.4f}",
            f"{sup_pos} ({sup_pos/tot*100:.2f}\\%)",
            f"{sup_neg} ({sup_neg/tot*100:.2f}\\%)",
            str(tot)
        ])

    # micro & macro rows
    for avg in ('micro', 'macro'):
        d = metrics_dict[avg]
        rows.append([
            avg,
            f"{d['accuracy']:.4f}",
            f"{d['precision']:.4f}",
            f"{d['recall']:.4f}",
            f"{d['f1']:.4f}",
            f"{d['auc']:.4f}",
            "", ""
        ])

    # plain header strings
    head = ["Receptor", "Accuracy", "Precision", "Recall", "F1 score", "AUROC",
            "Support$^{(+)}$", "Support$^{(-)}$", "Total"]

    # assemble tabular
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    latex = (
        "\\begin{tabular}{l|ccccc|ccc}\n"
        "\\toprule\n"
        + " & ".join(head) + " \\\\\n"
        "\\hline\n"
        + body + " \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    return latex

import json
import os

# --- 1.  Load the JSON -------------------------------------------------------
with open(dict_path, 'r') as f:
    full_dict = json.load(f)

# --- 2.  Build the dictionaries expected by the plotting routines ------------
cm_dict   = {k: {'confusion_matrix': np.array(v['confusion_matrix'])}
             for k, v in full_dict.items() if k in ('ER', 'PR', 'HER2')}

roc_dict  = {k: v for k, v in full_dict.items() if k in ('ER', 'PR', 'HER2', 'micro', 'macro')}

# --- 3.  Produce the outputs -------------------------------------------------
os.makedirs(f"{EXPERIMENT_DIR}/ResNet3D", exist_ok=True)

plot_confusion_matrices(cm_dict, ['ER', 'PR', 'HER2'], EXPERIMENT_DIR)
plot_roc_curves(roc_dict, ['ER', 'PR', 'HER2'], EXPERIMENT_DIR)

# --- 4.  Print the LaTeX table -----------------------------------------------
print(generate_metrics_table(full_dict, ['ER', 'PR', 'HER2']))