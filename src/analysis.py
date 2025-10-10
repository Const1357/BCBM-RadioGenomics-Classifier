# Quantitative Analysis of the Trial Results in experiments

import os
import torch
import optuna
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from collections import defaultdict

from utils.constants import EXPERIMENT_DIR

def parse_tensorboard_eventfile(event_file):
    """
    Parses a single TensorBoard event file and groups metrics by their prefix.
    Returns:
        {
          "train": {"accuracy": [(step, value), ...], "f1": ...},
          "value": {"auc": ...},
          "Loss": {"total": ...}
        }
    """
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = defaultdict(dict)

    for tag in ea.Tags().get("scalars", []):
        parts = tag.split("/", 1)
        if len(parts) == 2:
            prefix, name = parts
        else:
            prefix, name = "root", parts[0]

        data[prefix][name] = [(e.step, e.value) for e in ea.Scalars(tag)]

    return dict(data)

# Example usage:
# dct = parse_tensorboard_eventfile(os.path.join(EXPERIMENT_DIR, "UNet3D", "trial_0", "events.out.tfevents.1759612112.Const.1316969.0"))
# print(dct.keys())
# print(dct['train'].keys())
# print(dct['train']['ER_auc']) # list of (step, value) tuples

df = pd.DataFrame()

for trial_name in os.listdir(os.path.join(EXPERIMENT_DIR, 'UNet3D')):

    trial_num = int(trial_name.split('_')[-1])

    trial_dir = os.path.join(EXPERIMENT_DIR, 'UNet3D', trial_name)
    if not os.path.isdir(trial_dir):
        continue

    event_file = [f for f in os.listdir(trial_dir) if f.startswith("events.out.tfevents")][0]  # there should be only one

    if not event_file:
        print('[WARNING] No event file found in', trial_dir)
        continue

    event_file = os.path.join(trial_dir, event_file)

    data = parse_tensorboard_eventfile(event_file)

    num_epochs = len(data['val']['ER_auc'])
    if num_epochs < 60:
        # trial was pruned
        continue

    # prepare the validation data to a dataframe
    val_data = pd.DataFrame(data['val'])

    for key,value in val_data.items():
        val_data[key] = [v for s,v in value]  # keep only the values, discard the steps


    # compute composite score = (average AUC + min per-step AUC) / 2
    average_auc = val_data['auc_all']
    min_per_step_auc = val_data[['ER_auc', 'PR_auc', 'HER2_auc']].min(axis=1)
    composite_score = (average_auc + min_per_step_auc) / 2
    val_data['composite_score'] = composite_score

    best_index = val_data['composite_score'].argmax()  # index of the best composite score
    val_data = val_data.iloc[best_index]
    val_data = pd.DataFrame(val_data).T  # convert back to dataframe
    val_data.index = [trial_name]
    
    df = pd.concat([df, val_data], axis=0)

df = df.sort_values(by='composite_score', ascending=False)
df.to_csv("experiment_results_summary.csv")
print(df)

df_reduced = df[['ER_auc', 'PR_auc', 'HER2_auc', 'auc_all', 'composite_score']][:10]    # top 10
print(df_reduced)

#             ER_auc    PR_auc  HER2_auc   auc_all  composite_score
# trial_32  0.791667  0.796875  0.748148  0.778897         0.763522 <-- best overall & best for PR: 0.796875
# trial_46  0.763889  0.777344  0.759259  0.766831         0.763045
# trial_27  0.760417  0.742188  0.844444  0.782350         0.762269 <-- best for HER2: 0.844444
# trial_43  0.788194  0.769531  0.748148  0.768625         0.758386
# trial_42  0.864583  0.753906  0.711111  0.776534         0.743822 <-- best for ER: 0.864583
# trial_31  0.781250  0.726562  0.762963  0.756925         0.741744
# trial_16  0.760417  0.714844  0.759259  0.744840         0.729842
# trial_37  0.777778  0.710938  0.751852  0.746856         0.728897
# trial_44  0.732639  0.703125  0.777778  0.737847         0.720486
# trial_41  0.725695  0.710938  0.729630  0.722087         0.716512

# best trial overall: 32 with composite_score = 0.763522 | average_auc = 0.778897

# best trial for ER: 42 with ER_AUC = 0.864583          # easiest class
# best trial for PR: 32 with PR_AUC = 0.796875          # hardest class
# best trial for HER2: 27 with HER2_AUC = 0.844444      # intermediate class

# results are consistent with class ratios => the model is learning something meaningful

# Reminder:
# Label	Positive	Negative	Unknown	    Positive (%)	Negative (%)	Unknown (%)	    Pos/Neg Ratio
    # ER	98	        90	        5	        50.78%	        46.63%	        2.59%	        1.09
    # PR	63	        121	        9	        32.64%	        62.69%	        4.66%	        0.52
    # HER2  111	        70	        12	        57.51%	        36.27%	        6.22%	        1.59