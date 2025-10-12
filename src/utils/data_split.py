import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def extract_test_set():
    df = pd.read_csv('data_processed/clinical_labels.csv')

    complete_mask = (df[['ER', 'PR', 'HER2']] != -1).all(axis=1)
    complete_df = df[complete_mask].copy()
    
    # Ensure unique patients
    unique_patients = complete_df['ID'].unique()
    n_test = 24
    if n_test > len(unique_patients):
        raise ValueError("Not enough unique patients to select test set.")

    # Randomly select patients
    np.random.seed(12345)
    selected_patients = np.random.permutation(unique_patients)[:n_test]

    test_df = complete_df[complete_df['ID'].isin(selected_patients)].copy()
    remaining_df = complete_df[~complete_df['ID'].isin(selected_patients)].copy()

    return test_df, remaining_df

def split_complete_partial(df: pd.DataFrame):
    """
    Split the dataset into complete and partial samples.

    A complete sample has all three labels != -1.
    A partial sample has at least one label == -1.
    """
    complete_mask = (df[['ER', 'PR', 'HER2']] != -1).all(axis=1)
    complete_df = df[complete_mask].copy()
    partial_df = df[~complete_mask].copy()

    # print(f"Complete samples: {len(complete_df)}, Partial samples: {len(partial_df)}")
    return complete_df, partial_df



def stratified_multilabel_split(df: pd.DataFrame, n_splits=5, seed=12345):
    """
    Perform stratified K-fold splits for multilabel data using only complete samples for validation.

    Args:
        df (pd.DataFrame): Must contain columns ['ID', 'ER', 'PR', 'HER2'].
                           Missing labels are encoded as -1.
        n_splits (int): Number of folds.
        seed (int): Random seed.

    Returns:
        folds: list of (train_df, val_df) pairs.
               Each train_df includes: complete_train + all partial samples.
               Each val_df includes: complete_val only.
    """
    # --- Split into complete and partial ---
    complete_mask = (df[['ER', 'PR', 'HER2']] != -1).all(axis=1)
    complete_df = df[complete_mask].copy()
    partial_df = df[~complete_mask].copy()

    # --- Prepare label codes for stratification ---
    labels = complete_df[['ER', 'PR', 'HER2']].values
    y_code = np.sum(labels * (2 ** np.arange(labels.shape[1])), axis=1)

    # --- Stratified splits ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(y_code)), y_code):
        train_complete = complete_df.iloc[train_idx]
        val_complete = complete_df.iloc[val_idx]

        # Training always includes partials
        train_fold = pd.concat([train_complete, partial_df], ignore_index=True)
        folds.append((train_fold, val_complete))

    return folds
