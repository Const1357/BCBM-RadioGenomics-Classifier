import torch

def check_params_for_nan(model):
    """
    Check all parameters of the model for NaNs or Infs.
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[NaN] Parameter '{name}' contains NaN")
        if torch.isinf(param).any():
            print(f"[Inf] Parameter '{name}' contains Inf")
