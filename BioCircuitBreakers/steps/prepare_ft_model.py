import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Subset

import model.train as model_train
from model.util import quantize_tensor

SAMPLE_RATE = 4e-3

def PrepareFinetuneModel(proj, input_dim, TCL, device):
    Model = getattr(model_train, proj.network)
    model_kwargs = {k.replace("model_", ""): v for k, v in vars(proj).items() if k.startswith("model_")}
    net = Model(
        input_dim=input_dim if "M1Only" not in proj.preprocessors else input_dim / 2, 
        **model_kwargs
        )
                                            
    # Load weights
    net.load_state_dict(torch.load(os.path.join(TCL.ckpt_dir, f"ckpt_epoch_{proj.best_epoch}.pth"), map_location=device), strict=False)

    # Prune 
    parameters_to_prune = []
    pruned_module_names = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear) and "decoder" not in name and "logvar" not in name:
            parameters_to_prune.append((module, "weight"))
            pruned_module_names.append(name)
        if isinstance(module, nn.GRUCell):
            parameters_to_prune.append((module, "weight_ih"))
            parameters_to_prune.append((module, "weight_hh"))
            pruned_module_names.append(name)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )
    return net, pruned_module_names