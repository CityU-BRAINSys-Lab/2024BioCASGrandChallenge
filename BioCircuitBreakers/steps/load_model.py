import os
import torch
from model import train as model_train

def LoadModel(proj, input_dim, finetune_from=None):
    
    model = getattr(model_train, proj.network)(input_dim=input_dim / 2 if "M1Only" in proj.preprocessors else input_dim,
                                               **proj.get_model_arch_params()
                                               )

    if finetune_from:
        model.load_state_dict(torch.load(os.path.join(finetune_from, f"ckpt_epoch_{proj.best_epoch}.pth")), strict=False)
        # Freeze all layers except the last one
        # for param in model.parameters():
        #     param.requires_grad = False
        # last_layer_name = '.'.join(list(model.named_parameters())[-1][0].split('.')[:-1])
        # for name, param in model.named_parameters():
        #     if name.startswith(last_layer_name):
        #         param.requires_grad = True
        #         print(f"Unfreezing {name}")
        proj.add_attr(finetune_from=finetune_from)
    
    
    return model