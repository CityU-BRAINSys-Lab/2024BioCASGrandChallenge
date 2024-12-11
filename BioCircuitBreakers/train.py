"""
train.py 

Author: Yuanxi
Date: 01.08.2024
Version: "3.0"

This script is used to train a model on the Primate Reaching dataset. 

This verision can be used to train multiple types of models given the model
architecture defined in the `model_train` folder and configurations defined in the 
`hyperparams` folder. 

Use pruning and fake quantization tricks to reduce the model size and complexity.

Final models are named as `final_ckpt.pth` and can be found in the directory 
defined in `-r` argument.
"""
import torch
from utils import CSVLogger, FixSeed

from steps.proj_params import ProjectParams
from steps.proj_log import TrainingCheckpointLogger
from steps.load_data import LoadData
from steps.load_model import LoadModel
from steps.load_preprocessors import LoadPreprocessors
from steps.load_augmentators import LoadAugmentators
from steps.train_model import TrainModel
from steps.prepare_ft_model import PrepareFinetuneModel
from steps.finetune_model import FinetuneModel
from steps.test_model import TestModel

FixSeed(83)

if __name__ == "__main__":
    # Set up the project parameters
    proj = ProjectParams(
        # data_dir = "neurobench/data/primate_reaching/PrimateReachingDataset/",
        # monkey = "loco",
        # result_dir = "results_recwise/",
        # network = "AEGRU",  # ANNModel2D, GRURNN, ANNModel3D, GRUbiRNN
        # all_files = [
        # #     # "indy_20160622_01", 
        # #    # "indy_20160630_01",
        # #     # "indy_20170131_02", 
        # #     # "loco_20170210_03",
        #  "loco_20170215_02",
        # #  "loco_20170301_05"
        # ],
    )
    proj.print()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim, data = LoadData(proj)
    model = LoadModel(proj, input_dim)
    ppc = LoadPreprocessors(proj)
    aug = LoadAugmentators(proj)
    TCL = TrainingCheckpointLogger(proj)
    TrainModel(proj, data, model, ppc, aug, device, TCL)
    model, pruned_module_names = PrepareFinetuneModel(proj, input_dim, TCL, device)
    FinetuneModel(proj, data, model, ppc, aug, device, TCL)
    TestModel(proj, TCL, device, pruned_module_names)