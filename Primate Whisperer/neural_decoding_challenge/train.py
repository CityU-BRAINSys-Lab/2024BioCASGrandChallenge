import math
import os
from typing import OrderedDict
import joblib
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from snntorch import surrogate
from torch.utils.data import DataLoader, Subset
import numpy as np

from neurobench.examples.primate_reaching.neural_decoding_challenge.batched_dataset import (
    BatchedPrimateReaching,
)
from neurobench.datasets import PrimateReaching
from neurobench.examples.primate_reaching.neural_decoding_challenge.pl_snn import (
    SpikingNetwork,
)
from neurobench.examples.primate_reaching.neural_decoding_challenge.neurons import (
    SleepyStdLIF,
    SleepyLogLIF,
)
from neurobench.examples.primate_reaching.neural_decoding_challenge.networks import *

torch.set_float32_matmul_precision("medium")
from sklearn.preprocessing import StandardScaler, RobustScaler

from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

import argparse


def weight_init(model, mean, std, print_changes=False):
    """function for weight initialization; CAUTION: only implemented for fully-connected layers for now."""
    for name, param in model.named_parameters():
        if name[0:2] == "fc":
            if print_changes:
                old_mean = param.data.mean().item()
                old_var = param.data.var().item()
            param.data.normal_(mean=mean, std=std)
            if print_changes:
                new_mean = param.data.mean().item()
                new_var = param.data.var().item()
                print(
                    f"Parameter {name}: Mean changed from {old_mean} to {new_mean} and variance changed from {old_var} to {new_var}."
                )


def main(filename, gpu_id):
    data_dir = "/shared/datasets/primate_reaching/PrimateReachingDataset"  # replace with your data directory

    load_checkpoint_folder = None
    TRAIN = True
    BENCHMARK = True

    batch_size = 2048
    seq_len = 1024
    drop_rate = 0.2
    reset_mem_every_step = True
    loss_weight_type = "equal"  # "lin", "equal", "sig", "last"
    output_type = "full"  # "displacement", "full"
    scaler_type = "StandardScaler"  # StandardScaler, RobustScaler
    linear_interpolation = True
    random_init = True
    gru_hidden_size = 20
    kernel_size = 3
    conv_channels = 10

    mean = 0
    std = 0.1
    lr = 0.001 * 3

    pr_dataset = PrimateReaching(
        file_path=data_dir,
        filename=filename,
        num_steps=1,
        train_ratio=0.5,
        label_series=True,
        bin_width=0.004,
        biological_delay=0,
        remove_segments_inactive=False,
    )

    # Create scaler for dataset labels
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()

    if output_type == "full":
        labels = scaler.fit_transform(pr_dataset.labels.T).T
        pr_dataset.labels = torch.tensor(labels)
    else:
        displacement_labels = pr_dataset.labels[:, 1:] - pr_dataset.labels[:, :-1]
        displacement_labels = torch.tensor(
            scaler.fit_transform(displacement_labels.T).T
        )
        displacement_labels = torch.cat(
            (torch.zeros((displacement_labels.shape[0], 1)), displacement_labels), dim=1
        )
        pr_dataset.labels = displacement_labels

    train_step = (
        8  # use only powers of 2 otherwise it breaks the r2 computation during training
    )
    train_dataset = BatchedPrimateReaching(
        Subset(pr_dataset, pr_dataset.ind_train),
        seq_len,
        overlap=int(seq_len - train_step),
    )
    val_dataset = BatchedPrimateReaching(
        Subset(pr_dataset, pr_dataset.ind_val), seq_len, overlap=0
    )
    test_dataset = BatchedPrimateReaching(
        Subset(pr_dataset, pr_dataset.ind_test), seq_len, overlap=0
    )

    def collate_fn(batch):
        """
        Used to make sure sequence length comes as the first dimension.
        """
        data, targets, pr_index = zip(*batch)
        data = torch.stack(data, dim=0).permute(1, 0, 2)
        targets = torch.stack(targets, dim=0).permute(1, 0, 2)
        pr_index = torch.stack(pr_index, dim=0).permute(1, 0)
        return data, targets, pr_index

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    network = ConvSNN

    tbl = TensorBoardLogger(save_dir=os.getcwd())
    cvl = CSVLogger(save_dir=os.getcwd(), version=tbl.version)
    # save checkpoint with lowest val_loss and highest val_r2

    checkpoint_callback_val_loss = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss for improvement
        dirpath=tbl.log_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",  # Filename format
        save_top_k=1,  # Save only the best model
        mode="min",  # Minimize validation loss
        save_last=True,  # Additionally, always save the last model
    )

    checkpoint_callback_r2 = ModelCheckpoint(
        monitor="val_r2_seq",  # Monitor r2 for improvement
        dirpath=tbl.log_dir,
        filename="model-{epoch:02d}-{val_r2_seq:.2f}",  # Filename format
        save_top_k=1,  # Save only the best model
        mode="max",  # Maximize r2
        save_last=True,  # Additionally, always save the last model
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=30,
            verbose=False,
            mode="min",
        ),
        checkpoint_callback_val_loss,
        checkpoint_callback_r2,
    ]

    scaler_path = f"{tbl.log_dir}/scaler.save"
    hyperparameters_path = f"{tbl.log_dir}/hyperparameters.save"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    hyperparameters = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "drop_rate": drop_rate,
        "reset_mem_every_step": reset_mem_every_step,
        "loss_weight_type": loss_weight_type,
        "output_type": output_type,
        "scaler_type": scaler_type,
        "scaler_path": scaler_path,
        "linear_interpolation": linear_interpolation,
        "mean": mean,
        "std": std,
        "lr": lr,
        "gru_hidden_size": gru_hidden_size,
        "kernel_size": kernel_size,
        "conv_channels": conv_channels,
        "random_init": random_init,
        "train_step": train_step,
    }

    joblib.dump(hyperparameters, hyperparameters_path)
    scaler = joblib.load(hyperparameters["scaler_path"])

    net = network(
        input_dim=train_dataset.input_feature_size,
        seq_len=hyperparameters["seq_len"],
        output_dim=2,
        batch_size=hyperparameters["batch_size"],
        drop_rate=hyperparameters["drop_rate"],
        reset_mem_every_step=hyperparameters["reset_mem_every_step"],
        linear_interpolation=hyperparameters["linear_interpolation"],
        gru_hidden_size=hyperparameters["gru_hidden_size"],
        kernel_size=hyperparameters["kernel_size"],
        conv_channels=hyperparameters["conv_channels"],
        random_init=hyperparameters["random_init"],
    )

    weight_init(
        net, hyperparameters["mean"], hyperparameters["std"], print_changes=True
    )

    model = SpikingNetwork(
        net,
        lr=hyperparameters["lr"],
        spike_regu=0,
        target_spike_share=0,
        scaler=scaler,
        output_type=hyperparameters["output_type"],
        loss_weight_type=hyperparameters["loss_weight_type"],
        loss_weight_a=1,
        hyperparameters=hyperparameters,
    )

    # 0 -> cuda:3
    # 1 -> cuda:4
    # 2 -> cuda:5
    # 3 -> cuda:6
    # 4 -> cuda:0
    # 5 -> cuda:1
    # 6 -> cuda:2
    devices = [gpu_id]

    trainer = Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=devices,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=[tbl, cvl],
    )

    if load_checkpoint_folder:
        # list all files in the folder that end with .ckpt
        checkpoint_files = [
            f for f in os.listdir(load_checkpoint_folder) if f.endswith(".ckpt")
        ]
        best_val_loss_checkpoint = [f for f in checkpoint_files if "val_loss" in f][0]
        best_val_r2_checkpoint = [f for f in checkpoint_files if "val_r2" in f][0]
        checkpoint_callback_r2.best_model_path = (
            f"{load_checkpoint_folder}/{best_val_r2_checkpoint}"
        )
        checkpoint_callback_val_loss.best_model_path = (
            f"{load_checkpoint_folder}/{best_val_loss_checkpoint}"
        )

    if TRAIN:
        trainer.fit(model, train_loader, val_loader)

    # load model with best val_loss
    model_dict = torch.load(checkpoint_callback_val_loss.best_model_path)
    model.load_state_dict(model_dict["state_dict"])
    model.tested_model = "best_val_loss"

    device = "cuda:" + str(devices[0])

    model.reset_model(
        sequential_testing=False, batch_size=len(test_loader.dataset), device=device
    )
    trainer.test(model, dataloaders=test_loader)
    model.reset_model(sequential_testing=True, batch_size=1, device=device)
    trainer.test(model, dataloaders=test_loader)

    # load model with best val_r2
    model_dict = torch.load(checkpoint_callback_r2.best_model_path)
    model.load_state_dict(model_dict["state_dict"])
    model.tested_model = "best_val_r2"

    model.reset_model(
        sequential_testing=False, batch_size=len(test_loader.dataset), device=device
    )
    trainer.test(model, dataloaders=test_loader)
    model.reset_model(sequential_testing=True, batch_size=1, device=device)
    trainer.test(model, dataloaders=test_loader)

    best_model_path = checkpoint_callback_val_loss.best_model_path

    if BENCHMARK:
        dataset = PrimateReaching(
            file_path=data_dir,
            filename=filename,
            num_steps=1,
            train_ratio=0.5,
            bin_width=0.004,
            biological_delay=0,
            remove_segments_inactive=False,
        )

        test_set_loader = DataLoader(
            Subset(dataset, dataset.ind_test),
            batch_size=1024,
            shuffle=False,
            drop_last=True,
        )

        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        model.reset_model(sequential_testing=True, batch_size=1, device="cpu")
        torch_model = TorchModel(model)

        static_metrics = ["footprint", "connection_sparsity"]
        workload_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

        # Benchmark expects the following:
        benchmark = Benchmark(
            torch_model, test_set_loader, [], [], [static_metrics, workload_metrics]
        )
        results = benchmark.run(device="cpu")
        results["best_model_path"] = best_model_path
        print(results)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--file_id", default=0, type=int)
    args = parser.parse_args()

    all_files = [
        "indy_20160622_01",
        "indy_20160630_01",
        "indy_20170131_02",
        "loco_20170210_03",
        "loco_20170215_02",
        "loco_20170301_05",
    ]

    file = all_files[args.file_id]

    footprint_array = []
    con_s_array = []
    act_s_array = []
    dense_ops_array = []
    eff_macs_array = []
    eff_acs_array = []
    r2_array = []

    num_models = 5
    for i in range(num_models):
        # seed everything training for reproducibility
        torch.manual_seed(i + 42)
        result = main(file, args.gpu_id)
        footprint, con_s, act_s, r2, synaptic_ops, best_model_path = list(
            result.values()
        )
        eff_macs, eff_acs, dense_ops = list(synaptic_ops.values())

        footprint_array.append(footprint)
        con_s_array.append(con_s)
        act_s_array.append(act_s)
        dense_ops_array.append(dense_ops)
        eff_macs_array.append(eff_macs)
        eff_acs_array.append(eff_acs)
        r2_array.append(r2)

        text_results = (
            f"Aggregated results after run {i+1}\n"
            + f"Footprint: {np.mean(footprint_array)}, {np.std(footprint_array)}\n"
            + f"Connection Sparsity: {np.mean(con_s_array)}, {np.std(con_s_array)}\n"
            + f"Activation Sparsity: {np.mean(act_s_array)}, {np.std(act_s_array)}\n"
            + f"Dense Operations: {np.mean(dense_ops_array)}, {np.std(dense_ops_array)}\n"
            + f"Effective MACs: {np.mean(eff_macs_array)}, {np.std(eff_macs_array)}\n"
            + f"Effective ACs: {np.mean(eff_acs_array)}, {np.std(eff_acs_array)}\n"
            + f"R2: {np.mean(r2_array)}, {np.std(r2_array)}\n"
        )

        print(text_results)
        folder = f"/shared/work/phd/neurobench/neurobench/examples/primate_reaching/benchmark_models/{file}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/results.txt", "a") as f:
            f.write(text_results)

        # copy the best model to the folder, with hyperparameters and scaler
        model_folder = f"{folder}/model_{i}/"
        # ensure folder exists
        os.makedirs(model_folder, exist_ok=True)
        os.system(f"cp {best_model_path} {model_folder}model.ckpt")
        hyperparameters_path = best_model_path.split("/")[:-1] + [
            "hyperparameters.save"
        ]
        hyperparameters_path = "/".join(hyperparameters_path)
        os.system(f"cp {hyperparameters_path} {model_folder}hyperparameters.save")
        hyperparameters_path = best_model_path.split("/")[:-1] + ["hparams.yaml"]
        hyperparameters_path = "/".join(hyperparameters_path)
        os.system(f"cp {hyperparameters_path} {model_folder}hparams.yaml")
        scaler_path = best_model_path.split("/")[:-1] + ["scaler.save"]
        scaler_path = "/".join(scaler_path)
        os.system(f"cp {scaler_path} {model_folder}scaler.save")

    np.savetxt(f"{folder}/file_{file}_footprint_results.txt", np.array(footprint_array))
    np.savetxt(
        f"{folder}/file_{file}_connection_sparsity_results.txt", np.array(con_s_array)
    )
    np.savetxt(
        f"{folder}/file_{file}_activation_sparsity_results.txt", np.array(act_s_array)
    )
    np.savetxt(f"{folder}/file_{file}_dense_ops_results.txt", np.array(dense_ops_array))
    np.savetxt(f"{folder}/file_{file}_eff_macs_results.txt", np.array(eff_macs_array))
    np.savetxt(f"{folder}/file_{file}_eff_acs_results.txt", np.array(eff_acs_array))
    np.savetxt(f"{folder}/file_{file}_r2_results.txt", np.array(r2_array))

# Baseline model:
# Footprint: [24972, 24972, 24972, 43020, 43020, 43020] 33996
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.6453041225691365, 0.6873141238678099, 0.919185901113612, 0.9157824548894764, 0.7036032468324952, 0.8569459155448776] 0.788022627469568
# Dense: [32928.0, 32928.0, 32928.0, 54432.0, 54432.0, 54432.0] 43680.0
# MACs: [21504.0, 21503.999401096742, 21504.0, 43007.99836552424, 43007.99707097652, 43007.99943442413] 32255.999045336936
# ACs: [6321.460861433107, 7215.396642398608, 5009.047845227062, 4860.221515246594, 5612.495449552801, 5969.860638570507] 5831.413825404779
# R2: [0.6967655420303345, 0.5771909952163696, 0.6517471075057983, 0.6225693821907043, 0.56768798828125, 0.681292712688446] 0.6328756213188171

# Model results mean:
# Footprint: [20808, 20808, 20808, 32328, 32328, 32328] 26568
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# Activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# Dense: [3507.24609375, 3507.24609375, 3507.24609375, 6387.24609375, 6387.24609375, 6387.24609375] 4947.24609375
# MACs: [627.24609375, 627.24609375, 627.24609375, 627.24609375, 627.24609375, 627.24609375] 627.24609375
# ACs: [238.00213481104652, 146.31901667668268, 111.3298828125, 306.1044630363806, 335.4614917652027, 350.47505326704544] 247.948673728143
# R2: [0.7211523175239563, 0.5768382906913757, 0.6696014523506164, 0.5853629112243652, 0.5020735859870911, 0.6704336643218994] 0.620910370349884

# Model results std:
# Footprint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# Dense: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# MACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# R2: [0.00904376168580114, 0.006927160391411863, 0.016704759753601285, 0.013605330829100327, 0.03371670752065806, 0.0037871622061744293] 0.013964147064457852


# Comparison ours vs. baseline:
# Footprint: 26568 / 33996
# Connection sparsity: 0.0 / 0.0
# Activation sparsity: 0.0 / 0.788022627469568
# Dense: 4947.24609375 / 43680.0
# MACs: 627.24609375 / 32255.999045336936
# ACs: 247.948673728143 / 5831.413825404779
# R2: 0.620910370349884 / 0.6328756213188171
