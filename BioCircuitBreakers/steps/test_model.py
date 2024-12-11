import os
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

import model.test as model_test
from steps.load_preprocessors import LoadPreprocessors

SAMPLE_RATE = 4e-3

def TestModel(proj, trainning_logger, device, pruned_module_names):
    ckpt_dir = trainning_logger.ckpt_dir

    proj.print()

    Model = getattr(model_test, proj.network)

    preprocessors = LoadPreprocessors(proj)

    footprint = []
    connection_sparsity = []
    activation_sparsity = []
    dense = []
    macs = []
    acs = []
    r2 = []

    for filename in proj.all_files:
        print("Processing {}".format(filename))

        # The dataloader and preprocessor has been combined together into a single class
        data_dir = proj.data_dir
        dataset = PrimateReaching(file_path=data_dir, filename=filename,
                                num_steps=1, train_ratio=0.5, bin_width=SAMPLE_RATE,
                                biological_delay=0, remove_segments_inactive=False)

        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), 
                                    batch_size=proj.test_dataloader_batch_size, 
                                    shuffle=False)

        testmodeladd_kwargs = {k.replace("testmodeladd_", ""): v for k, v in vars(proj).items() if k.startswith("testmodeladd_")}
        model_kwargs = {k.replace("model_", ""): v for k, v in vars(proj).items() if k.startswith("model_")}
        net = Model(
            **(testmodeladd_kwargs | {'bin_width': proj.dataset_bin_width, 'num_steps': proj.dataset_num_steps}),
            preprocessors=preprocessors,
            input_dim=dataset.input_feature_size if "M1Only" not in proj.preprocessors else dataset.input_feature_size / 2, 
            **model_kwargs
            )
        
        parameters_to_prune = []
        for name, module in net.named_modules():
            if name in pruned_module_names:
                if isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))
                if isinstance(module, torch.nn.GRUCell):
                    parameters_to_prune.append((module, "weight_ih"))
                    parameters_to_prune.append((module, "weight_hh"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.5,
        )
        
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ft_latest.pth"), map_location=device), strict=False)
        
        for name, module in net.named_modules():
            if name in pruned_module_names:
                if isinstance(module, torch.nn.Linear):
                    prune.remove(module, "weight")
                if isinstance(module, torch.nn.GRUCell):
                    prune.remove(module, "weight_ih")
                    prune.remove(module, "weight_hh")

        net.fake_quantize_weights()

        torch.save(net.state_dict(), os.path.join(ckpt_dir, f"final_ckpt.pth"))

        model = TorchModel(net)

        static_metrics = ["footprint", 
                          "connection_sparsity",
                          ]
        workload_metrics = ["r2", 
                            "activation_sparsity", 
                            "synaptic_operations",
                            ]

        # Benchmark expects the following:
        benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, workload_metrics])
        results = benchmark.run(device=device)
        print(results)

        footprint.append(results['footprint'])
        connection_sparsity.append(results['connection_sparsity'])
        activation_sparsity.append(results['activation_sparsity'])
        dense.append(results['synaptic_operations']['Dense'])
        macs.append(results['synaptic_operations']['Effective_MACs'])
        acs.append(results['synaptic_operations']['Effective_ACs'])
        r2.append(results['r2'])

    print("Footprint: {}".format(footprint))
    print("Connection sparsity: {}".format(connection_sparsity))
    print("Activation sparsity: {}".format(activation_sparsity), sum(activation_sparsity)/len(activation_sparsity))
    print("Dense: {}".format(dense), sum(dense)/len(dense))
    print("MACs: {}".format(macs), sum(macs)/len(macs))
    print("ACs: {}".format(acs), sum(acs)/len(acs))
    print("R2: {}".format(r2), sum(r2)/len(r2))

    results = {"footprint": footprint, 
            "connection_sparsity": connection_sparsity, 
            "activation_sparsity": activation_sparsity, 
            "dense": dense, "macs": macs, "acs": acs, 
            "r2": r2}

    trainning_logger.add_info(benchmark_results=results)