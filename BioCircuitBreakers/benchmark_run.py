import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, Subset

from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark
from neurobench.datasets import PrimateReaching

import model.test as model_test
from steps.proj_params import ProjectParams
from steps.load_preprocessors import LoadPreprocessors

SAMPLE_RATE = 4e-3

if __name__ == "__main__":
    ## Receive seed id
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    seed = args.seed

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    ## Get project configuration -> proj: ProjectParams
    ckpt_dir = f"example_ckpts/seed{seed}"
    proj = ProjectParams(ckpt_dir)
    proj.print()

    ## Prepare Preprocessors
    preprocessors = LoadPreprocessors(proj)

    ## Initialize benchmark containers
    footprint = []
    connection_sparsity = []
    activation_sparsity = []
    dense = []
    macs = []
    acs = []
    r2 = []
    static_metrics = [
        "footprint", 
        "connection_sparsity",
        ]
    workload_metrics = [
        "r2", 
        "activation_sparsity",
        "synaptic_operations",
        ]

    for filename in proj.all_files:
    ## Load data
        dataset = PrimateReaching(file_path=proj.data_dir, filename=filename,
                                num_steps=1, train_ratio=0.5, bin_width=SAMPLE_RATE,
                                biological_delay=0, remove_segments_inactive=False)

        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), 
                                    batch_size=proj.test_dataloader_batch_size, 
                                    shuffle=False)

    ## Load models
        # Initiate network architecture
        Model = getattr(model_test, proj.network)
        testmodeladd_kwargs = {k.replace("testmodeladd_", ""): v for k, v in vars(proj).items() if k.startswith("testmodeladd_")}
        model_kwargs = {k.replace("model_", ""): v for k, v in vars(proj).items() if k.startswith("model_")}
        net = Model(
            **(testmodeladd_kwargs | {'bin_width': proj.dataset_bin_width, 'num_steps': proj.dataset_num_steps}),
            preprocessors=preprocessors,
            input_dim=dataset.input_feature_size if "M1Only" not in proj.preprocessors else dataset.input_feature_size / 2, 
            **model_kwargs,
            )
        # Load weights
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ckpt_{filename}.pth"), map_location=device), strict=False)

    ## Proceed benchmark
        model = TorchModel(net)
    
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

    ## Print out benchmark results
    print("======Benchmark results with======")
    print("Footprint: {}".format(footprint))
    print("Expected footprint: {}".format(proj.benchmark_results["footprint"]))
    
    print("Connection sparsity: {}".format(connection_sparsity))
    print("Expected connection sparsity: {}".format(proj.benchmark_results["connection_sparsity"]))

    print("Activation sparsity: {}".format(activation_sparsity), sum(activation_sparsity)/len(activation_sparsity))
    print("Expected activation sparsity: {}".format(proj.benchmark_results["activation_sparsity"]), sum(proj.benchmark_results["activation_sparsity"])/len(proj.benchmark_results["activation_sparsity"]))

    print("Dense: {}".format(dense), sum(dense)/len(dense))
    print("Expected dense: {}".format(proj.benchmark_results["dense"]), sum(proj.benchmark_results["dense"])/len(proj.benchmark_results["dense"]))

    print("MACs: {}".format(macs), sum(macs)/len(macs))
    print("Expected MACs: {}".format(proj.benchmark_results["macs"]), sum(proj.benchmark_results["macs"])/len(proj.benchmark_results["macs"]))

    print("ACs: {}".format(acs), sum(acs)/len(acs))
    print("Expected ACs: {}".format(proj.benchmark_results["acs"]), sum(proj.benchmark_results["acs"])/len(proj.benchmark_results["acs"]))

    print("R2: {}".format(r2), sum(r2)/len(r2))
    print("Expected R2: {}".format(proj.benchmark_results["r2"]), sum(proj.benchmark_results["r2"])/len(proj.benchmark_results["r2"]))

## SEED 83
    # footprint: [39184, 39184, 39184, 51856, 51856, 51856] 
    # connection_sparsity: [0.497, 0.4913, 0.492, 0.4959, 0.4947, 0.4955]
    # activation_sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # dense: [46610.250691174806, 46607.01299281486, 46588.979046741886, 61966.35294630848, 61952.286397428696, 61962.64000844803]  54281.25368048612
    # MACs: [22052.175547101586, 22095.455418636684, 21997.39926318213, 29606.194998000654, 29846.127833596824, 29689.321729029532] 25881.112464924572
    # ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # R2: [0.7381970286369324, 0.6382004618644714, 0.7529817819595337, 0.6813970804214478, 0.620822548866272, 0.6817165017127991] 0.685552567243576
## SEED 109
    # footprint: [39184, 39184, 39184, 51856, 51856, 51856]
    # connection sparsity: [0.5017, 0.4978, 0.495, 0.5215, 0.509, 0.5116]
    # activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # dense: [46610.250691174806, 46607.01299281486, 46588.979046741886, 61966.35294630848, 61952.286397428696, 61962.64000844803] 54281.25368048612
    # MACs: [21619.23756073675, 21742.572167826962, 21799.515849259344, 28178.652419208258, 28971.586226269646, 28693.675314161006] 25167.539922910324
    # ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # R2: [0.7744131088256836, 0.653124213218689, 0.7669975757598877, 0.6596226692199707, 0.6305672526359558, 0.7006524801254272] 0.6975628832976023
## SEED 10100
    # footprint: [39184, 39184, 39184, 51856, 51856, 51856]
    # connection sparsity: [0.5048, 0.4964, 0.4948, 0.5108, 0.5163, 0.5114]
    # activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # dense: [46610.250691174806, 46607.01299281486, 46588.979046741886, 61966.35294630848, 61952.286397428696, 61962.64000844803] 54281.25368048612
    # MACs: [21680.277426645072, 21900.548453147687, 21837.504566735744, 28783.460045803193, 28427.834629002966, 28697.67013974445] 25221.215876846516
    # ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # R2: [0.7589291334152222, 0.6572209596633911, 0.7663106918334961, 0.700482964515686, 0.6362364888191223, 0.7220180034637451] 0.7068663736184438
## SEED 13144
    # footprint: [39184, 39184, 39184, 51856, 51856, 51856]
    # connection sparsity: [0.5023, 0.4957, 0.4967, 0.514, 0.5075, 0.5161]
    # activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # dense: [46610.250691174806, 46607.01299281486, 46588.979046741886, 61966.35294630848, 61952.286397428696, 61962.64000844803] 54281.25368048612
    # MACs: [21594.250849372856, 21939.528386880607, 21831.560979353748, 28583.518470318806, 28983.547813475834, 28486.770178464572] 25236.529446311066
    # ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # R2: [0.775389552116394, 0.6527516841888428, 0.7598301768302917, 0.6820827722549438, 0.5991934537887573, 0.6984744071960449] 0.6946203410625458
## SEED 44685
    # footprint: [39184, 39184, 39184, 51856, 51856, 51856]
    # connection sparsity: [0.5011, 0.498, 0.4963, 0.5183, 0.5208, 0.5136]
    # activation sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # dense: [46610.250691174806, 46607.01299281486, 46588.979046741886, 61966.35294630848, 61952.286397428696, 61962.64000844803] 54281.25368048612
    # MACs: [21652.23313119138, 21802.568519414766, 21770.553457671347, 28339.58971972809, 28277.97291506069, 28611.713259882432] 25075.771833824783
    # ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
    # R2: [0.7691452503204346, 0.6628929376602173, 0.768081545829773, 0.6859934329986572, 0.6350768804550171, 0.7184968590736389] 0.706614484389623
