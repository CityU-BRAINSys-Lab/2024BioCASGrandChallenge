# Code Repository for submission to the IEEE BioCAS 2024 Grand Challenge on Neural Decoding for Motor Control of non-Human Primates

This repository contains the code for the submission to the IEEE BioCAS 2024 Grand Challenge on Neural Decoding for Motor Control of non-Human Primates. 

Team name: **ZenkeLab-QAAS**

Team members: **Tengjun Liu, Julia Gygax, Julian Rossbroich**

## About this submission

We have constructed two spiking neural network (SNN) models for the decoding of hand kinematics from neural activity data. The two models correspond to the two tracks of the challenge:

- `maxR2`: a large, high-performance recurrent SNN model for **Track 1** (Obtaining the highest R2 score)
- `tinyRSNN`: a small, resource-efficient recurrent SNN model for **Track 2** (Obtaining the best trade-off between R2 score and solution complexity)

## Model Descriptions

All models were implemented using the [`stork`](https://github.com/fmi-basel/stork) library for training spiking neural networks and trained using surrogate gradients. Hyperparameters were optimized based on the average validation set performance across all sessions.

### maxR2 (Track 1)

The `maxR2` model was designed to maximize the R2 score on the validation set regardless of the computational resources required.
The model consists of a single recurrent spiking neural network (SNN) layer with 1024 LIF neurons. The input size corresponds to the number of electrode channels for each monkey. The readout layer consists of five independent readout heads with 2 leaky integrator readout units each. 
The final output for X and Y coordinates is obtained by averaging the predictions of the five readout heads. 
Synaptic and membrane time constants are heteregeneous for each hidden and readout unit and were optimized during training.

### tinyRSNN (Track 2)

The `tinyRSNN` model was designed to achieve a good trade-off between R2 score and computational complexity. 
It consists of a single recurrent spiking neural network (SNN) layer with 64 LIF neurons. 
The input layer size matches the number of electrode channels for each monkey.
The readout layer consists of 2 leaky integrator units for each of the X and Y coordinates. 
As in the `maxR2` model, synaptic and membrane time constants are heteregeneous for each hidden and readout unit and were optimized during training. 

To further reduce the computational complexity of the `tinyRSNN` model, we applied an additional activity regularization loss acting on hidden layer spike trains during training, which penalizes firing rates above 10 Hz. 
To enforce connection sparsity, we implemented an iterative pruning strategy of synaptic weights during training. 
At each iteration of the pruning procedure, the $N$ smallest synaptic weights in each weight matrix were set to zero and the network was re-trained.
Finally, the model is set to half-precision floating point format after training to reduce the memory footprint and speed up inference.

## Organization
The code is organized as follows:

- `/challenge`: contains source code for data loaders, models, training and evaluation
- `/stork`: contains a custom version of the [`stork` library](https://github.com/fmi-basel/stork) for the training of spiking neural networks (SNNs)
- `/conf`: contains configuration files for training and evaluation scripts (uses the [`hydra`](https://github.com/facebookresearch/hydra) framework)
- `/models`: contains model state dictionaries for the best models obtained during training, with the format: `/models/session_name/model_name-rand_seed.pth`
- `/results`: contains evaluation results for each model & session independently. Each `.json` file summarizes model performance across five random seeds.

#### Training and evaluation scripts

The scripts used for training the submitted models are `train-maxR2.py` and `train-tinyRSNN.py`. The evaluation script used to run [NeuroBench](https://github.com/NeuroBench/neurobench) benchmarks is `evaluate.py`. Configuration files for these scripts are located in the `/conf` directory.

#### Results
The files `results_summary_maxR2.json` and `results_summary_tinyRSNN.json` hold a summary of the results as submitted to the challenge (averaged across five random seeds). For results corresponding to individual seeds, please refer to the `/results` folder.


## Installation

We used Python 3.10.12 for the development of this code. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Reproducing the results

To reproduce the results, first go to the `/conf/data/data-default.yaml` file and set the `data_dir` parameter to the path of the data directory containing the challenge data. Because models are pre-trained on all publicly available sessions where the same number of electrodes were used for each monkey, the data directory should contain all files from the ["Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology" dataset](https://zenodo.org/records/583331). For only pretraining on the sessions used in the challenge, set **data.pretrain_filenames=challenge-data**.

Second, set the output directory for `hydra` in the `/conf/config.yaml` file to the desired output directory. Each run of the training script will create a new subdirectory in the output directory to store configuration files, logs, results, plots, and a copy of model state dictionaries. This defaults to `./outputs`.

### Training

To train the models from scratch, run the following commands:

```bash
python train-maxR2.py --multirun seed=1,2,3,4,5
```

and

```bash
python train-tinyRSNN.py --multirun seed=1,2,3,4,5
```

This will train the models on the training set with the specified random seeds and overwrite the model state dicts in the `/models` directory. 

To train only one model with a specific seed, run the following commands:

```bash
python train-maxR2.py seed=1
```

and

```bash
python train-tinyRSNN.py seed=1
```

By default, benchmarking is run after training and results are recorded in the `hydra` generated output log file. 
To obtain a summary of the results from the log file, without re-running evaluation (see below), please refer to the `results_extract_from_logs.ipynb` notebook.

**Note**: Training the models from scratch requires a GPU and significant computational resources. Training the `maxR2` model with one initial random seed (pre-training & fine-tuning on each session) takes approximately 12 hours on an NVIDIA RTX A4000 GPU. Training the `tinyRSNN` model with one initial random seed (pre-training & fine-tuning on each session) takes approximately 6 hours on an NVIDIA RTX A4000 GPU.

### Evaluation and benchmarking

We supplied the model state dictionaries for the best models obtained during training in the `/models` directory. Models are sorted into subdirectories by session and monkey:

- `/loco01`: models trained on session `loco_20170210_03`
- `/loco02`: models trained on session `loco_20170215_02`
- `/loco03`: models trained on session `loco_20170301_05`
- `/indy01`: models trained on session `indy_20160622_01`
- `/indy02`: models trained on session `indy_20160630_01`
- `/indy03`: models trained on session `indy_20170131_02`

There are five `tinyRSNN` models and five `maxR2` models for each session, corresponding to five different initializations. To evaluate these models on the test set, run the following commands:

```bash
python evaluate.py modelname=maxR2
```

and

```bash
python evaluate.py modelname=tinyRSNN
```

By default, the evaluation script uses a custom wrapper for stork models to be compatible with the [NeuroBench](https://github.com/NeuroBench/neurobench) benchmarking suite (see code in the `/challenge/neurobench` directory). Alternatively, the user can set the `use_snnTorch_model` flag to `True`, to convert the original `stork` model to an equivalent model using the [snnTorch](https://github.com/jeshraghian/snntorch) library and run evaluation using the unmodified Neurobench code, which leads equivalent results.

The evaluation scripts saves results in the `json` format. Results for individual models and sessions are saved in the `/results` folder. Additionally, summaries for each model are saved in the root directory.

