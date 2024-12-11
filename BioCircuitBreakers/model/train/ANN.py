import torch
import torch.nn as nn

import numpy as np

## Define model ##
# The model defined here is a vanilla Fully Connected Network
class ANNModel2D(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 bin_window=0.2, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)

    def single_forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

    # def forward(self, x):
    #     predictions = []
    #     acc_spikes_batch = []

    #     seq_length = x.shape[0]
    #     for seq in range(seq_length):
    #         current_seq = x[seq, :, :]
    #         self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

    #         if self.data_buffer.shape[0] <= self.bin_window_size:
    #             predictions.append(torch.zeros(1, self.output_dim).to(x.device))
    #         else:
    #             # Only pass input into model when the buffer size == bin_window_size
    #             if self.data_buffer.shape[0] > self.bin_window_size:
    #                 self.data_buffer = self.data_buffer[1:, :]

    #             # Accumulate
    #             spikes = self.data_buffer.clone()
    #             acc_spikes = torch.sum(spikes, dim=0)
    #             acc_spikes_batch.append(acc_spikes)

    #     acc_spikes_batch = torch.stack(acc_spikes_batch)
    #     predictions = torch.cat(predictions, dim=0) if len(predictions) > 0 else torch.zeros(0, self.output_dim).to(x.device)
    #     predictions = torch.cat((predictions, self.single_forward(acc_spikes_batch)), dim=0)

    #     return predictions

    def forward(self, x):

        return self.single_forward(x.squeeze(dim=1))
    


# To be modified for training
class ANNModel3D(nn.Module):
    def __init__(self, input_dim, num_steps=7, layer1=32, layer2=48, output_dim=2, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.num_steps = num_steps
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.sampling_rate = 0.004

        self.fc1 = nn.Linear(self.input_dim*self.num_steps, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.activation(self.fc1(x.view(x.size(0), -1)))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x