import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np

from neurobench.examples.primate_reaching.neural_decoding_challenge.neurons import (
    Integrator,
)


class FeedForwardSNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers,
        num_neurons,
        neuron_type,
        neuron_args,
        actual_batch_size,  # batch_size actually represents sequence length
        output_dim=2,
        batch_size=256,
        bin_window=0.2,
        num_steps=7,
        drop_rate=0.5,
        mem_thresh=0.5,
        spike_grad=surrogate.atan(alpha=2),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.neuron_args = neuron_args
        self.drop_rate = drop_rate
        self.actual_batch_size = actual_batch_size

        self.batch_size = batch_size
        self.bin_window_time = bin_window
        self.num_steps = num_steps
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)
        self.step_size = self.bin_window_size // self.num_steps

        self.fc = nn.ModuleDict()
        self.fc["in"] = nn.Linear(self.input_dim, self.num_neurons, bias=False)
        for i in range(self.num_layers - 1):
            self.fc[str(i)] = nn.Linear(self.num_neurons, self.num_neurons, bias=False)
        self.fc["out"] = nn.Linear(self.num_neurons, self.output_dim, bias=False)

        self.dropout = nn.Dropout(self.drop_rate)
        # self.norm_layer = nn.LayerNorm([self.num_steps, self.actual_batch_size, self.input_dim])

        self.mem_thresh = mem_thresh
        self.spike_grad = spike_grad

        self.lifs = nn.ModuleDict()
        for i in range(self.num_layers):
            self.lifs[str(i)] = self.neuron_type(
                **self.neuron_args,
                spike_grad=self.spike_grad,
                learn_beta=False,
                learn_threshold=False,
                init_hidden=True,
                reset_mechanism="zero",
            )
        # self.lifs["out"] = Integrator(
        #     spike_grad=self.spike_grad,
        #     learn_beta=False,
        #     learn_threshold=False,
        #     init_hidden=True,
        #     reset_mechanism="none",
        # )
        self.lifs["out"] = self.neuron_type(
            **self.neuron_args,
            spike_grad=self.spike_grad,
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
            reset_mechanism="none",
        )

        self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))

        self.register_buffer(
            "data_buffer",
            torch.zeros(1, self.actual_batch_size, input_dim).type(torch.float32),
            persistent=False,
        )

    def reset_mem(self):
        for l in self.lifs.values():
            l.reset_hidden()

    def single_forward(self, x):
        # x = self.norm_layer(x)
        self.reset_mem()
        for step in range(self.num_steps):
            cur = self.dropout(self.fc["in"](x[step, :]))
            spk = self.lifs["0"](cur)
            for i in range(self.num_layers - 1):
                cur = self.fc[str(i)](spk)
                spk = self.lifs[str(i + 1)](cur)
            cur = self.fc["out"](spk)
            spk = self.lifs["out"](cur)

        return self.lifs["out"].mem.clone()

    def forward(self, x):
        predictions = []

        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :].unsqueeze(
                0
            )  # unsqueeze retains 0th dimension for data buffering
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :, :]

            # Accumulate
            spikes = self.data_buffer.clone()

            acc_spikes = torch.zeros(
                (self.num_steps, self.actual_batch_size, self.input_dim)
            )
            for i in range(self.num_steps):
                temp = torch.sum(
                    spikes[
                        self.step_size * i : self.step_size * i + (self.step_size), :, :
                    ],
                    dim=0,
                )
                acc_spikes[i, :] = temp

            pred = self.single_forward(acc_spikes.to(x.device)) 
            U_x = self.v_x * pred[:, 0]
            U_y = self.v_y * pred[:, 1]
            out = torch.stack((U_x, U_y), 0).permute(1, 0)
            predictions.append(out)

        predictions = torch.stack(predictions).squeeze(
            dim=1
        )  # squeeze is a remnant of the original code to get rid of the batch_size=1 dimension, here it does nothing

        return predictions


class AlternativeFeedForwardSNN(FeedForwardSNN):
    def __init__(
        self,
        input_dim,
        num_ff_layers,
        hidden_size,
        neuron_type,
        neuron_args,
        seq_len,  # batch_size actually represents sequence length
        output_dim=2,
        batch_size=256,
        bin_window=0.2,
        num_steps=None,
        drop_rate=0.5,
        mem_thresh=0.5,
        spike_grad="atan",
        grad_slope=1.0,
        decoder="last",
        num_conv_layers=None,
        conv_channels=None,
        conv_kernel_size=None,
    ):
        if spike_grad == "atan":
            self.spike_grad = surrogate.atan(alpha=2*grad_slope)
        elif spike_grad == "sigmoid":
            self.spike_grad = surrogate.sigmoid(slope=int(25 * grad_slope))
        elif spike_grad == "fast_sigmoid":
            self.spike_grad = surrogate.fast_sigmoid(slope=int(25 * grad_slope))
        else:
            raise ValueError("Invalid spike_grad")
        super().__init__(
            input_dim=input_dim,
            num_layers=num_ff_layers,
            num_neurons=hidden_size,
            neuron_type=neuron_type,
            neuron_args=neuron_args,
            actual_batch_size=seq_len,  # batch_size actually represents sequence length
            output_dim=output_dim,
            batch_size=batch_size,
            bin_window=bin_window,
            num_steps=batch_size,
            drop_rate=drop_rate,
            mem_thresh=mem_thresh,
            spike_grad=self.spike_grad,
        )
        self.decoder = decoder
        self.seq_len = seq_len

    def single_forward(self, x):
        # x = self.norm_layer(x)

        outs = []

        seq_length = x.shape[0]
        self.reset_mem()
        cur, spk = dict(), dict()
        for i in range(seq_length):
            cur["in"] = self.fc["in"](x[i, :, :])
            spk[str(0)] = self.lifs["0"](cur["in"])
            for j in range(self.num_layers - 1):
                cur[str(j)] = self.fc[str(j)](spk[str(j)])
                spk[str(j + 1)] = self.lifs[str(j + 1)](cur[str(j)])
            cur["out"] = self.fc["out"](spk[str(self.num_layers - 1)])
            spk["out"] = self.lifs["out"](cur["out"])
            #outs.append(cur["out"].clone())
            outs.append(self.lifs["out"].mem.clone())

        outs = torch.stack(outs)
        if self.decoder == "max":
            outs = torch.max(outs, dim=0).values
        elif self.decoder == "mean":
            outs = torch.mean(outs, dim=0)
        elif self.decoder == "last":
            outs = outs[-1]
        elif self.decoder == 'seq':
            outs = outs
        else:
            raise ValueError("Invalid decoder")

        return outs, spk

    def forward(self, x):
        predictions = []

        pred, spikes = self.single_forward(x)
        U_x = self.v_x * pred[:, :, 0].to(self.v_x.device)
        U_y = self.v_y * pred[:, :, 1].to(self.v_x.device)
        predictions = torch.stack((U_x, U_y), 0).permute(1, 2, 0)
        # print(predictions.shape)

        return predictions, spikes

class Seq2SeqSNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_ff_layers, # Number of feedforward layers
        hidden_size,
        neuron_type,
        neuron_args,
        seq_len,
        spike_steps=7,
        output_dim=2,
        batch_size=256,
        drop_rate=0.5,
        spike_grad="atan",
        grad_slope=1.0,
        decoder="mean",
        reset_mechanism="zero",
        output_layer="integrator", # "d_sum", "lif_no_reset"
        reset_mem_every_step=False,
        input_layer_norm=False,
        use_grus=False,
        use_velocity_multiplier=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ff_layers = num_ff_layers
        self.num_layers = self.num_ff_layers
        self.hidden_size = hidden_size
        self.neuron_type = neuron_type
        self.neuron_args = neuron_args
        self.drop_rate = drop_rate
        self.seq_len = seq_len
        self.spike_steps = spike_steps
        self.reset_mechanism = reset_mechanism
        self.batch_size = batch_size
        self.output_layer = output_layer
        self.reset_mem_every_step = reset_mem_every_step
        self.input_layer_norm = input_layer_norm
        self.use_grus = use_grus
        self.grad_slope = grad_slope
        self.use_velocity_multiplier = use_velocity_multiplier
        if spike_grad == "atan":
            self.spike_grad = surrogate.atan(alpha=2*grad_slope)
        elif spike_grad == "sigmoid":
            self.spike_grad = surrogate.sigmoid(slope=int(25 * grad_slope))
        elif spike_grad == "fast_sigmoid":
            self.spike_grad = surrogate.fast_sigmoid(slope=int(25 * grad_slope))
        else:
            raise ValueError("Invalid spike_grad")

        self.decoder = decoder
        self.dropout = nn.Dropout(self.drop_rate)

        # Init FF Layers
        self.fc = nn.ModuleDict()
        self.lifs = nn.ModuleDict()
        if self.use_grus:
            self.layernorms = nn.ModuleDict()
            self.grus = nn.ModuleDict()
            self.gru_mems = dict()

        if self.input_layer_norm:
            self.layernorm = nn.LayerNorm(self.input_dim)
        
        for i in range(self.num_ff_layers):
            if i == 0:
                self.fc[str(i)] = nn.Linear(self.input_dim, self.hidden_size, bias=False)
            else:
                self.fc[str(i)] = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

            self.lifs[str(i)] = self.neuron_type(
                **self.neuron_args,
                spike_grad=self.spike_grad,
                learn_beta=True,
                learn_threshold=True,
                init_hidden=True,
                reset_mechanism=self.reset_mechanism,
            )
            if use_grus:
                self.layernorms[str(i)] = nn.LayerNorm(self.hidden_size)
                self.grus[str(i)] = nn.GRUCell(self.hidden_size, self.hidden_size)
                self.gru_mems[str(i)] = None

        self.fc["out"] = nn.Linear(self.hidden_size, self.output_dim, bias=False) 
        if self.output_layer == "integrator":
            self.lifs["out"] = Integrator(
                spike_grad=self.spike_grad,
                learn_beta=False,
                learn_threshold=False,
                init_hidden=True,
                reset_mechanism="none",
            )
        elif self.output_layer == "lif_no_reset":
            self.lifs["out"] = self.neuron_type(
                **self.neuron_args,
                spike_grad=self.spike_grad,
                learn_beta=False,
                learn_threshold=False,
                init_hidden=True,
                reset_mechanism="none",
            )
        elif self.output_layer == "d_sum":
            self.lifs["out"] = self.neuron_type(
                **self.neuron_args,
                spike_grad=self.spike_grad,
                learn_beta=False,
                learn_threshold=False,
                init_hidden=True,
                reset_mechanism="none",
            )

        if self.use_velocity_multiplier:
            self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
            self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))

    def reset_mem(self):
        for l in self.lifs.values():
            l.reset_hidden()
        if self.use_grus:
            for k in self.gru_mems.keys():
                self.gru_mems[k] = None
    
    def decode(self, x):
        if self.decoder == "max":
            x = torch.max(x, dim=0).values
        elif self.decoder == "mean":
            x = torch.mean(x, dim=0)
        elif self.decoder == "last":
            x = x[-1]
        elif self.decoder == "sum":
            x = torch.sum(x, dim=0)
        elif self.decoder == 'seq':
            x = x
        else:
            raise ValueError("Invalid decoder")
        return x

    def single_forward(self, x):
        outs = []
        cur, spk = dict(), dict()

        for j in range(self.num_ff_layers):
            spk[str(j)] = dict()

        spk["out"] = dict()

        for k in range(self.spike_steps):
            for j in range(self.num_ff_layers):
                if j==0:
                    cur[str(j)] = self.dropout(self.fc[str(j)](x))
                else:
                    cur[str(j)] = self.dropout(self.fc[str(j)](z))

                spk[str(j)][str(k)] = self.lifs[str(j)](cur[str(j)])

                if self.use_grus:
                    self.gru_mems[str(j)] = self.grus[str(j)](spk[str(j)][str(k)], self.gru_mems[str(j)])
                    z = self.layernorms[str(j)](self.gru_mems[str(j)])
                else:
                    z = spk[str(j)][str(k)]

            cur["out"] = self.fc["out"](z)
            spk["out"][str(k)] = self.lifs["out"](cur["out"])

            if self.output_layer == "d_sum":
                outs.append(cur["out"].clone())
            else:
                outs.append(self.lifs["out"].mem.clone())

        outs = torch.atleast_3d(torch.stack(outs))
        outs = self.decode(outs)

        for j in range(self.num_ff_layers):
            spk[str(j)] = torch.stack([spk[str(j)][str(k)] for k in range(self.spike_steps)], dim=0)
        spk["out"] = torch.stack([spk["out"][str(k)] for k in range(self.spike_steps)], dim=0)

        return outs, spk

    def forward(self, x, test=False):
        seq_length, batch_size, features = x.shape

        predictions = []
        spikes = dict()

        if self.input_layer_norm:
            x = self.layernorm(x)

        for i in range(seq_length):
            if self.reset_mem_every_step:
                self.reset_mem()
            elif test:
                pass
            elif i == 0:
                self.reset_mem()
                
            pred, spk = self.single_forward(x[i])
            predictions.append(pred)
            for k, v in spk.items():
                if k not in spikes:
                    spikes[k] = []
                spikes[k].append(v.to(dtype=torch.bool))

        predictions = torch.stack(predictions)
        for k, v in spikes.items():
            spikes[k] = torch.stack(spikes[k])

        if self.use_velocity_multiplier:
            U_x = self.v_x * predictions[:, :, 0].to(self.v_x.device)
            U_y = self.v_y * predictions[:, :, 1].to(self.v_x.device)
            predictions = torch.stack((U_x, U_y), 0).permute(1, 2, 0)

        return predictions, spikes

class ConvSNN(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len=1024,
        output_dim=2,
        batch_size=256,
        drop_rate=0.5,
        reset_mem_every_step=True,
        linear_interpolation=True,
        random_init=True,
        kernel_size=20,
        gru_hidden_size=128,
        conv_channels=32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_rate = drop_rate
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dropout = nn.Dropout(self.drop_rate)
        self.reset_mem_every_step = reset_mem_every_step
        self.linear_interpolation = linear_interpolation
        self.random_init = random_init
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.GRU_HIDDEN_SIZE = gru_hidden_size
        self.CONV_CHANNELS = conv_channels

        self.conv = nn.ModuleDict()
        self.maxp = nn.ModuleDict()
        self.lynm = nn.ModuleDict()

        # input shape is [batch_size, input_dim, 1024]
        self.conv[str(0)] = nn.Conv1d(in_channels=self.input_dim, out_channels=self.CONV_CHANNELS, kernel_size=self.kernel_size, stride=1, padding=self.padding + (2 if self.linear_interpolation else 0))
        # shape after conv1 is [batch_size, self.CONV_CHANNELS, 1028]
        self.maxp[str(0)] = nn.MaxPool1d(kernel_size=2, stride=2)
        # shape after maxp1 is [batch_size, self.CONV_CHANNELS, 514]
        self.conv[str(1)] = nn.Conv1d(in_channels=self.CONV_CHANNELS, out_channels=self.CONV_CHANNELS, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # shape after conv2 is [batch_size, self.CONV_CHANNELS, 514]
        self.maxp[str(1)] = nn.MaxPool1d(kernel_size=2, stride=2)
        # shape after maxp2 is [batch_size, self.CONV_CHANNELS, 257]
        #self.conv[str(2)] = nn.Conv1d(in_channels=self.CONV_CHANNELS, out_channels=self.CONV_CHANNELS, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        ## shape after conv3 is [batch_size, self.CONV_CHANNELS, 257]
        #self.maxp[str(2)] = nn.MaxPool1d(kernel_size=2, stride=2)
        ## shape after maxp3 is [batch_size, self.CONV_CHANNELS, 129]

        self.after_conv_dim, self.after_conv_seq_len = self.get_dim_after_conv()
        self.time_steps_per_keypoint = int(self.seq_len / (self.after_conv_seq_len-(1 if self.linear_interpolation else 0)))

        # Init GRU Layers
        self.grus = nn.ModuleDict()
        self.gru_mems = dict()
        self.gru_mems[str(0)] = None
        self.grus[str(0)] = nn.GRUCell(self.after_conv_dim, self.GRU_HIDDEN_SIZE)
        self.lynm[str(0)] = nn.LayerNorm(self.GRU_HIDDEN_SIZE)

        # Init FF Layers
        self.fc = nn.ModuleDict()
        if linear_interpolation:
            self.fc["x"] = nn.Linear(self.GRU_HIDDEN_SIZE, 1)
            self.fc["y"] = nn.Linear(self.GRU_HIDDEN_SIZE, 1)
        else:
            self.fc["x"] = nn.Linear(self.GRU_HIDDEN_SIZE, self.time_steps_per_keypoint)
            self.fc["y"] = nn.Linear(self.GRU_HIDDEN_SIZE, self.time_steps_per_keypoint)

    def forward(self, x):
        predictions = []

        # x.shape is [1024, batch_size, input_dim]
        seq_length, batch_size, features = x.shape
        self.batch_size = batch_size
        self.device = x.device

        x = x.permute(1, 2, 0)
        
        if self.reset_mem_every_step:
            self.reset_mem()

        k = self.conv_forward(x)
        # k.shape is [batch_size, after_conv_dim, seq_len_after_conv]
        for i in range(self.after_conv_seq_len):
            z = self.single_forward(k[:,:,i])
            # z.shape is [batch_size, self.GRU_HIDDEN_SIZE]
            z = self.ff_forward(z)
            # z.shape is [batch_size, 2]
            if self.linear_interpolation:
                if i == 0:
                    self.last_pred = z
                else:
                    # linearly interpolate between the previous and current prediction by the number of time steps per keypoint
                    step_size = (z - self.last_pred) / self.time_steps_per_keypoint
                    #self.last_pred = self.last_pred - step_size # begin with the previous prediction
                    for j in range(self.time_steps_per_keypoint):
                        preds = self.last_pred + step_size
                        self.last_pred = preds
                        predictions.append(preds)
            else:
                predictions.append(z)

        predictions = torch.cat(predictions, dim=1).permute(1, 0, 2).to(torch.float64)

        assert predictions.shape[0] == self.seq_len

        return predictions

    def conv_forward(self, x):
        for i in range(len(self.conv.keys())):
            x = self.conv[str(i)](x)
            x = self.maxp[str(i)](x)
            x = self.dropout(x)
        return x
    
    def single_forward(self, x):
        # x.shape is [batch_size, after_conv_dim]
        self.gru_mems[str(0)] = self.grus[str(0)](x, self.gru_mems[str(0)])
        z = self.lynm[str(0)](self.gru_mems[str(0)])
        z = self.dropout(z)
        return z

    def ff_forward(self, x):
        z_1 = self.fc["x"](x)
        z_2 = self.fc["y"](x)
        
        out = torch.stack((z_1, z_2), 2)
        return out

    def get_dim_after_conv(self):
        with torch.no_grad():
            x = torch.zeros(1, self.input_dim, self.seq_len)
            x = self.conv_forward(x)
            return x.shape[1], x.shape[2]
        
    def reset_mem(self):
        self.last_pred = torch.zeros(self.batch_size, 1, 2).to(torch.float32).to(self.device)
        for k in self.gru_mems.keys():
            if self.random_init:
                self.gru_mems[k] = torch.randn(self.batch_size, self.GRU_HIDDEN_SIZE).to(torch.float32).to(self.device)
            else:
                self.gru_mems[k] = torch.zeros(self.batch_size, self.GRU_HIDDEN_SIZE).to(torch.float32).to(self.device)
    