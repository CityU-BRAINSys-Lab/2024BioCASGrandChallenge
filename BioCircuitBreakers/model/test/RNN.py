"""
RNN model specifically for testing purposes
"""
import torch
from ..train import GRURNN as GRURNN_train
from ..train import GRUbiRNN as GRUbiRNN_train

class GRURNNv1(GRURNN_train):
    def __init__(self, num_steps, bin_width, **kwargs):
        super().__init__(**kwargs)
        self.step_size = int(bin_width // 4e-3)
        self.num_steps = num_steps
        self.bin_window_size = self.step_size*self.num_steps
        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)
    
    def forward(self, x):
        predictions = []

        # The dataloader for test set must have shuffle=False
        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]

            # Accumulate
            spikes = self.data_buffer.clone()
            
            # acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            if spikes.shape[0] % self.step_size == 0:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size, self.input_dim))
            else:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size + 1, self.input_dim))
            for i in range(acc_spikes.shape[0]):
            # for i in range(self.num_steps):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp
            torch.flip(acc_spikes, dims=[0])

            pred = super().forward(acc_spikes.unsqueeze(dim=0).to(x.device))
            # U_x = self.v_x*pred[0]
            # U_y = self.v_y*pred[1]
            # out = torch.stack((U_x, U_y), 0).permute(1, 0)
            predictions.append(pred[:, -1,:] if self.output_series else pred)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions # dim: len_data, output_dim(2)
        
        
class GRURNNv0(GRURNN_train):
    def __init__(self, num_steps, bin_width, **kwargs):
        super().__init__( **kwargs)
        self.step_size = int(bin_width // 4e-3)
        self.num_steps = 1
        self.bin_window_size = self.step_size*self.num_steps
        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)

    def preprocess(self, x):
        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]
            
            spikes = self.data_buffer.clone()
            x[seq, :, :] = torch.sum(spikes, dim=0, keepdim=True)
        x = x.permute(1, 0, 2).contiguous()

        return x # dim: 1, seq(dataset time), input_dim(96/192)

    def postprocess(self, x):
        x = x.squeeze(dim=0)

        return x 
    
    def forward(self, x):
        x = self.preprocess(x)
        x = super().forward(x)
        x = self.postprocess(x)

        return x # dim: len_data, output_dim(2)
    

class GRUbiRNN(GRUbiRNN_train):
    def __init__(self, num_steps, bin_width,
                 log_train=False, focus_factor=None, 
                 preprocessors=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.step_size = int(bin_width // 4e-3)
        self.num_steps = num_steps
        self.bin_window_size = self.step_size*self.num_steps
        self.log_train = log_train
        self.focus_factor = focus_factor
        self.preprocessors = preprocessors
        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)
        # self.register_buffer("pred_buffer", torch.zeros(round(self.num_steps/2)+1, self.output_dim).type(torch.float32), persistent=False)

        self.output_series = False

    def forward(self, x):
        predictions = []
        # flag = 0

        # The dataloader for test set must have shuffle=False
        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]
                # flag = 1

            # Accumulate
            spikes = self.data_buffer.clone()
            
            # acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            if spikes.shape[0] % self.step_size == 0:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size, self.input_dim))
            else:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size + 1, self.input_dim))
            # for i in range(self.num_steps):
            for i in range(acc_spikes.shape[0]):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp
            # torch.flip(acc_spikes, dims=[0])

            net_input = acc_spikes.unsqueeze(dim=0).to(x.device)
            for ppc in self.preprocessors:
                net_input, _ = ppc((net_input, None))

            pred = super().forward(net_input)
            # U_x = self.v_x*pred[0]
            # U_y = self.v_y*pred[1]
            # out = torch.stack((U_x, U_y), 0).permute(1, 0)

            # if flag:
            #     pred_seq = pred[:, acc_spikes.shape[0]//2:, :]
            #     pred_seq_prev = self.pred_buffer.clone()

            #     self.pred_buffer = torch.cat((self.pred_buffer, pred), dim=0)
            # else:
            # predictions.append(pred[:, acc_spikes.shape[0]//2, :])
            predictions.append(pred)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions # dim: len_data, output_dim(2)


class GRUbiRNNv1(GRUbiRNN_train):
    def __init__(self, num_steps, bin_width, 
                 log_train=False, focus_factor=None, 
                 preprocessors=[],
                 **kwargs):
        assert log_train == (focus_factor is not None)
        super().__init__(**kwargs)
        self.step_size = int(bin_width // 4e-3)
        self.num_steps = num_steps
        self.bin_window_size = self.step_size*self.num_steps
        self.log_train = log_train
        self.focus_factor = focus_factor
        self.preprocessors = preprocessors
        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)
        # self.register_buffer("pred_buffer", torch.zeros(round(self.num_steps/2)+1, self.output_dim).type(torch.float32), persistent=False)

        # In this test method, model must be trained with output_series=True
        assert self.output_series

    def forward(self, x):
        predictions = []
        # flag = 0

        # The dataloader for test set must have shuffle=False
        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]
                # flag = 1

            # Accumulate
            spikes = self.data_buffer.clone()
            
            # acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            if spikes.shape[0] % self.step_size == 0:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size, self.input_dim))
            else:
                acc_spikes = torch.zeros((spikes.shape[0] // self.step_size + 1, self.input_dim))
            # for i in range(self.num_steps):
            for i in range(acc_spikes.shape[0]):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp
            # torch.flip(acc_spikes, dims=[0])
            net_input = acc_spikes.to(x.device)
            for ppc in self.preprocessors:
                net_input, _ = ppc((net_input, None))

            pred = self.single_forward(net_input)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        return predictions # dim: len_data, output_dim(2)
    
    def single_forward(self, x):
        # bi-directional GRU
        seq_length, _ = x.size()
        h_f = [torch.zeros(self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]
        h_b = [torch.zeros(self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]

        # forward pass
        for t in range(seq_length):
            h_f[0] = self.lns[0](self.gru_cells_f[0](x[t, :], h_f[0]))
            # h_f[0] = self.bns[0](self.gru_cells_f[0](x[t, :], h_f[0]))
            for i in range(1, self.num_gru_layers):
                h_f[i] = self.lns[i](self.gru_cells_f[i](h_f[i-1], h_f[i]))
                # h_f[i] = self.bns[i](self.gru_cells_f[i](h_f[i-1], h_f[i]))

        # backward pass
        t = seq_length-1

        h_b[0] = self.lns[0](self.gru_cells_b[0](x[t, :], h_b[0]))
        # h_b[0] = self.bns[0](self.gru_cells_b[0](x[t, :], h_b[0]))
        for i in range(1, self.num_gru_layers):
            h_b[i] = self.lns[i](self.gru_cells_n[i](h_b[i-1], h_b[i]))
            # h_b[i] = self.bns[i](self.gru_cells_b[i](h_b[i-1], h_b[i]))
        h_out = torch.cat((h_f[-1], h_b[-1]), dim=-1)

        x = self.fc(h_out)

        if self.log_train:
            x = x.sign() * x.abs().expm1() / self.focus_factor

        return x