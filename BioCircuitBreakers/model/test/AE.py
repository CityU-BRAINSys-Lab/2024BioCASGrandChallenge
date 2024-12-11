import torch
import torch.nn as nn
from model.train import AEGRU as AEGRU_train

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1_mu = nn.Linear(input_dim, latent_dim)
        self.bn_mu = nn.BatchNorm1d(latent_dim)
        # self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.softplus(x).log()
        mu = self.fc1_mu(x)
        mu = self.bn_mu(mu.permute(0, 2, 1)).permute(0, 2, 1)
        return mu, None

class AEGRU(AEGRU_train):
    def __init__(self, num_steps, bin_width,
                 preprocessors=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.step_size = int(bin_width // 4e-3)
        self.num_steps = num_steps
        self.bin_window_size = self.step_size*self.num_steps
        self.preprocessors = preprocessors
        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)
            
        self.encoder = Encoder(self.input_dim, self.latent_dim)
        self.decoder = None

        self.output_series = False

    def forward(self, x):
        predictions = []
        # flag = 0
        for ppc in self.preprocessors:
            x, _ = ppc((x, None))
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

            pred = self.single_forward(net_input)
            pred = self.post_process(pred)
            predictions.append(pred)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions # dim: len_data, output_dim(2)
    
    def single_forward(self, x):
        batch_size, seq_length, _ = x.size()

        mu, _ = self.encoder(x)

        # h_out_series = []
        h = [torch.zeros(batch_size, self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]

        for t in range(seq_length):
            h[0] = self.lns[0](self.gru_cells[0](mu[:, t, :], h[0]))
            for i in range(1, self.num_gru_layers):
                h[i] = self.lns[i](self.gru_cells[i](h[i-1], h[i]))
            # h_out_series.append(h[-1])
        v = self.fc(h[-1])

        return v, None, mu, None
