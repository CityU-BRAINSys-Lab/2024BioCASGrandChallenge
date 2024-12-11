import torch
from torch import nn

class GRURNN(nn.Module):
    def __init__(self, input_dim, 
                 gru_layer_dims=[128, 64], 
                 output_dim=2,
                 output_series=False, 
                 drop_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_gru_layers = len(gru_layer_dims)
        self.layer_dims = [input_dim] + gru_layer_dims
        self.output_series = output_series
        self.output_dim = output_dim

        # GRU Cells
        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size=self.layer_dims[i], 
                                                   hidden_size=self.layer_dims[i+1]) for i in range(self.num_gru_layers)])

        # Layer Normalization
        self.lns = nn.ModuleList([nn.LayerNorm(self.layer_dims[i+1]) for i in range(self.num_gru_layers)])

        # Fully Connected Layer
        self.fc = nn.Sequential()
        if self.layer_dims[-1] != output_dim:
            self.fc.add_module('drop_out', nn.Dropout(drop_rate))
            self.fc.add_module('fc_layer', nn.Linear(in_features=self.layer_dims[-1], out_features=output_dim, bias=True))

        # self.h_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, self.layer_dims[i+1]), requires_grad=True) for i in range(self.num_gru_layers)])


    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h_out_series = []
        # h = [h_bias.expand(batch_size, -1).contiguous() for h_bias in self.h_biases]
        h = [torch.zeros(batch_size, self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]

        for t in range(seq_length):
            h[0] = self.lns[0](self.gru_cells[0](x[:, t, :], h[0]))
            for i in range(1, self.num_gru_layers):
                h[i] = self.lns[i](self.gru_cells[i](h[i-1], h[i]))
            h_out_series.append(h[-1])

        if self.output_series: # dim: batch_size, seq_length, output_dim
            x = self.fc(torch.stack(h_out_series, dim=1))
        else: # dim: batch_size, output_dim
            x = self.fc(h[-1])

        return x
    
class GRUbiRNN(nn.Module):
    def __init__(self, input_dim, 
                 gru_layer_dims=[128, 64], 
                 output_dim=2,
                 output_series=False, 
                 drop_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_gru_layers = len(gru_layer_dims)
        self.layer_dims = [input_dim] + gru_layer_dims
        self.output_series = output_series
        self.output_dim = output_dim

        # GRU Cells
        self.gru_cells_f = nn.ModuleList([nn.GRUCell(input_size=self.layer_dims[i], 
                                                   hidden_size=self.layer_dims[i+1]) for i in range(self.num_gru_layers)])
        self.gru_cells_b = nn.ModuleList([nn.GRUCell(input_size=self.layer_dims[i], 
                                                   hidden_size=self.layer_dims[i+1]) for i in range(self.num_gru_layers)])
        
        # Layer Normalization
        self.lns = nn.ModuleList([nn.LayerNorm(self.layer_dims[i+1]) for i in range(self.num_gru_layers)])

        # Batch Normalization
        # self.bns = nn.ModuleList([nn.BatchNorm1d(self.layer_dims[i+1]) for i in range(self.num_gru_layers)])

        # Fully Connected Layer
        self.fc = nn.Sequential()
        if self.layer_dims[-1] != output_dim:
            self.fc.add_module('drop_out', nn.Dropout(drop_rate))
            self.fc.add_module('fc_layer', nn.Linear(in_features=self.layer_dims[-1]*2, out_features=self.layer_dims[-1], bias=True))
            self.fc.add_module('relu', nn.ReLU())
            self.fc.add_module('drop_out2', nn.Dropout(drop_rate))
            self.fc.add_module('fc_layer2', nn.Linear(in_features=self.layer_dims[-1], out_features=output_dim, bias=True))
            

    def forward(self, x):
        # bi-directional GRU
        batch_size, seq_length, _ = x.size()
        h_out_series_f, h_out_series_b = [], []
        # h_out_series = []
        h_f = [torch.zeros(batch_size, self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]
        h_b = [torch.zeros(batch_size, self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]

        # forward pass
        for t in range(seq_length):
            h_f[0] = self.lns[0](self.gru_cells_f[0](x[:, t, :], h_f[0]))
            # h_f[0] = self.bns[0](self.gru_cells_f[0](x[:, t, :], h_f[0]))
            for i in range(1, self.num_gru_layers):
                h_f[i] = self.lns[i](self.gru_cells_f[i](h_f[i-1], h_f[i]))
                # h_f[i] = self.bns[i](self.gru_cells_f[i](h_f[i-1], h_f[i]))
            h_out_series_f.append(h_f[-1])
            # h_out_series.append(h_f[-1])

        # backward pass
        for t in range(seq_length-1, -1, -1):
            h_b[0] = self.lns[0](self.gru_cells_b[0](x[:, t, :], h_b[0]))
            # h_b[0] = self.bns[0](self.gru_cells_b[0](x[:, t, :], h_b[0]))
            for i in range(1, self.num_gru_layers):
                h_b[i] = self.lns[i](self.gru_cells_n[i](h_b[i-1], h_b[i]))
                # h_b[i] = self.bns[i](self.gru_cells_b[i](h_b[i-1], h_b[i]))
            h_out_series_b.append(h_b[-1])
            # h_out_series[t] = torch.cat((h_out_series[t], h_b[-1]), dim=1)
        h_out_series_b.reverse()

        if self.output_series: # dim: batch_size, seq_length, output_dim
            h_out_series = [torch.cat((h_out_series_f[i], h_out_series_b[i]), dim=-1) for i in range(seq_length)]
            x = self.fc(torch.stack(h_out_series, dim=1))
        else: # dim: batch_size, output_dim
            # x = self.fc(h_out_series[seq_length//2])
            x = self.fc(torch.cat((h_out_series_f[-1], h_out_series_b[0]), dim=-1))
        return x