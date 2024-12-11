import torch
import torch.nn as nn
from model.util import quantize_tensor, hardsigmoid

# Define the encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1_logvar = nn.Linear(input_dim, hidden_dim)
        self.fc1_mu = nn.Linear(input_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # self.bn_logvar = nn.BatchNorm1d(latent_dim)
        self.bn_mu = nn.BatchNorm1d(latent_dim)

        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(x).log()
        h = self.relu(self.fc1_logvar(x))
        logvar = self.fc2_logvar(h)
        # logvar = self.bn_logvar(logvar.permute(0, 2, 1)).permute(0, 2, 1).clamp(min=-10, max=10)
        mu = self.fc1_mu(x)
        mu = self.bn_mu(mu.permute(0, 2, 1)).permute(0, 2, 1).clamp(min=-10, max=10)
        return mu, logvar

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        x_recon = torch.exp(self.fc3(z))
        return x_recon

# Define the VAE model
class AEGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32,
                 log_focus_factor=None,
                 gru_layer_dims=[128, 64], 
                 output_dim=2,
                 output_series=False, 
                 drop_rate=0.5,
                 loss_ratio=[0.5,0.0,0.5]):
        super(AEGRU, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.log_focus_factor = log_focus_factor
        self.num_gru_layers = len(gru_layer_dims)
        self.layer_dims = [latent_dim] + gru_layer_dims
        self.output_series = output_series
        self.output_dim = output_dim
        self.loss_ratio = loss_ratio

        # GRU Cells
        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size=self.layer_dims[i], 
                                                   hidden_size=self.layer_dims[i+1]) for i in range(self.num_gru_layers)])
        
        # Layer Normalization
        self.lns = nn.ModuleList([nn.LayerNorm(self.layer_dims[i+1]) for i in range(self.num_gru_layers)])

        # Fully Connected Layer
        self.fc = nn.Sequential()
        self.fc.add_module('drop_out', nn.Dropout(drop_rate))
        self.fc.add_module('fc_layer', nn.Linear(in_features=self.layer_dims[-1], out_features=self.output_dim, bias=True))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        mu, logvar = self.encoder(x)
        # mu = torch.log(expmu)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)

        h_out_series = []
        h = [torch.zeros(batch_size, self.layer_dims[i+1]).to(x.device) for i in range(self.num_gru_layers)]

        for t in range(seq_length):
            h[0] = self.lns[0](self.gru_cells[0](z[:, t, :], h[0]))
            for i in range(1, self.num_gru_layers):
                h[i] = self.lns[i](self.gru_cells[i](h[i-1], h[i]))
            h_out_series.append(h[-1])

        if self.output_series: # dim: batch_size, seq_length, output_dim
            v = self.fc(torch.stack(h_out_series, dim=1))
        else:
            v = self.fc(h[-1])

        return v, x_reconstructed, mu, logvar


    # Loss function
    def loss_function(self, net_output, ground_truth):
        """
        loss fuction specifically for the model
        net_output: exaclty the output of the model, Tensor / Tuple, depends on the model
        ground_truth: Tuple of (x, y), input and output from the dataset
        """
        v_pred, x_reconstructed, mu, logvar = net_output
        x, v_label = ground_truth

        v_pred = v_pred.view(-1, self.output_dim)
        v_label = v_label.view(-1, self.output_dim)

        if self.log_focus_factor:
            v_label = v_label.sign() * (v_label.abs() * self.log_focus_factor).log1p()

        criterion_mse = nn.MSELoss(reduction='mean') 
        criterion_nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')

        # def criterion_nll(x_reconstructed, x, nll):
        #     # p(z|x) = N(z; mu_z(x), sigma_z(x)), \Sigma_z(x) = diag(sigma_z1(x), sigma_z2(x), ...)
        #     # f = Wz, def: \mu_f(x) = W\mu_z(x)
        #     # r = exp(f), def: \mu_r(x) = exp(\mu_f(x))
        #     # x = Poisson(r)
        #     # -E_q(z|x)[log p(x|z)]   = -\sum N(z; \mu_z, \Sigma_z) * log Poisson(x; r)
        #                               = \sum_i \frac{\mu_r(x) * exp(diag(W \Sigma_z(x) W^T) / 2)}{\prod_j W[:,j]}_i - x_i * \mu_f(x)_i

        #     Sigma_z = torch.diag_embed(logvar.exp()) # B * num_steps * latent_dim * latent_dim
        #     W = self.decoder.fc3.weight # input_dim * latent_dim
            
        #     mu_r = x_reconstructed # B * num_steps * input_dim
        #     pow_exp_term = torch.diagonal(torch.matmul(torch.matmul(W, Sigma_z), W.t()) / 2, dim1=-2, dim2=-1) # B * num_steps * input_dim 
        #     prod_term = torch.prod(torch.where((W >= -1) & (W <= 1), 1.0, W), dim=-1) # input_dim

        #     nll_loss = torch.mean(mu_r * torch.exp(pow_exp_term) / prod_term - x * mu_r) # sum of: B * num_steps * input_dim


            # nll_loss = ((-nll.sum(dim=-1)).exp() * (x * x_reconstructed.log() - x_reconstructed).sum(dim=-1)).sum()
            
            # if nll_loss.isnan():
                # a = 1
                # raise ValueError("NLL Loss is NaN")

            # return nll_loss

            

        recon_loss = criterion_nll(x_reconstructed, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        pred_loss = criterion_mse(v_pred, v_label)

        return self.loss_ratio[0]*recon_loss + self.loss_ratio[1]*kl_loss +self.loss_ratio[2]*pred_loss

    
    def post_process(self, net_output):
        """
        post process the output of the model to produce the final velocity prediction
        net_output: exaclty the output of the model, Tensor / Tuple, depends on the model
        """
        v_pred, *trivial = net_output
        if self.log_focus_factor:
            v_pred = v_pred.sign() * v_pred.abs().expm1() / self.log_focus_factor
        return v_pred
    
    def fake_quantize_weights(self, qiw=1, qfw=7, module_type=[nn.Linear, nn.GRUCell]):
        if nn.GRUCell in module_type:
            for i in range(self.num_gru_layers):
                self.gru_cells[i].weight_ih.data = quantize_tensor(self.gru_cells[i].weight_ih.data, qiw, qfw)
                self.gru_cells[i].weight_hh.data = quantize_tensor(self.gru_cells[i].weight_hh.data, qiw, qfw)
        if nn.Linear in module_type:
            self.fc.fc_layer.weight.data = quantize_tensor(self.fc.fc_layer.weight.data, qiw, qfw)
            self.encoder.fc1_mu.weight.data = quantize_tensor(self.encoder.fc1_mu.weight.data, qiw, qfw)