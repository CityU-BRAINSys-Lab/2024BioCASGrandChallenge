"""
ANN model specifically for testing purposes
"""
import torch
from ..train import ANNModel2D as ANNModel2D_train
from ..train import ANNModel3D as ANNModel3D_train

## to be modified to the realtime style
class ANNModel2D(ANNModel2D_train):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)
    

class ANNModel3D(ANNModel3D_train):
    def __init__(self, bin_width, **kwargs):
        super().__init__(**kwargs)
        self.step_size = int(bin_width // self.sampling_rate)
        self.bin_window_size = self.step_size*self.num_steps

        self.register_buffer("data_buffer", torch.zeros(1, self.input_dim).type(torch.float32), persistent=False)

    def forward(self, x):
        predictions = []
        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]

            # Accumulate
            spikes = self.data_buffer.clone()
            
            acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            for i in range(self.num_steps):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp

            pred = super().forward(acc_spikes.unsqueeze(dim=0).to(x.device))
            predictions.append(pred)

        predictions = torch.stack(predictions).squeeze(dim=1)
    
        return predictions