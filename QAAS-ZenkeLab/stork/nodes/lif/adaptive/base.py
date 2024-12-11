import numpy as np
import torch
from torch.nn import Parameter
from stork.nodes.lif.base import LIFGroup


class AdaptiveLIFGroup(LIFGroup):
    """
    Base class for LIF neurons with adaptive threshold

    Args: 
        shape: The neuron group shape
        tau_ada: The adaptation time constant
        adapt_a: The adaptation strength
    """

    def __init__(self, shape, tau_ada=100e-3, adapt_a=0.1, learn_timescales_ada=False, het_timescales_ada=False, TC_ada_het_init='Uniform', ada_bandpass_high_ratio_cut=2, is_delta_syn = False, ada_het_forward_method='bandpass', **kwargs): #ltj
        super().__init__(shape, **kwargs)
        self.tau_ada = tau_ada
        self.adapt_a = adapt_a
        self.learn_timescales_ada = learn_timescales_ada #ltj
        self.het_timescales_ada = het_timescales_ada #ltj
        self.ada_bandpass_high_ratio_cut = ada_bandpass_high_ratio_cut #ltj
        self.TC_ada_het_init = TC_ada_het_init #ltj
        self.ada_het_forward_method = ada_het_forward_method
        self.ada = None
        self.name = 'AdaptiveLIFGroup'

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_ada = float(np.exp(-time_step / self.tau_ada))
        self.scl_ada = 1.0 - self.dcy_ada
        if self.learn_timescales_ada: #ltj
            size_tc = self.shape[0] if self.het_timescales_ada else 1 #ltj
            if self.TC_ada_het_init == 'Uniform':
                ada_param = torch.rand(
                    size_tc, device=device, dtype=dtype, requires_grad=True) #ltj
            elif self.TC_ada_het_init == 'Gaussian':
                ada_param = torch.randn(
                    size_tc, device=device, dtype=dtype, requires_grad=True) #ltj
            elif self.TC_ada_het_init == 'Constant':
                ada_param = torch.ones(
                    size_tc, device=device, dtype=dtype, requires_grad=True) #ltj
            # elif self.TC_ada_het_init == 'XavierUniform':
            #     ada_param = torch.empty(size_tc, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_uniform_(ada_param)  # Xavier uniform initialization
            # elif self.TC_ada_het_init == 'XavierGassian':
            #     ada_param = torch.empty(size_tc, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_normal_(ada_param)  # Xavier gaussian initialization
            elif self.TC_ada_het_init == 'logNormal':
                ada_param = torch.empty(size_tc, device=device, dtype=dtype, requires_grad=True)
                mean = -4.3  # Example mean of the underlying normal distribution
                std = 2.5   # Example standard deviation of the underlying normal distribution
                torch.nn.init.normal_(ada_param, mean=mean, std=std)
                ada_param = ada_param.exp()  # Converting normal distribution to log-normal
            elif self.TC_ada_het_init == 'logspace':
                ada_param = np.logspace(np.log10(1), np.log10(10), num=size_tc)
                ada_param = torch.tensor(ada_param, device=device, dtype=dtype) #ltj
            self.ada_param = Parameter(ada_param, requires_grad=self.learn_timescales)
        elif self.het_timescales_ada:
            size_tc = self.shape[0]
            if self.TC_ada_het_init == 'Uniform':
                ada_param = torch.rand(size_tc, device=device, dtype=dtype) #ltj
            elif self.TC_ada_het_init == 'Gaussian':
                ada_param = torch.randn(size_tc, device=device, dtype=dtype) #ltj
            elif self.TC_ada_het_init == 'Constant':
                ada_param = torch.ones(size_tc, device=device, dtype=dtype) #ltj
            elif self.TC_ada_het_init == 'logNormal':
                ada_param = torch.empty(size_tc, device=device, dtype=dtype)
                mean = -4.3  # Example mean of the underlying normal distribution
                std = 2.5   # Example standard deviation of the underlying normal distribution
                torch.nn.init.normal_(ada_param, mean=mean, std=std)
                ada_param = ada_param.exp()  # Converting normal distribution to log-normal
            elif self.TC_ada_het_init == 'logspace':
                ada_param = np.logspace(np.log10(1), np.log10(10), num=size_tc)
                ada_param = torch.tensor(ada_param, device=device, dtype=dtype) #ltj

            if self.ada_het_forward_method == 'bandpass':
                self.dcy_ada = torch.exp(-time_step /
                                        (self.ada_bandpass_high_ratio_cut * self.tau_ada * torch.sigmoid(ada_param)))
            elif self.ada_het_forward_method == 'highpass':
                softplus = torch.nn.Softplus()
                self.dcy_ada = torch.exp(-time_step /
                                        (self.tau_ada * softplus(ada_param)))
            elif self.ada_het_forward_method == 'original':
                self.dcy_ada = torch.exp(-time_step /
                                        (self.tau_ada * ada_param))
            self.scl_ada = 1.0 - self.dcy_ada
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_timescales_ada: #ltj
            if self.ada_het_forward_method == 'bandpass':
                self.dcy_ada = torch.exp(-self.time_step /
                                        (self.ada_bandpass_high_ratio_cut * self.tau_ada * torch.sigmoid(self.ada_param)))
            elif self.ada_het_forward_method == 'highpass':
                softplus = torch.nn.Softplus()
                self.dcy_ada = torch.exp(-self.time_step /
                                        (self.tau_ada * softplus(self.ada_param)))
            elif self.ada_het_forward_method == 'original':
                self.dcy_ada = torch.exp(-self.time_step /
                                        (self.tau_ada * self.ada_param))
            self.scl_ada = 1.0 - self.dcy_ada
        self.ada = self.get_state_tensor("ada", state=self.ada)

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        if not self.is_delta_syn:
            new_syn = self.dcy_syn * self.syn + self.input
            new_mem = (self.dcy_mem * self.mem + self.scl_mem *
                    (self.syn - self.adapt_a * self.ada)) * (1.0 - rst)
        else:
            new_mem = (self.dcy_mem * self.mem + self.scl_mem *
                    (self.input - self.adapt_a * self.ada)) * (1.0 - rst)  # multiplicative reset
            
        new_ada = self.dcy_ada * self.ada + self.scl_ada * self.out #ltj

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        if not self.is_delta_syn:
            self.syn = self.states["syn"] = new_syn
        self.ada = self.states["ada"] = new_ada
