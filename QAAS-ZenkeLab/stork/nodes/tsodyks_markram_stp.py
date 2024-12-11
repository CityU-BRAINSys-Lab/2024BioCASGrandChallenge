import numpy as np
import torch
from torch.nn import Parameter
from stork.nodes.base import CellGroup


class TsodyksMarkramSTP(CellGroup):
    def __init__(self, parent_group, tau_d=100e-3, tau_f=50e-3, U=0.2):
        super(TsodyksMarkramSTP, self).__init__(parent_group.shape, stateful=parent_group.stateful)
        # self.states = parent_group.states # Share states
        self.parent_group = parent_group
        self.default_target = self.parent_group.default_target
        self.u_jump = U
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.U = U

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_x = float(np.exp(-time_step / self.tau_d))
        self.scl_x = 1.0 - self.dcy_x
        self.dcy_u = float(np.exp(-time_step / self.tau_f))
        self.scl_u = 1.0 - self.dcy_u

    def clear_input(self):
        pass

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.state_x = torch.ones(self.int_shape, device=self.device, dtype=self.dtype)
        self.state_u = self.U * torch.ones(self.int_shape, device=self.device, dtype=self.dtype)
        self.out = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)
        for key in self.parent_group.states:
            self.states[key] = self.parent_group.states[key]

    def add_to_state(self, target, x):
        """ Just implement pass through to parent group. """
        self.parent_group.add_to_state(target, x)

    def forward(self):
        x = self.state_x
        u = self.state_u
        spikes = self.parent_group.out

        stp_w = x * u * spikes
        new_x = x + self.scl_x * (1.0 - x) - stp_w
        new_u = u + self.scl_u * (self.u_jump - u) + self.u_jump * (1.0 - u) * spikes

        self.state_x = new_x
        self.state_u = new_u
        self.out = stp_w / self.u_jump
        # self.out_seq.append(stp_w)


class TsodyksMarkramLearnSTP(TsodyksMarkramSTP):
    def __init__(self, parent_group, tau_d_max=100e-3, tau_f_max=500e-3, U=0.2, learn=True):
        super(TsodyksMarkramLearnSTP, self).__init__(parent_group, tau_d_max, tau_f_max, U)
        # self.states = parent_group.states # Share states

        self.tau_d_max = tau_d_max
        self.tau_f_max = tau_f_max
        self.parent_group = parent_group
        self.learn = learn

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dt_ = torch.tensor(time_step, device=device, dtype=dtype)
        d_var = torch.randn(self.nb_units, device=device, dtype=dtype, requires_grad=self.learn)
        f_var = torch.randn(self.nb_units, device=device, dtype=dtype, requires_grad=self.learn)
        self.d_var = Parameter(d_var, requires_grad=self.learn)
        self.f_var = Parameter(f_var, requires_grad=self.learn)
        self.u_var = Parameter(torch.randn(self.nb_units, device=device, dtype=dtype, requires_grad=self.learn))
        # running configure here at the end because it invokes reset state
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.dcy_x = torch.exp(-self.dt_ / (torch.sigmoid(self.d_var) * self.tau_d_max))
        self.scl_x = 1.0 - self.dcy_x
        self.dcy_u = torch.exp(-self.dt_ / (torch.sigmoid(self.f_var) * self.tau_f_max))
        self.scl_u = 1.0 - self.dcy_u

    def forward(self):
        x = self.state_x
        u = self.state_u

        spikes = self.parent_group.out

        stp_w = x * u * spikes
        new_x = torch.relu(x + self.scl_x * (1.0 - x) - stp_w)  # relu to prevent negativ values
        u_jump = self.u_jump * torch.sigmoid(self.u_var)
        new_u = torch.relu(u + self.scl_u * (u_jump - u) + torch.sigmoid(u_jump) * (1.0 - u) * spikes)

        self.state_x = new_x
        self.state_u = new_u
        self.out = stp_w / self.u_jump
        # self.out_seq.append(stp_w)