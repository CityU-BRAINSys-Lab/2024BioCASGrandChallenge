import torch
import snntorch as snn
from snntorch._neurons.neurons import _SpikeTensor, _SpikeTorchConv

def init_sleepy():
    """
    Used to initialize mem and deltat as an empty SpikeTensors.
    ``init_flag`` is used as an attribute in the forward pass to convert
    the hidden states to the same as the input.
    """
    mem = _SpikeTensor(init_flag=False)
    deltat = _SpikeTensor(init_flag=False)

    return mem, deltat

class SleepyStdLIF(snn.Leaky):
    """
    right now this is neuron class is only implemented for init_hidden=True.
    """
    def __init__(self, beta, threshold=1, spike_grad=None, surrogate_disable=False, init_hidden=False, inhibition=False, learn_beta=False, learn_threshold=False, reset_mechanism="subtract", state_quant=False, output=False, graded_spikes_factor=1, learn_graded_spikes_factor=False):
        super().__init__(beta, threshold, spike_grad, surrogate_disable, init_hidden, inhibition, learn_beta, learn_threshold, reset_mechanism, state_quant, output, graded_spikes_factor, learn_graded_spikes_factor)
        if self.init_hidden:
            self.mem, self.deltat = init_sleepy()

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
            deltat = _SpikeTorchConv(deltat, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)
            self.deltat = _SpikeTorchConv(self.deltat, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 /
        # self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as
        # initial beta
        # beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._leaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _base_state_function_hidden(self, input_):
        hot = torch.Tensor(input_ != 0).to(torch.float32)  # which neurons are activated
        cold = 1 - hot  # which neurons are cold
        base_fn = self.mem * (1 - 1 / self.beta) ** (hot * (self.deltat + 1)) + input_  # update mem according to neuron model
        self.deltat = cold * (self.deltat + cold)  # update sleeping times

        return base_fn

class SleepyLogLIF(snn.Leaky):
    """
    right now this is neuron class is only implemented for init_hidden=True.
    """
    def __init__(
        self, 
        a, 
        b,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        self.loglif_a = a
        self.loglif_b = b
        beta = 1.  # assumed that beta is not used bc forward functions are overwritten
        super(SleepyLogLIF, self).__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )
        if self.init_hidden:
            self.mem, self.deltat = init_sleepy()

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
            deltat = _SpikeTorchConv(deltat, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)
            self.deltat = _SpikeTorchConv(self.deltat, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 /
        # self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as
        # initial beta
        # beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._leaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _base_state_function_hidden(self, input_):
        hot = torch.Tensor(input_ != 0).to(torch.float32)  # which neurons are activated
        cold = 1 - hot  # which neurons are cold
        base_fn = self.mem - torch.sign(self.mem) * self.loglif_a * torch.log2(1e-5 + self.loglif_b * torch.minimum(hot * (self.deltat + 1), 2 ** (torch.abs(self.mem) / self.loglif_a) - 1) + 1) + input_  # update mem according to neuron model
        self.deltat = cold * (self.deltat + cold)  # update sleeping times

        return base_fn


class Integrator(snn.Leaky):
    """
    right now this is neuron class is only implemented for init_hidden=True.
    """
    def __init__(
        self,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="none",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super(Integrator, self).__init__(
            1.,  # beta, not used here
            1e6,  # threshold, should be unreachably high
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

    def _base_state_function_hidden(self, input_):
        base_fn = self.mem + input_  # update mem according to neuron model

        return base_fn
        