import numpy as np
import time

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import stork.plotting as splt

import stork.nodes.base
from . import generators
from . import loss_stacks
from . import monitors

import logging

logger = logging.getLogger(__name__)

from torchaudio.functional import lowpass_biquad


class RecurrentSpikingModel(nn.Module):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
        filter_params=None,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs

        self.device = device
        self.dtype = dtype

        self.fit_runs = []

        self.groups = []
        self.connections = []
        self.devices = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None
        self.sparse_input = sparse_input

        self.filter_params = filter_params
     

    def configure(
        self,
        input,
        output,
        n_hidden_layer=None,  # ltj
        hidden_size=None,  # ltj
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        generator=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb
        self.n_hidden_layer = n_hidden_layer  # ltj
        self.hidden_size = hidden_size  # ltj

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.TemporalCrossEntropyReadoutStack()

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        # configure data generator
        self.data_generator_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)

        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.configure_scheduler(self.scheduler_class, self.scheduler_kwargs)
        
        self.to(self.device)
        
    def set_nb_steps(self, nb_time_steps):
        self.nb_time_steps = nb_time_steps
        
        # configure data generator
        self.data_generator_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )
        
        for g in self.groups:
            g.set_nb_steps(nb_time_steps)
            
        self.reset_states()

    def time_rescale(self, time_step=1e-3, batch_size=None):
        """Saves the model then re-configures it with the old hyper parameters, but the new timestep.
        Then loads the model again."""
        if batch_size is not None:
            self.batch_size = batch_size
        saved_state = self.state_dict()
        saved_optimizer = self.optimizer_instance
        saved_scheduler = self.scheduler_instance
        self.nb_time_steps = int(self.nb_time_steps * self.time_step / time_step)
        self.configure(
            self.input_group,
            self.output_group,
            loss_stack=self.loss_stack,
            generator=self.data_generator_,
            time_step=time_step,
            optimizer=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            wandb=self.wandb,
        )
        # Commenting this out will re-init the optimizer
        self.optimizer_instance = saved_optimizer
        self.scheduler_instance = saved_scheduler
        self.load_state_dict(saved_state)

    def configure_optimizer(self, optimizer_class, optimizer_kwargs):
        if optimizer_kwargs is not None:
            self.optimizer_instance = optimizer_class(
                self.parameters(), **optimizer_kwargs
            )
        else:
            self.optimizer_instance = optimizer_class(self.parameters())
            
    def configure_scheduler(self, scheduler_class, scheduler_kwargs):
        if scheduler_class is None:
            self.scheduler_instance = None
        else:
            if scheduler_kwargs is not None:
                self.scheduler_instance = scheduler_class(
                    self.optimizer_instance, **scheduler_kwargs
                )
            else:
                self.scheduler_instance = scheduler_class(self.optimizer_instance)

    def reconfigure(self):
        """Runs configure and replaces arguments with default from last run.

        This should reset the model an reinitialize all trainable variables.
        """
        if self.input_group is None or self.output_group is None:
            print(
                "Warning! No input or output group has been assigned yet. Run configure first."
            )
            return

        for o in self.groups + self.connections:
            o.configure(self.batch_size, self.device, self.dtype)

        # Re-init optimizer
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)

    def prepare_data(self, dataset):
        return self.data_generator_.prepare_data(dataset)

    def data_generator(self, dataset, shuffle=True):
        return self.data_generator_(dataset, shuffle=shuffle)

    def add_group(self, group):
        self.groups.append(group)
        self.add_module("group%i" % len(self.groups), group)
        return group

    def add_connection(self, con):
        self.connections.append(con)
        self.add_module("con%i" % len(self.connections), con)
        return con

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        return monitor

    def reset_states(self, batch_size=None):
        for g in self.groups:
            g.reset_state(batch_size)

    def evolve_all(self):
        for g in self.groups:
            g.evolve()
            g.clear_input()

    def apply_constraints(self):
        for c in self.connections:
            c.apply_constraints()

    def propagate_all(self):
        for c in self.connections:
            c.propagate()

    def execute_all(self):
        for d in self.devices:
            d.execute()

    def monitor_all(self):
        for m in self.monitors:
            m.execute()

    def compute_regularizer_losses(self):
        reg_loss = torch.zeros(1, device=self.device)
        for g in self.groups:
            reg_loss += g.get_regularizer_loss()
        for c in self.connections:
            reg_loss += c.get_regularizer_loss()
        return reg_loss

    def remove_regularizers(self):
        for g in self.groups:
            g.remove_regularizers()
        for c in self.connections:
            c.remove_regularizers()

    def run(self, x_batch, cur_batch_size=None, record=False):
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch)
        for t in range(self.nb_time_steps):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all()
            self.propagate_all()
            self.execute_all()
            if record:
                self.monitor_all()
        self.out = self.output_group.get_out_sequence()

        if self.filter_params is not None:
            logger.info("Applying lowpass filter to output")
            
            # Apply lowpass filter
            x0_filtered = lowpass_biquad(
                self.out[..., 0],
                sample_rate=self.filter_params["sampling_rate"],
                cutoff_freq=self.filter_params["cutoff"],
            )
            x1_filtered = lowpass_biquad(
                self.out[..., 1],
                sample_rate=self.filter_params["sampling_rate"],
                cutoff_freq=self.filter_params["cutoff"],
            )

            # Stack the filtered channels back together
            self.out = torch.stack((x0_filtered, x1_filtered), dim=-1)

        return self.out

    def forward_pass(self, x_batch, cur_batch_size, record=False):
        # run recurrent dynamics
        us = self.run(x_batch, cur_batch_size, record=record)
        return us

    def get_example_batch(self, dataset, **kwargs):
        self.prepare_data(dataset)
        for batch in self.data_generator(dataset, **kwargs):
            return batch

    def get_total_loss(self, output, target_y, regularized=True):
        if type(target_y) in (list, tuple):
            target_y = [ty.to(self.device) for ty in target_y]
        else:
            target_y = target_y.to(self.device)

        self.out_loss = self.loss_stack(output, target_y)

        if regularized:
            self.reg_loss = self.compute_regularizer_losses()
            total_loss = self.out_loss + self.reg_loss
        else:
            total_loss = self.out_loss

        return total_loss

    def evaluate(self, test_dataset, train_mode=False, one_batch=False):
        self.train(train_mode)
        self.prepare_data(test_dataset)
        metrics = []
        loss_min_batch = np.inf
        for local_X, local_y in self.data_generator(test_dataset, shuffle=False):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )
            if self.out_loss < loss_min_batch:
                loss_min_batch = self.out_loss
                pre_batch_best = (
                    output.cpu().detach().numpy()[0, :, :]
                )  # ltj, Choose the first sample in the batch with minimal loss as an example
                gt_batch_best = (
                    local_y.cpu().detach().numpy()[0, :, :]
                )  # ltj, Choose the first sample in the batch with minimal loss as an example
            if one_batch:
                pre_batch_best = output.cpu().detach().numpy()
                gt_batch_best = local_y.cpu().detach().numpy()
                break
        return np.mean(np.array(metrics), axis=0) , pre_batch_best, gt_batch_best  # ltj

    def regtrain_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            self.reg_loss = self.compute_regularizer_losses()

            total_loss = self.get_total_loss(output, local_y)
            loss = self.reg_loss

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            loss.backward()
            self.optimizer_instance.step()
            self.apply_constraints()

        if self.scheduler_instance is not None:
            self.scheduler_instance.step()
            
        return np.mean(np.array(metrics), axis=0)

    def train_epoch(self, dataset, shuffle=True, mask=None):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            if mask is not None:
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        param.grad.data.mul_(mask[name])

            self.optimizer_instance.step()
            self.apply_constraints()
            
        if self.scheduler_instance is not None:
            self.scheduler_instance.step()

        return np.mean(np.array(metrics), axis=0)

    def get_metric_names(self, prefix="", postfix=""):
        metric_names = ["loss", "reg_loss"] + self.loss_stack.get_metric_names()
        return ["%s%s%s" % (prefix, k, postfix) for k in metric_names]

    def get_metrics_string(self, metrics_array, prefix="", postfix=""):
        s = ""
        names = self.get_metric_names(prefix, postfix)
        for val, name in zip(metrics_array, names):
            s = s + " %s=%.3g" % (name, val)
        return s

    def get_metrics_dict(self, metrics_array, prefix="", postfix=""):
        s = {}
        names = self.get_metric_names(prefix, postfix)
        for val, name in zip(metrics_array, names):
            s[name] = val
        return s

    def get_metrics_history_dict(self, metrics_array, prefix="", postfix=""):
        " Create metrics history dict. " ""
        s = ""
        names = self.get_metric_names(prefix, postfix)
        history = {name: metrics_array[:, k] for k, name in enumerate(names)}
        return history

    def prime(self, dataset, nb_epochs=10, verbose=True, wandb=None):
        self.hist = []
        for ep in range(nb_epochs):
            t_start = time.time()
            ret = self.regtrain_epoch(dataset)
            self.hist.append(ret)

            if self.wandb is not None:
                self.wandb.log(
                    {key: value for (key, value) in zip(self.get_metric_names(), ret)}
                )

            if verbose:
                t_iter = time.time() - t_start
                print(
                    "%02i %s t_iter=%.2f" % (ep, self.get_metrics_string(ret), t_iter)
                )

        self.fit_runs.append(self.hist)
        history = self.get_metrics_history_dict(np.array(self.hist))
        return history

    def fit(self, dataset, nb_epochs=10, verbose=True, shuffle=True, wandb=None):
        self.hist = []
        self.wall_clock_time = []
        self.train()
        for ep in range(nb_epochs):
            t_start = time.time()
            ret = self.train_epoch(dataset, shuffle=shuffle)
            self.hist.append(ret)

            if self.wandb is not None:
                self.wandb.log(
                    {key: value for (key, value) in zip(self.get_metric_names(), ret)}
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s t_iter=%.2f" % (ep, self.get_metrics_string(ret), t_iter)
                )

        self.fit_runs.append(self.hist)
        history = self.get_metrics_history_dict(np.array(self.hist))
        return history

    def fit_validate(
        self,
        dataset,
        valid_dataset,
        n_timesteps_train=None,
        n_timesteps_val=None,
        nb_epochs=10,
        verbose=True,
        wandb=None,
        is_save_model_every_epoch=False,
        model_name_prefix="./models/",
        is_FR_every_epoch=False,
        is_raster_every_epoch=False,
        raster_name_prefix="./figures/rasters/",
        nb_samples=2,
        mask=None,
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []
        loss_min_epoch = np.inf

        if is_FR_every_epoch:
            # Add monitors for spikes and membrane potential
            for i in range(self.n_hidden_layer):
                self.add_monitor(stork.monitors.SpikeCountMonitor(self.groups[1 + i]))
            for i in range(self.n_hidden_layer):
                self.add_monitor(stork.monitors.StateMonitor(self.groups[1 + i], "out"))
            self.hist_val_avg_hidden_FR = []

        for ep in range(nb_epochs):
            t_start = time.time()
            if n_timesteps_train is not None:
                self.nb_time_steps = n_timesteps_train
            self.train()
            ret_train = self.train_epoch(dataset, mask=mask)
            self.train(False)

            if is_raster_every_epoch:
                plt.figure(dpi=200)
                splt.plot_activity_snapshot(
                    self, data=dataset, nb_samples=nb_samples, point_alpha=0.3
                )
                plt.savefig(raster_name_prefix + " -epoch " + str(ep) + ".png")
                plt.close()

            if n_timesteps_val is not None:
                self.nb_time_steps = n_timesteps_val
            ret_valid, pred_best_batch, gt_best_batch = self.evaluate(valid_dataset)  # ltj
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            if ret_valid[0] < loss_min_epoch:
                loss_min_epoch = ret_valid[0]
                pred_best_epoch = pred_best_batch
                gt_best_epoch = gt_best_batch

            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    }
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)

                logger.info(
                    "%02i %s --%s t_iter=%.2f"
                    % (
                        ep,
                        self.get_metrics_string(ret_train),
                        self.get_metrics_string(ret_valid, prefix="val_"),
                        t_iter,
                    )
                )

            if is_save_model_every_epoch:  #  ltj
                # torch.save(self, model_name_prefix + ' -epoch '+ str(ep) + '.pth')
                torch.save(
                    self.state_dict(), model_name_prefix + " -epoch " + str(ep) + ".pth"
                )

            if is_FR_every_epoch:
                if n_timesteps_val is not None:
                    self.nb_time_steps = n_timesteps_val
                res = self.monitor(valid_dataset)
                total_spikes_per_layer = [
                    torch.sum(res[i]).item() for i in range(self.n_hidden_layer)
                ]
                avg_layer_freq = [
                    nb
                    / len(valid_dataset)
                    / (self.nb_time_steps * self.time_step)
                    / self.hidden_size
                    for nb in total_spikes_per_layer
                ]
                self.hist_val_avg_hidden_FR.append(avg_layer_freq)

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        if is_FR_every_epoch:  # ltj
            history["val_avg_hidden_FR"] = np.array(self.hist_val_avg_hidden_FR)
        return history, pred_best_epoch, gt_best_epoch

    def get_probabilities(self, x_input):
        probs = []
        # we don't care about the labels, but want to use the generator
        fake_labels = torch.zeros(len(x_input))
        self.prepare_data((x_input, fake_labels))
        for local_X, local_y in self.data_generator((x_input, fake_labels)):
            output = self.forward_pass(local_X, cur_batch_size=len(local_y))
            tmp = torch.exp(self.loss_stack.log_py_given_x(output))
            probs.append(tmp)
        return torch.cat(probs, dim=0)

    def get_predictions(self, dataset):
        self.prepare_data(dataset)

        pred = []
        for local_X, _ in self.data_generator(dataset, shuffle=False):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            pred_labels = self.loss_stack.predict(output).cpu().numpy()
            pred.append(pred_labels)

        return np.concatenate(pred)

    def predict(self, data, train_mode=False):
        self.train(train_mode)
        if type(data) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(data, cur_batch_size=len(data))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            self.prepare_data(data)
            pred = []
            for local_X, _ in self.data_generator(data, shuffle=False):
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def monitor(self, dataset):
        self.prepare_data(dataset)

        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for local_X, local_y in self.data_generator(dataset, shuffle=False):
            for m in self.monitors:
                m.reset()

            output = self.forward_pass(
                local_X, record=True, cur_batch_size=len(local_X)
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def monitor_backward(self, dataset):
        """
        Allows monitoring of gradients with GradientMonitor
            - If there are no GradientMonitors, this runs the usual `monitor` method
            - Returns both normal monitor output and backward monitor output
        """

        # if there is a gradient monitor
        if any([isinstance(m, monitors.GradientMonitor) for m in self.monitors]):
            self.prepare_data(dataset)

            # Set monitors to record gradients
            gradient_monitors = [
                m for m in self.monitors if isinstance(m, monitors.GradientMonitor)
            ]
            for gm in gradient_monitors:
                gm.set_hook()

            # Prepare a list for each monitor to hold the batches
            results = [[] for _ in self.monitors]
            for local_X, local_y in self.data_generator(dataset, shuffle=False):
                for m in self.monitors:
                    m.reset()

                # forward pass
                output = self.forward_pass(
                    local_X, record=True, cur_batch_size=len(local_X)
                )

                # compute loss
                total_loss = self.get_total_loss(output, local_y)

                # Use autograd to compute the backward pass.
                self.optimizer_instance.zero_grad()
                total_loss.backward()

                # do not call an optimizer step as that would update the weights!

                # Retrieve data from monitors
                for k, mon in enumerate(self.monitors):
                    results[k].append(mon.get_data())

            # Turn gradient recording off
            for gm in gradient_monitors:
                gm.remove_hook()

            return [torch.cat(res, dim=0) for res in results]

        else:
            return self.monitor(dataset)

    def record_group_outputs(self, group, x_input):
        res = []
        # we don't care about the labels, but want to use the generator
        fake_labels = torch.zeros(len(x_input))
        self.prepare_data((x_input, fake_labels))
        for local_X, _ in self.data_generator((x_input, fake_labels)):
            output = self.forward_pass(local_X)
            res.append(group.get_out_sequence())
        return torch.cat(res, dim=0)

    def evaluate_ensemble(
        self, dataset, test_dataset, nb_repeats=5, nb_epochs=10, callbacks=None
    ):
        """Fits the model nb_repeats times to the data and returns evaluation results.

        Args:
            dataset: Training dataset
            test_dataset: Testing data
            nb_repeats: Number of repeats to retrain the model (default=5)
            nb_epochs: Train for x epochs (default=20)
            callbacks: A list with callbacks (functions which will be called as f(self) whose return value
                       is stored in a list and returned as third return value

        Returns:
            List of learning training histories curves
            and a list of test scores and if callbacks is not None an additional
            list with the callback results
        """
        results = []
        test_scores = []
        callback_returns = []
        for k in range(nb_repeats):
            print("Repeat %i/%i" % (k + 1, nb_repeats))
            self.reconfigure()
            self.fit(dataset, nb_epochs=nb_epochs, verbose=(k == 0))
            score = self.evaluate(test_dataset)
            results.append(np.array(self.hist))
            test_scores.append(score)
            if callbacks is not None:
                callback_returns.append([callback(self) for callback in callbacks])

        if callbacks is not None:
            return results, test_scores, callback_returns
        else:
            return results, test_scores

    def summary(self):
        """Print model summary"""

        print("\n# Model summary")
        print("\n## Groups")
        for group in self.groups:
            if group.name is None or group.name == "":
                print("no name, %s" % (group.shape,))
            else:
                print("%s, %s" % (group.name, group.shape))

        print("\n## Connections")
        for con in self.connections:
            print(con)
            
            
    def half(self):
        """ 
        Convert model to half precision. 
        Because stork does not treat group states as parameters,
        we have to convert the model to half precision manually.
        """
        
        # Convert group states to half precision
        for group in self.groups:
            group.half()

        # This will convert parameters to half precision
        super().half()
        
        self.set_dtype(torch.float16)
         
        return self
    
    
    def set_dtype(self, dtype):
        self.dtype = dtype
        
        for g in self.groups:
            g.set_dtype(dtype)
        for c in self.connections:
            c.set_dtype(dtype)
            
        self.to(self.device)
        return self
            

class DoubleInputRecSpikingModel(RecurrentSpikingModel):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        nb_outputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.device = device
        self.dtype = dtype

        self.fit_runs = []

        self.groups = []
        self.connections = []
        self.devices = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None
        self.sparse_input = sparse_input

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        generator1=None,
        generator2=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.TemporalCrossEntropyReadoutStack()

        if generator1 is None:
            self.data_generator1_ = generators.StandardGenerator()
        else:
            self.data_generator1_ = generator1

        if generator2 is None:
            self.data_generator2_ = generators.StandardGenerator()
        else:
            self.data_generator2_ = generator2

        # configure data generator
        self.data_generator1_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )
        self.data_generator2_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)
        self.to(self.device)

    def monitor(self, datasets):
        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(datasets[0], shuffle=False),
            self.data_generator2(datasets[1], shuffle=False),
        ):
            for m in self.monitors:
                m.reset()

            local_X = torch.cat((local_X1, local_X2), dim=2)

            output = self.forward_pass(
                local_X,
                record=True,
                cur_batch_size=len(local_X),
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def data_generator1(self, dataset, shuffle=True):
        return self.data_generator1_(dataset, shuffle=shuffle)

    def data_generator2(self, dataset, shuffle=True):
        return self.data_generator2_(dataset, shuffle=shuffle)

    def predict(self, datasets, train_mode=False):
        self.train(train_mode)
        print("predicting")
        if type(datasets) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(datasets, cur_batch_size=len(datasets))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            # self.prepare_data(data)
            pred = []
            for (local_X1, local_y1), (local_X2, local_y2) in zip(
                self.data_generator1(datasets[0], shuffle=False),
                self.data_generator2(datasets[1], shuffle=False),
            ):
                local_X = torch.cat((local_X1, local_X2), dim=2)
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        # self.prepare_data(dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def evaluate(self, dataset, train_mode=False):
        self.train(train_mode)
        # self.prepare_data(test_dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

        return np.mean(np.array(metrics), axis=0)