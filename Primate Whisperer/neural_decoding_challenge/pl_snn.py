import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from tqdm import tqdm
from neurobench.benchmarks.workload_metrics import r2
from neurobench.utils import check_shape
from sklearn.metrics import r2_score as r2_score_sk

def plot_output(a_targets, b_preds, logdir, name="train_results_fig"):
    fig = plt.figure()
    # plot targets and preds on same figure
    plt.plot(
        a_targets[:, 0],
        a_targets[:, 1],
        label="target",
        ls="--",
        marker=".",
    )
    plt.plot(
        b_preds[:, 0],
        b_preds[:, 1],
        label="pred",
        alpha=0.75,
    )
    plt.legend()
    plt.savefig(os.path.join(logdir, name))
    plt.close()

class MyR2(r2):
    """version of the r2 class that works for batched data."""

    def __init__(self, seq_to_seq=True):
        super().__init__()
        self.y_pred_stacked = None
        self.y_true_stacked = None
        self.seq_to_seq = seq_to_seq

    def reset(self):
        """Reset metric state."""
        self.y_pred_stacked = None
        self.y_true_stacked = None

    def __call__(self, y_pred, y_true):
        check_shape(y_pred, y_true)
        # shape: batch, features

        if self.y_pred_stacked is None:
            self.y_pred_stacked = y_pred
            self.y_true_stacked = y_true
        else:
            self.y_pred_stacked = np.concatenate((self.y_pred_stacked, y_pred), axis=1)
            self.y_true_stacked = np.concatenate((self.y_true_stacked, y_true), axis=1)

        #return self.compute()

    def compute(self):
        if not self.seq_to_seq:
            x_r2 = r2_score_sk(
                self.y_true_stacked[:, 0],
                self.y_pred_stacked[:, 0],
                multioutput="raw_values",
            )
            y_r2 = r2_score_sk(
                self.y_true_stacked[:, 1],
                self.y_pred_stacked[:, 1],
                multioutput="raw_values",
            )
        else:
            x_r2 = r2_score_sk(
                self.y_true_stacked[:, :, 0],
                self.y_pred_stacked[:, :, 0],
                multioutput="raw_values",
            )
            y_r2 = r2_score_sk(
                self.y_true_stacked[:, :, 1],
                self.y_pred_stacked[:, :, 1],
                multioutput="raw_values",
            )

        r2 = (x_r2 + y_r2) / 2
        r2_mean = r2.mean()  # mean over all batches

        return r2_mean


class SpikingNetwork(LightningModule):
    def __init__(
        self, net, lr=1e-3, spike_regu=0.0, target_spike_share=0.1, weight_regu=0.0, scaler=None, output_type="displacement", loss_weight_type='lin', 
        loss_weight_a=None, loss_weight_b=None, hyperparameters=None
    ):
        super().__init__()
        self.net = net
        self.lr = lr
        self.spike_regu = spike_regu
        self.target_spike_share = target_spike_share
        self.weight_regu = weight_regu
        self.loss_weight_type = loss_weight_type
        self.loss_weight_a = loss_weight_a
        self.loss_weight_b = loss_weight_b
        self.scaler = scaler
        self.output_type = output_type
        self.sequential_testing = False

        self.seq_len = net.seq_len
        if loss_weight_type == 'lin':
            self.weighting_function = torch.linspace(0, 1, self.seq_len)
        elif loss_weight_type == 'equal':
            self.weighting_function = torch.ones(self.seq_len)
        elif loss_weight_type == 'sig':
            x = torch.linspace(0, 1, self.seq_len)
            self.weighting_function = 1 / (1 + torch.exp(-loss_weight_a * (x - loss_weight_b)))
        elif loss_weight_type == 'last':
            x = np.zeros(self.seq_len)
            x[-self.loss_weight_a:] = 1
            self.weighting_function = torch.tensor(x)
        else:
            raise Exception(f"weighting_type {loss_weight_type} is unknown.")

        # sum up to one
        self.weighting_function = self.weighting_function / torch.sum(self.weighting_function)

        self.train_y_hat = np.array([])
        self.train_y = np.array([])
        self.train_pr_idxs = np.array([])

        self.val_y_hat = np.array([])
        self.val_y = np.array([])
        self.val_pr_idxs = np.array([])

        self.test_y_hat = np.array([])
        self.test_y = np.array([])
        self.test_pr_idxs = np.array([])

        self.r2 = MyR2(seq_to_seq=True)

        if hyperparameters is not None:
            self.save_hyperparameters(hyperparameters)

    def set_sequential_testing(self, value):
        self.sequential_testing = value
        self.net.reset_mem_every_step = not value

    def sparsity_loss(self, preds, targets):
        """preds is the spk dictionary containing all layer spikes outputs."""
        spikes = preds.values()
        spike_shares = torch.tensor([0]).to(preds["out"].device)
        for s in spikes:
            spike_shares = torch.add(
                spike_shares,
                torch.nn.functional.l1_loss(torch.sum(s) / (torch.prod(torch.tensor(s.shape))), torch.tensor(self.target_spike_share).to(self.device)),
            )
        spike_loss = spike_shares / len(spikes)
        return spike_loss

    def scheduled_loss(self, epoch):
        idx = int(epoch / 10) + 1
        x = np.zeros(self.seq_len)
        x[-idx:] = 1
        self.weighting_function = torch.tensor(x)
    
    def weighted_mse(self, y_hat, y):
        batch_size, out_size = y_hat.shape[1], y_hat.shape[2]
        #self.scheduled_loss(self.trainer.current_epoch)  # creates epoch number-dependent loss weighting function
        final_weighting_function = self.weighting_function.to(y_hat.device).repeat(out_size, batch_size, 1).permute(2, 1, 0)
        mse_loss_unweighted = torch.nn.functional.mse_loss(y_hat, y, reduction='none')
        weighted_mse = final_weighting_function * mse_loss_unweighted
        mse_loss = weighted_mse.sum(0).mean(0).sum()
        return mse_loss
    
    def reset_model(self, sequential_testing=False, batch_size=1, device="cpu"):
        self.set_sequential_testing(sequential_testing)
        self.net.batch_size = batch_size
        self.net.device = device
        self.net.reset_mem()
        self.last_pos = None

    def forward(self, x):
        result = self.net(x)

        result = self.inverse_scale(result.cpu().detach().numpy())
        result = torch.tensor(result).to(x.device)
    
        if self.output_type == "displacement":
            result = torch.cumsum(result, dim=0)
            result = self.last_pos + result if self.last_pos is not None else result
            self.last_pos = result[-1]

        if x.shape[1] == 1:
            result = result.squeeze(1)
        
        return result
        

    def forward_scale(self, y):
        """Scale the labels from original scale to normalized scale"""
        return self.scaler.transform(y.reshape(-1,y.shape[-1])).reshape(*y.shape)
    
    def inverse_scale(self, y):
        """Scale the labels from normalized scale to original scale"""
        return self.scaler.inverse_transform(y.reshape(-1,y.shape[-1])).reshape(*y.shape)
    
    def training_step(self, batch, batch_idx):
        x, y, pr_idxs = batch
        y_hat = self.net(x)

        mse_loss = self.weighted_mse(y_hat, y)
        loss = mse_loss

        self.train_y_hat = np.concatenate((self.train_y_hat, y_hat.detach().cpu().numpy()), axis=1) if self.train_y_hat.size else y_hat.detach().cpu().numpy()
        self.train_y = np.concatenate((self.train_y, y.cpu().numpy()), axis=1) if self.train_y.size else y.cpu().numpy()
        self.train_pr_idxs = np.concatenate((self.train_pr_idxs, pr_idxs.cpu().numpy()), axis=1) if self.train_pr_idxs.size else pr_idxs.cpu().numpy()

        self.log("train_loss", loss, batch_size=y.shape[0])
        return loss

    def on_train_epoch_end(self):
        r2_scores = []
        r2_scores_seq = []
        self.train_y_hat = self.inverse_scale(self.train_y_hat)
        self.train_y = self.inverse_scale(self.train_y)

        seq_len, _, _ = self.train_y_hat.shape

        # sort by pr_idx
        sort_idx = np.argsort(self.train_pr_idxs[0,:])
        step = self.train_pr_idxs[:, sort_idx][0,1] - self.train_pr_idxs[:, sort_idx][0,0]
        step = int(seq_len / step)

        self.train_y_hat = self.train_y_hat[:, sort_idx, :]
        self.train_y = self.train_y[:, sort_idx, :]

        for i in range(step):
            sample_hat = self.train_y_hat[:, i::step]
            sample_hat_seq = sample_hat.transpose(1,0,2).reshape(-1, 1, sample_hat.shape[-1])

            sample = self.train_y[:, i::step]
            sample_seq = sample.transpose(1,0,2).reshape(-1, 1, sample.shape[-1])

            if self.output_type == "displacement":
                sample_hat = np.cumsum(sample_hat, axis=0)
                sample = np.cumsum(sample, axis=0)

                sample_hat_seq = np.cumsum(sample_hat_seq, axis=0)
                sample_seq = np.cumsum(sample_seq, axis=0)
            
            self.r2.reset()
            self.r2(sample_hat, sample)
            r2_score = self.r2.compute()
            r2_scores.append(r2_score)

            self.r2.reset()
            self.r2(sample_hat_seq, sample_seq)
            r2_score_seq = self.r2.compute()
            r2_scores_seq.append(r2_score_seq)
        
        r2_score = np.mean(r2_scores)
        r2_score_seq = np.mean(r2_scores_seq)

        self.log("train_r2", r2_score, batch_size=step, on_epoch=True)
        self.log("train_r2_seq", r2_score_seq, batch_size=step, on_epoch=True)

        #fig_save_dir = os.path.join(self.logger.log_dir, "figures")
        #os.makedirs(fig_save_dir, exist_ok=True)
        #fig_train_save_dir = os.path.join(fig_save_dir, "train")
        #os.makedirs(fig_train_save_dir, exist_ok=True)

        #num_steps_to_eval = 1024
        #num_figs = 1
        #start_at_fig = 0
        #for i in range(start_at_fig, start_at_fig + num_figs):
        #    plot_output(
        #        sample[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
        #        sample_hat[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
        #        fig_train_save_dir, 
        #        f"train_e_{self.current_epoch}_{i}"
        #    )
        
        self.train_y_hat = np.array([])
        self.train_y = np.array([])
        self.train_pr_idxs = np.array([])
        

    def validation_step(self, batch, batch_idx):
        x, y, pr_idxs = batch
        y_hat = self.net(x)

        mse_loss = self.weighted_mse(y_hat, y)
        loss = mse_loss

        self.val_y_hat = np.concatenate((self.val_y_hat, y_hat.detach().cpu().numpy()), axis=1) if self.val_y_hat.size else y_hat.detach().cpu().numpy()
        self.val_y = np.concatenate((self.val_y, y.cpu().numpy()), axis=1) if self.val_y.size else y.cpu().numpy()
        self.val_pr_idxs = np.concatenate((self.val_pr_idxs, pr_idxs.cpu().numpy()), axis=1) if self.val_pr_idxs.size else pr_idxs.cpu().numpy()

        self.log("val_loss", loss, batch_size=y.shape[0])
        return loss
    
    def on_validation_epoch_end(self):
        self.val_y_hat = self.inverse_scale(self.val_y_hat)
        self.val_y = self.inverse_scale(self.val_y)
        
        self.val_y_seq = self.val_y.transpose(1,0,2).reshape(-1, 1, self.val_y.shape[-1])
        self.val_y_hat_seq = self.val_y_hat.transpose(1,0,2).reshape(-1, 1, self.val_y_hat.shape[-1])

        if self.output_type == "displacement":
            self.val_y_hat = np.cumsum(self.val_y_hat, axis=0)
            self.val_y = np.cumsum(self.val_y, axis=0)

            self.val_y_seq = np.cumsum(self.val_y_seq, axis=0)
            self.val_y_hat_seq = np.cumsum(self.val_y_hat_seq, axis=0)

        self.r2.reset()
        self.r2(self.val_y_hat, self.val_y)
        r2_score_batched = self.r2.compute()

        self.r2.reset()
        self.r2(self.val_y_hat_seq, self.val_y_seq)
        r2_score_seq = self.r2.compute()

        self.log("val_r2_seq", r2_score_seq, batch_size=self.val_y.shape[0], on_epoch=True)
        self.log("val_r2", r2_score_batched, batch_size=self.val_y.shape[0], on_epoch=True)

        #num_steps_to_eval = 1024
        #num_figs = 1

        #fig_save_dir = os.path.join(self.logger.log_dir, "figures")
        #os.makedirs(fig_save_dir, exist_ok=True)
        #fig_val_save_dir = os.path.join(fig_save_dir, "val")
        #os.makedirs(fig_val_save_dir, exist_ok=True)

        #for i in range(num_figs):
        #    plot_output(
        #        self.val_y[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
        #        self.val_y_hat[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
        #        f"val_e_{self.current_epoch}_{i}"
        #    )

        self.val_y_hat = np.array([])
        self.val_y = np.array([])
        self.val_pr_idxs = np.array([])

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.net.reset_mem()

        x, y, pr_idxs = batch

        if self.sequential_testing:
            seq_len, batch_size, _ = x.shape
            
            y_hat = []

            for i in tqdm(range(batch_size)):
                y_hat_tmp = self.net(x[:,i].reshape(seq_len,1,-1))
                y_hat.append(y_hat_tmp)
                
            y_hat = torch.cat(y_hat, dim=1)
        else:
            y_hat = self.net(x)

        mse_loss = self.weighted_mse(y_hat, y)
        loss = mse_loss
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.detach().cpu().numpy()), axis=1) if self.test_y_hat.size else y_hat.detach().cpu().numpy()
        self.test_y = np.concatenate((self.test_y, y.cpu().numpy()), axis=1) if self.test_y.size else y.cpu().numpy()
        self.test_pr_idxs = np.concatenate((self.test_pr_idxs, pr_idxs.cpu().numpy()), axis=1) if self.test_pr_idxs.size else pr_idxs.cpu().numpy()

        self.log("test_loss", loss, batch_size=y.shape[0])

        return loss

    def on_test_epoch_end(self):
        self.r2.reset()
        self.test_y_hat = self.inverse_scale(self.test_y_hat)
        self.test_y = self.inverse_scale(self.test_y)

        self.test_y = self.test_y.transpose(1,0,2).reshape(-1, 1, self.test_y.shape[-1])
        self.test_y_hat = self.test_y_hat.transpose(1,0,2).reshape(-1, 1, self.test_y_hat.shape[-1])

        if self.output_type == "displacement":
            self.test_y_hat = np.cumsum(self.test_y_hat, axis=0)
            self.test_y = np.cumsum(self.test_y, axis=0)

        self.r2(self.test_y_hat, self.test_y)
        r2_score = self.r2.compute()

        self.log("test_" + self.tested_model + "_r2_" + str(self.sequential_testing), r2_score, batch_size=self.test_y.shape[0], on_epoch=True)     

        fig_save_dir = os.path.join(self.logger.log_dir, "figures")
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_val_save_dir = os.path.join(fig_save_dir, "test")
        os.makedirs(fig_val_save_dir, exist_ok=True)
        fig_val_save_dir = os.path.join(fig_val_save_dir, self.tested_model + "_seq_testing=" + str(self.sequential_testing))
        os.makedirs(fig_val_save_dir, exist_ok=True)

        num_steps_to_eval = 1024
        num_figs = 10
        for i in range(num_figs):
            plot_output(
                self.test_y[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
                self.test_y_hat[num_steps_to_eval*i:num_steps_to_eval*(i+1), 0, :],
                fig_val_save_dir, 
                f"test_" + self.tested_model + str(self.sequential_testing) + f"_{i}"
            )

        self.test_y_hat = np.array([])
        self.test_y = np.array([])
        self.test_pr_idxs = np.array([])  
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_regu
        )

    def reset(self):
        self.r2.reset()
