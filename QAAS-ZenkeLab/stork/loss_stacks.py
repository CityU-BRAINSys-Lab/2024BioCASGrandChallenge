import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score


class LossStack:
    def __init__(self):
        pass

    def log_py_given_x(self, output):
        raise NotImplemented()

    def get_metric_names(self):
        raise NotImplemented()

    def compute_loss(self, output, targets):
        raise NotImplemented()

    def predict(self, output):
        raise NotImplemented()


class MaxOverTimeCrossEntropy(LossStack):
    """Readout stack that employs the max-over-time reduction strategy paired with categorical cross entropy."""

    def __init__(self, time_dimension=1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()
        self.time_dim = time_dimension

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MaxOverTimeFocalLoss(LossStack):
    """Readout stack that employs the max-over-time reduction strategy paired with focal loss."""

    def __init__(
        self, gamma=0.0, eps=1e-7, samples_per_class=None, beta=0.99, time_dimension=1
    ):
        super().__init__()
        self.time_dim = time_dimension
        self.eps = eps
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.weights = None
        if samples_per_class is not None:
            eff_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / np.array(eff_num)
            weights = weights / np.sum(weights) * len(samples_per_class)
            print("weights: ", weights)
            self.weights = torch.tensor(weights).float()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        # reduce along time with max
        ma, _ = torch.max(output, dim=self.time_dim)
        y = F.one_hot(targets, ma.size(-1))

        logit = F.softmax(ma, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)

        loss = -1.0 * y * torch.log(logit)  # cross entropy loss
        loss = loss * (1.0 - logit) ** self.gamma  # change to focal loss

        if self.weights is not None:
            w = self.weights.to(loss.device)
            w = w.unsqueeze(0)
            w = w.repeat(y.shape[0], 1) * y
            w = w.sum(1)
            w = w.unsqueeze(1)
            w = w.repeat(1, ma.size(-1))
            loss = w * loss

        acc_val = self.acc_fn(logit, targets)
        self.metrics = [acc_val.item()]
        return torch.mean(loss)

    def log_py_given_x(self, output):
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class SumOverTimeCrossEntropy(LossStack):
    """Loss stack that employs the sum-over-time reduction strategy paired with categorical cross entropy."""

    def __init__(self, time_dimension=1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()
        self.time_dim = time_dimension

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over sum over time"""
        su = torch.sum(output, self.time_dim)  # reduce along time with sum
        log_p_y = self.log_softmax(su)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        su = torch.sum(output, self.time_dim)  # reduce along time with sum
        log_p_y = self.log_softmax(su)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class LastStepCrossEntropy(LossStack):
    """Computes crossentropy loss on last time frame of the network"""

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=1)
        a = pred_labels == target_labels
        return a.cpu().numpy().mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        log_p_y = self.log_softmax(output[:, -1])
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        log_p_y = self.log_softmax(output[:, -1])
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class EveryStepCrossEntropy(LossStack):
    """Computes crossentropy loss on every time frame of the network"""

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=1)
        a = pred_labels == target_labels
        return a.cpu().numpy().mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        log_p_y = self.log_softmax(output)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        log_p_y = self.log_softmax(output)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class CSTLossStack(LossStack):
    def __init__(self, 
                 mask=None, 
                 density_weighting_func=False):
        super().__init__()
        self.mask = mask
        self.loss_fn = None  # to be defined in the child class
        self.density_weighting_func = density_weighting_func

    def get_R2(self, pred, target):
        # Julian Rossbroich
        # modified july 2024
        """
        Args:
            pred: Predicted series of the model (batch_size * timestep * nb_outputs),
            target: Ground truth series (batch_size * timestep * nb_outputs).

        Return:
            r2: R-squared between the inputs along consecutive axis, over a batch.
        """

        # For each feature, calculate R2
        # We use the mean across all samples to calculate sst
        ssr = torch.sum((target - pred) ** 2, dim=(0, 1))
        sst = torch.sum((target - torch.mean(target, dim=(0, 1))) ** 2, dim=(0, 1))
        r2 = (1 - ssr / sst).detach().cpu().numpy()

        return [float(r2[0].round(3)), float(r2[1].round(3)), float(r2.mean().round(3))]

    def get_metric_names(self):
        # Julian Rossbroich
        # modified july 2024
        return ["r2x", "r2y", "r2"]

    def compute_loss(self, output, target):
        """Computes MSQE loss between output and target."""

        if self.mask is not None:
            output = output * self.mask.expand_as(output)
            target = target * self.mask.expand_as(output)
            
        if self.density_weighting_func:
            weight = self.density_weighting_func(target)
        else:
            weight = None

        self.metrics = self.get_R2(output, target)
        return self.loss_fn(output, target, weight=weight)

    def predict(self, output):
        return output

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MeanSquareError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MSEloss
        
    def _weighted_MSEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * (output - target) ** 2)
        else:
            return torch.mean((output - target) ** 2)


class RootMeanSquareError(CSTLossStack):

    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_RMSEloss

    def _weighted_RMSEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.sqrt(torch.mean(weight * (output - target) ** 2))
        else:
            return torch.sqrt(torch.mean((output - target) ** 2))


class MeanAbsoluteError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MAEloss

    def _weighted_MAEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * torch.abs(output - target))
        else:
            return torch.mean(torch.abs(output - target))


class HuberLoss(CSTLossStack):
    def __init__(self, delta=1.0, mask=None, density_weighting_func=False):
        
        if density_weighting_func:
            raise ValueError("Density weighting not supported for Huber loss.")
        
        super().__init__(mask=mask)
        self.loss_fn = nn.SmoothL1Loss(beta=delta)
        self.delta = delta


class DictMeanSquareError(MeanSquareError):
    # Julian Rossbroich refactor:
    # DEPRECATED!

    """Like MeanSquareError, but uses a dictionary of possible output
    patterns which can be kept in the GPU memory."""

    def __init__(self, target_patterns, mask=None):
        super().__init__(mask)
        self.dict_ = target_patterns

    def compute_loss(self, output, targets):
        """Computes MSQE loss between output and target."""
        local_targets = [self.dict_[idx] for idx in targets]
        return super().compute_loss(output, torch.stack(local_targets, dim=0))

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)
