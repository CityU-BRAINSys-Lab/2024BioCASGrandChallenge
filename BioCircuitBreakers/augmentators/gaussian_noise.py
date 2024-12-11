import torch


class GaussianNoise(object):

    def __init__(self, proj, p=0.5, std=0.1):
        # data jittering with probability p
        self.probability = p
        self.std = std * torch.load(f"saves/{proj.all_files[0]}_{int(proj.dataset_bin_width / 4e-3)}_std_spikes.pt")

    def __call__(self, sample):
        with torch.no_grad():
            data, label = sample
            prob = torch.rand(data.shape)
            mask = prob < self.probability  # mask of elements that will be jittered
            noise = torch.normal(mean=0, std=self.std.expand(data.shape))
            data[mask] = (data + noise)[mask]
        return (data, label)