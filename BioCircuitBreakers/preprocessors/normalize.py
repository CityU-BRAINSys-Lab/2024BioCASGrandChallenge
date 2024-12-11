import torch

class Normalize(object):

    def __init__(self, proj):
        super(Normalize, self).__init__()
        if len(proj.all_files) == 1:
            self.mean = torch.load(f"saves/{proj.all_files[0]}_{int(proj.dataset_bin_width / 4e-3)}_mean_spikes.pt")
            self.std = torch.load(f"saves/{proj.all_files[0]}_{int(proj.dataset_bin_width / 4e-3)}_std_spikes.pt")
        else:
            self.mean = torch.load(f"saves/{proj.monkey}_{int(proj.dataset_bin_width / 4e-3)}_mean_spikes.pt")
            self.std = torch.load(f"saves/{proj.monkey}_{int(proj.dataset_bin_width / 4e-3)}_std_spikes.pt")

    def __call__(self, sample):
        with torch.no_grad():
            data, label = sample
            shape = data.shape
            assert shape[-1] == self.mean.shape[-1]
            assert shape[-1] == self.std.shape[-1]

            data = (data - self.mean.to(data.device)) / (self.std.to(data.device) + 1e-6)
        return (data, label)