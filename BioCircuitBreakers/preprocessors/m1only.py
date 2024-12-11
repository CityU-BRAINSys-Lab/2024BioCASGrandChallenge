import torch

class M1Only(object):

    def __init__(self, proj):
        super(M1Only, self).__init__()

    def __call__(self, sample):
        with torch.no_grad():
            data, label = sample
            data = data[..., :96]

        return (data, label)