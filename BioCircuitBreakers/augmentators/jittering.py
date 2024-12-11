import torch


class Jittering(object):

    def __init__(self, proj, p=0.5):
        # data jittering with probability p
        self.probability = p

    def __call__(self, sample):
        with torch.no_grad():
            data, label = sample
            prob = torch.rand(data.shape)
            mask = prob > self.probability  # mask of elements that will keep the original value
            jittered = torch.poisson(data)
            jittered[mask] = data[mask]
        return (jittered, label)