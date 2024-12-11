import os
import csv

import torch
import numpy as np
import random



class CSVLogger(object):
    def __init__(self, keys, path, append=False):
        super(CSVLogger, self).__init__()
        self._keys = keys
        self._path = path
        if append is False or not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                w = csv.DictWriter(f, self._keys)
                w.writeheader()

    def write(self, logs):
        with open(self._path, 'a') as f:
            w = csv.DictWriter(f, self._keys)
            w.writerow(logs)

    def get_ckpt_dir(self):
        return os.path.dirname(self._path)
    
    def get_column(self, key):
        with open(self._path, 'r') as f:
            reader = csv.DictReader(f)
            return [row[key] for row in reader]

def FixSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)