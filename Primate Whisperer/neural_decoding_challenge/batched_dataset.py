from torch.utils.data import Dataset
from torch import tensor
class BatchedPrimateReaching(Dataset):
    def __init__(self, primate_reaching, seq_len, overlap=0):
        """
        Args:
            primate_reaching (PrimateReaching): subset of PrimateReaching object
            seq_len (int): sequence length
            overlap (int): overlap between sequences (between 0 and seq_len)
        """
        self.primate_reaching = primate_reaching
        self.input_feature_size = primate_reaching.dataset.input_feature_size
        self.seq_len = seq_len
        self.overlap = overlap

        self.primate_reaching_indexes = list(range(len(primate_reaching)))
        if seq_len == len(primate_reaching) or seq_len == 0 or seq_len == -1:
            self.seq_len == len(primate_reaching)
            self.valid_indexes = [0]
        else:
            self.valid_indexes = self.primate_reaching_indexes[:-seq_len][::seq_len - overlap]

    def __len__(self):
        return len(self.valid_indexes)

    def __getitem__(self, idx):
        sample = self.primate_reaching[self.valid_indexes[idx]:self.valid_indexes[idx] + self.seq_len]
        # Preprocess the sample if necessary
        return (sample[0], sample[1], tensor(range(self.valid_indexes[idx],self.valid_indexes[idx] + self.seq_len)))

