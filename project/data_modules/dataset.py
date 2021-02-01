import numpy as np
from torch.utils.data import Dataset

class GrenadeDataset(Dataset):

    def __init__(self, X, y):
        self.X = X.copy().values.astype(np.float32)
        self.y = y.copy().values.astype(np.int64)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




