from torch.utils.data import Dataset

class GrenadeDataset(Dataset):

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features.iloc[idx], self.targets.iloc[idx]




