import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import GrenadeDataset

pl.seed_everything(42)

class GrenadeDataModule(pl.LightningDataModule):

    def __init__(self, hparams):

        super().__init__()
        self.csv_file = hparams["csv_file"]
        self.batch_size = hparams["batch_size"]
        self.test_size = hparams["test_size"]

    def setup(self, stage = None):
        raw_data = pd.read_csv(self.csv_file, index_col = 0)
        data_dropped =  raw_data.drop(columns = ["demo_id", "demo_round_id", "weapon_fire_id", "map_name", "round_start_tick"])
        data_dummies = pd.get_dummies(data_dropped, columns = ["team", "LABEL", "TYPE"], drop_first = True)
        X = data_dummies.drop(columns = ["LABEL_True"])
        y = data_dummies["LABEL_True"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)

        self.train_dataset = GrenadeDataset(X_train, y_train)
        self.val_dataset = GrenadeDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)

