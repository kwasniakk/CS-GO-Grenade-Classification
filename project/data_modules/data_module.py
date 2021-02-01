import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import sys
from torch.utils.data import DataLoader
from .dataset import GrenadeDataset
sys.path.insert(1, "C:/CS-GO-Grenade-Classification/project/utils")
from data_utils import preprocess


COLUMNS_TO_DROP = ["demo_id", "demo_round_id", "weapon_fire_id", "round_start_tick"]
DUMMY_COLS = ["LABEL", "team", "TYPE", "map_name"]
pl.seed_everything(42)

class GrenadeDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.mirage_csv_file = "C:/CS-GO-Grenade-Classification/project/data/train-grenades-de_mirage.csv"
        self.inferno_csv_file = "C:/CS-GO-Grenade-Classification/project/data/train-grenades-de_inferno.csv"
        self.batch_size = 16
        self.test_size = 0.25

    def setup(self, stage = None):
        inferno = pd.read_csv(self.mirage_csv_file, index_col = 0)
        mirage = pd.read_csv(self.inferno_csv_file, index_col = 0)
        raw_data = pd.concat([inferno, mirage])
        X, y = preprocess(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = self.test_size, random_state = 42)

        self.train_dataset = GrenadeDataset(X_train, y_train)
        self.val_dataset = GrenadeDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)




