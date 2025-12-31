import pandas as pd
import numpy as np
from mindspore.dataset import GeneratorDataset

# -----------------------------------
# Dataset MindSpore
# -----------------------------------
class AlcoholDataset:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.X = df.drop(columns=["abodalc"]).values.astype(np.float32)
        self.y = df["abodalc"].values.astype(np.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# -----------------------------------
# Création dataset MindSpore
# -----------------------------------
def create_dataset(csv_path, batch_size=256, shuffle=False):
    return GeneratorDataset(
        AlcoholDataset(csv_path),
        ["features", "label"],
        shuffle=shuffle
    ).batch(batch_size)


# -----------------------------------
# Déséquilibre → pos_weight
# -----------------------------------
def compute_pos_weight(train_csv):
    df = pd.read_csv(train_csv)
    nb_pos = (df["abodalc"] == 1).sum()
    nb_neg = (df["abodalc"] == 0).sum()

    return nb_neg / nb_pos
