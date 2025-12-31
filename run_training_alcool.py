import pandas as pd
from sklearn.model_selection import train_test_split
from model.dataset import create_dataset, compute_pos_weight
from model.train import train_model

CSV_PATH = "data/alcohol_dataset_preprocessed.csv"

# -----------------------------------
# SPLIT UNIQUE → CSV
# -----------------------------------
df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["abodalc"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["abodalc"],
    random_state=42
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# -----------------------------------
# Dataset MindSpore
# -----------------------------------
train_ds = create_dataset("data/train.csv", shuffle=True)
val_ds = create_dataset("data/val.csv", shuffle=False)

input_dim = train_df.shape[1] - 1
pos_weight = compute_pos_weight("data/train.csv")

print("pos_weight =", pos_weight)

# -----------------------------------
# Entraînement
# -----------------------------------
model = train_model(
    train_ds,
    val_ds,
    input_dim=input_dim,
    pos_weight=pos_weight,
    epochs=10
)

print("🚀 Entraînement terminé")
