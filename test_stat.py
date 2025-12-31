import pandas as pd
df = pd.read_csv("data/alcohol_dataset_preprocessed.csv")
print(df["abodalc"].describe())