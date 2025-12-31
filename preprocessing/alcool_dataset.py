import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Colonnes à utiliser
cols = [
    'alcdays', 'alcpdang', 'alcemopb', 'alcserpb', 'alclimit', 'alccutdn', 
    'aldaypyr', 'abodalc', 'alcphlpb', 'alcbng30d', 'sexage', 
    'dstnrv30', 'dsthop30','dstngd30','dsteff30', 'health'
]

csv_file = "../../ICTData/nsduh/NSDUH_2015-2019.csv"
output_csv = "../data/alcohol_dataset_preprocessed.csv"
chunksize = 100_000

# Codes spéciaux
drop_codes = [91, 991]
map_zero_codes = [83, 93]
nan_codes_alc = [80, 85, 94, 97, 99, 98, 985, 989, 993, 994, 997, 998, 999]
nan_codes_dst = [85, 94, 97, 98, 99]
nan_codes_health = [94, 97]

first_chunk = True

for chunk in pd.read_csv(csv_file, usecols=cols, chunksize=chunksize):
    # supprimer les lignes sans target
    chunk = chunk.dropna(subset=["abodalc"])
    
    # Supprimer "never used alcohol"
    chunk = chunk[~chunk['alcdays'].isin(drop_codes)]
    chunk = chunk[~chunk['aldaypyr'].isin(drop_codes)]
    
    # Remplacer 83/93 par 0 et les autres codes spéciaux par NaN
    alc_cols = ['alcdays','alcpdang','alcemopb','alcserpb','alclimit',
                'alccutdn','aldaypyr','alcphlpb','alcbng30d']
    for col in alc_cols:
        chunk[col] = chunk[col].replace(map_zero_codes, 0)
        chunk[col] = chunk[col].replace(nan_codes_alc, np.nan)
    
    binary_cols = ['alclimit', 'alccutdn', 'alccutdn', 'alcserpb',
               'alcpdang', 'alcemopb', 'alcphlpb']

    for col in binary_cols:
        # Remplacer codes “never used / <6 days” par 0
        chunk[col] = chunk[col].replace([83, 93, 91], 0)
        # Les autres codes spéciaux -> NaN
        chunk[col] = chunk[col].replace([85, 94, 97, 98, 99], np.nan)
        # Puis convertir Yes/No en 1/0
        chunk[col] = chunk[col].replace({1: 1, 2: 0})
        # Imputer NaN par 0 (ou selon stratégie)
        chunk[col] = chunk[col].fillna(0)

    # DST variables
    dst_cols = ['dstnrv30','dsthop30','dstngd30','dsteff30']
    for col in dst_cols:
        chunk[col] = chunk[col].replace(nan_codes_dst, np.nan)
    
    # Health
    chunk['health'] = chunk['health'].replace(nan_codes_health, np.nan)
    
    # Imputation
    # Colonnes numériques liées à l'alcool → 0 pour NaN
    for col in ['alcdays','aldaypyr','alcbng30d']:
        chunk[col] = chunk[col].fillna(0)
    
    # Colonnes binaires → 0 pour NaN
    bin_cols = ['alcpdang','alcemopb','alcserpb','alclimit','alccutdn','alcphlpb']
    for col in bin_cols:
        chunk[col] = chunk[col].fillna(0)
    
    # DST variables → median
    for col in dst_cols:
        chunk[col] = chunk[col].fillna(chunk[col].median())
    
    # Health → median
    chunk['health'] = chunk['health'].fillna(chunk['health'].median())
    
    # Scaling Min-Max
    chunk['alcdays'] = chunk['alcdays'] / 30.0
    chunk['alcbng30d'] = chunk['alcbng30d'] / 30.0
    chunk['aldaypyr'] = chunk['aldaypyr'] / 365.0
    
    # Standardisation DST et health
    scaler_dst = StandardScaler()
    chunk[dst_cols + ['health']] = scaler_dst.fit_transform(chunk[dst_cols + ['health']])
    
    # One-hot encoding sexage
    sexage_map = {
        1: "male_12_17",
        2: "female_12_17",
        3: "male_18_25",
        4: "female_18_25",
        5: "other"
    }
    sexage_dummies = pd.get_dummies(chunk['sexage'])
    sexage_dummies = sexage_dummies.rename(columns=lambda x: f"sexage_{sexage_map.get(x, x)}")
    sexage_dummies = sexage_dummies.astype(int)

    chunk = pd.concat([chunk.drop(columns=['sexage']), sexage_dummies], axis=1)
    
    # Sauvegarde
    chunk.to_csv(output_csv, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
    first_chunk = False

print(f"Dataset alcool preprocessé créé : {output_csv}")
