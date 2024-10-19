import pandas as pd
import os

def load_data(data_dir, rpm):
    data = []
    for file in os.listdir(os.path.join(data_dir, f"rpm_{rpm}")):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, f"rpm_{rpm}", file))
            data.append(df)
    return pd.concat(data, ignore_index=True)