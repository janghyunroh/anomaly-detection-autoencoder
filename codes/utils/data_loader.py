import pandas as pd
import os
from utils.config import CONFIG

# data를 df 리스트 형태로 반환
def load_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            data.append(df)
    return data