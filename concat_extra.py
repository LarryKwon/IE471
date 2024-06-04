import pandas as pd
import numpy as np
import os
from functools import reduce

EXTRA_DATA_PATH = 'data/extra'
MERGED_DATA_PATH = 'data/processed_v2.0'
OUTPUT_PATH =  "data/processed_v3.0"
os.makedirs(OUTPUT_PATH, exist_ok=True)

extra_df_list = []
for file in os.listdir(EXTRA_DATA_PATH):
    if file.endswith(".csv"):
        extra_df_list.append(pd.read_csv(os.path.join(EXTRA_DATA_PATH, file)))
        
extra_merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date'), extra_df_list)

for file in os.listdir(MERGED_DATA_PATH):
    if file.startswith('merged_') and file.endswith(".csv"):
        file_path = os.path.join(MERGED_DATA_PATH, file)
        merged_data = pd.read_csv(file_path)
        merged_data = pd.merge(merged_data, extra_merged_df, on='Date')
        merged_data.to_csv(os.path.join(OUTPUT_PATH, f'extra_{file}'), index=False)