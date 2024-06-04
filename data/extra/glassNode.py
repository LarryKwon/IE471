# %%
import pandas as pd
import os


DATA_PATH = '../raw_v1.0'
OUTPUT_PATH =  "./"
os.makedirs(OUTPUT_PATH, exist_ok=True)

raw_data_list = ['min_1k_count.csv', 'realized_volatility_all.csv','active_count.csv']

for file in os.listdir(DATA_PATH):
    if file.endswith(".csv") and file in raw_data_list:
        file_path = os.path.join(DATA_PATH, file)
        raw_data = pd.read_csv(file_path).dropna()
        print(raw_data.columns)
        raw_data['Date'] = pd.to_datetime(raw_data['timestamp'], unit='s')
        raw_data.drop(columns=['timestamp'], inplace=True)
        if file == 'realized_volatility_all.csv':
            columns = raw_data.columns.tolist()
            columns.remove('Date')
            new_columns = {column: column + '_rv' for column in columns}
            raw_data.rename(columns=new_columns, inplace=True)
        else:
            raw_data.rename(columns={'value': file.split('.')[0]}, inplace=True)
        raw_data.to_csv(os.path.join(OUTPUT_PATH, file), index=False)