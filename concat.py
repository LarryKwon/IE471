import pandas as pd
import numpy as np
import os
DATA_PATH = 'data/raw_v2.0'
OUTPUT_PATH =  "data/processed_v2.0"
os.makedirs(OUTPUT_PATH, exist_ok=True)

bitcoin_data = pd.read_csv(os.path.join(DATA_PATH, 'bitcoin_bitstamp.csv'), index_col=0)
bitcoin_data['log_price'] = np.log(bitcoin_data['close'])
# Read and process CSV files
for file in os.listdir(DATA_PATH):
    if file.endswith(".csv") and 'bitcoin' not in file.lower():
        sector_name = file.split("_")[0]
        
        file_path = os.path.join(DATA_PATH, file)
        stock_data = pd.read_csv(file_path, index_col=0).dropna()
        stock_data['log_return'] = np.log(stock_data['Close']) - np.log(stock_data['Close'].shift(1))
        stock_data['realized_volatility'] = stock_data['log_return'].rolling(window=20).apply(lambda x: (252/20)* np.sqrt(np.sum(x**2)))
        merged_data = pd.merge(bitcoin_data, stock_data, on='Date')
        merged_data = merged_data.dropna()
        
        rename_dict = {}
        for col in merged_data.columns:
            if col == 'Date':
                continue
            if col in stock_data.columns:
                rename_dict[col] = f"{sector_name}_{col}"
            else:
                rename_dict[col] = f"bitcoin_{col}"
        merged_data = merged_data.rename(columns=rename_dict)
        # bitcoin_data.columns = bitcoin_data.columns + "_bitcoin"
        merged_data.to_csv(os.path.join(OUTPUT_PATH, f'merged_{file}'))