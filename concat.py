import pandas as pd
import numpy as np
import os
DATA_PATH = 'data/raw_v2.0'
OUTPUT_PATH =  "data/processed_v2.0"
os.makedirs(OUTPUT_PATH, exist_ok=True)

bitcoin_data = pd.read_csv(os.path.join(DATA_PATH, 'bitcoin_bitstamp.csv'))
bitcoin_data['log_price'] = np.log(bitcoin_data['close'])

# Read and process CSV files
for file in os.listdir(DATA_PATH):
    if file.endswith(".csv") and 'bitcoin' not in file.lower():
        file_path = os.path.join(DATA_PATH, file)
        stock_data = pd.read_csv(file_path).dropna()
        stock_data['log_return'] = np.log(stock_data['Close']) - np.log(stock_data['Close'].shift(1))
        stock_data['realized_volatility'] = stock_data['log_return'].rolling(window=20).apply(lambda x: (252/20)* np.sqrt(np.sum(x**2)))

        merged_data = pd.merge(bitcoin_data[['Date', 'log_price']], stock_data[['Date', 'realized_volatility']], on='Date')
        merged_data = merged_data.dropna()
        merged_data.to_csv( os.path.join(OUTPUT_PATH, f'{os.path.basename(file_path)}'))

#         if "bitcoin" in file.lower():  # Assuming the 'bitcoin' keyword might be case insensitive
#             df['Date'] = pd.to_datetime(df['time'])
#             df.set_index('Date', inplace=True)
#             df = df[df.index >=  pd.Timestamp('2018-07-01')]
#             data_frames["bitcoin"] = df
#         else:
#             df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split(" ")[0].strip( )))
#             df.set_index('Date', inplace=True)
#             df = df[df.index >= pd.Timestamp('2018-07-01')]
#             data_frames[os.path.basename(file)] = df

# # Perform left join using the 'bitcoin' DataFrame
# bitcoin_df = data_frames.get("bitcoin")

# if bitcoin_df is not None:
#     # Rename Bitcoin DataFrame columns to include 'bitcoin' prefix
#     bitcoin_df = bitcoin_df.rename(columns=lambda x: f'bitcoin_{x}')

#     for key, df in data_frames.items():
#         if key == "bitcoin":
#             continue
        
#         # Rename other DataFrame columns to include the key prefix
#         df = df.rename(columns=lambda x: f'{key}_{x}')
        
#         # Perform left join
#         merged_df = bitcoin_df.join(df, how='inner')
        
#         # Save the merged DataFrame
#         output_file_path = os.path.join(output_dir, f"merged_{key}")
#         merged_df.to_csv(output_file_path, index=True)
#         print(f"Saved merged DataFrame to {output_file_path}")
# else:
#     print("No 'bitcoin' DataFrame found. Ensure there is a CSV file containing 'bitcoin' in its filename.")
