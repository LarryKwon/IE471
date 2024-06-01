import pandas as pd
import os

data_dir = "./data"
output_dir = os.path.join(data_dir, "merged")
os.makedirs(output_dir, exist_ok=True)

data_frames = {}

# Read and process CSV files
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df = df.dropna()


        if "bitcoin" in file.lower():  # Assuming the 'bitcoin' keyword might be case insensitive
            df['Date'] = df['time']
            df.set_index('Date', inplace=True)
            data_frames["bitcoin"] = df
        else:
            df['Date'] = df['Date'].apply(lambda x: x.split(" ")[0].strip( ).replace("-", "."))
            df.set_index('Date', inplace=True)
            data_frames[os.path.basename(file)] = df
        
# Perform left join using the 'bitcoin' DataFrame
bitcoin_df = data_frames.get("bitcoin")

if bitcoin_df is not None:
    # Rename Bitcoin DataFrame columns to include 'bitcoin' prefix
    bitcoin_df = bitcoin_df.rename(columns=lambda x: f'bitcoin_{x}')

    for key, df in data_frames.items():
        if key == "bitcoin":
            continue
        
        # Rename other DataFrame columns to include the key prefix
        df = df.rename(columns=lambda x: f'{key}_{x}')
        
        # Perform left join
        merged_df = bitcoin_df.join(df, how='inner')
        
        # Save the merged DataFrame
        output_file_path = os.path.join(output_dir, f"merged_{key}")
        merged_df.to_csv(output_file_path, index=True)
        print(f"Saved merged DataFrame to {output_file_path}")
else:
    print("No 'bitcoin' DataFrame found. Ensure there is a CSV file containing 'bitcoin' in its filename.")
