import pandas as pd
import os

augmento_data = pd.read_csv('../raw_v2.0/augmento_summed_data.csv')
augmento_data = augmento_data.set_index(pd.DatetimeIndex(augmento_data['date']))
augmento_data = augmento_data.resample('D').sum()
augmento_data.drop(columns=['Unnamed: 0','date'], inplace=True)
augmento_data.reset_index(inplace=True)
augmento_data.rename(columns = {'date': 'Date'}, inplace=True)
augmento_data.to_csv('augmento_daily_summed_data.csv',index=False)

print(augmento_data.head())