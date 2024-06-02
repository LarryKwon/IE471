import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator

# Load data from a CSV file into a PandasDataset
df = pd.read_csv(
    'data/merged/merged_S&P 500 Financials.csv',
    index_col=0,
    parse_dates=True,
)

duplicate_rows = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
# Ensure uniformly spaced index by reindexing
all_times = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(all_times)
df.interpolate(method='linear')

dataset = PandasDataset(df, freq='D',target="S&P 500 Financials.csv_Open")

# Split the data for training and testing
training_data, test_gen = split(dataset, date=pd.Period('2023-01-01'))
test_data = test_gen.generate_instances(prediction_length=12, windows=3)

# Train the model and make predictions
model = DeepAREstimator(
    prediction_length=12, freq="D", trainer_kwargs={"max_epochs": 20}
).train(training_data)

forecasts = list(model.predict(test_data.input))
