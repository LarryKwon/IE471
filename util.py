
from easydict import EasyDict as edict
import yaml
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


# 모델에 weight load 
def load_models(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path}")
    return model, epoch, loss

# 모델에 weight save
def save_models(model, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved at {path}")

#   test 결과를 plot 합니다. 

from easydict import EasyDict as edict
import yaml
import torch
import matplotlib.pyplot as plt
import os

def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


# 모델에 weight load 
def load_models(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path}")
    return model, epoch, loss

# 모델에 weight save
def save_models(model, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved at {path}")

#   test 결과를 plot 합니다. 
def plot_results(df, forecasts, name, target_column):   
    plt.figure(figsize=(20,15))
    plotting_df = df.loc['2019-01-01':, [target_column]]
    plotting_df.plot(color="black")
    plt.ylim(0, 1.5 * plotting_df.max().values[0])

    for forecast in forecasts:
        # print(len(forecast.mean))
        # print(forecast.start_date)
        # forecast_index = pd.date_range(start=forecast.start_date.to_timestamp(), periods=len(forecast.mean), freq='D')
        # print(forecast.mean)
        # plt.plot(forecast_index, forecast.mean, label=f'Forecast {name}', alpha=0.7, color='blue')
        forecast.plot(color='blue')

    plt.legend(["True values"] + [f"Forecast {i}" for i in range(len(forecasts))], loc="upper left", fontsize="small")
    plt.title(f'Forecast vs Actual for {name}')
    plt.savefig(f"{name}.png")
    plt.show()


def load_models(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ckpt"):
            yield os.path.join(directory, filename)

    
    


def load_models(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ckpt"):
            yield os.path.join(directory, filename)

    