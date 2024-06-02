
from easydict import EasyDict as edict
import yaml
import torch
import matplotlib.pyplot as plt

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

    # Plot predictions
    plt.figure(figsize=(20,15))
    plotting_df = df.loc['2021-01-01':, [target_column]]
    plotting_df.plot(color="black")
    # plt.plot(df, color="black")
    for forecast in forecasts:
        forecast.plot()

    plt.legend(["True values"], loc="upper left", fontsize="small")
    plt.show()
    plt.savefig(f"{name}")
    