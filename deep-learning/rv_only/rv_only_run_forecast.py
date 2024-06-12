from ast import mod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('../../')
# from util import *
import os 
import pickle

# 필요한 모델 로드 
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator, DLinearEstimator, TemporalFusionTransformerEstimator, PatchTSTEstimator,SimpleFeedForwardEstimator
# from gluonts.mx import TransformerEstimator



#   해당 경로에 있는 데이터셋을 받아 train dataset 과 test Dataset 으로 나눕니다.
def split_dataset(dataset_path, target_column, feature_columns=[]):
    df = pd.read_csv(
        dataset_path,
        index_col=0,
        parse_dates=True,
    )
   
    duplicate_rows = df[df.duplicated(keep=False)]
    df = df.drop_duplicates()

    # 모든 날짜가 균일하게 분포되도록 interpolate 진행 
    all_times = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_times)
    df = df.interpolate(method='linear')
    df = df.astype({col: 'float32' for col in df.select_dtypes(include='float64').columns})
    feature_list = [target_column]+feature_columns
    df = df[feature_list]
    print(df.head())
    dataset = PandasDataset(df, freq='D',target=target_column)

    # 22 년도 기준으로 데이터 분할, training dataset
    training_data, test_gen = split(dataset,date=pd.Period('2020-12-31'))
    
    return training_data, test_gen, df

# 모델을 학습합니다. 
def train_models(sector_name, training_data, prediction_length=20, train=True):
    if train:
        directory_path = f'{sector_name}/model/'
        try:
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' is created or already exists.")
        except Exception as e:
            print(f"An error occurred while creating directory '{directory_path}': {e}")

    # ## Estimator 는 Predic
        arModel = DeepAREstimator(
            prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 30}, context_length=30
        ).train(training_data)
        with open(directory_path+"arModel.pkl", "wb") as f:
            pickle.dump(arModel, f)

        tftModel = TemporalFusionTransformerEstimator(
            prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 30}, context_length=30
        ).train(training_data)
        
        with open(directory_path+"tftModel.pkl", "wb") as f:
            pickle.dump(tftModel, f)
            
        ffModel = SimpleFeedForwardEstimator(
            prediction_length=prediction_length, trainer_kwargs={"max_epochs": 30}, context_length=30
        ).train(training_data)
        
        with open(directory_path+"simpleFFModel.pkl", "wb") as f:
            pickle.dump(ffModel, f)
            
        # tfModel = TransformerEstimator(
        #     prediction_length=prediction_length, freq="D", context_length=30
        # ).train(training_data)
        
        # with open(directory_path+"tfModel.pkl", "wb") as f:
        #     pickle.dump(tfModel, f)

        
    else:
        directory_path = f'{sector_name}/model/'
        with open(directory_path+"arModel.pkl", "rb") as f:
            arModel = pickle.load(f)
        with open(directory_path+"tftModel.pkl", "rb") as f:
            tftModel = pickle.load(f)
        with open(directory_path+"simpleFFModel.pkl", "rb") as f:
            ffModel = pickle.load(f)
        # with open(directory_path+"tfModel.pkl", "rb") as f:
        #     tfModel = pickle.load(f)
        
    return [arModel, tftModel, ffModel]#, tfModel] 
#  trained 된 모델들을 테스트 합니다. 
def test_and_plot(df, models, test_gen, target_column, prediction_length=20, windows=1, ):
    # Generate test instances
    windows = (len(df.loc['2021-01-01':, [target_column]])-1)//prediction_length 
    print(prediction_length)
    print(windows)
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)
    print(test_data)
    for idx, model in enumerate(models):
        forecasts = list(model.predict(test_data.input))
        print(forecasts)
        plot_results(df, forecasts, idx, target_column)    
        
def test_and_combine_forecasts(df, models, test_gen, target_column, prediction_length=20, windows=1):
    windows = (len(df.loc['2021-01-01':, [target_column]])-1)//prediction_length 
    print(prediction_length)
    print(windows)
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)
    
    model_predicitons = []
    for idx, model in enumerate(models):
        combined_forecasts = pd.Series(dtype='float64')
        forecasts = list(model.predict(test_data.input))
        for forecast in forecasts:
            forecast_index = pd.date_range(start=forecast.start_date.to_timestamp(), periods=len(forecast.mean), freq='D')
            forecast_series = pd.Series(forecast.mean, index=forecast_index)
            combined_forecasts = pd.concat([combined_forecasts, forecast_series])
        model_predicitons.append((combined_forecasts, model))
    
    return model_predicitons

def plot_results(df, forecasts, model, target_column, figure_path):
        plt.figure(figsize=(20, 15))
        # print(combined_forecasts)
        plotting_df = df.loc['2021-01-01':, [target_column]]
        plotting_df.plot(color="black")
        plt.plot(forecasts.index, forecasts, color="blue", label="Combined Forecast")
        plt.legend(["True values", "Combined Forecast"], loc="upper left", fontsize="small")
        plt.title(f'Combined Forecast vs Actual')
        plt.savefig(os.path.join(figure_path,f'{model.prediction_net.__class__.__name__}.png'))
        # plt.show()


def evaluate(df, target_column, model_predictions):
    test_data = df.loc['2021-01-01':, [target_column]][target_column]
    result = []
    for prediction in model_predictions:
        model = prediction[1]
        forecast = prediction[0]
        # rrmse = np.sqrt(((forecast - test_data) ** 2).mean() / (test_data ** 2).sum())
        sse = ((forecast - test_data) ** 2).sum()
        mse = ((forecast - test_data) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = (forecast - test_data).abs().mean()
        mape = ((forecast - test_data).abs() / test_data * 100).mean()
        result.append({
            'model': model.prediction_net.__class__.__name__,
            'sse': sse,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })
    return result
            
def save_evaluate(results, output_path, output_name='evaluation_results.txt', ):
    with open(os.path.join(output_path,output_name), 'w') as f:
        for metrics in results:
            f.write(f"Model: {metrics['model']}\n")
            f.write(f"SSE: {metrics['sse']}\n")
            f.write(f"MSE: {metrics['mse']}\n")
            f.write(f"RMSE: {metrics['rmse']}\n")
            f.write(f"MAE: {metrics['mae']}\n")
            f.write(f"MAPE: {metrics['mape']}\n")
            f.write("\n")

if __name__ == "__main__":

    # args = argparse.ArgumentParser()
    # # ## 데이터셋 경로 입력
    # args.add_argument("--dataset_path", type=str, default='data/processed/merged_S&P 500 Financials.csv')
    
    # # ## 모델 선택
    # args.add_argument("--models", type=str, nargs='+', help="List of model YAML files")
    
    # ## 이미 학습된 모델이 있으면 불러옴
    # args.add_argument("--model_path", type=str, default=None) 
    # args.add_argument("--model_path", type=str, nargs='+', help="List of model checkpoint files")

    # # ## train , test df 가져오기
    # args.add_argument("--train", type=str, default='data')

    TARGET_COLUMN = 'composite_realized_volatility'
    DATASET_DIR_PATH = 'data/processed_v3.0/'
    DATASET_LIST = []
    for file in os.listdir(path=DATASET_DIR_PATH):
        if file.endswith(".csv"):
            file_path = os.path.join(DATASET_DIR_PATH, file)
            DATASET_LIST.append(file_path)
    
    CKPT_SAVE_PATH = 'model_checkpoints/'
    FIG_SAVE_PATH = 'fig/'
    METRIC_SAVE_PATH = 'metric/'
    MODEL_SAVE_PATH = 'model/'
    
    for DATASET_PATH in DATASET_LIST:
        
        sector_name = DATASET_PATH.split(DATASET_DIR_PATH)[1].split('.csv')[0]
        if sector_name != 'merged_composite_data':
            continue
        print(sector_name)
        try:
            os.makedirs(sector_name, exist_ok=True)
            os.makedirs(os.path.join(sector_name, 'predict_30', MODEL_SAVE_PATH), exist_ok=True)
            os.makedirs(os.path.join(sector_name,'predict_30', FIG_SAVE_PATH),exist_ok=True)
            os.makedirs(os.path.join(sector_name,'predict_30', METRIC_SAVE_PATH),exist_ok=True)


            print(f"Directory '{sector_name}' is created or already exists.")
        except Exception as e:
            print(f"An error occurred while creating directory '{sector_name}': {e}")
        
        df = pd.read_csv(DATASET_PATH)
        TARGET_COLUMN = [column for column in df.columns if 'realized_volatility' in column][0]
        print(TARGET_COLUMN)
        training_data, test_generator, df =  split_dataset(dataset_path=DATASET_PATH, target_column=TARGET_COLUMN)
        # print(len(training_data))

        trained_models = train_models(sector_name, training_data=training_data, prediction_length=10, train=True)
        model_predictions = test_and_combine_forecasts(df, trained_models, test_generator, TARGET_COLUMN, prediction_length=10)
        for prediction in model_predictions:
            model = prediction[1]
            model_name = model.prediction_net.__class__.__name__
            forecast = prediction[0]
            forecast.to_csv(os.path.join(sector_name,f'{sector_name}_{model_name}_forecast.csv'))
            plot_results(df,forecast, prediction[1], TARGET_COLUMN, os.path.join(sector_name,FIG_SAVE_PATH))
            metrics = evaluate(df,TARGET_COLUMN, model_predictions)
            save_evaluate(metrics,os.path.join(sector_name,'metric'), f'{model_name}_evaluation_results.txt',)
    # ## 선택한 모델들에 대해서 돌리기 
    # if args.train:
    #     trained_models = train_models(training_data)
