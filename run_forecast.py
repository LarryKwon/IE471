import pandas as pd
import matplotlib.pyplot as plt
import argparse
from util import *
import os 
import pickle

# 필요한 모델 로드 
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator, DLinearEstimator, TemporalFusionTransformerEstimator, PatchTSTEstimator
# from gluonts.mx import TransformerEstimator



#   해당 경로에 있는 데이터셋을 받아 train dataset 과 test Dataset 으로 나눕니다.
def split_dataset(dataset_path, target_column):
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
    dataset = PandasDataset(df, freq='D',target=target_column)

    # 22 년도 기준으로 데이터 분할, training dataset
    training_data, test_gen = split(dataset,date=pd.Period('2019-01-01'))
    
    return training_data, test_gen, df

# 모델을 학습합니다. 
def train_models(training_data, prediction_length=20, train=True):
    if train:
    # ## Estimator 는 Predic
        arModel = DeepAREstimator(
            prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 5}, context_length=5*prediction_length
        ).train(training_data)
        with open("model/arModel.pkl", "wb") as f:
            pickle.dump(arModel, f)

        tftModel = TemporalFusionTransformerEstimator(
            prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 2}, context_length=5*prediction_length
        ).train(training_data)
        with open("model/tftModel.pkl", "wb") as f:
            pickle.dump(tftModel, f)
        # transformerModel = TransformerEstimator(
        #     prediction_length=prediction_length, freq="D"
        # ).train(training_data)
        
    else:
        with open("model/arModel.pkl", "rb") as f:
            arModel = pickle.load(f)
        with open("model/tftModel.pkl", "rb") as f:
            tftModel = pickle.load(f)
            
        
    return [arModel, tftModel] 
#  trained 된 모델들을 테스트 합니다. 
def test_and_plot(df, models, test_gen, target_column, prediction_length=20, windows=1, ):
    # Generate test instances
    windows = (len(df.loc['2019-01-01':, [target_column]])-1)//prediction_length 
    print(prediction_length)
    print(windows)
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)
    print(test_data)
    for idx, model in enumerate(models):
        forecasts = list(model.predict(test_data.input))
        print(forecasts)
        plot_results(df, forecasts, idx, target_column)    
        
def test_and_combine_forecasts(df, models, test_gen, target_column, prediction_length=20, windows=1):
    windows = (len(df.loc['2019-01-01':, [target_column]])-1)//prediction_length 
    print(prediction_length)
    print(windows)
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)

    for idx, model in enumerate(models):
        combined_forecasts = pd.Series(dtype='float64')
        forecasts = list(model.predict(test_data.input))
        for forecast in forecasts:
            forecast_index = pd.date_range(start=forecast.start_date.to_timestamp(), periods=len(forecast.mean), freq='D')
            forecast_series = pd.Series(forecast.mean, index=forecast_index)
            combined_forecasts = pd.concat([combined_forecasts, forecast_series])

        plt.figure(figsize=(20, 15))
        print(combined_forecasts)
        plotting_df = df.loc['2019-01-01':, [target_column]]
        plotting_df.plot(color="black")
        plt.plot(combined_forecasts.index, combined_forecasts, color="red", label="Combined Forecast")
        plt.legend(["True values", "Combined Forecast"], loc="upper left", fontsize="small")
        plt.title(f'Combined Forecast vs Actual')
        plt.savefig(f"{idx}.png")    
        plt.show()
    return combined_forecasts




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
    DATASET_PATH = 'data/processed_v2.0/merged_composite_data.csv'
    CKPT_SAVE_PATH = 'model_checkpoints/'
    FIG_SAVE_PATH = 'fig/'
    training_data, test_generator, df =  split_dataset(dataset_path=DATASET_PATH, target_column=TARGET_COLUMN)
    print(len(training_data))

    train_models = train_models(training_data,prediction_length=1, train=True)
    test_and_combine_forecasts(df, train_models, test_generator, TARGET_COLUMN, prediction_length=1)
    # ## 선택한 모델들에 대해서 돌리기 
    # if args.train:
    #     trained_models = train_models(training_data)
