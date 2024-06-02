import pandas as pd
import matplotlib.pyplot as plt
import argparse
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator, DLinearEstimator, TemporalFusionTransformerEstimator, PatchTSTEstimator

TARGET_COLUMN = 'S&P 500 Financials.csv_Open'
DATASET_PATH = 'data/merged/merged_S&P 500 Financials.csv'

'''
  해당 경로에 있는 데이터셋을 받아 train dataset 과 test Dataset 으로 나눕니다.
'''
def split_dataset(dataset_path = DATASET_PATH, target_column = TARGET_COLUMN):
    # Load data from a CSV file into a PandasDataset
    df = pd.read_csv(
        dataset_path,
        index_col=0,
        parse_dates=True,
    )

    # 중복 데이터 제거
    duplicate_rows = df[df.duplicated(keep=False)]
    df = df.drop_duplicates().drop(columns=['bitcoin_time'])

    # 모든 날짜가 균일하게 분포되도록 interpolate 진행 
    all_times = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_times)
    df = df.interpolate(method='linear')
    dataset = PandasDataset(df, freq='D',target=target_column)

    # 22 년도 기준으로 데이터 분할, training dataset
    training_data, test_gen = split(dataset,date=pd.Period('2022-01-01'))
    
    return training_data, test_gen


'''
    trained 된 모델들을 테스트 합니다. 
'''
def test_model(model, test_gen, prediction_length=14, windows=10):
    # Generate test instances

    test_gen = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)
    forecasts = list(model.predict(test_gen.input))
    return forecasts
      
'''
    test 결과를 plot 합니다. 
'''    
def plot_results(df, forecasts, target_column = TARGET_COLUMN):   
    # Plot predictions
    plotting_df = df.loc[:, [target_column]]
    plotting_df.plot(color="black")
    
    # plt.plot(df, color="black")
    for forecast in forecasts:
        forecast.plot()
        plt.legend(["True values"], loc="upper left", fontsize="small")
        plt.show()


'''
    모델을 학습합니다. 
'''
def train_models(training_data, prediction_length=14):
    ## Estimator 는 Predic
    arModel = DeepAREstimator(
        prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 20}
    ).train(training_data)

    tftModel = TemporalFusionTransformerEstimator(
        prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 400}
    ).train(training_data)

    # patchModel = PatchTSTEstimator(
    #     patch_len=10, prediction_length=15, trainer_kwargs={"max_epochs": 20}
    # ).train(training_data)
    return [arModel, tftModel]

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    ## 데이터셋 경로 입력
    args.add_argument("--dataset_path", type=str, default='data/processed/merged_S&P 500 Financials.csv')
    
    ## 모델 선택
    args.add_argument("--models", type=str, default='all') 
    
    ## train , test df 가져오기
    training_data, test_generator =  split_dataset()
    
    ## 선택한 모델들에 대해서 돌리기 
    train_models = train_models(training_data)
    for model in train_models:
        forecast = test_model(model, test_generator)
        plot_results(training_data, forecast)