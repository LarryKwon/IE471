from abc import abstractclassmethod, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from util import *
import os 

# 필요한 모델 로드 
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator, DLinearEstimator, TemporalFusionTransformerEstimator, PatchTSTEstimator


TARGET_COLUMN = 'composite_realized_volatility'
DATASET_PATH = 'data/processed_v2.0/merged_composite_data.csv'
CKPT_SAVE_PATH = 'model_checkpoints/'
FIG_SAVE_PATH = 'fig/'


#   해당 경로에 있는 데이터셋을 받아 train dataset 과 test Dataset 으로 나눕니다.
def split_dataset(dataset_path = DATASET_PATH, target_column = TARGET_COLUMN):
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
def train_models(training_data, prediction_length=20):
    # ## Estimator 는 Predic
    # arModel = DeepAREstimator(
    #     prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 200}
    # ).train(training_data)
    # with open("arModel.pkl", "wb") as f:
    #     pickle.dump(arModel, f)

    tftModel = TemporalFusionTransformerEstimator(
        prediction_length=prediction_length, freq="D", trainer_kwargs={"max_epochs": 40}
    ).train(training_data)
    return [tftModel]

#  trained 된 모델들을 테스트 합니다. 
def test_and_plot(df, models, test_gen, prediction_length=20, windows=1):
    # Generate test instances
    test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)

    for idx, model in enumerate(models):
        forecasts = list(model.predict(test_data.input))
        plot_results(df, forecasts, idx, TARGET_COLUMN)    



class ParserStrategy:
    def __init__(self):
        self.mode = {
            "train_with_hyperparm": TrainWithHyperparmStrategy,
            "train_from_scratch": TrainModelStrategy,
            "test": TestModelStrategy
        }
        self.TARGET_COLUMN = 'composite_realized_volatility'
        self.DATASET_PATH = 'data/processed_v2.0/merged_composite_data.csv'
        self.CKPT_SAVE_PATH = 'model_checkpoints/'
        self.FIG_SAVE_PATH = 'fig/'
    
    @abstractmethod
    def execute(self):
        pass
    
    def procedure(self, mode_option, parser):
        return self.mode[mode_option](parser)
            
            
class TrainWithHyperparmStrategy(ParserStrategy):
    
    def __init__(self, parser):
        self.parser = parser
    
    def execute(self):
        args = self.parser.parse_args()
        model_params = args.model_params
        model_params = yaml.safe_load(open(model_params))
        model_path = args.model_path
        
        # if(dataset):
        #     split_dataset
        # else:
        #     train, test 
        #     train_models()
        #     test_and_plot()
        # ### load dataset
        
        
        ### load test data
        
        ### test_and_plot


class TrainModelStrategy(ParserStrategy):
    
    def __init__(self, parser):
        self.parser = parser
    
    def execute(self):
        args = self.parser.parse_args()
        
        if args.dataset_path:
            dataset_path = args.dataset_path
            training_data, test_generator, df =  split_dataset(dataset_path, self.TARGET_COLUMN) 
            train_models = train_models(training_data)
            test_and_plot(df, train_models, test_generator)

        else:
            assert args.train and args.test, "train and test data should be provided"
            train_path = args.train
            train_df = pd.read_csv(train_path,index_col=0,parse_dates=True)
            train_dataset = PandasDataset(train_df, freq='D',target=self.TARGET_COLUMN)
            model = train_models(train_dataset)
            
            test_path = args.test
            test_df = pd.read_csv(test_path,index_col=0,parse_dates=True)
            test_dataset = PandasDataset(test_df)
            
            for idx, model in enumerate(model):
                forecasts = list(model.predict(test_dataset.input))
                plot_results(train_df, forecasts, idx, TARGET_COLUMN)    
class TestModelStrategy(ParserStrategy):
    
    def __init__(self, parser):
        self.parser = parser
        
    def execute(self):
        args = self.parser.parse_args()
        model_params = args.model_params
        model_params = yaml.safe_load(open(model_params))
        model_path = args.model_path
        
        test_data = args.test
        
                
        ### load params and checkpoint
        
        ### load test data
        
        ### test_and_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## 데이터셋 경로 입력
    parser.add_argument("--dataset_path", type=str, default=None)
    
    ## 모델 선택
    parser.add_argument("--model_params", type=str, nargs='+', help="List of model YAML files")
    
    ## 이미 학습된 모델이 있으면 불러옴
    parser.add_argument("--model_path", type=str, nargs='*', help="List of model checkpoint files")

    ## train , test df 가져오기
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    
    strategy = ParserStrategy().procedure("train_from_scratch", parser)
    strategy.execute()
    
    
    
    

    # training_data, test_generator, df =  split_dataset()
    # print(len(training_data))

    # train_models = train_models(training_data)
    # test_and_plot(df, train_models, test_generator)
    # ## 선택한 모델들에 대해서 돌리기 
    # if args.train:
    #     trained_models = train_models(training_data)
