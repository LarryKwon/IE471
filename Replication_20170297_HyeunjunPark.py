#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

bitcoin_data = pd.read_csv('bitcoin_bitstamp.csv')
stock_data = pd.read_csv('composite_data.csv')

bitcoin_data['log_price'] = np.log(bitcoin_data['close'])

stock_data['log_return'] = np.log(stock_data['Close']) - np.log(stock_data['Close'].shift(1))

stock_data['realized_volatility'] = stock_data['log_return'].rolling(window=20).apply(lambda x: 14* np.sqrt(np.sum(x**2)), raw=False)

merged_data = pd.merge(bitcoin_data[['Date', 'log_price']], stock_data[['Date', 'realized_volatility']], on='Date')
merged_data = merged_data.dropna()


# In[41]:


merged_data.describe()


# In[45]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

break_dates = ['2019-02-01', '2020-02-27', '2020-10-08', '2021-05-21']
for i, date in enumerate(break_dates):
    merged_data[f'break_{i+1}'] = (merged_data['Date'] >= date).astype(int)

# 독립 변수와 종속 변수 설정
merged_data['btc_diff'] = merged_data['log_price'] - merged_data['log_price'].shift(1)
merged_data['log_price_lag'] = merged_data['log_price'].shift(1)

# 필요한 행만 선택 (NaN 값 제거)
filtered_data = merged_data.dropna(subset=['realized_volatility', 'log_price_lag', 'btc_diff'])

def nls_model(params, y, X):
    alpha, beta, gamma, rho = params[0], params[1], params[2], params[3]
    breaks = params[4:]
    y_pred = alpha + beta * X['log_price_lag'] + gamma * (X['log_price'] - rho * X['log_price_lag'])
    for i in range(len(break_dates)):
        y_pred += breaks[i] * X[f'break_{i+1}']
    residuals = y - y_pred
    return np.sum(residuals**2)

# 초기 추정값 설정
initial_params = np.zeros(4 + len(break_dates))
initial_params[0] = 0  # alpha
initial_params[1] = 0  # beta
initial_params[2] = 0  # gamma
initial_params[3] = 0.5  # 초기 rho 값

# 최적화 수행
result = minimize(nls_model, initial_params, args=(filtered_data['realized_volatility'], filtered_data),
                  method='L-BFGS-B')

# 최적화 결과
params = result.x
alpha, beta, gamma, rho = params[0], params[1], params[2], params[3]
breaks = params[4:]

print(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, rho: {rho}")
for i in range(len(break_dates)):
    print(f"break_{i+1}: {breaks[i]}")

# 예측값 계산
filtered_data['RV_t_predicted'] = alpha + beta * filtered_data['log_price_lag'] + \
                                  gamma * (filtered_data['log_price'] - rho * filtered_data['log_price_lag'])
for i in range(len(break_dates)):
    filtered_data['RV_t_predicted'] += breaks[i] * filtered_data[f'break_{i+1}']

# 예측값 출력
print(filtered_data[['Date', 'RV_t_predicted']])


# In[46]:


filtered_data['RV_t_predicted'] = alpha + beta * filtered_data['log_price_lag'] + \
                                  gamma * (filtered_data['log_price'] - rho * filtered_data['log_price_lag'])
for i in range(len(break_dates)):
    filtered_data['RV_t_predicted'] += breaks[i] * filtered_data[f'break_{i+1}']

# OLS 회귀 모델 설정
X_ols = filtered_data[['log_price_lag', 'btc_diff']]
for i in range(len(break_dates)):
    X_ols[f'break_{i+1}'] = filtered_data[f'break_{i+1}']
X_ols['const'] = 1  # 상수항 추가
y_ols = filtered_data['realized_volatility']

# OLS 회귀 분석 수행
ols_model = sm.OLS(y_ols, X_ols).fit()

# 회귀 결과 요약
print(ols_model.summary())


# In[51]:


import numpy as np
from scipy.stats import t

# CW test를 위한 함수 정의
def clark_west_test(actual, pred1, pred2):
    f = (actual - pred1) ** 2 - (actual - pred2) ** 2 + (pred1 - pred2) ** 2
    mean_f = np.mean(f)
    std_f = np.std(f, ddof=1)
    t_statistic = mean_f / (std_f / np.sqrt(len(f)))
    
    # 자유도 계산 (표본 크기 - 1)
    df = len(f) - 1
    
    # p-value 계산 (단측검정)
    p_value = 1 - t.cdf(t_statistic, df=df)
    
    print(f"Mean of f: {mean_f}")
    print(f"Standard deviation of f: {std_f}")
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")
    
    return t_statistic

cw_stat = clark_west_test(filtered_data['realized_volatility'], filtered_data['realized_volatility'].mean(), filtered_data['RV_t_predicted'])
print(cw_stat)


# In[52]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

filtered_data['RV_t_predicted'] = alpha + beta * filtered_data['log_price_lag'] + \
                                  gamma * (filtered_data['log_price'] - rho * filtered_data['log_price_lag'])
for i in range(len(break_dates)):
    filtered_data['RV_t_predicted'] += breaks[i] * filtered_data[f'break_{i+1}']

gamma_risk_aversion = 3  
theta_leverage = 6  
rf = 0.01  

filtered_data['sigma_sq'] = filtered_data['realized_volatility']**2
filtered_data['wt'] = (1 / gamma_risk_aversion) * ((theta_leverage * filtered_data['RV_t_predicted'] + (theta_leverage - 1) * rf) / (theta_leverage**2 * filtered_data['sigma_sq']))

filtered_data['Rp'] = filtered_data['wt'] * (filtered_data['RV_t_predicted'] - rf) + (1 - filtered_data['wt']) * rf
filtered_data['Var_Rp'] = filtered_data['wt']**2 * theta_leverage**2 * filtered_data['sigma_sq']
filtered_data['CER'] = filtered_data['Rp'] - 0.5 * (1 / gamma_risk_aversion) * filtered_data['Var_Rp']

print(filtered_data[['Date', 'wt', 'Rp', 'Var_Rp', 'CER']])

