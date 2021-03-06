# -*- coding: utf-8 -*-
"""FBD_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GQHYs63_7q0f9EUMYMT0YBMsaQ0l45h_
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time
import pyRMT
import glob

from torch import nn
from torch import optim
from torch import Tensor
import torch

DATA_PATH = "./data/raw/full-data-2000-2017"
RAW_PATH = "./data/raw/"
CLEAN_PATH = "./data/clean/"
FIG_PATH = './figures/'
GIT_PATH = 'https://raw.githubusercontent.com/palexlee/projectFBD/master/data/clean/'
GIT_RAW = 'https://raw.githubusercontent.com/palexlee/projectFBD/master/data/raw/'

sys.path.append(DATA_PATH)
sys.path.append(RAW_PATH)
sys.path.append(CLEAN_PATH)
sys.path.append('./lib')

from utils_data import *
from utils_clipping import *
from utils_portfolio import *
from utils_lstm import *
from utils_xboost import *

WRITE_RAW = False
FIT_LSTM = False # set to True to retrain LSTMs, ~7h
FIT_XGBOOST = False # set to True to retrain XGBoost, ~30min
FROM_GITHUB = True

window = 80

"""## Creating and cleaning forex dataset"""

if WRITE_RAW:
  data = load_data(DATA_PATH, time_agg='d')
  data.to_csv(RAW_PATH + 'raw_forex.csv')
else:
  raw_forex = pd.read_csv(RAW_PATH + 'raw_forex.csv', index_col=[0])

"""## Returns and ...."""

if not WRITE_RAW:
  raw_forex = pd.read_csv((GIT_RAW if FROM_GITHUB else RAW_PATH) + 'raw_forex.csv', index_col=[0])

stock_indexes = ["SPXUSD","JPXJPY","NSXUSD","FRXEUR","UDXUSD","UKXGBP","GRXEUR","AUXAUD","HKXHKD","ETXEUR","WTIUSD"]

clean_forex = raw_forex.dropna(axis=0, thresh=int(0.9*len(raw_forex.columns)))
clean_forex.columns = map(lambda x: x.replace(' CLOSE Bid Quote', ''), clean_forex.columns)
clean_forex.index = pd.to_datetime(clean_forex.index)
clean_forex = clean_forex.drop(columns=stock_indexes)
return_forex = np.log(clean_forex).diff()[1:].fillna(0.0)

return_forex.XAUEUR.cumsum().plot()


"""## 1) Portfolio of forex"""

rolled_return = return_forex.rolling(7).mean()

"""### 1.1) equally weighted"""

equal_weight = rolled_return.apply(lambda x:pd.Series([1./rolled_return.shape[1]]*rolled_return.shape[1]), axis=1)
equal_weight.columns = rolled_return.columns

performance_ew = portfolio_performance(equal_weight[window:], return_forex, 'Equally Weighted', log_ret=True, save=True)

"""### 1.2) Minimum variance"""

covariance = rolled_return.shift(1).rolling(window).cov().dropna(axis=0)

global_minimum_variance = covariance.groupby(level=0).apply(w_min)
global_minimum_variance.columns = return_forex.columns
global_minimum_variance.plot(legend=False)

performance_markovitz = portfolio_performance(global_minimum_variance, return_forex,'Markovitz', log_ret=True, save=True)

"""### 1.3) Risk parity"""

stds = rolled_return.shift(1).rolling(window).std().dropna()

risk_parity = 1./stds
risk_parity = risk_parity.div(risk_parity.sum(axis=1), axis=0)
risk_parity.plot(legend=False)

performance_rp = portfolio_performance(risk_parity, return_forex, 'Risk Parity', log_ret=True, save=True)

"""### 1.4) Strategies comparison"""

strategies = pd.concat([performance_ew, performance_rp, performance_markovitz], axis = 1)
(strategies * 100).plot()

title = 'Basic strategies comparison'
plt.title(title)
plt.ylabel('Cumulative return %')
plt.savefig(FIG_PATH+title+'.png')

"""In theory, we know that the risk parity and mean variance portfolio should outperform the equally weighted portfolio.

In practice, the equally weighted uses no knowledge of historical returns, risk parity uses the standard deviation of the returns, and the minimum variance uses the more data as it need to compute the covariance matrix of the pair of currencies. Therefore, we can see that the poorest strategies are the one that use the most data, and therefore are prone to fit more noise.

## 2) Correlation matrix cleaning
"""

N = rolled_return.shape[1]
T = window
q = N/T
correlation = rolled_return.shift(1).rolling(window).corr().dropna(axis=0)
C = correlation.loc[correlation.index.get_level_values(0).unique()[-1]] #taking the last correlation matrix
  
plot_eigenvalues_dist(C, N, T, q, FIG_PATH)

"""It looks like deviation from RMT is really minimum, suggesting that our data is really noisy.

### 2.1) MVP with eigenvalue clipping
"""
wrap_clipping = lambda x : corr_clipping(x, N, T, q, d=0.5)

clipped_correlation = correlation.groupby(level=0).apply(wrap_clipping)

clipped_covariance = pd.DataFrame()
for name, group in correlation.groupby(level=0):
  std = stds.loc[name].values
  new_cov = group.multiply(std @ std.T)
  clipped_covariance = pd.concat([clipped_covariance, new_cov], axis=0)

mvp_clipped = clipped_covariance.groupby(level=0).apply(w_min)
mvp_clipped.columns = rolled_return.columns
mvp_clipped.plot(legend=False)

performance_clipped = portfolio_performance(mvp_clipped, return_forex, 'markovitz clipped', log_ret=True, save=True)
strategies = pd.concat([strategies, performance_clipped], axis=1)

(strategies * 100).plot()
plt.title('Model comparison')
plt.xlabel('Cumulative return %')

"""The clipping here seems to be very efficient since this strategy outperforms the equally weighted portfolio wich fit the less noise.

### 2.2) pyRMT clipping
"""

rolled_return_adj = rolled_return.shift(1).dropna()

clipped_weight = get_weight(rolled_return_adj, pyRMT.clipped, window)

performance_clipped = portfolio_performance(clipped_weight, return_forex, 'min variance clipped pyRMT', log_ret=True, save=True)

"""###  2.3) pyRMT shrinking"""

shrinked_weight = get_weight(rolled_return_adj, pyRMT.optimalShrinkage, window)

performance_shrink = portfolio_performance(shrinked_weight, return_forex, 'min variance shrinkage', log_ret=True,save=True)

strategies_rmt = pd.concat([strategies, performance_clipped, performance_shrink], axis=1)
(strategies_rmt * 100).plot(figsize=(15, 10))
title='Model comparison (with correlation matrix cleaning)'
plt.title(title)
plt.ylabel('Performance %')
plt.savefig(FIG_PATH + title +'.png')

"""## 3) Alternative strategy

For each day, make forecast of price with an horizon of 7 days using LSTM.
The idea is to use those extra 7 days to compute a forecasted covariance matrix of the assets and plug it into the different portfolio stratgies done above, simulating a sort of look-ahead bias.

The predictions are computed as such, we take a rolling window of the previous month's prices and for each window train an lstm, then we make the 7 days forecast for the last day of the window.

The training is done the following way for each window:
- take a sub rolling window of 15 days of prices, each sample will correspond to a sequence as input to the lstm
- for each sample, take the following 7 days, this will correspond to the targets
- take the last pair of input, target,
"""

length_train = 15
horizon = 7

returns = clean_forex

if FIT_LSTM:

  h1 = 32
  num_layers = 2
  learning_rate = 1e-2
  dtype = torch.float

  model = bigLSTM(length_train, h1, batch_size=1, output_dim=horizon, num_layers=num_layers)

  epoch = 2
  use_gpu = torch.cuda.is_available()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.L1Loss()

  for asset_to_predict in clean_forex.columns:
    prediction = forecast_asset(model, optimizer, criterion, asset_to_predict, returns, window, length_train, horizon, epoch)
    filename = RAW_PATH + 'lstm_' + str(horizon) + '_' + asset_to_predict + '.csv'
    prediction.to_csv(filename)
    print(asset_to_predict + ' done')


  list_lstm_predictions = glob.glob(RAW_PATH + 'lstm_*')

  prediction_df = pd.DataFrame()
  for file in list_lstm_predictions:
      current_df = pd.read_csv(file, index_col=0, header=[0,1])
      prediction_df = pd.concat([prediction_df, current_df], axis=1)
      
  prediction_df.to_csv(CLEAN_PATH + 'forex_lstm_horizon7.csv')

"""## Predictions

### XGboost
"""

if FIT_XGBOOST:
  total_len_return_forex = return_forex.shape[0]
  xgb_returns = all_weighter_xgb(0,total_len_return_forex-1,90,return_forex,time=True)
  xgb_returns.to_csv(CLEAN_PATH + 'xgb_forex.csv')
else :

  xgb_returns = pd.read_csv((GIT_PATH if FROM_GITHUB else CLEAN_PATH) + 'xgb_forex.csv', index_col=[0])
  xgb_returns.index = pd.to_datetime(xgb_returns.index)

"""### LSTM"""

lstm_forex = pd.read_csv((GIT_PATH if FROM_GITHUB else CLEAN_PATH) + 'forex_lstm_horizon7.csv', index_col=[0], header=[0,1])
lstm_forex.index = pd.to_datetime(lstm_forex.index)
lstm_forex = lstm_forex.shift(-length_train-horizon)

"""drop FX pair that can't be correctly fitted the the lstm
 
 recompute the vanilla markovitz portfolio with the remaining FX pairs
"""

weird_tick = ['XAUCHF', 'XAUAUD', 'XAUUSD']
crazy_tick = ['EURDKK','USDHKD']
bad_tick = ['GBPJPY','CADJPY','EURJPY','XAUCHF', 'XAUAUD', 'XAUUSD', 'EURDKK','USDHKD']
pd.concat([clean_forex[crazy_tick[1]], lstm_forex[crazy_tick[1]]], axis=1).plot(figsize=(20,10))

title = 'LSTM Art'
plt.title(title)
plt.savefig(FIG_PATH + title+'.png')

lstm_forex = lstm_forex.drop(columns=(stock_indexes+bad_tick), level=0)
clean_forex = clean_forex.drop(columns=bad_tick)

tick = 'EURUSD'
tick_fit = pd.concat([clean_forex[tick], lstm_forex[tick]['0']], axis=1)
tick_fit.plot(figsize=(20, 10))
plt.ylabel('price')

title = tick + ' lstm fit'
plt.title(title)
plt.savefig(FIG_PATH+title+'.png')

tick_fit.loc['2015-04-01':'2015-08-01'].plot(figsize=(20,10))
plt.ylabel('price')
title = tick + ' lstm fit (4 months)'
plt.title(tick + ' lstm fit (4 months)')

plt.savefig(FIG_PATH+title+'.png')

"""The lstm seems to yield a smoothed version of the FX pair price.

### 'forecasted' covariance matrix
"""

clean_return = np.log(clean_forex).diff()[1:]
clean_rolled_return = rolled_return.drop(columns=bad_tick)

covariance = clean_rolled_return.shift(1).rolling(window).cov().dropna(axis=0)
gmv = covariance.groupby(level=0).apply(w_min)
gmv.columns = clean_rolled_return.columns

fwindow = window - 7

N = clean_forex.shape[1]
T = fwindow-1
q = N/T

weights = pd.DataFrame(columns=clean_forex.columns)
weights_clipped = weights.copy()

for i in tqdm(range(len(clean_forex) - window-horizon - length_train)):
  prices = clean_forex.loc[lstm_forex.index].iloc[i:i+fwindow]
  
  composed_covariance, composed_correlation, stds = forecast_covariance(prices, lstm_forex, clean_forex.columns)
  
  date = prices.index.values[-1]
  
  # compute the weights for the markovitz portfolio
  weight = w_min(composed_covariance)
  weight.index = clean_forex.columns 
  
  # compute the weights with correlation matrix clipping this time
  weight_clipped = w_min(corr_clipping(composed_correlation, N, T, q, d=0.5).multiply(stds @ stds.T))
  weight_clipped.index = clean_forex.columns
  
  weights_clipped.loc[date] = weight_clipped
  weights.loc[date] = weight

performance_lstm = portfolio_performance(weights, clean_return, 'markovitz lstm', log_ret=True, save=True)
performance_lstm_clipped = portfolio_performance(weights_clipped, clean_return, 'markovitz lstm with clipped correlation matrices', log_ret=True, save=True)

performance_gmv = portfolio_performance(gmv, clean_return, 'markovitz vanilla', log_ret=True, save=True)
pd.concat([performance_gmv, performance_lstm], axis=1).plot()
title = 'Vanilla vs lstm forecast'
plt.title(title)
plt.savefig(FIG_PATH + title+'.png', bbox_inches='tight')

pd.concat([performance_gmv, performance_lstm, performance_lstm_clipped], axis=1).plot()
title = 'Comparison with clipped correlation matrices'
plt.title(title)
plt.savefig(FIG_PATH + title+'.png', bbox_inches='tight')
plt.ylabel('Performance %')