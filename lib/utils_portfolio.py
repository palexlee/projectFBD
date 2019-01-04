import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def portfolio_performance(weights, returns, title, log_ret=False, save=False):
  """
  in sample: portfolio weights computed for time t are used with returns at time t
  out sample: portfolio weights computed for time t are used with returns at time t + 1
  
  In other words, for the out sample, the returns of the portfolio at time t+1 is calculated
  by using the 'optimum' weights for time t and assuming that the return at time
  t+1 is assumed to be the same as at time t
  """
  
  if log_ret:
    portfolio_returns = np.log(weights.multiply(np.exp(returns)).dropna().sum(axis=1))
    insample_returns = np.log(weights.shift(-1).multiply(np.exp(returns)).dropna().sum(axis=1))
    
    performance = portfolio_returns.cumsum()
    insample_performance = insample_returns.cumsum()
  else:
    portfolio_returns = weights.multiply(returns).dropna().sum(axis=1)
    insample_returns = weights.shift(-1).multiply(returns).dropna().sum(axis=1)
    
    performance = (1 + portfolio_returns).cumprod() - 1
    insample_performance = (1 + insample_returns).cumprod() -  1
    
  performance.name = title
  
  plt.figure(figsize=(10, 10))

  plt.subplot(211)
  plt.title(title)
  plt.ylabel('Portfolio return %')
  (portfolio_returns * 100).plot()

  plt.subplot(212)
  plt.ylabel('Cumulative return %')
  (performance * 100).plot(legend = True)
  insample_performance.name = 'in sample'
  (insample_performance * 100).plot(legend = True)
  
  if save:
    plt.savefig(title+'.png', bbox_inches='tight')
    
  return performance

def w_min(x, series=True):
    sigma_inv = np.linalg.inv(x)
    ones = np.ones([len(sigma_inv), 1])
    res = sigma_inv @ ones /(ones.T @ sigma_inv @ ones)
    if series:
        return pd.Series(res.flatten())
    else : 
        return res.flatten()