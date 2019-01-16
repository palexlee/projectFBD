import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_portfolio import w_min

def plot_eigenvalues_dist(C, N, T, q, path):
  lambda_, V = np.linalg.eig(C)

  lp = (1 + np.sqrt(q))**2
  lm = (1- np.sqrt(q))**2

  S= lambda x :1/(q*2*np.pi)*np.sqrt((lp - x)*(x - lm))/x

  plt.figure(figsize=(20, 5))
  plt.hist(lambda_, bins=40, density=True)
  x = np.arange(lm, lp, 0.0001)
  plt.plot(x, S(x))
  title = 'Eigenvalues distribution'
  plt.title(title)
  plt.savefig(path + title+'.png', bbox_inches='tight')

def corr_clipping(C, N, T, q, d=0.5):

  lambda_, V = np.linalg.eig(C)

  lp = (1 + np.sqrt(q))**2
  lm = (1- np.sqrt(q))**2

  lambda_[lambda_ < lp] = d #clipping eigenvalues

  clipped = V.T @ np.diag(lambda_) @ V

  np.fill_diagonal(clipped, 1)

  return pd.DataFrame(clipped, columns=C.columns, index=C.index)

def get_weight(returns, rmt_fun, window):
  weight = pd.DataFrame()

  for i in range(len(returns) - window):
    current = returns.iloc[i:i+window]
    cov = rmt_fun(current)
    weight = pd.concat([weight, pd.DataFrame(w_min(cov), columns=[current.iloc[-1].name]).transpose()], axis=0)
  
  weight.columns = returns.columns
  return weight