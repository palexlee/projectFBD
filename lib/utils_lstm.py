from torch import nn
from torch import optim
from torch import Tensor
import torch

import pandas as pd
import numpy as np
from tqdm import tqdm

use_gpu = torch.cuda.is_available()

def small_lstm(window):
  model = nn.Sequential(
      nn.LSTM(input_size=window, hidden_size=100, num_layers=2),
      nn.LSTM(input_size=100, hidden_size=10, num_layers=2),
      nn.Linear(32, 1)
  )
  return model

class bigLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(bigLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

def train_model(model, inputs, labels, tofit, optimizer, criterion, batch_size, epoch):
  
  #model.hidden = model.init_hidden()
  
  for i in range(epoch):
    for j, data in enumerate(inputs):
      data = data.unsqueeze(0)

      optimizer.zero_grad()

      outputs = model(data)

      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      with torch.no_grad():  
        print('L1 loss :', loss.data.item(), flush=True)
          
          
  with torch.no_grad():
    model.eval()
    fitted = model(tofit)
    
    return fitted.data.numpy()

def forecast_asset(model, optimizer, criterion, asset_to_predict, returns, window, batch_size, horizon, epoch):
  dataset = pd.DataFrame(columns=np.arange(horizon))

  for t0 in tqdm(range(0, len(returns)-window)):

    model.hidden = model.init_hidden()


    t1 = t0 + window

    data_in_sample=returns.iloc[t0:t1]
    data_in_sample=data_in_sample.fillna(axis=1,method='ffill')

    predictors=data_in_sample[asset_to_predict]

    data_input = np.array([predictors.values[i:i + batch_size] for i in range(window-batch_size-horizon)])

    if horizon > 1:
      data_label = np.array([predictors.values[i + batch_size :i + batch_size + horizon] for i in range(window-batch_size-horizon)]) ## TODO forecast 7 day
    else:
      data_label = predictors.values[batch_size:]

    train_input = data_input[:-1]
    train_label = data_label[:-1]
    tofit = data_input[-1]


    inputs = torch.Tensor(train_input).unsqueeze(2)
    labels = torch.Tensor(train_label)

    tofit = torch.Tensor(tofit).unsqueeze(0).unsqueeze(2)

    if use_gpu:
      inputs = inputs.cuda()
      labels = labels.cuda()
      tofit = tofit.cuda()

    result = train_model(model, inputs, labels, tofit, optimizer, criterion, batch_size, epoch)

    dataset.loc[predictors.index.values[-horizon]]=result
    
  dataset.columns = pd.MultiIndex.from_product([[asset_to_predict], dataset.columns])
  
  return dataset

def forecast_covariance(prices, predictions, columns, log_ret=True):

  current_date = prices.index.values[-1]
  stacked_forecast = predictions.loc[[current_date]].stack()
  stacked_forecast.index = stacked_forecast.index.droplevel(0)
  
  #replace the last price with the forecast ones
  composed = pd.concat([prices.iloc[:-1], stacked_forecast[prices.columns]], axis=0)[columns].reset_index(drop=True)
  
  if log_ret:
    composed_return = np.log(composed).diff()[1:]
  else:
    composed_return = composed.pct_change()[1:]
  
  return composed_return.cov(), composed_return.corr(), composed_return.std(axis=0).values.reshape(-1, 1)
