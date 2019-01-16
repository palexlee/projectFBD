import xgboost as xgb


def one_weighter_xgb(taille0,taille1,Tin,returns,asset_to_predict,time=False): 
  taille = range(taille0,taille1)
  
  #g_regr=[]
  #x_regr=[]
  w = pd.DataFrame()
  acc = 0
  for t0 in taille:
      t1=t0+Tin
      data_in_sample=returns.iloc[t0:t1]
      data_in_sample=data_in_sample.dropna(axis=1)

      what_to_predict=data_in_sample[asset_to_predict].shift(-1).iloc[:-1] # Y
      
      
      predictors=data_in_sample
      if(time): 
        predictors["t"]=range(len(predictors))
      lastpredictors=predictors.tail(1)
      predictors=predictors.iloc[0:(len(predictors)-1)]

      num_round = 10
      param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
      param['nthread'] = 4
      param['eval_metric'] = 'rmse'

      #myRF=RandomForestRegressor(n_jobs=-1,n_estimators=n_est).fit(predictors,what_to_predict)
      #mypredictions=myRF.predict(lastpredictors)
      
      dtrain = xgb.DMatrix(predictors, label=what_to_predict)
      bst = xgb.train(param, dtrain, num_round)
      topredict = xgb.DMatrix(lastpredictors)
      mypredictions = bst.predict(topredict)
      

      x_t=np.sign(mypredictions[0])
      w_t = mypredictions[0]
      w = pd.concat([w,pd.Series(w_t,index=lastpredictors.index)],axis=0)
  return w




def all_weighter_xgb(taille0,taille1,Tin,returns,time=False):
  w_tot =  pd.DataFrame()
  for i in tqdm(returns.columns): 
    print(i)
    w_i = one_weighter_xgb(taille0,taille1,Tin,returns,i,time)
    w_tot = pd.concat([w_tot,w_i],axis=1)
  
  return w_tot
