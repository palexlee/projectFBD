# How to use (UNIX)

First, you need to get the data from https://github.com/philipperemy/FX-1-Minute-Data and copy the folder `full-data-2000-2017` under the folder `data`.

Some package such as PyTorch and XGBoost are required to run the code. To install the necessary package run:

```bash
pip install -r requirement.txt
```

Then, to run the code:

```bash
python main.py
```

The following variables can be changed if needed:

`WRITE_RAW` set to True to merge all the files and form the raw dataset

`FIT_LSTM` set to True to retrain LSTMs, ~7h

`FIT_XGBOOST` set to True to retrain XGBoost, ~30min

`FROM_GITHUB` set To True to load the data from this github repo

