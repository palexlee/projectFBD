from zipfile import ZipFile
import glob
import pandas as pd
import numpy as np

def load_data(folder_path, time_agg=None):
    list_paths = glob.glob(folder_path + '/**/', recursive=True)
    list_years = glob.glob(folder_path + '/**/*.zip', recursive=True)
    

    df = pd.DataFrame()
    for path in tqdm(list_paths[1:]):
        list_years = glob.glob(path+'*.zip', recursive=True)

        df_tmp = pd.DataFrame()
        for year in list_years:
            ticker = year[len(folder_path + "/eurczk/DAT_ASCII_"):][:len("EURCZK")]
            filename = year[len(folder_path+"/eurczk/"):-len('.zip')]

            columns = ['DateTime Stamp', ticker, ticker + ' Volume']
            zip_file = ZipFile(year)
            df_tmp = pd.concat([df_tmp, pd.read_csv(zip_file.open(filename + '.csv'), sep=';', index_col=[0], usecols=[0, 4, 5], names=columns)], axis=0)
            
            if time_agg is not None:
              df_tmp = df_tmp.groupby(pd.to_datetime(df_tmp.index).floor(time_agg)).agg({ticker: 'last', ticker + ' Volume': 'sum'})

        df = pd.concat([df, df_tmp], axis=1)
        
    return df