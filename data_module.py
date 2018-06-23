#Import packages
import pandas as pd
import numpy as np
from glob import glob
import holidays
from datetime import timedelta, datetime

def market_dates(start, end, delta, holidays):
    date_list = []
    curr = start
    while curr < end:
        if((curr not in holidays) & (curr.weekday()<=4)):
            date_list.append(curr)
        curr += delta
    return date_list

def data_module(path, frequency, min_observations = 63):
    """
    Data manipulation module:

    - Aggregates data
    - Aligns indices to match US market dates
    - Fills na's forward
    """
    #Temporary
    #path = r'./Data/*.csv'
    #frequency = 'M'
    #min_observations = 36
    #Temporary

    #Aggregate data	
    csv_files = glob(path)
    data_df = []
    for file in csv_files:
        temp = pd.read_csv(file,index_col = 0, parse_dates = [0])
        temp['Asset'] = file.split('Data\\')[1].split('.csv')[0]
        temp.rename(columns = {temp.columns[0]:'Value'},inplace=True)
        data_df.append(temp)
        del temp
    data_df = pd.concat(data_df)
    data_df = pd.pivot(index = data_df.index, values = data_df.Value, columns = data_df.Asset)

    #Match dates to US market trading days
    us_holidays = holidays.UnitedStates()
    start_date = data_df.index.min()
    end_date = data_df.index.max()
    us_market_dates = market_dates(start_date, end_date, timedelta(days = 1), us_holidays)
    data_df = data_df.loc[data_df.index.isin(us_market_dates),:].copy()

    #Fill prices forward
    data_df.fillna(method = 'ffill', inplace = True)

    #Save prices in a daily dataframe and fill nan prices with 0 so that backtester can handle unavailable indices
    daily_df = data_df.copy()
    daily_df.fillna(0,inplace=True)

    #Get simple rates of return with resampling at an user-defined frequency
    ror_df = data_df.resample(frequency).last().pct_change()

    #Select rates of return for price-based indices only
    ror_df = ror_df.loc[:,~data_df.columns.isin(['GSERCAUS','CESIUSD','GTII10','USGGBE10','ACMTP10','CPIAUCSL','GDP'])].copy()
    daily_df.to_excel(r'daily_features.xlsx')
    daily_features = daily_df.loc[:,daily_df.columns.isin(['GSERCAUS','CESIUSD','GTII10','USGGBE10','ACMTP10','CPIAUCSL','GDP'])].copy()
    daily_df = daily_df.loc[:,~daily_df.columns.isin(['GSERCAUS','CESIUSD','GTII10','USGGBE10','ACMTP10','CPIAUCSL','GDP'])].copy()
    
    #Replace inf numbers by nan (inf may appear because of resampling)
    ror_df.replace([np.inf, -np.inf],np.nan, inplace=True)

    #Extract labels from indices returns
    labels_df = ror_df.copy()
    labels_df[labels_df<0] = -1
    labels_df[labels_df==0] = 0
    labels_df[labels_df>0] = 1
    
    #Keep real y's
    real_y = labels_df.copy()

    #Normalize all data into expanding rolling zscores with a threshold for min number of observations
    features_df = pd.DataFrame(index = data_df.index)
    for c in data_df.columns:
        m = data_df.loc[:,c].expanding(min_periods = min_observations).mean()
        s = data_df.loc[:,c].expanding(min_periods = min_observations).std()
        features_df[c] = (data_df.loc[:,c] - m) / s

    features_df = features_df.resample(frequency).last().copy()
    
    #Offset labels forward (avoid situation of predicting what has already occurred)
    labels_df = labels_df.shift(-1)

    #Drop nan's
    daily_df.dropna(inplace = True)
    features_df.dropna(inplace = True)
    labels_df.dropna(inplace = True)

    #Ensure that features and labels tables match in terms of dates		
    features_df = features_df.loc[features_df.index.isin(labels_df.index),:].copy()
    labels_df = labels_df.loc[labels_df.index.isin(features_df.index),:].copy()

    #Ensure last date on both features and labels dataframes is prior period end (relative to date of script run)
    features_df = features_df.loc[features_df.index < datetime.today(),:].copy()
    labels_df = labels_df.loc[labels_df.index < datetime.today(),:].copy()

    #Ensure that daily price dataframe ends at the same date as both features and labels dataframes
    daily_df = daily_df.loc[daily_df.index < features_df.index.max(),:].copy()

	#Output dataframes
    daily_df.to_excel(r'data.xlsx')
    features_df.to_excel(r'features_df.xlsx')
    real_y.to_excel(r'labels.xlsx')
    #daily_features.to_excel(r'daily_features.xlsx')
    ror_df.to_excel(r'ror_df.xlsx')

    return daily_df, features_df, labels_df, real_y

if __name__=='__main__':
    data_module(path, frequency, min_observations = 63)