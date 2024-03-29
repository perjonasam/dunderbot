import pandas as pd
import ta
from src.util.config import get_config

config = get_config()


def load_OHLC_data():
    """ Load data file specified in config. Currently consists of a couple of files from Cryptodownload.com.
    Will be mch better shortly. Manual prepping of csv data has occurred."""

    data_folder = 'data/input'
    filepath = f'{data_folder}/{config.input_data.source}_{config.input_data.asset}_{config.input_data.tempres}.pickle'
    df = pd.read_pickle(filepath)
    print(f'LOAD: Loading data with {len(df)} samples')
    return df




# def manual_prep_raw_data():
#     """ not for running, only to help import new data sources"""
#     # Example 1: Process for data from cryptodownload.com -- hourly data
#     df = pd.read_csv(filepath, header=1)
#     df['Timestamp'] = pd.to_datetime(df.Date, format=('%Y-%m-%d %I-%p'))
#     df = df.rename(columns={'Volume BTC': 'VolumeBTC', 'Volume USD': 'VolumeUSD'})
#     df = df.sort_values('Timestamp').reset_index(drop=True)
#     df = df.drop(columns=['Date'])

