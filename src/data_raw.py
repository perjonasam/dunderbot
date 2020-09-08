import os
import pandas as pd

import sys
import gzip
import requests

from src.util.config import get_config
config = get_config()


def check_if_spec_data_exists():
    filename = f'./data/input/{config.input_data.source}_{config.input_data.asset}_{config.input_data.tempres}.pickle'
    return os.path.isfile(filename)


def download_data():
    """
    Downloding data if the specified data details are not previously downloaded, and save to raw data folder
    Source: http://api.bitcoincharts.com/v1/csv/
    """
    oldfashion_currency = config.input_data.asset[-3:]
    filename = f'{config.input_data.source}{oldfashion_currency}.csv.gz'
    url = f'http://api.bitcoincharts.com/v1/csv/{filename}'
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        open(f'data/raw/{filename}', 'wb').write(r.content)
        print(f'{config.input_data.source}_{oldfashion_currency}.csv.gz downloaded and saved to ./data/raw')
    else:
        print(f'The specified data details {config.input_data.source}/{config.input_data.asset}/{config.input_data.tempres} are unavailable both locally and at bitcoincharts. Update.')
        print(f'Aborting.')
        sys.exit()


def load_downloaded_data():
    # Unpack gz and load
    oldfashion_currency = config.input_data.asset[-3:]
    filename = f'./data/raw/{config.input_data.source}{oldfashion_currency}.csv.gz'
    print(filename)
    with gzip.open(filename) as f:
        df = pd.read_csv(f, header=None)
    print(f'Raw data file loaded...')
    return df


def prepare_raw_data(*, df):
    """
    Convert the file from raw transactions (per second) to OHLC data, and set correct dtypes and order etc.
    """
    print(f'Preparing the raw data...')
    # Safety measure: drop NaNs
    if df.isna().sum().max() > 0:
        print(f'Dropping {df.isna().sum().max()} NaN rows in raw data')
    df = df.dropna()

    df.columns = ['Timestamp', 'Price', 'VolumeBTC']
    df = df.groupby('Timestamp').agg(
        Open=('Price', 'first'), 
        High=('Price', 'max'), 
        Low=('Price', 'min'), 
        Close=('Price', 'last'),
        VolumeBTC=('VolumeBTC', 'sum'))

    # Ensuring order in data
    df = df.sort_index()

    df = df.reset_index(drop=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df['Symbol'] = config.input_data.asset

    print(f'Done.')
    return df


def save_processed_data(df):
    filename = f'./data/input/{config.input_data.source}_{config.input_data.asset}_{config.input_data.tempres}.pickle'
    df.to_pickle(filename)
    print(f'Processed data file saved in .data/input/.')


def download_and_process():
    # check if file exists, or trigger download and processing
    if not check_if_spec_data_exists():
        print('Will download and process raw data, since data specified in config is not available')
        download_data()
        df = load_downloaded_data()
        df = prepare_raw_data(df=df)
        save_processed_data(df=df)
    else:
        print(f'Processed data avilable locsally, no downloading and raw data processing needed.')
