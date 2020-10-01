import numpy as np
import pandas as pd
import ta

from src.util.config import get_config
config = get_config()


def drop_nans_from_data(df):
    """
    A number of preprocessing steps we want/need to take.
    """

    # Dropping time buckets with no trading. Retaining index as we depend on it being incremental integer.
    print(f'PREPROCESS: Dropping {df["Open"].isna().sum()} NaNs out of {len(df)} samples ({round(df["Open"].isna().sum()/len(df)*100, 2)}%) from input file')
    df = df.dropna(how='any').reset_index(drop=True)

    return df


def trim_df(df):
    """
    Drop everything we won't use, for more efficient data processing. Only disadvantage: slightly less flexibility when training (n timesteps).
    """
    timesteps_needed = config.train_predict.train_timesteps + config.train_predict.predict_timesteps + config.data_params.data_n_timesteps
    starting_index = df.index.max() - timesteps_needed + 1
    starting_index = max(starting_index, 0)
    print(f'PREPROCESS: Dropping unused data, {len(df)} -> {df.index.max()-starting_index+1} samples')
    df = df.loc[starting_index:]

    return df


def add_technical_features(df):
    """
    Adding a couple of handpicked, very commonly utilized TIs for BTC. 
    More info on each indicator: https://technical-analysis-library-in-python.readthedocs.io/en/latest
    NOTE: column names must be prefixed with `ti_`, since that is used to define obs space in env.

    TODO: move some recurring settings to config, like n.
    """
    print(f'PREPROCESS: Adding technical features...')
    orig_len = len(df)
    
    # Add Bollinger Bands features
    BB = ta.volatility.BollingerBands(close=df['Close'], n=20, ndev=2, fillna=False)
    df['ti_bb_hind'] = BB.bollinger_hband_indicator()
    df['ti_bb_lind'] = BB.bollinger_lband_indicator()
    # NOTE: too many inf/nan on 1s-data
    #df['ti_bb_pband'] = BB.bollinger_pband()
    df['ti_bb_wband'] = BB.bollinger_wband()

    # Ichimoku
    II = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=9, n2=26, n3=52, visual=False, fillna=False)
    df['ti_ii_senkou_a'] = II.ichimoku_a()
    df['ti_ii_senkou_b'] = II.ichimoku_b()
    df['ti_ii_kijun_sen'] = II.ichimoku_base_line()

    # Relative Strength Index (RSI)
    # TODO: consider adding binary features for above 70 and below 30 (i.e. standard interpretation)
    RSI = ta.momentum.RSIIndicator(close=df['Close'], n=14, fillna=False)
    df['ti_rsi'] = RSI.rsi()

    # Moving Average Convergence Divergence (MACD)
    MACD = ta.trend.MACD(close=df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
    df['ti_macd_hist'] = MACD.macd_diff()

    # Parabolic Stop and Reverse (Parabolic SAR)
    # NOTE: removed because too slow
    # PSAR = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2, fillna=False)
    # df['ti_psar_dind'] = PSAR.psar_down_indicator()
    # df['ti_psar_uind'] = PSAR.psar_up_indicator()
    
    # Average Directional Movement Index (ADX)
    ADX = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], n=14, fillna=False)    
    df['ti_adx'] = ADX.adx()
    df['ti_adx_neg'] = ADX.adx_neg()
    df['ti_adx_pos'] = ADX.adx_pos()

    # Commodity Channel Index (CCI)
    # NOTE: too many infs on 1s data
    # CCI = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], n=20, c=0.015, fillna=False)
    # df['ti_cci'] = CCI.cci()

    # Chaikin Money Flow (CMF)
    CMF = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['VolumeBTC'], n=20, fillna=False)
    df['ti_cmf'] = CMF.chaikin_money_flow()

    assert orig_len == len(df), f'Length of df has changed when adding TI features, from {orig_len} to {len(df)}.'
    print('Done.')
    
    return df


def perform_nan_check(*, df):
    """Make sure there are no NaNs or ±infs in data that will be used."""
    print(f'PREPROCESS: Performing NaN/inf check on data...')
    assert df.iloc[config.data_params.ti_nan_timesteps:].replace([np.inf, -np.inf], np.nan).isna().sum().sum() == 0, \
        f'PREPROCESS: Found Nan/inf in data, aborting...'
    print(f'Done.')


def preprocess_data(df):
    """
    Run all preprocessing steps
    """
    df = drop_nans_from_data(df=df)
    df = trim_df(df=df)
    df = add_technical_features(df=df)
    perform_nan_check(df=df)

    return df
