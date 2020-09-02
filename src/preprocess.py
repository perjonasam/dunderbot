import pandas as pd
import ta

from src.util.config import get_config
config = get_config()

def clean_data(df):
    """
    A number of preprocessing steps we want/need to take.
    """

    # Dropping time buckets with no trading. Retaining index as we depend on it being incremental integer.
    print(f'PREPROCESS: Dropping {df["Open"].isna().sum()} NaNs out of {len(df)} samples ({round(df["Open"].isna().sum()/len(df)*100, 2)}%) from input file')
    df = df.dropna(how='any').reset_index(drop=True)

    # Drop everything we won't use, for more efficient data processing. Only disadvantage: slightly less flexibility when training (n timesteps).
    timesteps_needed = config.train_predict.train_timesteps + config.train_predict.predict_timesteps + config.data_params.data_n_timesteps
    starting_index = df.index.max() - timesteps_needed + 1
    print(f'PREPROCESS: Dropping unused data, {len(df)} -> {timesteps_needed} samples')
    df = df.loc[starting_index:]

    return df


def add_technical_features(df):
    """
    Adding a couple of handpicked, very commonly utilized TIs for BTC. 
    More info on each indicator: https://technical-analysis-library-in-python.readthedocs.io/en/latest
    
    TODO: move some recurring settings to config
    """
    print(f'PREPROCESS: Adding technical features...')
    orig_len = len(df)
    
    # Add Bollinger Bands features
    BB = ta.volatility.BollingerBands(close=df['Close'], n=20, ndev=2, fillna=False)
    df['bb_hind'] = BB.bollinger_hband_indicator()
    df['bb_lind'] = BB.bollinger_lband_indicator()
    df['bb_pband'] = BB.bollinger_pband()
    df['bb_wband'] = BB.bollinger_wband()

    # Ichimoku
    II = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=9, n2=26, n3=52, visual=False, fillna=False)
    df['ii_senkou_a'] = II.ichimoku_a()
    df['ii_senkou_b'] = II.ichimoku_b()
    df['ii_kijun_sen'] = II.ichimoku_base_line()

    # Relative Strength Index (RSI)
    # TODO: consider adding binary features for above 70 and below 30 (i.e. standard interpretation)
    RSI = ta.momentum.RSIIndicator(close=df['Close'], n=14, fillna=False)
    df['rsi'] = RSI.rsi()

    # Moving Average Convergence Divergence (MACD)
    MACD = ta.trend.MACD(close=df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
    df['macd_hist'] = MACD.macd_diff()

    # Parabolic Stop and Reverse (Parabolic SAR)
    PSAR = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2, fillna=False)
    df['psar_dind'] = PSAR.psar_down_indicator()
    df['psar_uind'] = PSAR.psar_up_indicator()
    
    # Average Directional Movement Index (ADX)
    ADX = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], n=14, fillna=False)    
    df['adx'] = ADX.adx()
    df['adx_neg'] = ADX.adx_neg()
    df['adx_pos'] = ADX.adx_pos()

    # Commodity Channel Index (CCI)
    CCI = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], n=20, c=0.015, fillna=False)
    df['cci'] = CCI.cci()

    # Chaikin Money Flow (CMF)
    CMF = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['VolumeBTC'], n=20, fillna=False)
    df['cmf'] = CMF.chaikin_money_flow()

    assert orig_len == len(df), f'Length of df has changed when adding TI features, from {orig_len} to {len(df)}.'
    print('Done.')
    return df


def preprocess_data(df):
    """
    Run all preprocessing steps
    """
    df = clean_data(df=df)
    df = add_technical_features(df=df)
    
    return df
