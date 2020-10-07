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


def add_technical_features(df, include_slow_features=False, verbose=True):
    """
    Adding a couple of handpicked, very commonly utilized TIs for BTC.
    More info on each indicator: https://technical-analysis-library-in-python.readthedocs.io/en/latest
    NOTE: column names must be prefixed with `ti_`, since that is used to define obs space in env.

    TODO: move some recurring settings to config, like n.
    """
    if verbose: print(f'PREPROCESS: Adding technical features...')
    orig_len = len(df)

    # Add Bollinger Bands features
    BB = ta.volatility.BollingerBands(close=df['Close'], n=20, ndev=2, fillna=False)
    df['ti_bb_hind'] = BB.bollinger_hband_indicator()
    df['ti_bb_lind'] = BB.bollinger_lband_indicator()
    # NOTE: too many inf/nan on 1s-data
    # df['ti_bb_pband'] = BB.bollinger_pband()
    df['ti_bb_wband'] = BB.bollinger_wband()

    # Ichimoku
    # TODO: test ALT settings: 20,60, 120 (https://medium.com/@coinloop/technical-analysis-indicators-and-how-to-use-them-aa0fa706051)
    II = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], n1=9, n2=26, n3=52, visual=False, fillna=False)
    df['ti_ii_senkou_a'] = II.ichimoku_a()
    df['ti_ii_senkou_b'] = II.ichimoku_b()
    df['ti_ii_kijun_sen'] = II.ichimoku_base_line()

    # Relative Strength Index (RSI)
    RSI = ta.momentum.RSIIndicator(close=df['Close'], n=14, fillna=False)
    df['ti_rsi'] = RSI.rsi()

    # Moving Average Convergence Divergence (MACD)
    MACD = ta.trend.MACD(close=df['Close'], n_slow=26, n_fast=12, n_sign=9, fillna=False)
    df['ti_macd_hist'] = MACD.macd_diff()

    # Parabolic Stop and Reverse (Parabolic SAR)
    # NOTE: very slow (1h on 1.75M obs)
    if include_slow_features:
        PSAR = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2, fillna=False)
        df['ti_psar_dind'] = PSAR.psar_down_indicator()
        df['ti_psar_uind'] = PSAR.psar_up_indicator()

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

    # StochasticOscillator (SO)
    # NOTE: too many NaN on 1s data
    # TODO: test alt settings: Length = 10 K = 5 D = 3 (https://medium.com/@coinloop/technical-analysis-indicators-and-how-to-use-them-aa0fa706051)
    # SO = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], n=14, d_n=3, fillna=False)
    # df['ti_so'] = SO.stoch()
    # df['ti_so_signal'] = SO.stoch_signal()

    # AwesomeOscillator
    AO = ta.momentum.AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], s=5, len=34, fillna=False)
    df['ti_ao'] = AO.ao()

    # EMA8
    span = 8
    df['ti_EMA8'] = df['Close'].ewm(span=span, min_periods=span, adjust=False).mean()

    # EMA50
    span = 50
    df['ti_EMA50'] = df['Close'].ewm(span=span, min_periods=span, adjust=False).mean()

    # EMA100
    span = 100
    df['ti_EMA100'] = df['Close'].ewm(span=span, min_periods=span, adjust=False).mean()

    # EMA200
    span = 200
    df['ti_EMA200'] = df['Close'].ewm(span=span, min_periods=span, adjust=False).mean()

    assert orig_len == len(df), f'Length of df has changed when adding TI features, from {orig_len} to {len(df)}.'
    if verbose: print('Done.')

    return df


def perform_nan_check(*, df):
    """Make sure there are no NaNs or ±infs in data that will be used."""
    print(f'PREPROCESS: Performing NaN/inf check on data...')
    assert df.iloc[config.data_params.ti_nan_timesteps:].replace([np.inf, -np.inf], np.nan).isna().sum().sum() == 0, \
        f'PREPROCESS: Found Nan/inf in data, aborting...'
    print(f'Done.')
    return


def perform_TIs_forward_lookingness_check(df):
    """ Check for potential forward lookingness of TIs """
    print('PREPROCESS: Performing check of forward lookingness of TIs')
    test_len = 5000 + int(config.data_params.ti_nan_timesteps)
    df_small = df.iloc[:test_len*2].copy()
    df_smaller = df.iloc[:test_len].copy()

    df_small = add_technical_features(df=df_small, include_slow_features=False, verbose=False)
    df_smaller = add_technical_features(df=df_smaller, include_slow_features=False, verbose=False)

    ti_cols = [col for col in df_small.columns if 'ti_' in col]
    for ti_col in ti_cols:
        assert df_smaller[ti_col].equals(df_small[ti_col].iloc[:test_len]), f'TI {ti_col} potentially forward looking'
    return


def preprocess_data(df, include_slow_features=False):
    """
    Run all preprocessing steps
    """
    df = drop_nans_from_data(df=df)
    df = trim_df(df=df)
    _ = perform_TIs_forward_lookingness_check(df=df)
    df = add_technical_features(df=df, include_slow_features=include_slow_features)
    _ = perform_nan_check(df=df)

    return df
