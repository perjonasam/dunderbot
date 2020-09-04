import ta
import pandas as pd
from enum import Enum


class SIGNALS(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


def trade_strategy(prices, initial_balance, commission, signal_fn):
    net_worths = [initial_balance]
    balance = initial_balance
    amount_held = 0

    for i in range(0, len(prices)-1):
        if amount_held > 0:
            net_worths.append(balance + amount_held * prices.iloc[i])
        else:
            net_worths.append(balance)

        signal = signal_fn(i)

        if signal == SIGNALS.SELL and amount_held > 0:
            balance = amount_held * (prices.iloc[i] * (1 - commission))
            amount_held = 0
        elif signal == SIGNALS.BUY and amount_held == 0:
            amount_held = balance / (prices.iloc[i] * (1 + commission))
            balance = 0

    return net_worths


def buy_and_hold(prices, initial_balance, commission):
    def signal_fn(i):
        return SIGNALS.BUY

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def rsi_divergence(prices, initial_balance, commission, period=3):
    rsi = ta.momentum.rsi(prices)

    def signal_fn(i):
        if i >= period:
            rsiSum = sum(rsi.iloc[i - period:i + 1].diff().cumsum().fillna(0))
            priceSum = sum(prices.iloc[i - period:i + 1].diff().cumsum().fillna(0))

            if rsiSum < 0 and priceSum >= 0:
                return SIGNALS.SELL
            elif rsiSum > 0 and priceSum <= 0:
                return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def sma_crossover(prices, initial_balance, commission):
    macd = ta.trend.macd(prices)

    def signal_fn(i):
        if macd.iloc[i] > 0 and macd.iloc[i - 1] <= 0:
            return SIGNALS.SELL
        elif macd.iloc[i] < 0 and macd.iloc[i - 1] >= 0:
            return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)
