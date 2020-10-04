import numpy as np

from typing import Tuple, Callable


class TradeStrategy():
    def __init__(self,
                 commission_percent: float,
                 max_slippage_percent: float,
                 base_precision: int,
                 asset_precision: int,
                 min_cost_limit: float,
                 min_amount_limit: float):
        self.commission_percent = commission_percent
        self.max_slippage_percent = max_slippage_percent
        self.base_precision = base_precision
        self.asset_precision = asset_precision
        self.min_cost_limit = min_cost_limit
        self.min_amount_limit = min_amount_limit


    def trade(self,
              buy_amount: float,
              sell_amount: float,
              balance: float,
              asset_held: float,
              current_price) -> Tuple[float, float, float, float]:

        commission = self.commission_percent / 100
        slippage = np.random.uniform(0, self.max_slippage_percent) / 100

        asset_bought, asset_sold, purchase_cost, sale_revenue = buy_amount, sell_amount, 0, 0

        if buy_amount > 0 and balance >= self.min_cost_limit:
            price_adjustment = (1 + commission) * (1 + slippage)
            buy_price = round(current_price * price_adjustment, self.base_precision)
            purchase_cost = round(buy_price * buy_amount, self.base_precision)
        elif sell_amount > 0 and asset_held >= self.min_amount_limit:
            price_adjustment = (1 - commission) * (1 - slippage)
            sell_price = round(current_price * price_adjustment, self.base_precision)
            sale_revenue = round(sell_amount * sell_price, self.base_precision)

        return asset_bought, asset_sold, purchase_cost, sale_revenue


    def trade_new(self,
              action_type: str,
              action_amount: float,
              balance: float,
              asset_held: float,
              current_price) -> Tuple[float, float, float, float]:


        # amount_asset_to_buy = 0
        # amount_asset_to_sell = 0

        # if action_type == 'buy' and self.balance >= self.min_cost_limit:
        #     price_adjustment = (1 + (self.commission_percent / 100)) * (1 + (self.max_slippage_percent / 100))
        #     buy_price = round(self.current_price * price_adjustment, self.base_precision)
        #     amount_asset_to_buy = round(self.balance * action_amount / buy_price, self.asset_precision)
        # elif action_type == 'sell' and self.asset_held >= self.min_amount_limit:
        #     amount_asset_to_sell = round(self.asset_held * action_amount, self.asset_precision)
        # return amount_asset_to_buy, amount_asset_to_sell

        commission = self.commission_percent / 100
        # TODO: model slippage non-uniform
        slippage = np.random.uniform(0, self.max_slippage_percent) / 100
        asset_bought, asset_sold, purchase_cost, sale_revenue = 0, 0, 0, 0

        if action_type == 'buy' and balance >= self.min_cost_limit:
            price_adjustment = (1 + commission) * (1 + slippage)
            buy_price = round(current_price * price_adjustment, self.base_precision)
            asset_bought = round(balance * action_amount / buy_price, self.asset_precision)
            purchase_cost = round(buy_price * asset_bought, self.base_precision)
        elif action_type == 'sell' and asset_held >= self.min_amount_limit:
            price_adjustment = (1 - commission) * (1 - slippage)
            sell_price = round(current_price * price_adjustment, self.base_precision)
            asset_sold = round(asset_held * action_amount, self.asset_precision)
            sale_revenue = round(asset_sold * sell_price, self.base_precision)

        return asset_bought, asset_sold, purchase_cost, sale_revenue
