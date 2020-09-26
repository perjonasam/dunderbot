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
        # self.asset_precision = asset_precision
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
