import numpy as np
from typing import Tuple
from src.util.config import get_config
config = get_config()


class TradeStrategy():
    def __init__(self,
                 base_precision: int,
                 asset_precision: int,
                 min_cost_limit: float,
                 min_amount_limit: float):
        self.base_precision = base_precision
        self.asset_precision = asset_precision
        self.min_cost_limit = min_cost_limit
        self.min_amount_limit = min_amount_limit
        self.commission_percent = config.trading_params.commission
        self.max_slippage_percent = config.trading_params.max_slippage

    def trade(self,
              action_type: str,
              action_amount: float,
              balance: float,
              asset_held: float,
              current_price) -> Tuple[float, float, float, float]:

        commission = self.commission_percent / 100
        slippage = np.random.beta(1, 3) * self.max_slippage_percent / 100

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
