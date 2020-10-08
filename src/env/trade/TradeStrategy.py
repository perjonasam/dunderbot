import numpy as np
from typing import Tuple
from src.util.config import get_config
config = get_config()


class TradeStrategyBase():
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
        self.action_n_bins = int(config.action_strategy.n_value_bins)

    def trade(self,
              action: int,
              balance: float,
              asset_held: float,
              current_price) -> Tuple[float, float, float, float]:

        # Translate action
        action_type, action_amount = self._translate_action(action)

        commission = self.commission_percent / 100
        slippage = np.random.beta(1, 3) * self.max_slippage_percent / 100

        # Calculate buy/sell values
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

        return asset_bought, asset_sold, purchase_cost, sale_revenue, action_type, action_amount


class TradeStrategyRatio(TradeStrategyBase):
    def _translate_action(self, action):
        """There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.

        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold."""
        # TODO: test impact of different orders of actions
        if action < self.action_n_bins:
            action_type = 'buy'
            action_amount = 1/(action + 2)  # +1 for 0 first of array, and +1 for 1/2 as max ratio
        elif action >= self.action_n_bins and action < 2 * self.action_n_bins:
            action_type = 'sell'
            action_amount = 1/((action-self.action_n_bins) + 2)
        elif action == 2 * self.action_n_bins:
            action_type = 'hold'
            action_amount = None
        return action_type, action_amount

    def get_strategy_action(self):
        pass
