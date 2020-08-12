import pickle
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from .reward_schema import RewardSchema
from src.env.render.TradingChartStatic import TradingChartStatic
from src.env.render.ActionDistribution import ActionDistribution
from src.env.trade.TradeStrategy import TradeStrategy
from src.env.rewards import IncrementalNetWorth, RiskAdjustedReturns #, BaseRewardStrategy

from src.util.config import get_config
config = get_config()

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_ASSET = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000.0
class DunderBotEnv(gym.Env):
    """The Dunderbot class"""
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, **kwargs):
        super(DunderBotEnv, self).__init__()

        self.df = df
        # -1 due to inclusive slicing and 0-indexing
        self.data_n_timesteps = int(config.data_n_timesteps)
        self.data_n_indexsteps = self.data_n_timesteps - 1
        
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        #self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])

        # n_value_bins of different ratio
        self.action_n_bins = int(config.action_strategy.n_value_bins)
        self.action_space = spaces.Discrete(2 * self.action_n_bins + 1)

        # Prices contains the OHCL values for the last data_n_timesteps prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.data_n_timesteps), dtype=np.float16)

        # Set trade strategy with some constants
        # TODO: select best values here (now using default in RLTrader)
        # TODO: move some to config
        self.base_precision = 2
        self.asset_precision = 8
        self.min_cost_limit = 1E-3
        self.min_amount_limit = 1E-3
        self.commission_percent = 0
        self.max_slippage_percent = 0

        self.trade_strategy = TradeStrategy(commission_percent=self.commission_percent,
                                             max_slippage_percent=self.max_slippage_percent,
                                             base_precision=self.base_precision,
                                             asset_precision=self.asset_precision,
                                             min_cost_limit=self.min_cost_limit,
                                             min_amount_limit=self.min_amount_limit)

        # Set Reward Strategy
        #TODO: move this choice to config somehow
        self.reward_strategy = IncrementalNetWorth()



    def _next_observation(self):
        # Get the stock data points for the last data_n_indexsteps days and scale to between 0-1
        obs = np.array([
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'VolumeUSD'].values / MAX_NUM_ASSET,
        ])

        # Append additional data and scale each value to between 0-1
        # obs = np.append(obs, [[
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.asset_held / MAX_NUM_ASSET,
        #     self.cost_basis / MAX_SHARE_PRICE,
        #     self.total_asset_sold / MAX_NUM_ASSET,
        #     self.total_sales_value / (MAX_NUM_ASSET * MAX_SHARE_PRICE),
        # ]])

        return obs


    def translate_action(self, action):
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


    def _get_trade(self, action_type: str, action_amount: float):

        amount_asset_to_buy = 0
        amount_asset_to_sell = 0

        if action_type == 'buy' and self.balance >= self.min_cost_limit:
            price_adjustment = (1 + (self.commission_percent / 100)) * (1 + (self.max_slippage_percent / 100))
            buy_price = round(self.current_price * price_adjustment, self.base_precision)
            amount_asset_to_buy = round(self.balance * action_amount / buy_price, self.asset_precision)
        elif action_type == 'sell' and self.asset_held >= self.min_amount_limit:
            amount_asset_to_sell = round(self.asset_held * action_amount, self.asset_precision)
        return amount_asset_to_buy, amount_asset_to_sell


    def _take_action(self, action):
        """
        There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.
        
        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold.
        """

        # TODO: rename for more clarity
        
        

        # Set the current price to a random price within the time step
        self.current_price = self.df.loc[self.current_step, "Close"]
        
        action_type, action_amount = self.translate_action(action)
        amount_asset_to_buy, amount_asset_to_sell = self._get_trade(action_type, action_amount)

        asset_bought, asset_sold, purchase_cost, sale_revenue = self.trade_strategy.trade(buy_amount=amount_asset_to_buy,
                                                                                          sell_amount=amount_asset_to_sell,
                                                                                          balance=self.balance,
                                                                                          asset_held=self.asset_held,
                                                                                          current_price=self.current_price)

        if asset_bought:
            self.asset_held += asset_bought
            self.balance -= purchase_cost

            self.trades.append({'step': self.current_step,
                                'amount': asset_bought,
                                'total': purchase_cost,
                                'type': 'buy',
                                'action_amount': action_amount})
        elif asset_sold:
            self.asset_held -= asset_sold
            self.balance += sale_revenue

            # TODO: lookup this thing copied from RLTrader
            #self.reward_strategy.reset_reward()

            self.trades.append({'step': self.current_step,
                                'amount': asset_sold,
                                'total': sale_revenue,
                                'type': 'sell',
                                'action_amount': action_amount})
        elif action_type == 'hold':
            self.trades.append({'step': self.current_step,
                            'amount': None,
                            'total': None,
                            'type': 'hold',
                            'action_amount': None})

        current_net_worth = round(self.balance + self.asset_held * self.current_price, self.base_precision)
        self.net_worths.append(current_net_worth)
        self.account_history = self.account_history.append({
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': asset_bought,
            'purchase_cost': purchase_cost,
            'asset_sold': asset_sold,
            'sale_revenue': sale_revenue,
        }, ignore_index=True)



        ## OLD STARTS HERE
        # if trade == 'buy':  # Buy
        #     total_possible = self.balance / self.current_price
        #     asset_bought = total_possible * ratio
           
        #     prev_cost = self.cost_basis * self.asset_held
        #     additional_cost = asset_bought * self.current_price

        #     self.balance -= additional_cost
        #     self.cost_basis = (
        #         prev_cost + additional_cost) / (self.asset_held + asset_bought)

        #     # Save the trade for rendering
        #     self.trades.append({'step': self.current_step,
        #                         'amount': asset_bought,
        #                         'total': additional_cost,
        #                         'type': trade,
        #                         'action_ratio': ratio})

        # elif trade == 'sell':  # Sell
        #     asset_sold = self.asset_held * ratio

        #     self.balance += asset_sold * self.current_price
        #     self.asset_held -= asset_sold
        #     self.total_asset_sold += asset_sold
        #     self.total_sales_value += asset_sold * self.current_price

        #     # Save the trade for rendering
        #     self.trades.append({'step': self.current_step,
        #                         'amount': asset_sold,
        #                         'total': self.total_sales_value,
        #                         'type': trade,
        #                         'action_ratio': ratio})
        
        # elif trade == 'hold':  # Hold
        #     # Save the trade for rendering
        #     self.trades.append({'step': self.current_step,
        #                         'amount': None,
        #                         'total': None,
        #                         'type': trade,
        #                         'action_ratio': None})

        # # Handle net worth variants
        # self.net_worth = self.balance + self.asset_held * self.current_price
        # self.net_worths.append(self.net_worth)
        # if self.net_worth > self.max_net_worth:
        #     self.max_net_worth = self.net_worth
        
        # self.asset_held_hist.append(self.asset_held)
        # if self.asset_held == 0:
        #     self.cost_basis = 0


    def _reward(self):
        reward = self.reward_strategy.get_reward(net_worths=self.net_worths)

        reward = float(reward) if np.isfinite(float(reward)) else 0

        self.rewards.append(reward)

        #if self.stationarize_rewards:
        #    rewards = difference(self.rewards, inplace=False)
        #else:
        rewards = self.rewards

        #if self.normalize_rewards:
        #    mean_normalize(rewards, inplace=True)

        rewards = np.array(rewards).flatten()

        return float(rewards[-1])


    def _get_reward(self):

        Reward = RewardSchema(balance=self.balance, 
                                current_step=self.current_step, 
                                MAX_STEPS=MAX_STEPS)
        return Reward.get_reward_strategy()


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        # Start over again when data is out
        if self.current_step > self.df.index.max():
            self.current_step = self.data_n_indexsteps

        #reward = self._get_reward()
        reward = self._reward()
        
        # DoD: if we don't have any money, we can't trade
        # TODO: add time series has run out in DoD (see RLTrader for example)?
        done = self.net_worth <= 0

        # Next observation
        obs = self._next_observation()

        return obs, reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.asset_held = 0
        self.cost_basis = 0
        self.total_asset_sold = 0
        self.total_sales_value = 0
        self.trades = []
        
        # NOTE: Assess whether this should be padded with 0:s to retain timestep index between classes at first use
        self.rewards = [0]

        # Add data_n_indexsteps dummy net_worths to retain consistency in current_step between classes (since first index of df == 0 but first index of used data point == data_n_indexsteps != 0)
        self.net_worths = [INITIAL_ACCOUNT_BALANCE] * (self.data_n_indexsteps)
        self.asset_held_hist = [0.0] * (self.data_n_indexsteps)

        # Set the starting step to first useable value
        self.current_step = self.data_n_indexsteps

        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': 0,
            'purchase_cost': 0,
            'asset_sold': 0,
            'sale_revenue': 0,
        }]  * (self.data_n_indexsteps))

        return self._next_observation()


    def render(self, mode='human'):
        # Render the environment to the screen
        # TODO: when plot is good and has stabilized, rm this
        all_dict={  'current_step': self.current_step,
                    'net_worths': self.net_worths,
                    'trades': self.trades,
                    'account_history': self.account_history}
        with open('all_dict_pred.pickle', 'wb') as handle:
            pickle.dump(all_dict, handle)

        if mode == 'system':
            print('Price: ' + str(self.current_price))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            # Render static TradingChart
            self.viewer = TradingChartStatic(self.df)
            self.viewer.render(self.current_step,
                            self.net_worths,
                            self.trades,
                            self.account_history)

            # Render action distribution
            self.viewer = ActionDistribution(self.trades)
            self.viewer.render()

            