import pickle
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from .reward_schema import RewardSchema
from src.env.render.TradingChartStatic import TradingChartStatic

from src.util.config import get_config
config = get_config()

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
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
            low=0, high=1, shape=(5, self.data_n_timesteps), dtype=np.float16)


    def _next_observation(self):
        # Get the stock data points for the last data_n_indexsteps days and scale to between 0-1
        obs = np.array([
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'VolumeUSD'].values / MAX_NUM_SHARES,
        ])

        return obs


    def get_action_from_action(self, action):
        """There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.
        
        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold."""
        
        if action < self.action_n_bins:
            action_type = 'buy'
            ratio = 1/(action + 2)  # +1 for 0 first of array, and +1 for 1/2 as max ratio
        elif action >= self.action_n_bins and action < 2 * self.action_n_bins:
            action_type = 'sell'
            ratio = 1/((action-self.action_n_bins) + 2)
        elif action == 2 * self.action_n_bins:
            action_type = 'hold'
            ratio = None
        return action_type, ratio


    def _take_action(self, action):
        """
        There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.
        
        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold.
        """
        # TODO: test impact of different orders of actions
        # TODO: rename for more clarity

        # Set the current price to a random price within the time step
        self.current_price = self.df.loc[self.current_step, "Close"]
        
        # Fetch the trade
        action_type, ratio = self.get_action_from_action(action)

        if action_type == 'buy':  # Buy
            total_possible = self.balance / self.current_price
            shares_bought = total_possible * ratio
           
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)

            # Save the trade for rendering
            self.trades.append({'step': self.current_step,
                                'amount': shares_bought,
                                'total': additional_cost,
                                'type': 'buy',
                                'action_ratio': ratio})

        elif action_type == 'sell':  # Sell
            shares_sold = self.shares_held * ratio

            self.balance += shares_sold * self.current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * self.current_price

            # Save the trade for rendering
            self.trades.append({'step': self.current_step,
                                'amount': shares_sold,
                                'total': self.total_sales_value,
                                'type': 'sell',
                                'action_ratio': ratio})
        
        elif action_type == 'hold':  # Hold
            # Save the trade for rendering
            self.trades.append({'step': self.current_step,
                                'amount': None,
                                'total': None,
                                'type': 'hold',
                                'action_ratio': None})

        # Handle net worth variants
        self.net_worth = self.balance + self.shares_held * self.current_price
        self.net_worths.append(self.net_worth)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        self.shares_held_hist.append(self.shares_held)
        if self.shares_held == 0:
            self.cost_basis = 0


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

        reward = self._get_reward()
        
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
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trades = []

        # Add data_n_indexsteps dummy net_worths to retain consistency in current_step between classes (since first index of df == 0 but first index of used data point == data_n_indexsteps != 0)
        self.net_worths = [INITIAL_ACCOUNT_BALANCE] * (self.data_n_indexsteps)
        self.shares_held_hist = [0.0] * (self.data_n_indexsteps)

        # Set the starting step to first useable value
        self.current_step = self.data_n_indexsteps

        return self._next_observation()


    def render(self, mode='human'):
        # Render the environment to the screen
        # TODO: make system mode runable (currently throws errors)
        if mode == 'system':
            print('Price: ' + str(self.current_price))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            # TODO: when plot is good and has stabilized, rm this
            # all_dict={  'current_step': self.current_step,
            #             'net_worths': self.net_worths,
            #             'trades': self.trades,
            #             'shares_held_hist': self.shares_held_hist}
            # with open('all_dict_pred.pickle', 'wb') as handle:
            #     pickle.dump(all_dict, handle)

            # Render static TradingChart
            self.viewer = TradingChartStatic(self.df)
            self.viewer.render(self.current_step,
                            self.net_worths,
                            self.trades,
                            self.shares_held_hist)

            # Render action distribution
            #self.viewer = ActionDistribution(self.df)
            