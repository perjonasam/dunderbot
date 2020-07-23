import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from .reward_schema import RewardSchema

from src.util.config import get_config
config = get_config()

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000
class DunderBotEnv(gym.Env):
    """The Dunderbot class"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(DunderBotEnv, self).__init__()

        self.df = df
        # -1 due to inclusive slicing and 0-indexing
        self.data_n_timesteps = int(config.data_n_timesteps)
        self.data_n_indexsteps = self.data_n_timesteps - 1
        
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # n_value_bins of different ratio
        self.action_n_bins = int(config.action_strategy.n_value_bins)
        self.action_space = spaces.Discrete(2 * self.action_n_bins + 1)

        # Prices contains the OHCL values for the last data_n_timesteps prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, self.data_n_timesteps), dtype=np.float16)


    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
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
                        , 'Volume'].values / MAX_NUM_SHARES,
        ])

        return obs

    def _take_action(self, action):
        """
        There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.
        
        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold.
        """
        # TODO: test impact of different orders of actions
        
        # TODO: consider setting this to closing price instead
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        if action < self.action_n_bins:  # Buy
            total_possible = int(self.balance / current_price)
            ratio = 1/(action + 2)  # +1 for 0 first of array, and +1 for 1/2 as max ratio
            shares_bought = int(total_possible * ratio)
           
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action >= self.action_n_bins and action < 2 * self.action_n_bins:  # Sell
            ratio = 1/((action-self.action_n_bins) + 2)
            shares_sold = int(self.shares_held * ratio)

            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

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

        # TODO: should be df index?
        self.current_step += 1

        #TODO: handle timeline properly
        if self.current_step > self.df.index.max():
            self.current_step = self.data_n_indexsteps

        reward = self._get_reward()
        
        # DoD: if we don't have any money, we can't trade
        # TODO: add time series has run out in DoD (see RLTrader for example)
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

        # Set the starting step to a random point within the data frame
        self.current_step = random.randint(
            self.data_n_indexsteps, self.df.index.max())

        return self._next_observation()


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')