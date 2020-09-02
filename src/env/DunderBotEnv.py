import pickle
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from stable_baselines.common import set_global_seeds
from src.env.render import TradingChartStatic, ActionDistribution, RewardDevelopment
from src.env.trade.TradeStrategy import TradeStrategy
from src.env.rewards import IncrementalNetWorth, RiskAdjustedReturns

from src.util.config import get_config
config = get_config()

INITIAL_ACCOUNT_BALANCE = 10000.0
class DunderBotEnv(gym.Env):
    """The Dunderbot class"""
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, train_predict):
        super(DunderBotEnv, self).__init__()

        self.df = df
        # -1 due to inclusive slicing and 0-indexing
        self.data_n_timesteps = int(config.data_params.data_n_timesteps)
        self.data_n_indexsteps = self.data_n_timesteps - 1

        set_global_seeds(config.random_seed)
        
        self.reward_range = (0, 2147483647)

        # actions: buy/sell n_value_bins of different ratio of balance/assets held + hold
        self.action_n_bins = int(config.action_strategy.n_value_bins)
        self.action_space = spaces.Discrete(2 * self.action_n_bins + 1)

        # Observations are price and volume data the last data_n_timesteps, portfolio features, and ti features
        # Exploit ti_ prefix to locate ti features
        n_ti_features = len([col for col in self.df.columns if 'ti_' in col])
        self.obs_array_length = self.data_n_timesteps*2 + int(config.data_params.n_portfolio_features) + n_ti_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_array_length,), dtype=np.float16)

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
        #TODO: move this choice to config
        self.reward_strategy = IncrementalNetWorth()

        # Are we training or predicting? Decides starting timestep.
        assert train_predict in ['train', 'predict'], f'Train_predict can only be `train` or `predict`.'
        self.train_predict = train_predict

        # Calculate train/predict breaking point from some setting in config
        self.traintest_breaking_point_timestep = df.index.max() - int(config.train_predict.predict_timesteps)


    def _next_observation(self):
        # Get the price+volume data points for the last data_n_indexsteps days and scale to between 0-1
        obs = self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Close'].values
        obs = np.append(obs,
            self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'VolumeBTC'].values
        )

        # Non-price/volume features
        # NOTE: if changed, don't forget to set n_features in config
        obs = np.append(obs, 
            [self.balance,
            self.net_worths[-1],
            self.asset_held]
        )

        # Technical indicator features (supposedly pd bug: dtype becomes object, enforcing float)
        ti_cols = [col for col in self.df.columns if 'ti_' in col]
        ti_features = self.df.loc[self.current_step, ti_cols].astype(float)
        obs = np.append(obs,
                        ti_features)

        # Tests
        assert not np.isnan(np.sum(obs)), f'Observation contains nan'
        assert not np.isinf(obs).any(), f'Observation contains inf'
        assert len(obs) == self.obs_array_length, \
            f'Actual obs array is {len(obs)} long, but specified to be {self.obs_array_length}'

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


    def _reward(self):
        reward = self.reward_strategy.get_reward(net_worths=self.net_worths)

        reward = float(reward) if np.isfinite(float(reward)) else 0

        self.rewards.append(reward)

        # TODO: evaluate if we should stationarize_rewards i addition to normalizing.
        #if self.stationarize_rewards:
        #    rewards = difference(self.rewards, inplace=False)
        #else:
        rewards = self.rewards

        rewards = np.array(rewards).flatten()

        return float(rewards[-1])


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        # Start over again when data is out for the train case (test case will set done to True when data is out)
        # TODO: consider ending the episode for this scenario
        if self.train_predict == 'train' and self.current_step >= self.traintest_breaking_point_timestep:
            print(f'Resetting current step to start_step ({self.start_step}) in train env')
            self.current_step = self.start_step
        
        reward = self._reward()
        
        done = False
        # DoD1: if we don't have any money, we can't trade
        done = self.net_worths[-1] <= 0
        # DoD2: When data is out during prediction, halt.
        if self.train_predict == 'predict':
            done = self.current_step >= self.df.index.max()
        if done:
            print(f'Env calls done')
        done = bool(done)

        # Next observation
        obs = self._next_observation()

        return obs, reward, done, {}


    def reset(self):
        """ 
        Reset the state of the environment to an initial state 
        """
        
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.asset_held = 0
        self.trades = []
        self.net_worths = [INITIAL_ACCOUNT_BALANCE]
        self.asset_held_hist = [0.0]
        self.rewards = [0]

        self.account_history = pd.DataFrame([{
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': 0,
            'purchase_cost': 0,
            'asset_sold': 0,
            'sale_revenue': 0,
        }] )

        # Set starting and ending time steps (some calculation in __init__)
        self.train_timesteps = int(config.train_predict.train_timesteps)
        if self.train_predict == 'train':
            # Starting step cannot be smaller than set by data avilability. In addition, offset by max TI data lag 
            # (to most easily avoid NaNs from lagging TIs). 
            calculated_min = self.traintest_breaking_point_timestep - self.train_timesteps
            self.start_step = max(calculated_min, self.df.index.min()) + config.data_params.ti_nan_timesteps
            self.end_step = self.traintest_breaking_point_timestep
        elif self.train_predict == 'predict':
            self.start_step = self.traintest_breaking_point_timestep + self.data_n_timesteps
            self.end_step = self.df.index.max()
        
        self.current_step = self.start_step
        print(f'Resetting to timesteps: start {self.start_step}, end {self.end_step}.')
        
        return self._next_observation()


    def render(self, mode='human'):
        # Render the environment to the screen
        # TODO: when plot is good and has stabilized, rm this
        all_dict={  'current_step': self.current_step,
                    'net_worths': self.net_worths,
                    'trades': self.trades,
                    'account_history': self.account_history,
                    'rewards': self.rewards}
        with open('all_dict_pred.pickle', 'wb') as handle:
            pickle.dump(all_dict, handle)

        if mode == 'system':
            print('Price: ' + str(self.current_price))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            # Render static TradingChart
            print(f'Rendering TradingChartStatic for index steps {self.start_step} through {self.current_step}')
            self.viewer = TradingChartStatic(self.df, 
                                            self.start_step, 
                                            self.current_step)
            
            self.viewer.render(self.net_worths,
                            self.trades,
                            self.account_history)

            # Render action distribution
            self.viewer = ActionDistribution(self.trades)
            self.viewer.render()

            # Render reward output
            self.viewer = RewardDevelopment(self.rewards)
            self.viewer.render()