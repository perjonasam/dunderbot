import pickle
import gym
from datetime import datetime
from gym import spaces
import pandas as pd
import numpy as np
from stable_baselines.common import set_global_seeds
from src.env.render import TradingChartStatic, ActionDistribution, RewardDevelopment
from src.env.trade.TradeStrategy import TradeStrategy
from src.env.rewards import RiskAdjustedReturns, IncrementalNetWorth

from src.util.config import get_config
config = get_config()


class DunderBotEnv(gym.Env):
    """The Dunderbot class"""
    metadata = {'render.modes': ['plots', 'system', 'none']}
    viewer = None

    def __init__(self, df, train_predict, record_steptime=False):
        super(DunderBotEnv, self).__init__()

        self.timer = []
        self.df = df
        self.record_steptime = record_steptime
        # -1 due to inclusive slicing and 0-indexing
        self.data_n_timesteps = int(config.data_params.data_n_timesteps)
        self.data_n_indexsteps = self.data_n_timesteps - 1

        set_global_seeds(config.random_seed)

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
        # TODO: review values
        self.base_precision = 3
        self.asset_precision = 8
        self.min_cost_limit = 1E-3
        self.min_amount_limit = 1E-3

        self.trade_strategy = TradeStrategy(base_precision=self.base_precision,
                                            asset_precision=self.asset_precision,
                                            min_cost_limit=self.min_cost_limit,
                                            min_amount_limit=self.min_amount_limit)

        # Set Reward Strategy
        self.reward_strategy = eval(config.reward.strategy)
        self.reward_range = self.reward_strategy.get_reward_range()

        # Are we training or predicting? Decides starting timestep.
        assert train_predict in ['train', 'predict'], f'Train_predict can only be `train` or `predict`.'
        self.train_predict = train_predict

        # Calculate train/predict breaking point from settings in config
        self.traintest_breaking_point_timestep = df.index.max() - int(config.train_predict.predict_timesteps)
        self.n_cpu = config.n_cpu

    def _next_observation(self):
        # Get the price+volume data points for the last data_n_indexsteps days and scale to between 0-1
        obs = self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'Close'].to_list()
        temp = self.df.loc[self.current_step - self.data_n_indexsteps: self.current_step
                        , 'VolumeBTC'].to_list()
        obs.extend(temp)

        # Non-price/volume features
        # NOTE: if changed, don't forget to set n_features in config
        obs.extend([self.balance,
            self.net_worths[-1],
            self.asset_held])

        # Technical indicator features (supposedly pd bug: dtype becomes object, enforcing float)
        ti_cols = [col for col in self.df.columns if 'ti_' in col]
        ti_features = self.df.loc[self.current_step, ti_cols].astype(float).to_list()
        obs.extend(ti_features)

        obs = np.array(obs)

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

    def _take_action(self, action):
        """
        There are n_value_bins (=self.action_n_bins) actions per buy and sell, marking a range of ratios of possible assets to buy and sell,
        according to 1/(bin_value+1) since we want a buy/sell ratio of max 1/2. Hold uses only one action.
        
        First n_value_bins actions are buy, next n_value_bins sell, and lastly single hold.

        Saving info during prediction for rendering (similar info is logged using tensorboard for training)
        """
        # Set the current price to a random price within the time step
        self.current_price = self.df.loc[self.current_step, "Close"]

        action_type, action_amount = self.translate_action(action)

        assets_bought, assets_sold, purchase_cost, sale_revenue = self.trade_strategy.trade(action_type=action_type,
                                                                                            action_amount=action_amount,
                                                                                            balance=self.balance,
                                                                                            asset_held=self.asset_held,
                                                                                            current_price=self.current_price)

        if assets_bought:
            self.asset_held += assets_bought
            self.balance -= purchase_cost

            if self.train_predict == 'predict':
                self.trades.append({'step': self.current_step,
                                    'amount': assets_bought,
                                    'total': purchase_cost,
                                    'type': 'buy',
                                    'action_amount': action_amount})
        elif assets_sold:
            self.asset_held -= assets_sold
            self.balance += sale_revenue

            if self.train_predict == 'predict':
                self.trades.append({'step': self.current_step,
                                    'amount': assets_sold,
                                    'total': sale_revenue,
                                    'type': 'sell',
                                    'action_amount': action_amount})
        elif action_type == 'hold':
            if self.train_predict == 'predict':
                self.trades.append({'step': self.current_step,
                                'amount': None,
                                'total': None,
                                'type': 'hold',
                                'action_amount': None})

        current_net_worth = round(self.balance + self.asset_held * self.current_price, self.base_precision)
        current_return = (current_net_worth-self.net_worths[-1])/(self.net_worths[-1]+1E-6)
        self.returns.append(current_return)
        self.net_worths.append(current_net_worth)

        if self.train_predict == 'predict':
            self.account_history.append(pd.DataFrame([{
                'balance': self.balance,
                'asset_held': self.asset_held,
                'asset_bought': assets_bought,
                'purchase_cost': purchase_cost,
                'asset_sold': assets_sold,
                'sale_revenue': sale_revenue,
            }]))

    def _reward(self):
        reward = self.reward_strategy.get_reward(returns=self.returns)

        reward = float(reward) if np.isfinite(float(reward)) else 0

        if self.train_predict == 'predict':
            self.rewards.append(reward)

        return reward

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

        # Timer for training slowdown analysis
        if self.record_steptime:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.timer.append(now)
        return obs, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """

        self.balance = config.trading_params.initial_account_balance
        self.asset_held = 0
        self.trades = []
        self.net_worths = [config.trading_params.initial_account_balance]
        self.asset_held_hist = [0.0]
        self.rewards = []
        self.returns = []
        self.save_dir = ''

        # TODO: why are we adding one timestep before anything?
        self.account_history = []
        self.account_history.append(pd.DataFrame([{
            'balance': self.balance,
            'asset_held': self.asset_held,
            'asset_bought': 0,
            'purchase_cost': 0,
            'asset_sold': 0,
            'sale_revenue': 0,
        }]))

        # Set starting and ending time steps (some calculation in __init__)
        # TODO: move n_steps to config
        self.train_timesteps = int(config.train_predict.train_timesteps)//128*128  # 128 is default n_steps
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
        print(f'Resetting to timesteps: start {self.start_step} ({self.df.loc[self.start_step]["Timestamp"]}), end {self.end_step} ({self.df.loc[self.end_step]["Timestamp"]})')

        return self._next_observation()

    def render(self, mode='plots'):
        # Render the environment to the screen
        account_history_df = pd.concat(self.account_history)
        # TODO: when plot is good and has stabilized, rm this
        all_dict = {'current_step': self.current_step,
                    'net_worths': self.net_worths,
                    'trades': self.trades,
                    'returns': self.returns,
                    'account_history': account_history_df,
                    'rewards': self.rewards}
        with open('all_dict_pred.pickle', 'wb') as handle:
            pickle.dump(all_dict, handle)

        if mode == 'system':
            print('Price: ' + str(self.current_price))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'plots':
            # Render static TradingChart
            print(f'Rendering TradingChartStatic for index steps {self.start_step} ({self.df.loc[self.start_step]["Timestamp"]}) through {self.current_step} ({self.df.loc[self.current_step]["Timestamp"]})')
            self.viewer = TradingChartStatic(df=self.df,
                                            start_step=self.start_step,
                                            end_step=self.current_step)

            self.viewer.render(net_worths=self.net_worths,
                            trades=self.trades,
                            account_history=account_history_df,
                            save_dir=self.save_dir)

            # Render action distribution
            self.viewer = ActionDistribution(trades=self.trades)
            self.viewer.render(save_dir=self.save_dir)

            # Render reward output
            self.viewer = RewardDevelopment(rewards=self.rewards)
            self.viewer.render(save_dir=self.save_dir)
