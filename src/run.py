import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import pandas as pd
from src.env.DunderBotEnv import DunderBotEnv

from src.util.config import get_config
config = get_config()

def preprocess(*, df):
    # The algorithms require a vectorized environment to run
    env = DunderBotEnv(df=df, train_test='train')
    #env = TrainTestWrapper(env, max_steps=100, train_test='train')
    env = DummyVecEnv([lambda: env])
    return env


def train(*, env, timesteps=20000):
    model = PPO2(MlpPolicy, env, tensorboard_log=config.monitoring.tensorboard.folder, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    return model


def predict(*, df, model, timesteps, rendermode='human'):
    # Same env as above, but with potentially different train_test setting
    env = DunderBotEnv(df=df, train_test='test')
    env = DummyVecEnv([lambda: env])
    obs = env.reset()
    for _ in range(timesteps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
            
    env.render(mode=rendermode)
