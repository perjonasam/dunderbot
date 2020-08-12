import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import pandas as pd
from src.env.dunderbot_env import DunderBotEnv

from src.util.config import get_config
config = get_config()

def preprocess(*, df):
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: DunderBotEnv(df)])
    return env


def train(*, env, total_timesteps=20000):
    model = PPO2(MlpPolicy, env, tensorboard_log=config.monitoring.tensorboard.folder, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    return model


def predict(*, env, model, total_timesteps=2000, rendermode='human'):
    obs = env.reset()
    for _ in range(total_timesteps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
            
    env.render(mode=rendermode)
