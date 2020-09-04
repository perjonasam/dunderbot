import json
import datetime as dt
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines import PPO2, A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import set_global_seeds

import numpy as np
import pandas as pd
from src.env.DunderBotEnv import DunderBotEnv

from src.util.config import get_config
config = get_config()


def setup_env(*, df):
    # The algorithms require a vectorized environment to run
    env = DunderBotEnv(df=df, train_predict='train')
    # check env is designed correctly, to catch some errors and bugs
    check_env(env)

    # Wrappers: Normalize observations and reards for more efficient learning, and check for nan and inf.
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=20)
    env = VecCheckNan(env, raise_exception=True, check_inf=True)
    return env


def _save(*, env, model, save_dir):
    print(f'RUN: Saving files to {save_dir}')
    model.save(save_dir + "PPO2")
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    env.save(stats_path)


def _load(*, df, train_predict, save_dir):
    print(f'RUN: Loading files from {save_dir}')
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    # Load the agent
    model = PPO2.load(save_dir + "PPO2")

    env = DunderBotEnv(df=df, train_predict=train_predict)
    env = DummyVecEnv([lambda: env])
    # Load the saved statistics from normalization
    env = VecNormalize.load(stats_path, env)
    env = VecCheckNan(env, raise_exception=True, check_inf=True)

    # Connect the environment with the model
    model.set_env(env)
    print(f'RUN: Model connected with env')
    return env, model


def train(*, env, timesteps, save_dir="/tmp/"):
    print(f'RUN: Training for {timesteps} timesteps...')
    policy = config.policy.network
    # NOTE: setting ent_coef to 0 to avoid unstable model during training. Subject to change.
    model = PPO2(policy, env, tensorboard_log=config.monitoring.tensorboard.folder, verbose=1, ent_coef=0, seed=config.random_seed)
    model.learn(total_timesteps=timesteps, log_interval=10)
    # Save model and env
    _save(env=env, model=model, save_dir=save_dir)
    print(f'Done.')
    return model


def predict(*, df, timesteps, save_dir="/tmp/", rendermode='human'):
    # Load model anf env stats from file
    env, model = _load(df=df, train_predict='predict', save_dir=save_dir)
    env.training = False
    env.norm_reward = False
    
    print(f'RUN: Predicting for {timesteps} timesteps')
    obs = env.reset()
    done = False
    for i in range(timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            print(f'Env done, loop {i+1}')
            break
    env.render(mode=rendermode)


