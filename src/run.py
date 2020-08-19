import json
import datetime as dt
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

import numpy as np
import pandas as pd
import gym
from src.env.DunderBotEnv import DunderBotEnv

from src.util.config import get_config
config = get_config()

def preprocess(*, df):
    # The algorithms require a vectorized environment to run
    env = DunderBotEnv(df=df, train_test='train')
    # check env is designed correctly
    #check_env(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=np.inf)
    return env


def _save(*, env, model, save_dir):
    print(f'Saving files to {save_dir}')
    model.save(save_dir + "PPO2")
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    env.save(stats_path)


def _load(*, df, train_test, save_dir):
    print(f'Loading files from {save_dir}')
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    # Load the agent
    model = PPO2.load(save_dir + "PPO2")
    
    env = DunderBotEnv(df=df, train_test=train_test)
    # Load the saved statistics
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats_path, env)
    model.set_env(env)
    print(f'Model connected with env')
    return env, model


def train(*, env, timesteps, save_dir="/tmp/"):
    model = PPO2(MlpPolicy, env, tensorboard_log=config.monitoring.tensorboard.folder, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=10)
    # Save model and env
    _save(env=env, model=model, save_dir=save_dir)
    return model


def predict(*, df, timesteps, save_dir="/tmp/", rendermode='human'):
    # Load model anf env stats from file
    env, model = _load(df=df, train_test='test', save_dir=save_dir)
    env.training = False
    env.norm_reward = False
    
    print(f'Predicting for {timesteps} timesteps')
    obs = env.reset()
    done = False
    obs_all=[]
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        obs_all.append(obs)
        if done:
            print(f'Env done, loop {i+1}')
            break
    
    env.render(mode=rendermode)
    return obs_all

