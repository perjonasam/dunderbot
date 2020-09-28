import os
import numpy as np
import pandas as pd

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from src.env.DebugEnv import DebugEnv
from src.env.callback.custom_callback import CustomCallback

from src.util.config import get_config
config = get_config()

# Filter tensorflow version warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


def setup_env(*, df):
    print(f'Setting up environment using {config.n_cpu} cores...')
    env = DebugEnv(df=df)
    # check env is designed correctly, to catch some errors and bugs
    #check_env(env)

    # Wrappers: Normalize observations and rewards for more efficient learning, and check for nan and inf.
    n_cpu = config.n_cpu
    if n_cpu == 1:
        env = DummyVecEnv([lambda: env])
    elif n_cpu > 1:
        # I benchmarked SubprocVecEnv to be notably faster than DummyVecEnv for equal cores
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
        # Give each process its own seed for robuster results
        env.seed(seed=int(config.random_seed))
    #env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=20)
    #env = VecCheckNan(env, raise_exception=True, check_inf=True)
    print(f'Done.')
    return env


def _save(*, env, model, save_dir):
    print(f'RUN: Saving files to {save_dir}')
    #model.save(save_dir + "PPO2")
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    #env.save(stats_path)


def _load(*, df, train_predict, save_dir):
    print(f'RUN: Loading files from {save_dir}')
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
    # Load the agent
    model = PPO2.load(save_dir + "PPO2")

    env = DunderBotEnv(df=df, train_predict=train_predict)
    # Prediction is only done on single core
    env = DummyVecEnv([lambda: env])
    # Load the saved statistics from normalization
    env = VecNormalize.load(stats_path, env)
    env = VecCheckNan(env, raise_exception=True, check_inf=True)

    # Connect the environment with the model
    model.set_env(env)
    print(f'RUN: Model connected with env')
    return env, model


def train(*, env, serial_timesteps=None, logging=False, save_dir="/tmp/"):
    if serial_timesteps is None:
        serial_timesteps = config.train_predict.train_timesteps

    policy = config.policy.network
    # NOTE: setting ent_coef to 0 to avoid unstable model during training. Subject to change.
    total_timesteps = serial_timesteps * config.n_cpu
    print(f'RUN: Training for {serial_timesteps} serial timesteps and {total_timesteps} total timesteps...')
    callback = None  # Alts: [None, CustomCallback()]
    if logging:
        tensorboard_log = config.monitoring.tensorboard.folder
    else:
        tensorboard_log = None
    
    model = PPO2(policy, env,
                tensorboard_log=tensorboard_log,
                verbose=0,
                #ent_coef=0,
                seed=config.random_seed)
    model.learn(total_timesteps=total_timesteps, log_interval=serial_timesteps/10, callback=callback)
    # Save model and env
    #_save(env=env, model=model, save_dir=save_dir)
    print(f'Done.')
    return model


def predict(*, df, timesteps=None, save_dir="/tmp/", rendermode='plots'):
    if timesteps is None:
        timesteps = config.train_predict.predict_timesteps - config.data_params.data_n_timesteps - 1
    else:
        timesteps_config = config.train_predict.predict_timesteps - config.data_params.data_n_timesteps - 1
        assert timesteps <= timesteps_config, \
            f'Number of predict timesteps requested larger than in config ({timesteps} > {timesteps_config})'

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

