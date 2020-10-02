import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from shutil import copyfile

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from src.env.DunderBotEnv import DunderBotEnv
from src.env.callback.custom_callback import CustomCallback

from src.util.run_util import retrieve_model_dir
from src.util.config import get_config
config = get_config()

# Filter tedious and uninformtive tensorflow version warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


def setup_env(*, df, record_steptime=False):
    """
    Environment is 1) set up, 2) parallelized (if n_cpu>1), 3) normalized and
    4) checked for NaN and for proper gym format.

    Arguments:
    - df: dataframe with olhc price data and timestamps
    - record_steptime: flag to render the time each step takes. Small, but noticable memory cost.
    """
    print(f'Setting up environment using {config.n_cpu} cores...')
    env = DunderBotEnv(df=df, train_predict='train', record_steptime=record_steptime)
    # check env is designed correctly, to catch some errors and bugs
    check_env(env)

    # Wrappers: Normalize observations and rewards for more efficient learning, and check for nan and inf.
    n_cpu = config.n_cpu
    if n_cpu == 1:
        env = DummyVecEnv([lambda: env])
    elif n_cpu > 1:
        # I benchmarked SubprocVecEnv to be notably faster than DummyVecEnv for equal cores
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
        # Give each process its own seed for robuster results
        env.seed(seed=int(config.random_seed))
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=20)
    env = VecCheckNan(env, raise_exception=True, check_inf=True)
    print(f'Done.')
    return env


def _save(*, env, model):
    """ Save model, but also some helpful meta data. The highest integer in folder structure
    is incremented and used."""
    model_dir = retrieve_model_dir(first=True)
    print(f'RUN: Saving files to {model_dir}')
    model.save(os.path.join(model_dir, "PPO2"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    Path(os.path.join(model_dir, datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.utc')).touch()
    copyfile('./config/config.yml', os.path.join(model_dir, 'config.yml'))


def _load(*, df, train_predict, model_dir):
    """ Load the either the specifed model or the latest (as given by folder increment),
    and set up environment. Prediction is always performed on single-core. Note that the
    environment is not saved and loaded, so to recreate any previous experiment, the env
    must be identical."""

    print(f'RUN: Loading files from {model_dir}')
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")
    # Load the agent
    model = PPO2.load(os.path.join(model_dir, "PPO2"))

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


def train(*, env, serial_timesteps=None, n_infoboxes=10):
    """
    Train the model that converts observations to actions.
    Arguments:
    - env: the custom gym environment we want to use
    - serial_timesteps: the number of timesteps stepped through in data during training. More on different timesteps in README.md.
    - n_infoboxes: the approximate number of infoboxes with training information that is printed

    A custom callback is one way to extract information during training, and can be specified here.
    """
    if serial_timesteps is None:
        serial_timesteps = config.train_predict.train_timesteps

    total_timesteps = serial_timesteps * config.n_cpu
    print(f'RUN: Training for {serial_timesteps} serial timesteps and {total_timesteps} total timesteps...')
    callback = None  # [None, CustomCallback()]
    tensorboard_log = config.monitoring.tensorboard.folder

    # screen log verbosity
    verbose = 1 if n_infoboxes > 0 else 0
    log_interval = 1 if n_infoboxes == 0 else round(serial_timesteps/128/n_infoboxes)  # per n updates

    policy = config.policy.network
    model = PPO2(policy, env,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                seed=config.random_seed)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callback)
    # Save model and env
    _save(env=env, model=model)
    print(f'Done.')
    return model


def predict(*, df, timesteps=None, rendermode='plots', model_dir=None):
    """
    Predicts using a trained model. This is performed on single core, and the normalization stats
    are not updated.
    Arguments:
    - df: the df with data. Be careful not to use data that has been trained on (although requires tinkering to achieve)
    - timesteps: Since single core, serial == total timesteps. If None, config number is used.
    - rendermode: whether to render and print plots, some stats, or nothing.
    """
    if timesteps is None:
        timesteps = config.train_predict.predict_timesteps - config.data_params.data_n_timesteps - 1
    else:
        timesteps_config = config.train_predict.predict_timesteps - config.data_params.data_n_timesteps - 1
        assert timesteps <= timesteps_config, \
            f'Number of predict timesteps requested larger than in config ({timesteps} > {timesteps_config})'

    # Load model and env stats from file
    if model_dir is None:
        model_dir = retrieve_model_dir(first=False)
    env, model = _load(df=df, train_predict='predict', model_dir=model_dir)
    env.training = False
    env.norm_reward = False

    print(f'RUN: Predicting for {timesteps} timesteps')
    obs = env.reset()
    done = False
    env.set_attr('save_dir', model_dir)
    for i in range(timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            print(f'Env done, loop {i+1}')
            break
    env.render(mode=rendermode)
