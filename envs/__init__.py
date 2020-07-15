from gym.envs.registration import register
register(id='mlfin-v1',
    entry_point='envs.mlfin_env.mlfin_env:MLFinEnv'
)