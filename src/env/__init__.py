from gym.envs.registration import register
register(id='dunderbot-v0',
    entry_point='envs.dunderbot_env.dunderbot_env:DunderBotEnv'
)