from src.util.config import get_config
config = get_config()

class RewardSchema():
    """ Contains all rewards, each reward as a method """
    def __init__(self, balance, current_step, MAX_STEPS):
        self.balance = balance
        self.current_step = current_step
        self.MAX_STEPS = MAX_STEPS


    def cumulative_reward(self):
        """ Poor cumulative reward fetched from https://github.com/notadamking/Stock-Trading-Environment """
        delay_modifier = (self.current_step / self.MAX_STEPS)
        cumulative_reward = self.balance * delay_modifier
        return cumulative_reward
    

    def get_reward_strategy(self):
        """ Fetch the reward specified in config. Fallback to simplest solution """
        if config.reward_strategy.cumulative_reward:
            reward = self.cumulative_reward()
        else:
            reward = self.cumulative_reward()
        return reward
