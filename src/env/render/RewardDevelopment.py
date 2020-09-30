import pandas as pd
import matplotlib.pyplot as plt


class RewardDevelopment:
    """An action/trade distribution visualization using matplotlib made to render gym environments"""

    def __init__(self, rewards):
        self.rewards = rewards

    def render(self, figwidth=15):
        # Convert to pandas df and plot. Simple lineplot with labels.
        rewards = pd.DataFrame(self.rewards, columns=['reward'])

        rewards['cumulative_reward'] = rewards.cumsum()

        ax = rewards.plot(figsize=(figwidth, figwidth/3), title='Reward development')
        ax.set_ylabel("Reward")
        ax.set_xlabel("Timestep")
        plt.show()
