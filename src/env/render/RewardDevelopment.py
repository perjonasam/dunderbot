import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt


class RewardDevelopment:
    """An action/trade distribution visualization using matplotlib made to render gym environments"""

    def __init__(self, rewards):
        self.rewards = rewards

    def render(self, save_dir, figwidth=15):
        # Convert to pandas df and plot. Simple lineplot with labels.
        rewards = pd.DataFrame(self.rewards, columns=['reward'])

        rewards['cumulative_reward'] = rewards.cumsum()

        ax = rewards.plot(figsize=(figwidth, figwidth/3), title='Reward development')
        ax.set_ylabel("Reward")
        ax.set_xlabel("Timestep")

        # Save to file
        fig = ax.get_figure()
        nowtime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        fig.savefig(os.path.join(save_dir, f'RewardDevelopment_{nowtime}.pdf'))

        plt.show()

    def close(self):
        plt.close()
