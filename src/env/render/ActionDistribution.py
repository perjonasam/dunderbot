import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

from src.util.config import get_config
config = get_config()


class ActionDistribution:
    """An action/trade distribution visualization using matplotlib made to render gym environments"""

    def __init__(self, trades):
        self.trades = trades

    def render(self, save_dir, figwidth=15):
        # Convert to pandas df and plot. Simple barplot with labels.
        df = pd.DataFrame(self.trades)
        df = df.fillna(0)  # Due to hold having some NaN fields
        df['action_amount'] = df['action_amount'].round(3)

        # Calculate 30-day overturn
        overturn_30 = df['action_amount'].sum()/(df.iloc[-1]['timestamp']-df.iloc[0]['timestamp']).total_seconds()*86400*30

        df_counts = df.groupby(['type', 'action_amount']).size()
        df_perc = df_counts/len(df)*100
        ax = df_perc.plot.bar(figsize=(figwidth, figwidth/3), title=f'Trade ratio distribution in % of {len(df)} trades | 30-day overturn: {round(overturn_30)} {config.input_data.asset[3:]}')
        for p in ax.patches:
            ax.annotate(str(round(p.get_height()*len(df)/100)), (p.get_x(), p.get_height() * 0.95))
        ax.set_ylabel("% of timesteps")
        ax.set_xlabel("trade with ratio")

        # Save to file
        fig = ax.get_figure()
        nowtime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        fig.savefig(os.path.join(save_dir, f'ActionDistribution_{nowtime}.pdf'))

        plt.show()

    def close(self):
        plt.close()
