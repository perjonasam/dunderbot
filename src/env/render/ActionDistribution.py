import pandas as pd
import matplotlib.pyplot as plt

class ActionDistribution:
    """An action/trade distribution visualization using matplotlib made to render gym environments"""

    def __init__(self, trades):
        self.trades = trades
    

    def render(self, figwidth=15):
        # Convert to pandas df and plot. Simple barplot with labels.
        df = pd.DataFrame(self.trades)
        df['action_ratio'] = df['action_ratio'].round(3)
        
        df_counts = df.groupby(['type', 'action_ratio']).size()
        df_perc = df_counts/len(df)*100
        ax = df_perc.plot.bar(figsize=(figwidth,figwidth/3), title='Trade ratio distribution in % of timesteps');
        ax.set_ylabel("% of timesteps")
        ax.set_xlabel("trade with ratio")

        plt.show()
