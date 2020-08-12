import pandas as pd

class IncrementalNetWorth():
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size = 1):
        self.window_size = window_size

    def reset(self):
        pass

    def get_reward(self, net_worths: list) -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window.
        Args:
            net_worth: history of net_worth
        Returns:
            The cumulative percentage change in net worth over the previous `window_size` timesteps.
        """
        net_worths = pd.Series(net_worths)
        returns = net_worths.pct_change().dropna()
        returns = (1 + returns[-self.window_size:]).cumprod() - 1
        return 0 if len(returns) < 1 else returns.iloc[-1]
