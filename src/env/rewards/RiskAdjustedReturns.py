import pandas as pd
import numpy as np

from typing import Callable

from src.util.config import get_config
config = get_config()


class RiskAdjustedReturns():
    """A reward scheme that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    """

    def __init__(self,
                 return_algorithm: str = 'sortino',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sortino'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """

        self._return_algorithm = self._return_algorithm_from_str(return_algorithm)
        self._risk_free_rate = risk_free_rate
        self._target_returns = target_returns
        self._window_size = config.reward.calculation_timesteps

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        assert algorithm_str in ['sharpe', 'sortino']

        if algorithm_str == 'sharpe':
            return self._sharpe_ratio
        elif algorithm_str == 'sortino':
            return self._sortino_ratio

    def _sharpe_ratio(self, returns: np.array) -> float:
        """Return the sharpe ratio for a given series of a returns.
        References:
            - https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1E-9) / (np.std(returns) + 1E-9)

    def _sortino_ratio(self, returns: np.array) -> float:
        """Return the sortino ratio for a given series of a returns.
        References:
            - https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = np.where(returns < self._target_returns, returns**2, returns)

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate) / (downside_std + 1E-9)

    def get_reward(self, returns: list) -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        risk_adjusted_return = self._return_algorithm(returns=np.array(returns[-self._window_size:]))

        return risk_adjusted_return

    def get_reward_range(self):
        return (-100000, 100000)
