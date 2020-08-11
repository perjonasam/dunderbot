

class BaseRewardStrategy(object):

    registered_name = "rewards"

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass


    def get_reward(self, net_worth: list) -> float:
        """
        Arguments:
            portfolio: The portfolio being used by the environment.
        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()
