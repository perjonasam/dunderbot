

class BaseRewardStrategy(object):

    registered_name = "rewards"

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass


    def get_reward(self, net_worth: list) -> float:
        """
        Arguments:
            net_worth: history of net_worth
        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()
