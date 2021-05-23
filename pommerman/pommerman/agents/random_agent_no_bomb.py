'''An agent that preforms a random action each step'''
from . import BaseAgent
import random
from .action_prune import get_filtered_actions

class RandomAgentNoBomb(BaseAgent):
    """The Random Agent that returns random actions(except placing bomb) given an action_space."""

    def act(self, obs, action_space):
        return random.randint(0,4)

class RandomAgentLimitNoBomb(BaseAgent):
    """The Random Agent that at the beginning returns random actions(except placing bomb) given an action_space.
    Later it will become static."""
    def __init__(self):
        super(RandomAgentLimitNoBomb, self).__init__()
        self.step_count=0

    def act(self, obs, action_space):
        if self.step_count < 5:
            self.step_count += 1
            return random.randint(1,4)
        return 0

    def episode_end(self, reward):
        self.step_count = 0

class RandomAgentSlowNoBomb(BaseAgent):
    """The Random Agent that every 15 steps returns random actions(except placing bomb or stop) given an action_space."""
    def __init__(self):
        super(RandomAgentSlowNoBomb, self).__init__()
        self.step_count=15

    def act(self, obs, action_space):
        if self.step_count == 15:
            self.step_count=0
            return random.randint(1,4)        
        self.step_count += 1
        return 0

    def episode_end(self, reward):
        self.step_count = 0

class RandomAgentSmart(BaseAgent):
    """The Random Agent that returns random actions from the filtered actions."""
    def __init__(self):
        super(RandomAgentSmart, self).__init__()
        self.prev_obs = [None,None]

    def act(self, obs, action_space):
        filtered_actions = get_filtered_actions(obs,(self.prev_obs[-2],self.prev_obs[-1]))
        self.prev_obs.append(obs)

        return random.choice(filtered_actions)

class RandomBomber(BaseAgent):
    """The Random Agent that returns random actions from the filtered actions, prefering bombs."""
    def __init__(self):
        super(RandomBomber, self).__init__()
        self.prev_obs = [None,None]

    def act(self, obs, action_space):
        filtered_actions = get_filtered_actions(obs,(self.prev_obs[-2],self.prev_obs[-1]))
        self.prev_obs.append(obs)
        if 5 in filtered_actions:
            return 5
        
        return random.choice(filtered_actions)