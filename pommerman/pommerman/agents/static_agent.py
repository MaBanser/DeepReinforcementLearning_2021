from pommerman.agents import BaseAgent

class StaticAgent(BaseAgent):
    def __init__(self):
        super(StaticAgent, self).__init__()

    def act(self, obs, action_space):
        return 0
