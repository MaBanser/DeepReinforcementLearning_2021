from pommerman.agents import ActorCriticAgent
from pommerman.agents import BaseAgent
import tensorflow as tf
import os

# Loads AC_agent and weights
class MyAgent(BaseAgent):
    def __init__(self):
        super(MyAgent, self).__init__()
        self.agent = ActorCriticAgent()
        # Load the model
        self.model_path = self.get_model_path(path='./././trained_model')
        model = tf.keras.models.load_model(self.model_path)
        self.agent.set_model(model)
    
    def act(self, obs, action_space):
        return self.agent.act(obs, action_space)
    
    def episode_end(self, reward):
        self.agent.episode_end(reward)

    def get_model_path(self, path='.'):
        result = []
        for folder in os.listdir(path):
            bd = os.path.join(path, folder)
            if os.path.isdir(bd):
                result.append(bd)                
        result = max(result, key=os.path.getmtime)
        return result
