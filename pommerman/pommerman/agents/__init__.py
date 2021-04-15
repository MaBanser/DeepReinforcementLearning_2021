'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent

from .actor_critic_model import ActorCritic
from .actor_critic_agent import ActorCriticAgent
from .my_agent import MyAgent
