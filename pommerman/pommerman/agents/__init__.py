'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent

from .static_agent import StaticAgent
from .random_agent_no_bomb import RandomAgentNoBomb
from .random_agent_no_bomb import RandomAgentLimitNoBomb
from .random_agent_no_bomb import RandomAgentSlowNoBomb
from .random_agent_no_bomb import RandomAgentSmart
from .random_agent_no_bomb import RandomBomber
from .simple_agent_no_bomb import SimpleAgentNoBomb

from .actor_critic_model import ActorCritic
from .actor_critic_model_dense import ActorCriticDense
from .actor_critic_agent import ActorCriticAgent
from .actor_critic_agent_dense import ActorCriticAgentDense

from .my_agent import MyAgent
