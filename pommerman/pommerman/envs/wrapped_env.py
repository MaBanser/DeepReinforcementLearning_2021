import gym
import numpy as np
import pommerman
from pommerman import constants

class WrappedEnv(gym.Wrapper):
    def __init__(self, agent_list, train_agent_pos, env_config="PommeFFACompetition-v0"):
        self.env = pommerman.make(env_config, agent_list)
        super(WrappedEnv, self).__init__(self.env)
        self.env.set_training_agent(train_agent_pos)

        # Variables for reward shaping
        self.agent = self.env._agents[train_agent_pos]
        self.ammo = self.agent.ammo
        self.max_ammo = self.agent.ammo
        self.blast_strength = self.agent.blast_strength
        self.can_kick = self.agent.can_kick
        self.num_wood = 0
        self.num_enem = 0
        self.pos_queue = []

        # Used for printing information while debugging
        self.step_cnt=0
        self.action_dicts = {
            0:'Stop',
            1:'Up',
            2:'Down',
            3:'Left',
            4:'Right',
            5:'Bomb',
        }

    def step(self, action):
        self.step_cnt+=1
        print(self.step_cnt,':',self.action_dicts[action])
        # Get current state for all agents
        obs = self.env.get_observations()
        # Get the actions of the other agents
        all_actions = self.env.act(obs)
        # Insert the train agents action into all actions
        all_actions.insert(self.env.training_agent, action)
        # Let the environment run for one step
        state, reward, done, info = self.env.step(all_actions)
        # Extract the state and reward of the train agent
        agent_state = state[self.env.training_agent]
        agent_reward = reward[self.env.training_agent]

        # Reward shaping

        # Reward for exploring new positions
        if agent_state['position'] not in self.pos_queue:
            agent_reward += 0.001
            self.pos_queue.append(agent_state['position'])

        # Reward for placing a bomb
        # if action == 5 and self.ammo != 0:
        #     agent_reward += 0.002

        # Reward for killing enemy
        num_enem = len(agent_state['alive']) - 1
        #if self.num_enem > num_enem > 0:
        #    agent_reward += 0.2

        # Reward for destroying wood
        num_wood = np.sum(agent_state['board'] == constants.Item.Wood.value)
        #if self.num_wood > num_wood:
        #    agent_reward += 0.01
        
        # Bonus for getting items
        if self.max_ammo < self.agent.ammo:
            self.max_ammo = self.agent.ammo
            agent_reward += 0.02

        if self.blast_strength < self.agent.blast_strength:
            agent_reward += 0.02

        if self.can_kick != self.agent.can_kick:
            agent_reward += 0.05

        self.ammo = self.agent.ammo
        self.blast_strength = self.agent.blast_strength
        self.can_kick = self.agent.can_kick
        self.num_wood = num_wood
        self.num_enem = num_enem
        
        return agent_state, agent_reward, done, info
    
    # Reset the environment only return observation of training agent
    def reset(self):
        all_obs = self.env.reset()
        self.step_cnt=0
        self.pos_queue = [all_obs[self.env.training_agent]['position']]
        self.ammo = self.agent.ammo
        self.max_ammo = self.agent.ammo
        self.blast_strength = self.agent.blast_strength
        self.can_kick = self.agent.can_kick
        return all_obs[self.env.training_agent]