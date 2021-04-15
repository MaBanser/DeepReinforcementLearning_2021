import gym
import pommerman

class WrappedEnv(gym.Wrapper):
    def __init__(self, agent_list, train_agent_pos, env_config="PommeFFACompetition-v0"):
        self.env = pommerman.make(env_config, agent_list)
        super(WrappedEnv, self).__init__(self.env)
        self.env.set_training_agent(train_agent_pos)
        self.agent = self.env._agents[train_agent_pos]
        self.ammo = self.agent.ammo
        self.blast_strength = self.agent.blast_strength
        self.can_kick = self.agent.can_kick

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
        print(all_actions)
        # Insert the train agents action into all actions
        all_actions.insert(self.env.training_agent, action)
        # Let the environment run for one step
        state, reward, done, info = self.env.step(all_actions)
        # Extract the state and reward of the train agent
        agent_state = state[self.env.training_agent]
        agent_reward = reward[self.env.training_agent]

        # # Reward shaping
        # agent_reward *= 200
        # # +1 if alive
        # # agent_reward += self.agent.is_alive
        # # Bonus for getting items
        # if self.ammo < self.agent.ammo:
        #     agent_reward += 1
        #     self.ammo = self.agent.ammo

        # if self.blast_strength < self.agent.blast_strength:
        #     agent_reward += 1
        #     self.blast_strength = self.agent.blast_strength

        # if not self.can_kick == self.agent.can_kick:
        #     agent_reward += 1
        #     self.can_kick = self.agent.can_kick
        
        return agent_state, agent_reward, done, info
    
    # Reset the environment only return observation of training agent
    def reset(self):
        all_obs = self.env.reset()
        self.step_cnt=0
        self.ammo = self.agent.ammo
        self.blast_strength = self.agent.blast_strength
        self.can_kick = self.agent.can_kick
        return all_obs[self.env.training_agent]