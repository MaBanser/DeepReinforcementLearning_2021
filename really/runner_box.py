import os

# only print error messages
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import gridworlds
import gym
import ray
from really.utils import discount_cumsum
from ray.tune.registry import register_env

@ray.remote
class RunnerBox:
    """
    Runner Box handling interaction between an instance of the Agent and an instance of the environment.

    @args:
        agent: Agent Object
        model: callable Model object
        environment: string specifying gym environment or class of Custom (gym-like) environment
        runner_position: int, index in list of remote runners
        returns: list of strings specifying what is to be returned by the box
                supported are: 'value_estimate', 'log_prob', 'monte_carlo'

    @kwargs:
        action_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continuous_normal_diagonal'
        temperature: float, temperature for thomson sampling, defaults to 1
        epsilon: epsilon for epsilon greedy sampling, defaults to 0.95
        weights: weights of the model, not needed if input_shape is given
        model_kwargs: dict, optional, model specificatins requried for initialization
        gamma: float, discount factor for monte carlo return, defaults to 0.99
        env_kwargs: dictionary, optional custom environment specifications
        input_shape: shape or boolean, if model needs input shape for initial call, defaults to the shape of the envs reset state
    """

    def __init__(
        self, agent, model, environment, runner_position, returns=[], **kwargs
        ):        
        
        if isinstance(environment, str) and kwargs['special_env']:
            exec(environment, globals())
            self.env = special_env
        else:
            self.env = environment
        kwargs.pop('special_env')

        self.runner_position = runner_position
        self.returns = returns

        self.return_log_prob = False
        self.return_value_estimate = False
        self.return_monte_carlo = False
        self.return_feature_state = False

        self.discrete_env = kwargs['discrete_env']
        kwargs.pop('discrete_env')

        # Initialize optional returns
        for key in self.returns:
            if key == "log_prob":
                self.return_log_prob = True
            if key == "value_estimate":
                self.return_value_estimate = True
                kwargs['value_estimate'] = True
            if key == "monte_carlo":
                self.return_monte_carlo = True
                if "gamma" in kwargs.keys():
                    self.gamma = kwargs["gamma"]
                    kwargs.pop("gamma")
                else:
                    self.gamma = 0.99
            if key == "feature_state":
                self.return_feature_state = True

        self.agent = agent(model, **kwargs)
        self.agent_kwargs = kwargs
        self.reset_data()

    def reset_data(self):
        # Initilize empty datasets aggregator
        self.data_agg = {}
        self.data_agg["action"] = []
        self.data_agg["state"] = []        
        self.data_agg["reward"] = []
        self.data_agg["state_new"] = []
        self.data_agg["not_done"] = []
        for r in self.returns:
            self.data_agg[r] = []

    def run_n_steps(self, num_steps, max_env=None):
        import tensorflow as tf

        if max_env is not None:
            self.env.__num_steps = max_env
        
        self.reset_data()
        state = self.env.reset()
        step = 0

        while step < num_steps:
            done = False
            new_state = self.env.reset()
            episode_rewards = []
            while not done:
                state = new_state
                agent_out = self.agent.act_experience(
                    np.expand_dims(state, axis=0), self.return_log_prob
                )
                # S
                self.data_agg["state"].append(state)                
                if self.return_feature_state:
                    self.data_agg["feature_state"].append(self.agent.get_state())
                # A
                action = agent_out["action"]
                if tf.is_tensor(action):
                    action = action.numpy()
                if self.discrete_env:
                    action = int(action)
                elif action.shape == ():
                    action = np.expand_dims(action,0)
                new_state, reward, done, _ = self.env.step(action)
                self.data_agg["action"].append(action)
                # R
                self.data_agg["reward"].append(reward)
                episode_rewards.append(reward)
                # S+1
                self.data_agg["state_new"].append(new_state)
                # info on terminal state
                self.data_agg["not_done"].append(float(int(not (done))))

                # append optional in time values to data
                if self.return_log_prob:
                    self.data_agg["log_prob"].append(agent_out["log_probability"])
                if self.return_value_estimate:
                    self.data_agg["value_estimate"].append(agent_out["value_estimate"])

                step += 1
                if step == num_steps:
                    break

            if self.return_monte_carlo:
                self.data_agg["monte_carlo"].extend(
                    discount_cumsum(episode_rewards, self.gamma)
                    )

        return self.data_agg, self.runner_position

    def run_n_episodes(self, num_episodes, max_env=None):
        import tensorflow as tf

        if max_env is not None:
            self.env.__num_steps = max_env

        self.reset_data()
        state = self.env.reset()

        for _ in range(num_episodes):
            done = False
            new_state = self.env.reset()
            episode_rewards = []
            while not done:
                state = new_state
                agent_out = self.agent.act_experience(
                    np.expand_dims(state, axis=0), self.return_log_prob
                )
    
                # S
                self.data_agg["state"].append(state)                
                if self.return_feature_state:
                    self.data_agg["feature_state"].append(self.agent.get_state())
                # A
                action = agent_out["action"]
                if tf.is_tensor(action):
                    action = action.numpy()
                # A
                if self.discrete_env:
                    action = int(action)
                elif action.shape == ():
                    action = np.expand_dims(action,0)
                new_state, reward, done, _ = self.env.step(action)
                self.data_agg["action"].append(action)
                # R
                self.data_agg["reward"].append(reward)
                episode_rewards.append(reward)
                # S+1
                self.data_agg["state_new"].append(new_state)
                # info on terminal state
                self.data_agg["not_done"].append(int(not (done)))

                # append optional in time values to data
                if self.return_log_prob:
                    self.data_agg["log_prob"].append(agent_out["log_probability"])
                if self.return_value_estimate:
                    self.data_agg["value_estimate"].append(agent_out["value_estimate"])

            if self.return_monte_carlo:
                self.data_agg["monte_carlo"].extend(
                    discount_cumsum(episode_rewards, self.gamma)
                    )


        return self.data_agg, self.runner_position

    def get_agent_kwargs(self):
        agent_kwargs = self.agent_kwargs
        return agent_kwargs
