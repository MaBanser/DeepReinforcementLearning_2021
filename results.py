import os
import gym
import ray
from really import SampleManager

import tensorflow as tf
from tensorflow.keras import Model

class QNet(Model):
    def __init__(self, layers, num_actions):
        super(QNet, self).__init__()        
        self.dense_layers = [tf.keras.layers.Dense(units=num_units,
                                                   activation='relu',
                                                   name=f'Dense_layer_{i}'
                                                  ) for i, num_units in enumerate(layers)]

        self.readout_layer = tf.keras.layers.Dense(units=num_actions,
                                                   activation=None,
                                                   name='Readout_layer'
                                                   )

    @tf.function
    def call(self, input_state):
        output = {}
        for layer in self.dense_layers:
            input_state = layer(input_state)
        output["q_values"] = self.readout_layer(input_state)
        return output

class A2C(Model):
    def __init__(self, layers, action_dim):
        super(A2C, self).__init__()
        self.mu_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_mu_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_mu = tf.keras.layers.Dense(units=action_dim,
                                                activation=None,
                                                name='Policy_mu_readout'
                                                )

        self.sigma_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_sigma_{i}'
                ) for i, num_units in enumerate(layers)]
                
        self.readout_sigma = tf.keras.layers.Dense(units=action_dim,
                                                   activation=None,
                                                   name='Policy_sigma_readout'
                                                   )

        self.value_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Value_layer_{i}'
                ) for i, num_units in enumerate(layers)]
                
        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation=None,
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, input_state):
        output = {}
        mu_pred = input_state
        sigma_pred = input_state
        value_pred = input_state
        for layer in self.mu_layer:
            mu_pred = layer(mu_pred)
        for layer in self.sigma_layer:
            sigma_pred = layer(sigma_pred)
        for layer in self.value_layer:
            value_pred = layer(value_pred)

        # Actor
        output["mu"] = tf.squeeze(self.readout_mu(mu_pred))
        output["sigma"] = tf.squeeze(tf.abs(self.readout_sigma(sigma_pred)))
        # Critic
        output["value_estimate"] = tf.squeeze(self.readout_value(value_pred))
        return output

if __name__ == "__main__":

    print('Prepare CartPole')
    env = gym.make("CartPole-v1")

    model_kwargs = {"layers": [16,16,16], "num_actions": env.action_space.n}

    kwargs = {
        "model": QNet,
        "environment": "CartPole-v1",
        "num_parallel": 1,
        "total_steps": 1000,
        "model_kwargs": model_kwargs,
    }

    # Initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # Where to load your results from
    loading_path = os.getcwd() + "/progress_CartPole"

    # Load model
    manager.load_model(loading_path)
    print("done")
    print("testing optimized agent")
    manager.test(
        1000, 
        test_episodes=10, 
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
        )
        
    print('Prepare LunarLander')
    env = gym.make("LunarLander-v2")

    model_kwargs = {"layers": [32,32,32], "num_actions": env.action_space.n}

    kwargs = {
        "model": QNet,
        "environment": "LunarLander-v2",
        "num_parallel": 1,
        "total_steps": 1000,
        "model_kwargs": model_kwargs,
    }

    # Initialize
    manager = SampleManager(**kwargs)

    # Where to load your results from
    loading_path = os.getcwd() + "/progress_LunarLander"

    # Load model
    manager.load_model(loading_path)
    print("done")
    print("testing optimized agent")
    manager.test(
        1000, 
        test_episodes=10, 
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
        )
    
    print('Prepare LunarLanderContinuous')
    env = gym.make("LunarLanderContinuous-v2")

    model_kwargs = {"layers": [32,32,32], "action_dim": env.action_space.shape[0]}

    kwargs = {
        "model": A2C,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 1,
        "total_steps": 1000,
        "model_kwargs": model_kwargs,
        "action_sampling_type": "continuous_normal_diagonal",
    }

    # Initialize
    manager = SampleManager(**kwargs)

    # Where to load your results from
    loading_path = os.getcwd() + "/progress_LunarLanderContinuous"

    # Load model
    manager.load_model(loading_path)
    print("done")
    print("testing optimized agent")
    manager.test(
        1000, 
        test_episodes=10, 
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
        )