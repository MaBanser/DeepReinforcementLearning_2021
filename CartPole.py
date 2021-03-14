import gym
import numpy as np
import ray
import os
from really import SampleManager
from really.utils import (
    dict_to_dataset,
)

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


if __name__ == "__main__":
    
    env = gym.make("CartPole-v1")

    model_kwargs = {"layers": [16,16,16], "num_actions": env.action_space.n}
    
    learning_rate = 0.001
    buffer_size = 30000
    max_episodes = 500
    sampled_batches = 16
    optimization_batch_size= 64
    epsilon = 1
    epsilon_decay = 0.95
    epsilon_min = 0.1
    gamma = 0.95

    kwargs = {
        "model": QNet,
        "environment": "CartPole-v1",
        "num_parallel": 2,
        "total_steps": 500,
        "model_kwargs": model_kwargs,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": epsilon
    }

    # Initialize the loss function
    loss_function = tf.keras.losses.MeanSquaredError()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # Where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_CartPole"

    # Initialize buffer
    manager.initilize_buffer(buffer_size)

    # Initialize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", 'reward']
    )

    rewards = []

    # Get initial agent
    agent = manager.get_agent()

    print('TRAINING')
    for e in range(max_episodes):
        
        # Experience replay
        print("collecting experience..")
        print('Epsilon: ',epsilon)
        data = manager.get_data()
        manager.store_in_buffer(data)

        # Sample data to optimize on from buffer
        sample_dict = manager.sample(sampled_batches*optimization_batch_size)
        samples,_ = dict_to_dataset(sample_dict,batch_size = optimization_batch_size)

        print("optimizing...")

        losses = []

        for s, a, r, s_1, not_done in samples:
            a = tf.cast(a, dtype = tf.int32)
            target = r + gamma*agent.max_q(s_1)*not_done
            with tf.GradientTape() as tape:
                pred = agent.q_val(s,a)
                loss = loss_function(target,pred)
                gradients = tape.gradient(loss, agent.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
            losses.append(loss)                

        # Set new weights
        manager.set_agent(agent.get_weights())

        # Let epsilon become smaller in later episodes
        if epsilon>epsilon_min:
            epsilon = epsilon*epsilon_decay

        manager.set_epsilon(epsilon=epsilon)

        print('TEST')

        # Update aggregator
        current_rewards = manager.test(
            max_steps=1000,
            test_episodes=10,
            render=False,
            evaluation_measure="time",
            )

        if (e+1) % 5 == 0:
            manager.test(
                max_steps=1000,
                test_episodes=1,
                render=True
                )
        manager.update_aggregator(loss=losses, reward=current_rewards)
        
        # Collect all rewards
        rewards.extend(current_rewards)
        # Average reward over last 100 episodes
        avg_reward = sum(rewards[-100:])/min(len(rewards),100)

        # Print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean(losses)}   avg_current_reward ::: {np.mean(current_rewards)}   avg_reward ::: {avg_reward}"
        )

        if avg_reward > env.spec.reward_threshold:
            print(f'\n\nEnvironment solved after {e+1} episodes!')
            # Save model
            manager.save_model(saving_path, e, model_name='CartPole')
            break

    print("testing optimized agent")
    manager.test(
        max_steps=1000,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time",
    )