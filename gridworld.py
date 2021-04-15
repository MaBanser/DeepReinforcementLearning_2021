import gym
import numpy as np
import ray
import os
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""

class TabularQ(object):
    def __init__(self, h, w, action_space):
        # Initialize Q table
        self.q_tab = np.zeros((h,w,action_space))

    def __call__(self, state):
        ## # TODO:
        h, w = np.squeeze(state).astype(int)
        output = {}
        output["q_values"] = np.expand_dims(self.q_tab[h, w],0)
        return output

    def get_weights(self):
        return self.q_tab

    def set_weights(self, q_vals):
        self.q_tab = q_vals

    # what else do you need?

    # A nice way to look at the found solution
    def print_optimal(self, action_dict):        
        # get optimal actions
        grid = np.argmax(self.q_tab,2)
        # Translate action values into the actual action
        print(np.asarray(list(action_dict.values()))[grid])



if __name__ == "__main__":
    action_dict = {0: " UP  ", 1: "RIGHT", 2: "DOWN ", 3: "LEFT "}

    maze = {
        (1,1),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),
        (2,1),(2,10),(2,12),
        (3,1),(3,6),(3,10),(3,12),(3,14),(3,15),(3,16),(3,17),
        (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,8),(4,10),(4,12),(4,15),
        (5,1),(5,8),(5,10),(5,12),(5,15),
        (6,1),(6,4),(6,5),(6,6),(6,7),(6,8),(6,12),(6,15),
        (7,1),(7,4),(7,8),(7,9),(7,10),(7,11),(7,12),(7,15),
        (8,1),(8,6),(8,8),(8,12),
        (9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,8),(9,10),(9,11),(9,12),(9,13),(9,14),(9,15),(9,16),(9,17),(9,18),
        (10,1),(10,8),
        (11,1),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,10),
        (12,1),(12,10),
        (13,1),(13,2),(13,3),(13,4),(13,5),(13,6),(13,7),(13,10),
        (14,10)
        }

    # Robocat maze

    env_kwargs = {
        "height": 15,
        "width": 19,
        "start_positions": [(h,0) for h in range(15)],
        "reward_positions": {(10, 11)},
        "block_positions": maze,
        "max_time_steps": 300
    }

    # Standard gridworld

    # env_kwargs = {
    #     "height": 3,
    #     "width": 4,
    #     "start_positions": [(2, 0)],
    #     "reward_positions": {(0, 3)},
    #     "block_positions": {(1,1)}
    # }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    # env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env_kwargs['height'], "w": env_kwargs['width'], "action_space": len(action_dict)}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 3,
        "total_steps": 2000,
        "model_kwargs": model_kwargs,
        "env_kwargs": env_kwargs,
        "num_episodes": 5
        # and more
    }

    # Initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=1,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    
    # Do the rest!!!!

    # Where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_TabQ"

    buffer_size = 50000
    epochs = 10
    sample_size = 10000
    step_size = 0.2
    gamma = 1

    assert(0<step_size<=1),"Step size should be between (0,1]"

    # Initialize buffer
    manager.initilize_buffer(buffer_size)

    # Initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["error", "steps"]
    )

    # Get initial agent
    agent = manager.get_agent()

    print('TRAINING')
    for e in range(epochs):
        
        # Experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # Sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        states, actions, rewards, new_states, not_dones = sample_dict.values()

        print("optimizing...")

        error_aggregator = []

        # TODO: iterate through your datasets
        for s, a, r, s_1, not_done in zip(states, actions, rewards, new_states, not_dones):
            old_q = agent.get_weights().copy()

            h, w = np.squeeze(s)
            h_1, w_1 = np.squeeze(s_1)
            error = r + step_size * (gamma * max(old_q[h_1, w_1]) - old_q[h, w, a]) * not_done
            error_aggregator.append(error)

            # TODO: optimize agent
            old_q[h, w, a] += error
            agent.set_weights(old_q)
                

        # Set new weights
        manager.set_agent(agent.get_weights())

        # Update aggregator
        steps = manager.test(
            max_steps=100,
            test_episodes=10,
            render=True,
            evaluation_measure="time",
            )
        manager.update_aggregator(error=error_aggregator, steps=steps)
        # Print progress
        print(
            f"epoch ::: {e}  error ::: {np.mean(error_aggregator)}   avg_timesteps ::: {np.mean(steps)}"
        )
        agent.model.print_optimal(action_dict)

    print("testing optimized agent")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time",
    )

    print('SOLUTION')
    agent.model.print_optimal(action_dict)