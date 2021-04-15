# DeepReinforcementLearning_2021
Homeworks and project for the Deep Reinforcement Learning seminar WS2021

Use the [env.yml](../master/env.yml) to setup an environment with the correct python version and dependencies.

## ReAllY
For ReAllY documentation see: https://github.com/geronimocharlie/ReAllY

## Pommerman
For Pommerman documentation see: https://github.com/MultiAgentLearning/playground

# Homework

## Homework 1 - Gridworld

For the first homework, we would like you to code an agent that is able to solve a simple gridworld.
A description of the gridworld and a visualization of it can be found in the gridworld.py script in the ReAllY repo:

[ReAllY](https://github.com/geronimocharlie/ReAllY)

We advise you to use the framework to solve the homework with the framework to get a feeling for it. As a Reinforcement Learning Algorithm we suggest to make use of tabular Q learning (Watkins).

### Install gridworlds

Within DeepReinforcementLearning_2021\gridworlds, run 
pip install -U .

## Homework 2 - Cartpole
The second challenge will be about solving the infamous [Cartpole-v0](https://gym.openai.com/envs/CartPole-v0/) environment from OpenAI gym with a deep reinforcement learning algorithm. Again we advise you to make use of the framework we provide: 

[ReAllY](https://github.com/geronimocharlie/ReAllY)

Our suggestion would be, due to reasons of simplicity, to try using a vanilla Deep Q-Network (or if you want combined with some of the rainbow deep q improvements) first.

### Homework 2 Optional - Cartpole-v1/LunarLander
Solve [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/) and [LunarLander](https://gym.openai.com/envs/LunarLander-v2/) using DQN

## Homework 3 - Lunar Lander
For your last homework we want you to implement something more sophisticated.
We want you to solve a continuous environment, for example the [continuous Lunar Lander](https://gym.openai.com/envs/LunarLanderContinuous-v2/) (one of the prettier gyms there are), where your agent should learn how to land a rocket. To solve this, we would like you to implement an algorithm of the policy gradient family. That could of course also be an advantage actor critic model!


# Final Project: Pommerman

## Install Pommerman
Within DeepReinforcementLearning_2021\pommerman, run 
pip install -U .



# Citation
Since we are using Pommerman environment in our research, we cite it using this [bibtex file](../master/pommerman/docs/pommerman.bib) in docs.