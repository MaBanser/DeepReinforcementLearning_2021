import random
import numpy as np
import gym
from gym import spaces
from cv2 import cv2

"""
 n x m gridworld
 The agent can move in the grid world.
 There is one block position where the agent cannot move to.
 There is one reward position where the agent gets a reward and is done.
 For each other move the agent gets a reward of 0.

"""


class GridWorld(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **config):

        self.config = {
            "height": 3,
            "width": 4,
            "block_positions": {(1, 1)},
            "reward_positions": {(2, 3)},
            "start_positions": [(0, 0)],
            "reward": 10,
            "max_time_steps": 100,
            "player_color": [1, 0, 0],
            "reward_color": [0, 1, 0],
            "block_color": [0, 0, 1],
            "action_dict": {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"},
        }
        self.config.update(config)

        # get correct actions and transitions
        self.action_dict = self.config["action_dict"]
        for k in self.action_dict.keys():
            if self.action_dict[k] == "UP":
                UP = k
            elif self.action_dict[k] == "RIGHT":
                RIGHT = k
            elif self.action_dict[k] == "DOWN":
                DOWN = k
            elif self.action_dict[k] == "LEFT":
                LEFT = k
            else:
                print(f"unsupported action {self.action_dict[k]} with key {k}")
                raise KeyError

        self.transitions = {UP: (-1, 0), DOWN: (1, 0), RIGHT: (0, 1), LEFT: (0, -1)}

        # get info on grid
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.max_time_steps = self.config["max_time_steps"]
        self.n_states = self.height * self.width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)

        # start state
        self.done = False
        self.position = random.choice(self.config["start_positions"])
        self.t = 0

        # grid info for rendering
        screen = np.zeros((self.height, self.width, 3))

        for h,w in self.config["reward_positions"]:
            screen[h,w] = self.config["reward_color"]

        for h,w in self.config["block_positions"]:
            screen[h,w] = self.config["block_color"]
        
        #screen[self.config["reward_positions"]] = self.config["reward_color"]
        #screen[self.config["block_positions"]] = self.config["block_color"]
        self.basic_screen = screen

        # for some reason gym wants that
        self._seed = random.seed(1234)

    def step(self, action):

        assert self.action_space.contains(action)

        off_x, off_y = self.transitions[action]
        new_position = self.move(off_x, off_y)

        if not (new_position in self.config["block_positions"]):
            self.position = new_position

        # done if terminal state is reached
        if new_position in self.config["reward_positions"]:
            self.done = True
            return self.position, self.config["reward"], self.done, None

        # done if max time steps reached
        if self.t + 1 == self.max_time_steps:
            self.done = True

        self.t += 1

        return self.position, 0, self.done, None

    def move(self, x_off, y_off):
        x, y = self.position

        # check for borders
        if ((x == 0) & (x_off == -1)) or ((x == self.height - 1) & (x_off == 1)):
            x = x
        else:
            x = x + x_off
        if ((y == 0) & (y_off == -1)) or ((y == self.width - 1) & (y_off == 1)):
            y = y
        else:
            y = y + y_off

        return (x, y)

    def reset(self):
        self.position = random.choice(self.config["start_positions"])
        self.done = False
        self.t = 0
        return self.position


    def render(self, mode="human"):
        screen = self.basic_screen.copy()
        screen[self.position] = self.config["player_color"]
        cv2.namedWindow("GridWorld environment",cv2.WINDOW_NORMAL)
        # Not needed, cv2.WINDOW_NORMAL allows resizing the window manually
        # Scale up to make it readable
        # cv2.resizeWindow("GridWorld environment", 400,300)
        cv2.imshow("GridWorld environment", screen)
        cv2.waitKey(100)
