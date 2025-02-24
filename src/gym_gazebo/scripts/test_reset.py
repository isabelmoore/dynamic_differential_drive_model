#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import time

import environment as env
from gym_env import MobileRobotEnv, heading_error

class Test:
    def __init__(self):
        self.env = MobileRobotEnv(max_episode_steps=1, bound=3.0,
                goal_distance=3.0, goal_radius=0.2, timestep=0.05,
                use_continuous_actions=True, use_shaped_reward=False,
                use_model=True, evaluate=False, debug=False)

    def reset(self, n=100):
        for i in range(n):
            info = self.env.random_goal()
            angle = info['angle']
            self.env.random_obstacle(angle=angle, p_inline=1.0)

if __name__ == '__main__':
    test = Test()
    test.reset()
