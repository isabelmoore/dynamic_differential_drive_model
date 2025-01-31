#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty

import sys
import multiprocessing as mp
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs import MobileRobotEnv
from command_publisher import CommandPublisher

import unittest

def unpause_physics():
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    try:
        unpause()
    except:
        pass

def test():
    command_queue = mp.Queue()
        
    # Create and start command manager
    command_publisher = CommandPublisher(command_queue, rate=50)
    command_publisher_process = mp.Process(target=command_publisher.publish)
    command_publisher_process.start()

    env = MobileRobotEnv(command_queue, max_episode_steps=300, bound=5.0,
            square_distance=3.0, goal_radius=0.2, timestep=0.05, use_model=False,
            evaluate=False)
    env.ros_init()

    env.reset()
    action = (1.0, 0.0)
    done = False
    while not done:
        obs, reward, done, info = env.step(action)

    env.reset()
    unpause_physics()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()

    command_publisher_process.join()

if __name__ == '__main__':
    test()
