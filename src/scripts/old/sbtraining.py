#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
from tb3env import ContinuousGym, DiscreteGym
from stable_baselines3 import PPO

import itertools
import numpy as np
import torch
import datetime
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create model and train
'''
def train(env, timesteps, model_path, log_path):
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)

if __name__ == '__main__':
    rospy.init_node('sbtrain', anonymous=True)
    
    continuous_env = rospy.get_param('~continuous_env', False)
    n_actions = rospy.get_param('~n_actions', 4)
    log_path = rospy.get_param('~log_path', './logs')
    model_dir = rospy.get_param('~model_dir', './models')
    model_name = rospy.get_param('~model_name', 'model')
    timesteps = rospy.get_param('~timesteps', 10000)
    model_path = model_dir + '/' + model_name

    if continuous_env:
        env = ContinuousGym()
    else:
        env = DiscreteGym(n_actions)
     
    train(env, timesteps, model_path, log_path)
    rospy.spin()
