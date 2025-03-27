#!/usr/bin/env python3

from tabnanny import verbose
# import rospy
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from sb3_contrib import RecurrentPPO

#from gym_env_her import MobileRobotEnv
from gym_env import MobileRobotPathTrackEnv
from callbacks import SaveCallback
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


'''
Create model and train
'''
def train(model_path, log_dir, timesteps=500000):
    # Create environment
    env = MobileRobotPathTrackEnv(timestep=1e-1, yaw_controller_frequency=50,
            path_length=100, trajectory_length=15, velocity=1.0,
            observation_lookahead=100, use_dubins=True, use_seed=False,
            evaluate=False,radiusOfCBin=2,use_model=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    #env.seed(0) # set seed
    env = Monitor(env, log_dir)
    
    # Filepaths
    save_model_path = model_path + '_best_'
    final_model_path = model_path + '_final_'
    # print(final_model_path)
    print("Timesteps: ",timesteps)
    save_callback = SaveCallback(10000, log_dir, save_model_path, verbose=True)
    # model = PPO("MultiInputPolicy", env, verbose=True, tensorboard_log=log_dir)
    model = SAC("MultiInputPolicy", env, verbose=True, tensorboard_log=log_dir)
    #model = PPO.load(final_model_path, env) # load from zip file
    #model = SAC.load(final_model_path, env) # load from zip file
    
    # Train
    model.learn(total_timesteps=timesteps, callback=save_callback)
    
    # Save final model
    model.save(final_model_path)
    
    print('Training complete')
    


if __name__ == '__main__':
	# train(model_path='models/MetaL_10H_Oct16', log_dir='logs', timesteps=5000000)
	train(model_path='/home/root/models/MetaL_10K_03272015', log_dir='logs', timesteps=1000000)
