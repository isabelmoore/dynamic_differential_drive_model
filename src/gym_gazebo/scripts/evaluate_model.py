from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

#from gym_env_her import MobileRobotEnv
from gym_env import MobileRobotPathTrackEnv
from callbacks import SaveCallback
#import matplotlib.pyplot as plt
#from matplotlib import cm


'''
Load model and evaluate
'''
def evaluate(model_path, episodes=50):
    # Create environment
    env = MobileRobotPathTrackEnv(timestep=1e-1, yaw_controller_frequency=50,
            path_length=250, trajectory_length=15, velocity=1.0,
            observation_lookahead=100, use_dubins=True, use_seed=True,
            evaluate=True,radiusOfCBin=5)

    # model = PPO.load(model_path)
    model = SAC.load(model_path)
    total_rewards = 0

    for i in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, obs = model.predict(obs, deterministic=True)
            # action = [0.01,0.005]
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                total_rewards += episode_reward

    print('mean episode reward: %f' %(total_rewards / episodes))
    

if __name__ == '__main__':
    # evaluate(model_path='models/MetaL_10H_July19_reducedTask_best__best_reward_10000', episodes=100) -> Best model
    #evaluate(model_path='models/MetaL_10H_Aug7_100m_best__best_reward_2460000', episodes=100)
    #evaluate(model_path='models/MetaL_10H_Aug7_100m_best__best_reward_10000', episodes=100)
    #evaluate(model_path='models/MetaL_10H_Oct10_best__best_reward_810000', episodes=100)
    evaluate(model_path='/home/root/models/MetaL_10H_Oct16MetaL_10K_03272015_best__best_reward_10000', episodes=100)

