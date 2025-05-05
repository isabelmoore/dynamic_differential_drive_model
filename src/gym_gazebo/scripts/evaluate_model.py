from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

#from gym_env_her import MobileRobotEnv
# from gym_env import MobileRobotPathTrackEnv
# from gym_env_ddd import MobileRobotPathTrackEnv
from gym_env_ddd_gazebo import MobileRobotPathTrackEnv

from callbacks import SaveCallback
import matplotlib.pyplot as plt
from matplotlib import cm


'''
Load model and evaluate
'''
def evaluate(model_path, episodes=50):
    # Create environment
    print('Creating environment...')
    env = MobileRobotPathTrackEnv(timestep=1e-1, yaw_controller_frequency=25,
            path_length=50, trajectory_length=15, velocity=0.1,
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
            print('action:', action)
            # action = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            # action = [-0.00902939 -0.00461888 -0.01180905  0.00549364  0.01214027 -0.00688547
            # action = [-0.01, -0.005, -0.01, 0.005, 0.01, -0.007]

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                total_rewards += episode_reward

    print('mean episode reward: %f' %(total_rewards / episodes))
    

if __name__ == '__main__':
    print('Evaluating model...')
    # evaluate(model_path='models/MetaL_10H_July19_reducedTask_best__best_reward_10000', episodes=100) -> Best model
    #evaluate(model_path='models/MetaL_10H_Aug7_100m_best__best_reward_2460000', episodes=100)
    #evaluate(model_path='models/MetaL_10H_Aug7_100m_best__best_reward_10000', episodes=100)
    #evaluate(model_path='models/MetaL_10H_Oct10_best__best_reward_810000', episodes=100)
    
    # evaluate(model_path='Apr16_bic_ent-auto_fc-nd_vect/models/MetaL_10H_Feb27__best_reward__10240', episodes=100)

    # evaluate(model_path='Apr17_ddd_ent-auto_fric-nd_vect_workingmodel/models/MetaL_10H_Feb27__best_reward__60160', episodes=100)
    # evaluate(model_path='models/MetaL_10H_Feb27__best_reward__70144', episodes=100)

    # evaluate(model_path='Apr24_ddd_ent-auto_fc-nd_vel-kp-1_omega_kp-3/models/MetaL_10H_Feb27__best_reward__70144', episodes=100)
    # evaluate(model_path='Apr26_ddd_ent-auto_fc-nd_vel-kp-1_omega_kp-5/models/MetaL_10H_Feb27__best_reward__1540096', episodes=100)
    # evaluate(model_path='Apr27_ddd_ent-auto_fc-nd_vel-kp-1_omega_kp-7/models/MetaL_10H_Feb27__best_reward__1480192', episodes=100)
    
    # evaluate(model_path='Apr27_ddd_ent-auto_fc-nd_vel-kp-2_omega_kp-3/models/MetaL_10H_Feb27__best_reward__1540096', episodes=100)
    # evaluate(model_path='Apr27_ddd_ent-auto_fc-nd_vel-kp-2_omega_kp-5/models/MetaL_10H_Feb27__best_reward__1540096', episodes=100)
    # evaluate(model_path='Apr28_ddd_ent-auto_fc-nd_vel-kp-2_omega_kp-7/models/MetaL_10H_Feb27__best_reward__1860096', episodes=100)
    
    # evaluate(model_path='Apr28_ddd_ent-auto_fc-nd_vel-kp-3_omega_kp-3/models/MetaL_10H_Feb27__best_reward__1670144', episodes=100)
    # evaluate(model_path='Apr28_ddd_ent-auto_fc-nd_vel-kp-3_omega_kp-5/models/MetaL_10H_Feb27__best_reward__1540096', episodes=100)
    # evaluate(model_path='Apr28_ddd_ent-auto_fc-nd_vel-kp-3_omega_kp-7/models/MetaL_10H_Feb27__best_reward__1540096', episodes=100)
    evaluate(model_path='MetaL__best_reward__2630144', episodes=100)
