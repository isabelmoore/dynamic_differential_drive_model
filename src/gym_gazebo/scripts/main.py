#!/usr/bin/env python3

from tabnanny import verbose
import rospy
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

#from gym_env_her import MobileRobotEnv
from gym_env import MobileRobotPathTrackEnv
from callbacks import SaveCallback
import matplotlib.pyplot as plt
from matplotlib import cm

'''
Create model and train
'''
def train(model_path, log_dir, timesteps=500000,use_model=True):
    # Create environment
    env = MobileRobotPathTrackEnv(timestep=1e-1, yaw_controller_frequency=50,
            path_length=250, trajectory_length=15, velocity=1.0,
            observation_lookahead=100, use_dubins=True, use_seed=False,
            evaluate=False,radiusOfCBin=2,use_model=use_model)
    
    #env.seed(0) # set seed
    env = Monitor(env, log_dir)
    
    # Filepaths
    save_model_path = model_path + '_best_'
    final_model_path = model_path + '_final_'
    print("save_model_path: ",save_model_path)
    save_callback = SaveCallback(1000, log_dir, save_model_path, verbose=True)
    # model = PPO("MultiInputPolicy", env, verbose=True, tensorboard_log=log_dir)
    model = SAC("MultiInputPolicy", env, verbose=True, tensorboard_log=log_dir)
    #model = PPO.load(final_model_path, env) # load from zip file
    #model = SAC.load(final_model_path, env) # load from zip file
    
    # Train
    model.learn(total_timesteps=timesteps, callback=save_callback)
    
    # Save final model
    model.save(final_model_path)
    
    print('Training complete')

'''
Load model and evaluate
'''
def evaluate(model_path, episodes=50,use_model=True):
    # Create environment
    env = MobileRobotPathTrackEnv(timestep=1e-1, yaw_controller_frequency=50,
            path_length=250, trajectory_length=15, velocity=1.0,
            observation_lookahead=100, use_dubins=True, use_seed=True,
            evaluate=True,radiusOfCBin=5,use_model=use_model)
    if not use_model:
        env.ros_init()
    # model = PPO.load(model_path)
    model = SAC.load(model_path)
    total_rewards = 0

    for i in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done and not rospy.is_shutdown():
            action, obs = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                total_rewards += episode_reward

    print('mean episode reward: %f' %(total_rewards / episodes))


'''
Load model and evaluate the best reward fn
'''
def evaluate_ppc(episodes=1):
    # Create environment
    radiusOfCBin = 3
    env = MobileRobotPathTrackEnv(timestep=0.1, yaw_controller_frequency=50,
            path_length=25, trajectory_length=15, velocity=1.0,
            observation_lookahead=2.5, use_dubins=True, use_seed=True,
            evaluate=True,radiusOfCBin=radiusOfCBin)

    # model = PPO.load(model_path)
    #model = SAC.load(model_path)
    # total_rewards = 0.0
    total_rewards_overall = []
    n_points_vel = 100
    n_points_la = 100
    vel = np.linspace(0.1,10,n_points_vel)
    la = np.linspace(0.1,10,n_points_la)

    # vel = [1,5,10]
    # la = [0.5,5,10]
    for v in vel:
        for l in la:
            action = [l,v]
            # print("action",action)
            total_rewards = 0.0
            for i in range(episodes):
                obs = env.reset()
                done = False
                episode_reward = 0
                while not done and not rospy.is_shutdown():
                    #action, obs = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    if done:
                        total_rewards += episode_reward
                        total_rewards_overall.append(total_rewards/episodes)
                        total_rewards = 0.0

    print("Total Rewards array",total_rewards_overall)
    la,vel = np.meshgrid(la,vel)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(la, vel, np.asarray(total_rewards_overall).reshape(n_points_la,n_points_vel))#, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
    ax.set_zlabel('$\mathbf{R}$')
    # ax.set_zlim(100,-2500)
    plt.title("3D Plot: Lookahead vs Reward for Radius of Curvature. (Bin n has a radius of curvature in the range: $[2^{n-1},2^{n}]$),n = %i"%radiusOfCBin)
    plt.xlabel('Look ahead')
    plt.yticks(np.arange(0,10, step=0.5))
    plt.ylabel('Velocity')
    # plt.zlabel('')
    plt.show()


def baseline(episodes=50):
    mean_rewards = {}

    for action in np.arange(-1., 1.1, 0.1):
        env = MobileRobotPathTrackEnv(timestep=0.1, yaw_controller_frequency=50,
                path_length=25, trajectory_length=15, velocity=1.0,
                observation_lookahead=2.5, use_dubins=True, use_seed=True,
                evaluate=False)
        lookahead = env.scale_action([action])
        total_rewards = 0
        for i in range(episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done and not rospy.is_shutdown():
                obs, reward, done, info = env.step([action])
                episode_reward += reward
                if done:
                    total_rewards += episode_reward
        mean_rewards[lookahead] = total_rewards / episodes

    print(mean_rewards)

    # Plot
    fig, ax = plt.subplots()
    lookaheads = mean_rewards.keys()
    rewards = [mean_rewards[la] for la in lookaheads]
    ax.scatter(lookaheads, rewards)
    ax.set_xlabel('lookahead [meters]')
    ax.set_ylabel('mean reward')
    ax.set_title('Fixed Lookahead Baselines')
    plt.show()
    fig.savefig('../plots/baseline.png')

if __name__ == '__main__':
    log_dir = rospy.get_param('rl_env/log_dir', './logs')
    model_dir = rospy.get_param('rl_env/model_dir', '~/home/root/models') # Set in ./train_model.sh
    model_name = rospy.get_param('rl_env/model_name', 'model')
    timesteps = rospy.get_param('rl_env/timesteps', 500000)

    action = rospy.get_param('rl_env/action', 'evaluate')
    use_model = rospy.get_param('rl_env/use_model', True)

    model_path = model_dir + '/' + model_name
    print("model_path",model_path)
    if action == 'train':
        train(model_path, log_dir, timesteps=timesteps)
    elif action == 'evaluate':
        evaluate(model_path, episodes=10,use_model=use_model)
    elif action == 'baseline':
        baseline()
    elif action == 'evaluate_ppc':
        evaluate_ppc()
