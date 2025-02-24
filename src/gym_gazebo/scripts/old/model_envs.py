#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

import gym
import math
import numpy as np
import random

from envs import GazeboEnv, euler_from_quaternion, distance

def heading_error(target, current):
    error_raw = current - target
    if abs(error_raw) > np.pi:
        error = 2 * np.pi - abs(error_raw) 
        if error_raw > 0:
            error = -error
    else:
        error = error_raw
    
    return error

class ModelEnv(GazeboEnv):
    def __init__(self):
        super(ModelEnv, self).__init__()

        metadata = {'render.modes': ['console']}

        self.max_episode_steps = 300
        self.bound = 3.0 # meters
        self.square_dist = 3.0 # meters # distance to edge from origin
        self.goal_radius = 0.1 # meters
        self.timestep = 0.05 # seconds

        # Continuous observation space
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,),
                dtype=np.float32)
        
        # Discrete action space
        self.actions = {}
        lx = np.linspace(0.0, 1.5, 12) # no reverse
        az = np.linspace(-2.5, 2.5, 5)
        self.actions['lx'] = lx
        self.actions['az'] = az
        rospy.loginfo('action space: lx = %s az = %s' %(lx, az))
        self.action_space = gym.spaces.MultiDiscrete([lx.shape[0], az.shape[0]])

        self.goal = np.zeros(2) # x, y
        self.target_heading = 0.0
        self.pose = np.zeros(3) # x, y, yaw
        self.action = np.zeros(2) # lx, az

    def observe(self):
        # distance error
        error_dist_norm = (self.pose[:2] - self.goal) / (self.square_dist +
                self.bound)
        
        # heading error
        error_yaw = heading_error(self.target_heading, self.pose[2])
        error_yaw_norm = error_yaw / np.pi
        
        obs = np.append(error_dist_norm, error_yaw_norm)
        return obs

    def reset_other(self):
        # Choose goal as random point on square
        p1 = self.square_dist * random.uniform(-1.0, 1.0)
        p2 = self.square_dist * random.choice([-1.0, 1.0])
        self.goal = random.choice([np.array([p1, p2]), np.array([p2, p1])])
        self.target_heading = math.atan2(self.goal[1], self.goal[0])
        rospy.loginfo('goal: %s' %self.goal)

        # Initial observation
        obs = self.observe()

        self.episode_steps = 0

        return obs

    def step(self, action):
        # Take action
        lx = self.actions['lx'][action[0]]
        az = self.actions['az'][action[1]]
        msg = Twist()
        msg.linear.x = lx
        msg.angular.z = az
        self.cmd_pub.publish(msg)
        rospy.sleep(self.timestep)
        self.episode_steps += 1
        
        # Make observation
        obs = self.observe()

        # Calculate reward
        reward = 0.0
        
        done = False
        info = {}

        # Check if max steps reached
        if self.episode_steps > self.max_episode_steps:
            rospy.loginfo('Max episode steps reached')
            done = True
            reward = -1.0
            self.stop_robot()
            return obs, reward, done, info

        # Check if robot out of bounds
        if abs(self.pose[0]) > self.bound or abs(self.pose[1]) > self.bound:
            rospy.loginfo('Robot out of bounds')
            done = True
            reward = -1.0
            self.stop_robot()
            return obs, reward, done, info

        # Check if goal reached
        if distance(self.pose, self.goal) < self.goal_radius:
            rospy.loginfo('Goal reached')
            done = True
            reward = 10.0
            self.stop_robot()
            return obs, reward, done, info

        # Normal step
        return obs, reward, done, info

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.pose = np.array([msg.pose.pose.position.x,
            msg.pose.pose.position.y, yaw])
