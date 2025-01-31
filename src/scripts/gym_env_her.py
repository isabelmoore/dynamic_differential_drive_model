#!/usr/bin/env python3

import rospy
#import tf # does not work for Python3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from controller_manager_msgs.srv import (SwitchController,
        SwitchControllerRequest, SwitchControllerResponse)
from gym_gazebo.srv import Step, StepRequest, StepResponse

import gym
from abc import abstractmethod
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from differential_drive_model import DifferentialDriveModel
import environment as env

def reset_world():
    rospy.wait_for_service('/gazebo/reset_world')
    reset = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    try:
        reset()
    except rospy.ServiceException as ex:
        rospy.logwarn('Gazebo reset unsuccessful: ' + str(ex))
        return False
    return True

def reset_simulation():
    rospy.wait_for_service('/gazebo/reset_simulation')
    reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    try:
        reset()
    except rospy.ServiceException as ex:
        rospy.logwarn('Gazebo reset unsuccessful: ' + str(ex))
        return False
    return True

def switch_controller(on=True):
    rospy.wait_for_service('/controller_manager/switch_controller')
    switch = rospy.ServiceProxy('/controller_manager/switch_controller',
            SwitchController)
    controller = 'jackal_velocity_controller'
    req = SwitchControllerRequest()
    if on:
        req.start_controllers.append(controller)
    else:
        req.stop_controllers.append(controller)
    try:
        res = switch(req)
    except rospy.ServiceException as ex:
        rospy.logwarn('Controller switch unsuccessful: ' + str(ex))
        return False
    return res.ok

'''
Uses pause and unpause physics services
'''
def naive_step_gazebo(timestep):
    rospy.wait_for_service('/gazebo/pause_physics')
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    try:
        unpause()
        rospy.sleep(timestep)
        pause()
    except rospy.exceptions.ROSTimeMovedBackwardsException:
        rospy.logdebug('Caught ROSTimeMovedBackwardsException')
    except rospy.ServiceException as ex:
        rospy.logwarn('Gazebo timestep unsuccessful: ' + str(ex))
        return False
    return True

'''
Uses custom Step plugin
'''
def step_gazebo(seconds, command):
    rospy.wait_for_service('/gazebo/step')
    step = rospy.ServiceProxy('/gazebo/step', Step)
    req = StepRequest()
    req.seconds = seconds
    req.command = command

    try:
        res = step(req)
        return res.odometry
    except rospy.ServiceException as ex:
        rospy.logwarn('Gazebo step unsuccessful: ' + str(ex))
        return None
    
def pose_twist_from_odometry(msg):
    q = msg.pose.pose.orientation
    r = Rotation.from_quat([q.x, q.y, q.z, q.w])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
    twist = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
    return pose, twist

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def heading_error(target, current):
    error_raw = current - target
    if abs(error_raw) > np.pi:
        error = 2 * np.pi - abs(error_raw) 
        if error_raw > 0:
            error = -error
    else:
        error = error_raw
    
    return error

'''
Assumes normalized [-1, 1] output
'''
def scale_offset_from_discrete_actions(actions):
    scale = abs(max(actions) - min(actions)) / 2
    offset = -1 / scale * (max(actions) - min(actions)) / 2
    return scale, offset

class MobileRobotEnv(gym.GoalEnv):
    def __init__(self, max_episode_steps=300, bound=10.0, square_distance=5.0,
            goal_radius=0.5, timestep=0.1, use_continuous_actions=True,
            discrete_actions=None, normalized_action_scale=None,
            normalized_action_offset=None, use_shaped_reward=False,
            use_model=False, evaluate=False):
        super(MobileRobotEnv, self).__init__()

        metadata = {'render.modes': ['console']}

        self.max_episode_steps = max_episode_steps
        self.bound = bound # meters
        self.square_distance = square_distance # meters
        self.goal_radius = goal_radius # meters
        self.timestep = timestep # seconds
        self.use_continuous_actions = use_continuous_actions
        self.use_shaped_reward = use_shaped_reward
        self.use_model = use_model
        self.evaluate = evaluate

        # Action space
        if use_continuous_actions:
            if (normalized_action_offset is not None and
                    normalized_action_scale is not None):
                self.normalized_action_scale = normalized_action_scale
                self.normalized_action_offset = normalized_action_offset
            else:
                # default action scale and offset
                self.normalized_action_scale = [3.0 / 4, 1.0]
                self.normalized_action_offset = [3.0 / 4, 0.0]
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,),
                    dtype=np.float32)
        else:
            if discrete_actions is not None:
                self.actions = discrete_actions
            else:
                # default actions
                lx = np.linspace(0.0, 1.4, 7) # no reverse
                az = np.linspace(-1.0, 1.0, 7)
                self.actions = {}
                self.actions['lx'] = lx
                self.actions['az'] = az
            # calculate action scale and offset
            lx_scale, lx_offset = scale_offset_from_discrete_actions(
                    self.actions['lx'])
            az_scale, az_offset = scale_offset_from_discrete_actions(
                    self.actions['az'])
            self.normalized_action_scale = [lx_scale, az_scale]
            self.normalized_action_offset = [lx_offset, az_offset]
            self.action_space = gym.spaces.MultiDiscrete([
                self.actions['lx'].shape[0], self.actions['az'].shape[0]])

        # Observation space
        #self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,),
        #        dtype=np.float32)

        '''
        self.observation_space = gym.spaces.Dict(spaces={
            'error': gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            #'twist': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            'costmap': gym.spaces.Box(low=False, high=True, shape=(40, 40),
                dtype=np.bool_)
            })
        '''

        self.observation_space = gym.spaces.Dict(spaces={
            'observation': gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            'desired_goal': gym.spaces.Box(low=-square_distance,
                high=square_distance, shape=(2,)),
            'achieved_goal': gym.spaces.Box(low=-bound, high=bound, shape=(2,))
            })
        
        self.goal = np.zeros(2) # x, y
        self.pose = np.zeros(3) # x, y, yaw
        self.twist = np.zeros(2) # lx, az
        self.costmap = np.zeros((40, 40), dtype=np.bool_)

        # Plotting
        if self.evaluate:
            self.plot_init()

        # Model
        if self.use_model:
            # vehicle model
            self.model = DifferentialDriveModel(wheel_radius=0.098,
                    track_width=0.37559)

            # environment model
            start = env.Pose2d(5.0, 5.0, 0.0) # assume 10 x 10 m space
            costmapResolution = 0.1
            costmapDimension = int(4.0 / costmapResolution) # cells
            self.env = env.Environment2d(start, costmapWidth=costmapDimension,
                    costmapHeight=costmapDimension,
                    costmapResolution=costmapResolution)
            shape = env.Rectangle(start.center, 0.420, 0.310, 0.0)
            self.env.updateFootprint(shape)
            self.costmapDistance, self.costmapHeading = \
                    self.env.getCostmapProperties()
            #self.env.plotInit() # intialize plotting
    
    def ros_init(self):
        rospy.init_node('rl_env')

    '''
    Customized for GoalEnv, HER
    '''
    def observe(self):
        obs = {}

        # normalized position
        position_norm = self.pose[:2] / (self.square_distance +
                self.bound)
        
        # normalized heading
        heading_norm = self.pose[2] / np.pi
 
        obs['observation'] = np.append(position_norm, heading_norm)

        # twist
        '''
        twist_norm = [(self.twist[0] - self.normalized_action_offset[0]) /
                self.normalized_action_scale[0],
                (self.twist[1] - self.normalized_action_offset[1]) /
                self.normalized_action_scale[1]]

        obs['twist'] = np.array(twist_norm)
        '''

        # costmap
        #obs['costmap'] = self.costmap

        # GoalEnv observations
        obs['desired_goal'] = self.goal
        obs['achieved_goal'] = self.pose[:2]

        return obs

    def reset(self):
        if not self.use_model:
            # Reset Gazebo
            #reset_simulation() # does not reset controller!
            reset_world() # resets controller
            naive_step_gazebo(0.001) # allow odometry to reset
            self.pose = np.zeros(3)
            self.twist = np.zeros(2)
        else:
            self.pose = np.zeros(3)
            self.twist = np.zeros(2)
            pose = env.Pose2d(*self.pose)
            #_, self.costmap = self.env.updateOdom(pose, dtype=np.bool_)
            #self.env.plot()

        if self.evaluate:
            self.path = []

        # Choose goal as random point on square
        p1 = self.square_distance * random.uniform(-1.0, 1.0)
        p2 = self.square_distance * random.choice([-1.0, 1.0])
        self.goal = random.choice([np.array([p1, p2]), np.array([p2, p1])])
        self.target_heading = math.atan2(self.goal[1], self.goal[0])
        if not self.evaluate:
            rospy.loginfo('goal: %s' %self.goal)
            if self.use_model:
                print('goal: %s' %self.goal)

        # Place obstacle at random point on offset square
        d1 = self.square_distance / 2 * random.uniform(-1.0, 1.0)
        d2 = self.square_distance / 2 * random.choice([-1.0, 1.0])
        d = random.choice([np.array([d1, d2]), np.array([d2, d1])])
        #d = np.array([1.5, 0])

        self.env.clearObstacles()
        '''
        self.env.addObstacle(env.Obstacle2d(env.Circle(env.Point(
            self.env.start.center.x + d[0], self.env.start.center.y + d[1]),
            0.25)))
        '''
        '''
        self.env.addObstacle(env.Obstacle2d(env.Circle(env.Point(6.5, 6.5),
            0.25)))
        '''
        _, self.costmap = self.env.updateOdom(pose, dtype=np.bool_)
        #self.env.plot()

        # Initial observation
        obs = self.observe()

        self.episode_steps = 0

        return obs
 
    '''
    Customized for GoalEnv, HER
    '''
    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 1:
            reward = 0
            if distance(desired_goal, achieved_goal) < self.goal_radius:
                reward = 10 #1
            return reward
        elif len(achieved_goal.shape) == 2:
            sq_diff = np.square(desired_goal - achieved_goal)
            dist = np.sum(sq_diff, axis=1)
            reward = 10 * (dist < self.goal_radius)
            return reward
        else:
            print('Unexpected input goal shape')

    def set_goal(self, goal):
        self.goal = np.array(goal)
        self.target_heading = math.atan2(self.goal[1], self.goal[0])
        rospy.loginfo('goal: %s' %self.goal)
 
    def step(self, action):
        # Get action
        if self.use_continuous_actions:
            lx = self.normalized_action_scale[0] * action[0] + \
                    self.normalized_action_offset[0] # [0.0, 1.5]
            az = self.normalized_action_scale[1] * action[1] + \
                    self.normalized_action_offset[1] # [-1.0, 1.0]
        else:
            lx = self.actions['lx'][action[0]]
            az = self.actions['az'][action[1]]

        # Simulate timestep
        if not self.use_model: 
            # Take action
            command = Twist()
            command.linear.x = lx
            command.angular.z = az
            odometry = step_gazebo(self.timestep, command)
            self.pose, self.twist = pose_twist_from_odometry(odometry)
        else:
            self.pose = self.model.step(self.pose, (lx, az), self.timestep)
            # twist is not updated
            pose = env.Pose2d(*self.pose)
            collision, self.costmap = self.env.updateOdom(pose, dtype=np.bool_)
            #self.env.plot()
        self.episode_steps += 1

        if self.evaluate:
            self.path.append(self.pose[:2])
        
        # Make observation
        obs = self.observe()

        # Calculate reward
        '''
        if self.use_shaped_reward:
            reward = self.shaped_reward(obs)
        else:
            reward = self.sparse_reward(obs)
        '''
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'],
                {})

        done = False
        info = {}

        # Check if max steps reached
        if self.episode_steps > self.max_episode_steps:
            rospy.loginfo('Max episode steps reached')
            if self.use_model:
                print('Max episode steps reached')
            done = True
            if self.evaluate:
                self.plot_result(success=False)
            return obs, reward, done, info

        # Check if robot out of bounds
        if abs(self.pose[0]) > self.bound or abs(self.pose[1]) > self.bound:
            rospy.loginfo('Robot out of bounds')
            if self.use_model:
                print('Robot out of bounds')
            done = True
            if self.evaluate:
                self.plot_result(success=False)
            return obs, reward, done, info

        # Check if goal reached; customized for GoalEnv, HER
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        if distance(desired_goal, achieved_goal) < self.goal_radius:
            rospy.loginfo('Goal reached')
            if self.use_model:
                print('Goal reached')
            done = True
            if self.evaluate:
                self.plot_result(success=True)
            return obs, reward, done, info

        # Check for collision
        if collision:
            rospy.loginfo('Collision')
            if self.use_model:
                print('Collision')
            done = True
            if self.evaluate:
                self.plot_result(success=False)
            return obs, reward, done, info

        # Normal step
        return obs, reward, done, info

    def get_eval_goals(self, episodes=50):
        perimeter = 4 * 2 * self.square_distance
        locations = np.linspace(0.0, perimeter, num=episodes, endpoint=False)
        corners = [perimeter / 8, 3 * perimeter/ 8, 5 * perimeter / 8, 7 *
                perimeter/ 8]
        middles = [perimeter / 4, perimeter/ 2, 3 * perimeter / 4, perimeter]
        goals = []
        for loc in locations:
            if loc < corners[0]:
                goals.append((self.square_distance, loc))
            elif loc < corners[1]:
                goals.append((middles[0] - loc, self.square_distance))
            elif loc < corners[2]:
                goals.append((-self.square_distance, middles[1] - loc))
            elif loc < corners[3]:
                goals.append((-(middles[2] - loc), -self.square_distance))
            else:
                goals.append((self.square_distance, -(perimeter - loc)))
        return goals

    def plot_init(self):
        #plt.ion() # interactive on
        fig, self.ax = plt.subplots()
        plt.title('Evaluation Results')
        plt.xlabel('meters')
        plt.ylabel('meters')
        self.ax.scatter(0, 0, marker='+', color='black')
        plt.draw()
        plt.pause(0.1)

    def plot_result(self, success=False, path=True):
        marker_val = 'x'
        if success:
            marker_val = 'o'
        self.ax.scatter(self.goal[0], self.goal[1], marker=marker_val)
        if path is not None:
            x = [a[0] for a in self.path]
            y = [a[1] for a in self.path]
            self.ax.plot(x, y)
        plt.draw()
        plt.pause(0.1)

    def plot_save(self, filename='plot'):
        plt.savefig(filename)
