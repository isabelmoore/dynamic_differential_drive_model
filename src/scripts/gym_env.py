#!/usr/bin/env python3

# ROS
#import rospy
#import tf # does not work for Python3
#from nav_msgs.msg import Odometry
#from geometry_msgs.msg import Twist
#from std_srvs.srv import Empty
#from controller_manager_msgs.srv import (SwitchController,
#        SwitchControllerRequest, SwitchControllerResponse)
#from gym_gazebo.srv import Step, StepRequest, StepResponse

# Python
import gym
from abc import abstractmethod
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation
import dubins


#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging

# Local
from vehicle_models_copy.differential_drive_model import DifferentialDriveModel
# import environment as env
from path_following.pure_pursuit import PurePursuit 
from pid import PID
# from vehicle_models_copy.bicycle_model import BicycleModel
from vehicle_models_copy.dynamic_differential_drive_model import BicycleModel

import unittest

def plot_init():
    plt.ion # interactive on
    fig, ax = plt.subplots()
    ax.set_title('Trajectory')
    ax.set_xlabel('meters')
    ax.set_ylabel('meters')
    ax.set_aspect('equal')
    plt.pause(0.1)
    return fig, ax

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def yaw_error(target, current):
    error_raw = target - current
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

'''
Returns list of waypoints
'''
def generate_waypoints(num_points, start_angle=0, radius=0.1):
    angle = start_angle
    x = []
    y = []
    x.append(0) # 0, 0 starting location
    y.append(0)
    for i in range(0, num_points):
        angle = random.uniform(-math.pi/8 + angle, math.pi/8 + angle)
        x.append(radius * np.cos(angle) + x[-1])
        y.append(radius * np.sin(angle) + y[-1])

    return np.array([x, y])

'''
Returns list of Dubins paths
'''
def generate_dubins(start_angle=0, length=50,bin = 1,interval1=20,interval2=10):
    count=0
    def choose_end(start,count):
        end = [0, 0, 0]
        # for i in range(2):
        #     # delta = random.choice([-1, 1]) * random.uniform(0, length/3)
        #     if(random.uniform(0,100)<=10):
        #         delta = -1 * random.uniform(0, length/5)
        #     else:
        #         delta = random.uniform(0, length/5)
        #     end[i] = start[i] + delta
        # end[2] = random.uniform(-math.pi, math.pi)
        if(count==0):
            end[2] = random.choice([-math.pi/4,math.pi/4])
        else:
            end[2] = random.uniform(-math.pi/3, math.pi/3)
        r = random.uniform(interval1,interval2)
        end[0] = start[0] + r*math.cos(end[2])
        end[1] = start[1] + r*math.sin(end[2])
        return tuple(end)

    p0 = (0, 0, start_angle)
    p1 = choose_end(p0,count)
    count = count + 1

    if(bin == 1):
        radius = (interval2-interval1)/1
    elif(bin == 2):
        radius = (interval2-interval1)/2
    elif(bin == 3):
        radius = (interval2-interval1)/4
    elif(bin == 4):
        radius = (interval2-interval1)/8
    elif(bin == 5):
        radius = (interval2-interval1)/16
    elif(bin == 6):
        radius = (interval2-interval1)/16

    current_length = 0
    paths = []
    while current_length < length:
        paths.append(dubins.shortest_path(p0, p1, radius))
        current_length += paths[-1].path_length()
        # Update start and end poses
        p0 = p1
        p1 = choose_end(p0,count)
        count = count + 1

    return paths

'''
Returns waypoints from list of Dubins paths
'''
def get_dubins_waypoints(dubins_paths, spacing=0.001):
    x = []
    y = []
    for path in dubins_paths:
        poses, _ = path.sample_many(spacing)
        for pose in poses:
            x.append(pose[0])
            y.append(pose[1])

    return np.array([x, y])

    '''
    # Examples of other Dubins methods
    print(path.path_length())
    print(path.path_type())
    for i in range(3):
        print(path.segment_length(i))
    print(path.sample(7.5))
    #print(path.sample(7.6))
    '''

class MobileRobotPathTrackEnv(gym.Env):
    def __init__(self, timestep=0.1, yaw_controller_frequency=50,
            path_length=10.0, trajectory_length=15, velocity=1.0,
            observation_lookahead=2.5, use_dubins=False, use_seed=False,
            evaluate=False,radiusOfCBin=1):
        # Yaw controller frequency in AGC stack is 50 Hz

        self.radiusOfCBin = radiusOfCBin
        self.yaw_controller_frequency = yaw_controller_frequency
        self.path_length_dubins = path_length
        self.trajectory_length = trajectory_length
        self.velocity = velocity
        self.observation_lookahead = observation_lookahead
        self.use_dubins = use_dubins
        self.use_seed = use_seed
        self.evaluate = evaluate

        self.waypoint_spacing = 1e-1
        self.num_path_points = int(self.path_length_dubins / self.waypoint_spacing)
        self.num_episode_steps = int(self.trajectory_length /
                self.waypoint_spacing)
        self.observation_points = int(self.observation_lookahead /
                self.waypoint_spacing)
        
        #Random Dubins path parameters
        self.interval1 = self.path_length_dubins/10
        self.interval2 = self.interval1*2
        
        self.seed = None
        if self.use_seed:
            self.seed = 8
        
        # Action space
        self.normalized_action_scale = [20.0,20.0]#[lookAhead,velocity]
        self.normalized_action_offset = [0.1,0.1]#[lookAhead,velocity]
        self.action_space = gym.spaces.Box(low=0., high=1., shape=(6,),
                    dtype=np.float16)

        self.history_length = 10
        # Observation space
        self.observation_space = gym.spaces.Dict(spaces={
            'relative_trajectory': gym.spaces.Box(low=-1., high=1.,
                shape=(2, self.observation_points), dtype=np.float16),
            'state': gym.spaces.Box(low=-1., high=1., shape=(7,),
                dtype=np.float16),
            'previous_state': gym.spaces.Box(low=-1., high=1., shape=(self.history_length,7),
                dtype=np.float16),
            'previous_input':gym.spaces.Box(low=-1., high=1., shape=(self.history_length,2),
                dtype=np.float16),
            'weights_fb':gym.spaces.Box(low=-1., high=1., shape=(1,4),
                dtype=np.float16),
        })
        self.previous_states = np.zeros((self.history_length,7),dtype=np.float16)
        self.previous_input = np.zeros((self.history_length,2), dtype=np.float16)
        self.weights_fb = np.zeros((1,4), dtype=np.float16)
        
        # Model Parameters
        self.lf = 1.0 # length from center of mass to front axle
        self.lr = 1.0 # length from center of mass to rear axle

        # Model
        self.model = BicycleModel(center_to_front=self.lf, 
                center_to_rear=self.lr, plot=False)

        self.num_episodes = 0

        #Model Parameters
        self.acceleration = 0.0

        #Reward Parameters
        self.delta_action = 0.0
        self.delta_previous = 0.0

        #Action Parameters
        self.desiredVelocity = 20.0*0.45 #convert to mpH

        #Time Parameters
        self.finalSimTime = 25.0
        self.timestep = timestep

        """Dynamic Plot"""
        if(self.evaluate):
            self.dynamicPlot = True
            self.finalSimTime = 150.0
        else:
            self.dynamicPlot = False

        """Scaling Factor"""
        self.scalingFactor = self.path_length_dubins * 10.0
    
    """Observation function"""
    def observe(self):
        obs = {}

        self.desiredPoseIdx = np.argmin(abs(self.path_length_current_time \
                                            - self.t_current))
        
        """ Get the distance to the closest point on the self.path """
        x = self.pose[0]
        y = self.pose[1]
        x_path = self.path[0]
        y_path = self.path[1]
        #
        dist = np.sqrt((x-x_path)**2 + (y-y_path)**2)
        closest_point_index = np.argmin(dist)
        #Get the cte and distance error
        self.cte = dist[closest_point_index]
        self.distError = abs(self.path_length[self.desiredPoseIdx] - \
                self.path_length[closest_point_index])
        
        if(closest_point_index + (self.observation_points)<=len(self.path[0])):
            obs_path_array = np.zeros((2,self.observation_points))
            obs_path_array[0,:] = self.path[0][closest_point_index:closest_point_index \
                                               + (self.observation_points)]
            obs_path_array[1,:] = self.path[1][closest_point_index:closest_point_index \
                                                  + (self.observation_points)]
        else:
            obs_path_array = np.zeros((2,int(self.observation_points)))
            obs_path_array[0,:] = self.path[0][-1]
            obs_path_array[1,:] = self.path[1][-1]

            obs_path_array[0,:len(self.path[0,closest_point_index:])] = \
                    self.path[0,closest_point_index:]
            obs_path_array[1,:len(self.path[1,closest_point_index:])] = \
                    self.path[1,closest_point_index:]
            

        self.obs_path_array = obs_path_array

        """Add the current state to the previous state buffer and remove the oldest state"""
        pose_ = np.reshape(self.pose,(1,7))/self.scalingFactor
        self.previous_states = np.append(pose_,self.previous_states,axis=0)
        self.previous_states = self.previous_states[:-1]

        """Add the current torque and steering to the previous torque buffer and remove the oldest torque"""
        t_ = np.reshape(self.actions,(1,2))/self.scalingFactor
        self.previous_input = np.append(t_,self.previous_input,axis=0)
        self.previous_input = self.previous_input[:-1]

        """Add the current torque and steering to the previous torque buffer and remove the oldest torque"""
        w_ = np.reshape(self.weights,(1,4))
        # self.weights_fb = np.append(t_,self.previous_input,axis=0)
        self.weights_fb = w_#self.weights_fb[:-1]

        obs['previous_input'] = self.previous_input

        obs['previous_state'] = self.previous_states

        obs['weights_fb'] = self.weights_fb

        obs['relative_trajectory'] = obs_path_array/self.scalingFactor

        obs['state'] = self.pose/self.scalingFactor

        return obs

    def reset(self):
        # Reset pose
        self.pose = np.zeros(7) #Bicycle model dynamic model
        self.prevPose = np.zeros(7) #Bicycle model dynamic model previous pose
        self.actions = np.zeros(2)
        self.weights = np.zeros(4)
        self.prevDeltaSteering = 0.0
        self.prevVelocity = 0.0
        # TODO: randomize starting yaw

        #reset steering wheel reward params
        self.delta_action = 0.0
        self.delta_previous = 0.0

        # Generate desired trajectory
        if self.use_seed:
            self.seed += 1 # reset the random seed
        if self.use_seed:
            random.seed(self.seed)

        if self.use_dubins:
            dubins_paths = generate_dubins(start_angle=self.pose[2],
                    length=self.path_length_dubins,bin=self.radiusOfCBin,\
                        interval1=self.interval1,interval2=self.interval2)
            self.path = get_dubins_waypoints(dubins_paths,
                    spacing=self.waypoint_spacing)
            self.path = self.path[:, :self.num_path_points]
        else:
            self.path = generate_waypoints(num_points=self.num_path_points,
                radius=self.waypoint_spacing)
        
        """Reset the timer"""
        self.t_current = 0.0

        """Generate the length vector of the path"""
        self.path_length = np.zeros(self.path.shape[1])
        for i in range(1,self.path.shape[1]):
            self.path_length[i] = self.path_length[i-1] + \
                    distance(self.path[:,i],self.path[:,i-1])
        #Path length current time vector
        self.path_length_current_time = self.path_length/self.desiredVelocity

        # Pure Pursuit controller
        self.ppc = PurePursuit(self.path, spacing = self.waypoint_spacing,dAnchor = -1 *self.lr)

        #Torque Controller
        self.tc = PID(kp=2000, ki=0.0, kd=0.0, satLower=-5880.0, satUpper=5880.0)

        # Initial observation
        self.episode_steps = 0
        obs = self.observe()

        self.episode_reward = 0
        # self.actions = np.zeros((1, self.num_episode_steps))
        self.lookahead_points = np.zeros((2, self.num_episode_steps))
        # self.poses = np.zeros((len(self.pose), self.num_episode_steps + 1))
        # self.poses[:, 0] = self.pose

        """Reset the data loggers"""
        self.F_trac =  [] #List to store the tractive force
        self.F_roll =  [] #List to store the rolling resistance force
        self.F_drag =  [] #List to store the drag force
        self.lambda_ = [] #List to store the slip ratio
        self.F_f = [] #List to store the front wheel force
        self.F_r = [] #List to store the rear wheel force
        self.alpha_f = [] #List to store the front wheel slip angle
        self.alpha_r = [] #List to store the rear wheel slip angle
        self.desired_torque = [] #List to store the desired torque
        self.steer_angle = [] #List to store the desired steering angle
        self.poses = [] #List to store the pose of the vehicle
        self.sideSlipRatio = [] #List to store the side slip ratio of the vehicle
        self.weights_overall = [] #List to store the weights of the model
        self.velSetPoint_RL = [] # List to store the velocity set point
        self.velSetPoint_RL.append(0.0)
        self.avgVel_RL = [] # List to store the average velocity of the vehicle
        self.smoothVel_RL = [] # List to store the smoothed velocity

        self.acceleration = 0.0 #Reset acceleration to 0.0

        self.velFiltered = 0.0 #Reset the filtered velocity value

        
        self.num_episodes += 1
        # print("In reset",self.pose)

        """Initialize the figure logger"""
        if(self.dynamicPlot == True):
            fig,self.ax = plot_init()

        self.obs_path_array = np.zeros((2,self.observation_points))
        self.previous_states = np.zeros((self.history_length,7),dtype=np.float16)
        self.previous_input = np.zeros((self.history_length,2),dtype=np.float16)
        self.weights_fb = np.zeros((1,4),dtype=np.float16)

        self.friction_surface = random.choice([0.1,0.4,0.7,1.0])
        self.friction_surface = 1.0

        # input("Press Enter to continue...")
        return obs

    def reward(self):
        reward_pose = 0.0
        reward_percentage = 0.0
        reward_percentage_torq = 0.0
        desired_pose = np.array([self.path[0][self.desiredPoseIdx],\
                                 self.path[1][self.desiredPoseIdx]])
        if(self.t_current>0.0):
            reward_pose = -1*(self.cte + self.distError)
            reward_pose = reward_pose/self.scalingFactor
            if(abs(self.prevDeltaSteering - self.delta_steering) > math.pi/4):
                #print("High change in steering")
                reward_percentage = -10e-2
            self.smooth()
            self.smoothVel_RL.append(self.velFiltered)
            if(abs(self.velFiltered - self.pose[3]) > 0.1):
                #print("High change in vel")
                reward_percentage_torq = -10e-2



        reward = (10*reward_pose)  + reward_percentage + reward_percentage_torq
        # print("Reward",reward)
        return reward


    def smooth(self):  # Weight between 0 and 1
        weight = 94e-2
        velocities = np.array(self.poses)[:,3]
        H = 4
        #if(len(velocities)>2*H+1):
        #    mean_vel = np.mean(velocities[-1-2*H:-1-H])

        #else:
        smoothed_val = self.velFiltered * weight + (1 - weight) * self.pose[3]  # Calculate smoothed value


        self.velFiltered = smoothed_val                        # Anchor the last smoothed value

            


    def scale_action(self, action):
        lookahead = self.normalized_action_scale[0] * action[0] + \
                self.normalized_action_offset[0] # [0.0, 2.5]
        
        velocity = self.normalized_action_scale[1] * action[1] + \
                self.normalized_action_offset[1]
        # velocity = action[1]
        # lookahead = action[0]

        return lookahead,velocity
    

    def compute_inputs(self,action):

        """Compute Torque input"""
        velocitySetPoint = action[1] * self.normalized_action_scale[1] #Convert to mpH
        v_x = self.pose[3]
        velocityError = velocitySetPoint - v_x
        desiredTorque = self.tc.computeControl(velocityError)

        """Compute Steering Angle input"""
        lookahead_point_input = (action[0] * self.normalized_action_scale[0]) + \
                self.normalized_action_offset[0]
        self.ppc.setLookahead(lookahead_point_input)
        curvature, lookahead_point, closest_distance = self.ppc.run(self.pose[:3])
        delta = math.atan(curvature * (self.lf + self.lr))
        delta = np.clip(delta,-math.pi/4,math.pi/4)

        # print("\n\n\n\n Actions : ",velocitySetPoint,lookahead_point_input)
        return desiredTorque,delta,lookahead_point_input,velocitySetPoint


    def step(self, action):
        
        #if(self.episode_steps > 150):
        #    self.friction_surface = 0.1
        """Get the desired action and pass it as an input to the model"""
        desiredTorque,delta,lookahead_point_input,velSetPoint = self.compute_inputs(action)
        self.delta_steering = delta
        self.actions = np.array([desiredTorque,delta])
        self.weights = np.array([action[2],action[3],action[4],action[5]])
        """Call the updated Runge-Kutta 4th order dynamic model"""
        self.pose, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r \
            = self.model.step_dynamic_torque_scipy_rk(self.pose, (desiredTorque, delta), \
                    timestep = self.timestep,t_init = self.t_current, \
                        t_bound = self.t_current + self.timestep,lambda_friction=self.friction_surface)
        # print("Pose inside step after model step",self.pose)    
        # self.actions[0, self.episode_steps] = lookahead_point_input
        # self.lookahead_points[:, self.episode_steps] = lookahead_point
        # self.poses[:, self.episode_steps + 1] = self.pose
        self.current_vel = self.pose[3]#Current velocity
        self.poses.append(self.pose)
        self.F_trac.append(f_trac)
        self.F_roll.append(f_roll)
        self.F_drag.append(f_drag)
        self.lambda_.append(slip_ratio)
        self.F_f.append(F_f)
        self.F_r.append(F_r)
        self.alpha_f.append(alpha_f*180/math.pi)
        self.alpha_r.append(alpha_r*180/math.pi)
        self.sideSlipRatio.append(math.atan2(self.pose[5],self.pose[3]))
        """Store the actions"""
        self.desired_torque.append(desiredTorque)
        self.steer_angle.append(delta*180/math.pi)
        self.weights_overall.append(self.weights)
        self.velSetPoint_RL.append(velSetPoint)

        self.episode_steps += 1
        self.t_current += self.timestep

        obs = self.observe()

        reward = self.reward()
        self.prevPose = self.pose
        self.prevDeltaSteering = self.delta_steering
        self.prevVelocity = self.current_vel
        self.episode_reward += reward

        if(self.dynamicPlot and self.episode_steps%10 == 0):
            self.plot_pose()

        done = False
        #if self.episode_steps == self.num_episode_steps:
        if (self.t_current >= self.finalSimTime):
            done = True
            reward = -100.0
            print("Time limit reached, friction surface: ",self.friction_surface," episode rwd: ",self.episode_reward)
            if(self.evaluate):
                self.plot()
            return obs, reward, done, {}
        
        if (self.getL2norm(self.pose[0],self.pose[1],self.path[0][-1],self.path[1][-1]) < 7e0):
            done = True
            reward = +1.0
            print("Reached the end of the path in :",self.t_current,"seconds"," friction surface: ",self.friction_surface," episode rwd: ",self.episode_reward)
            if(self.evaluate):
                self.plot()
            return obs, reward, done, {}
        
        elif (self.getL2norm(self.pose[0],self.pose[1],self.path[0][-1],self.path[1][-1]) > 10000.0):
            done = True
            reward = -1000.0
            print("Current time:",self.t_current)
            print("Out of bounds")
            print("Pose",self.pose)
            """Print last 15 poses"""
            print("Last 15 poses",self.poses[-150:])
            exit()
            return obs, reward, done, {}

            # print('episode reward: %f' %self.episode_reward)
            # print('actions: max: %f, min: %f' %(max(self.actions[0]),
            #     min(self.actions[0])))

        info = {}
        # print("Step done")
        # print("Observation",obs)
        return obs, reward, done, info
    
    def getL2norm(self,poseX,poseY,finalPointX,finalPointY):
        """Get the L2 norm of the pose and the final point"""
        # print("\n\n\n\n\n\n_____poses_____",poseX,poseY,finalPointX,finalPointY)
        return math.sqrt((poseX-finalPointX)**2 + (poseY-finalPointY)**2)

    def plot(self):

        poses = np.array(self.poses)

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(poses[:,3]*2.23,label='Velocity of the vehicle')
        plt.plot(np.array(self.smoothVel_RL)*2.23,label="Smoothed velocity of the vehicle")
        plt.legend()
        plt.grid()

        plt.subplot(2,1,2)
        plt.plot(self.steer_angle,label='Steering wheel angle')
        plt.legend()
        plt.grid()

        #plt.subplot(3,1,3)
        #avg_vel = np.convolve(poses[:,3],np.ones(len(poses[:,3]))/len(poses[:,3]),mode='valid')
        #plt.plot(avg_vel,label='Average velocity')
        #plt.legend()
        #plt.grid()


        
        plt.figure()
        #np.savetxt("RLPC_x_asphalt.txt",poses[:,0])
        #np.savetxt("RLPC_y_asphalt.txt",poses[:,1])       
        plt.plot(poses[:,0],poses[:,1],label="Trajectory vehicle (x,y))")
        plt.plot(self.path[0,:],self.path[1,:],label="Global Path - given")
        #plot the lookahead points
        # lookahead_points_ = np.array(lookahead_points_)
        # plt.plot(lookahead_points_[:,0],lookahead_points_[:,1],'.',label="lookahead points")
        plt.legend()
        plt.grid()

        plt.figure()
        
        poses = np.array(self.poses)
        plt.plot(poses[:,0][:-130],poses[:,1][:-130],label="Trajectory vehicle (x,y))")
        plt.plot(self.path[0,:-100],self.path[1,:-100],label="Global Path - given")
        #plot the lookahead points
        # lookahead_points_ = np.array(lookahead_points_)
        # plt.plot(lookahead_points_[:,0],lookahead_points_[:,1],'.',label="lookahead points")
        plt.legend()
        plt.grid()

        plt.figure()
        weights_ = np.array(self.weights_overall)

        plt.subplot(4,1,1)
        plt.plot(weights_[:,0][:],label='weight 1')
        plt.grid()
        plt.legend()

        plt.subplot(4,1,2)
        plt.plot(weights_[:,1][:],label='weight 2')
        plt.grid()
        plt.legend()
        plt.subplot(4,1,3)
        plt.plot(weights_[:,2][:],label='weight 3')
        plt.grid()
        plt.legend()
        plt.subplot(4,1,4)
        plt.plot(weights_[:,3][:],label='weight 4')
        plt.grid()
        plt.legend()



        plt.figure()

        plt.subplot(9,1,1)
        plt.plot(poses[:,3]*2.24,label="v_longitudinal in mpH")
        #add the velocity set point
        plt.plot(np.ones(len(poses[:,3]))*self.desiredVelocity*2.24,label="velocity set point in mpH")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,2)
        plt.plot(poses[:,4],label="omega_wheel")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,3)
        plt.plot(poses[:,5]*2.24,label="v_lateral in mpH")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,4)
        plt.plot(poses[:,6],label="omega_body")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,5)
        plt.plot(np.array(self.desired_torque),label="Torque")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,6)
        plt.plot(np.array(self.steer_angle),label="Steering Angle")
        plt.legend()
        plt.grid()
        
        plt.subplot(9,1,7)
        plt.plot(poses[:,0],label="x")
        plt.legend()
        plt.grid()

        plt.subplot(9,1,8)
        plt.plot(poses[:,1],label="y")
        plt.legend()
        plt.grid()
        
        plt.subplot(9,1,9)
        plt.plot(np.array(self.sideSlipRatio)*180/math.pi,label="side slip angle")
        plt.legend()
        plt.grid()



        """plot f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r as subplots"""
        plt.figure()
        plt.subplot(4,2,1)
        plt.plot(np.array(self.F_trac),label="f_trac")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,2)
        plt.plot(np.array(self.F_roll),label="f_roll")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,3)
        plt.plot(np.array(self.F_drag),label="f_drag")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,4)
        plt.plot(np.array(self.lambda_),label="slip_ratio")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,5)
        plt.plot(np.array(self.F_f),label="F_f")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,6)
        plt.plot(np.array(self.F_r),label="F_r")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,7)
        plt.plot(np.array(self.alpha_f),label="alpha_f")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,8)
        plt.plot(np.array(self.alpha_r),label="alpha_r")
        plt.legend()
        plt.grid()

        


        # fig.savefig('../plots/episode%d.png' %self.num_episodes)
        plt.show()

   

    def plot_pose(self):
        x = self.pose[0]
        y = self.pose[1]
        yaw = self.pose[2]
        lr = 1.0
        lf = 1.0
        # print("Pose",x,y,yaw)
        #print((yaw + steer_angle) * 180 / math.pi)

        # vehicle axles
        # xr = x - lr * math.cos(yaw)
        # yr = y - lr * math.sin(yaw)
        # xf = x + lf * math.cos(yaw)
        # yf = y + lf * math.sin(yaw)

        # obs_path_array = get_observation(path,desiredIdx,obs_distance,waypoint_spacing)

        line1 ,= self.ax.plot(self.path[0, :], self.path[1, :], color='red',label='Global Path') # path
        line2 ,= self.ax.plot(x, y,marker = '.' ,color='blue',label='Current position of the vehicle') # body center
        line3 ,= self.ax.plot(self.path[0,self.desiredPoseIdx],self.path[1,self.desiredPoseIdx],marker = '.',color='green',label='Desired position of the vehicle') # desired point
        # line4 ,= # self.ax.plot(self.path[0,self.desiredPoseIdx:self.desiredPoseIdx+1000],path[1,self.desiredPoseIdx:self.desiredPoseIdx+1000],color='black') # desired point
        line4 ,= self.ax.plot(self.obs_path_array[0,:],self.obs_path_array[1,:],color='black',label='Observation') # desired point
        #Display the current time instant, action and reward in the plot
        # self.ax.set_title('Trajectory')
        self.ax.set_xlabel('meters')
        self.ax.set_ylabel('meters')
        # self.ax.set_aspect('equal')
        # self.ax.text(0.0,0.0,'Time: '+str(round(self.t_current,2))+'s')
        # self.ax.text(0.0,0.0,'Action: '+str(round(self.desired_torque[-1],2))+'Nm'+str(round(self.steer_angle[-1],2))+'deg')
        # self.ax.text(0.0,0.0,'Reward: '+str(round(reward,2)))
        # ax.plot([xr, xf], [yr, yf], color='blue') # body
        # ax.arrow(xf, yf, math.cos(yaw + steer_angle),
                # math.sin(yaw + steer_angle), color='green') # front wheel vector
        # plt.legend()
        self.ax.legend(['Global Path', 'Current position of the vehicle', 'Desired position of the vehicle','ObsObservation'])
        plt.pause(0.1)
