#!/usr/bin/env python3

import math
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import logging
import dubins

# Local
from vehicle_models.dynamic_differential_drive_model import DifferentialDriveModel
from path_following.pure_pursuit import PurePursuit
from pid import PID


import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

def plot_init():
    plt.ion()  # interactive on
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title('Trajectory')
    ax.set_xlabel('meters')
    ax.set_ylabel('meters')
    ax.set_aspect('equal')
    plt.pause(0.1)
    return fig, ax

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def yaw_error(target, current):
    error_raw = target - current
    if abs(error_raw) > np.pi:
        error = 2*np.pi - abs(error_raw)
        if error_raw > 0:
            error = -error
    else:
        error = error_raw
    return error

def generate_waypoints(num_points, start_angle=0, radius=0.1):
    """
    Returns list of pseudo-random waypoints.
    """
    angle = start_angle
    x = [0.0]
    y = [0.0]
    for _ in range(num_points):
        angle = random.uniform(-math.pi/8 + angle, math.pi/8 + angle)
        x.append(radius * math.cos(angle) + x[-1])
        y.append(radius * math.sin(angle) + y[-1])
    return np.array([x, y])

def generate_dubins(start_angle=0, length=50, bin=1, interval1=20, interval2=10):
    """
    Returns list of Dubins paths for a random route of approximate length.
    """
    count = 0

    def choose_end(start, count):
        end = [0,0,0]
        if count == 0:
            end[2] = random.choice([-math.pi/4, math.pi/4])
        else:
            end[2] = random.uniform(-math.pi/3, math.pi/3)
        r = random.uniform(interval1, interval2)
        end[0] = start[0] + r*math.cos(end[2])
        end[1] = start[1] + r*math.sin(end[2])
        return tuple(end)

    p0 = (0, 0, start_angle)
    p1 = choose_end(p0, count)
    count += 1

    if bin == 1:
        radius = (interval2 - interval1)
    elif bin == 2:
        radius = (interval2 - interval1)/2
    elif bin == 3:
        radius = (interval2 - interval1)/4
    else:
        radius = 1.0

    current_length = 0
    paths = []
    while current_length < length:
        spath = dubins.shortest_path(p0, p1, radius)
        paths.append(spath)
        current_length += spath.path_length()
        p0 = p1
        p1 = choose_end(p0, count)
        count += 1

    return paths

def get_dubins_waypoints(dubins_paths, spacing=0.005):
    """
    Returns (2,N) array of x,y from list of Dubins paths.
    """
    x, y = [], []
    for path in dubins_paths:
        poses, _ = path.sample_many(spacing)
        for pose in poses:
            x.append(pose[0])
            y.append(pose[1])
    return np.array([x, y])


class MobileRobotPathTrackEnv(gym.Env):
    def __init__(
                    self, 
                    timestep=1e-1, 
                    yaw_controller_frequency=25,
                    path_length=50, 
                    trajectory_length=15, 
                    velocity=0.1,
                    observation_lookahead=100, 
                    use_dubins=True, 
                    use_seed=True,
                    evaluate=True,
                    radiusOfCBin=5):   
        """ 
        A path-tracking environment, originally for a bicycle model, now replaced
        by a differential-drive model but preserving the same structure, logs,
        prints, subplots, etc.
        """
        super(MobileRobotPathTrackEnv, self).__init__()

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
        self.num_episode_steps = int(self.trajectory_length / self.waypoint_spacing)
        self.observation_points = int(self.observation_lookahead / self.waypoint_spacing)

        self.interval1 = self.path_length_dubins/10
        self.interval2 = self.interval1*2

        self.seed_val = None
        if self.use_seed:
            self.seed_val = 8

        # ACTION SPACE: 6D -> [omega_L, omega_R, w1, w2, w3, w4]
        self.normalized_action_scale = [20.0, 20.0]
        self.normalized_action_offset = [0.1, 0.1]
        low  = np.array([-1.,-1.,-1.,-1.,-1.,-1., 0., 0., 0.], dtype=np.float32)
        high = np.array([ 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(9,), dtype=np.float32)

        # OBSERVATION SPACE
        self.history_length = 10
        self.observation_space = gym.spaces.Dict({
            'relative_trajectory': gym.spaces.Box(
                low=-1., high=1.,
                shape=(2, self.observation_points),
                dtype=np.float32
            ),
            'state': gym.spaces.Box(low=-1., high=1., shape=(5,), dtype=np.float32),
            'previous_state': gym.spaces.Box(
                low=-1., high=1., shape=(self.history_length,5), dtype=np.float32
            ),
            'previous_input': gym.spaces.Box(
                low=-1., high=1., shape=(self.history_length,2), dtype=np.float32
            ),
            'weights_fb': gym.spaces.Box(
                low=-1., high=1., shape=(1,4), dtype=np.float32
            ),
            'gains_fb': gym.spaces.Box(
                low=0., high=9., shape=(1,3), dtype=np.float32
            )
        })
        self.previous_states = np.zeros((self.history_length,5), dtype=np.float32)
        self.previous_input = np.zeros((self.history_length,2), dtype=np.float32)
        self.weights_fb = np.zeros((1,4), dtype=np.float32)
        self.gains_fb = np.zeros((1,3), dtype=np.float32)

        # MODEL: Differential Drive
        self.model = DifferentialDriveModel(
            center_to_front=0.262/2,                    # [m] from vehicle center to front
            center_to_rear=0.262/2,                     # [m] from vehicle center to rear 
            mass=17.0,                                  # [kg] typical Jackal mass
            inertia=1.0,                                # [kg m^2] much smaller than a car
            tire_radius=0.10,                           # [m] from drawing (R0.1000m)
            cornering_stiffness_front=580.0,            # [N/rad] # (17 kg [mass]* 9.8134.1 [g] * 0.7 [friction coeff]) / (2 wheels * 6 degrees [range]) 
            cornering_stiffness_rear=580.0,             # [N/rad]
            longitudinal_stiffness=1500.0,              # [N], for driving/braking
            plot=False,
            controlObjects=None, 
            title=None 
        )

        self.num_episodes = 0


        self.pose = np.zeros(5, dtype=np.float32) # [x, y, theta, wL, wR]

        # Reward, time, and other parameters
        # self.desiredVelocity = 20.0 * 0.45 ## 20 mph in m/s
        
        self.desiredVelocity = 1.0  # 1 m/s [max is 2 m/s]
        self.finalSimTime = 25.0
        self.timestep = timestep
        
        """Dynamic Plot"""
        if self.evaluate:
            self.dynamicPlot = True
            self.finalSimTime = 150.0
        else:
            self.dynamicPlot = False

        # For friction, etc. just keep the variable
        self.friction_surface = 1.0
        # self.friction_surface = random.choice([0.1,0.4,0.7,1.0])

        """Scaling Factor"""
        self.scalingFactor = self.path_length_dubins * 10.0

        self.episode_steps = 0
        self.episode_reward = 0
        self.t_current = 0.0

        # Large logs
        self.F_trac = []
        self.F_roll = []
        self.F_drag = []
        self.lambda_ = []
        self.F_f = []
        self.F_r = []
        self.alpha_f = []
        self.alpha_r = []
        self.desired_torque = []
        self.steer_angle = []
        self.poses = []
        self.sideSlipRatio = []
        self.weights_overall = []
        self.gains_overall = []
        self.velSetPoint_RL = []
        self.avgVel_RL = []
        self.smoothVel_RL = []
        self.ctes = []

        # Action-related
        self.actions = np.zeros(2)
        self.weights = np.zeros(4)
        self.delta_steering = 0.0
        
        # Debugging
        self.closest_idx_plot = None
        self.look_pt_plot = None

        self.vel_kp = 3
        self.omega_kp = 7
        self.psi_kp = 1.5
        
        # Initialize controllers
        self.vel_pid = PID(kp=self.vel_kp, ki=0.0, kd=0, satLower=-300.0, satUpper=300.0)
        self.omega_pid = PID(kp=self.omega_kp, ki=0.0, kd=0.0, satLower=-300.0, satUpper=300.0)

        init_gains = np.array([4.0, 9.0, 2], dtype=np.float32)    # [kp_vel, kp_omega, kp_psi]
        half_range = np.array([4.0, 4.0, 2.0], dtype=np.float32)   # how far up/down from init you’ll allow

        # store for later
        self.gains_scale  = half_range       # multiply normalized action by this
        self.gains_offset = init_gains       # then add this
        
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('differential_drive_eval', anonymous=True)

        self.pose = np.zeros(5)  # [x, y, yaw, wL, wR]
        self.odom_received = False

        self.sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def seed(self, seed=None):
        """
        If you want to implement seeding
        """
        pass

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        linear    = msg.twist.twist.linear
        angular   = msg.twist.twist.angular

        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quat)

        # Optional: estimate wheel speeds from linear/angular velocities
        r = self.model.r
        W = self.model.W
        v = linear.x
        w = angular.z
        wL = (v - w * W / 2) / r
        wR = (v + w * W / 2) / r

        self.pose = np.array([position.x, position.y, yaw, wL, wR])
        self.odom_received = True

    def observe(self):
        """
        Gather and return the current state of the environment, which is typically used by the RL agent to decide its next action.
        """
        
        obs = {}

        # Compute desired pose index based on the current time.
        self.desiredPoseIdx = np.argmin(abs(self.path_length_current_time \
                                            - self.t_current))
                
        # Get the distance from the current (x,y) to each point in the path.
        x = self.pose[0]
        y = self.pose[1]
        x_path = self.path[0]
        y_path = self.path[1]
        dist = np.sqrt((x - x_path)**2 + (y - y_path)**2)
        
        # Find the closest point on the path to the current pose.
        # avoids re-checking distances that have already been processed, improving performance
        # It updates the state (self.last_closest_idx) to ensure the process is incremental and avoids redundant work
        window = dist[self.last_closest_idx:]
        idx_offset = np.argmin(window)
        closest_point_index = self.last_closest_idx + idx_offset
        self.last_closest_idx = closest_point_index        
        self.closest_idx_plot = closest_point_index
        
        # Set cross-track error and distance error.
        self.cte = dist[closest_point_index]
        self.ctes.append(self.cte)
        self.distError = abs(self.path_length[self.desiredPoseIdx] - self.path_length[closest_point_index])
        
        # Build the observation path array (the relative trajectory).
        if (closest_point_index + self.observation_points) <= len(self.path[0]):
            obs_path_array = np.zeros((2, self.observation_points))
            obs_path_array[0, :] = self.path[0][closest_point_index : closest_point_index + self.observation_points]
            obs_path_array[1, :] = self.path[1][closest_point_index : closest_point_index + self.observation_points]
        else:
            obs_path_array = np.zeros((2, int(self.observation_points)))
            obs_path_array[0, :] = self.path[0][-1]
            obs_path_array[1, :] = self.path[1][-1]
            obs_path_array[0, :len(self.path[0, closest_point_index:])] = self.path[0, closest_point_index:]
            obs_path_array[1, :len(self.path[1, closest_point_index:])] = self.path[1, closest_point_index:]
        
        self.obs_path_array = obs_path_array

        # Update previous state buffer.
        # 5-dimensional self.pose => [x, y, theta, wL, wR]
        pose_ = np.reshape(self.pose, (1, 5)) / self.scalingFactor
        self.previous_states = np.append(pose_, self.previous_states, axis=0)
        self.previous_states = self.previous_states[:-1]

        # Update previous input buffer (for torque/steering, which remains 2-dimensional).
        t_ = np.reshape(self.actions, (1, 2)) / self.scalingFactor
        self.previous_input = np.append(t_, self.previous_input, axis=0)
        self.previous_input = self.previous_input[:-1]

        # Update weight buffer.
        w_ = np.reshape(self.weights, (1, 4))
        self.weights_fb = w_
        
        g_ = np.reshape(self.gains, (1, 3))
        self.gains_fb = g_

        # Build the observation dictionary.
        obs['previous_input'] = self.previous_input.astype(np.float32)
        obs['previous_state'] = self.previous_states.astype(np.float32)
        obs['weights_fb'] = self.weights_fb.astype(np.float32)
        obs['relative_trajectory'] = (obs_path_array / self.scalingFactor).astype(np.float32)
        obs['state'] = (self.pose / self.scalingFactor).astype(np.float32)
        obs['gains_fb'] = self.gains_fb.astype(np.float32)

        return obs

    def reset(self):
        """
        Reset the environment to its initial state, every time a new episode starts.
        """
        while not self.odom_received and not rospy.is_shutdown():
            rospy.sleep(0.01)
            
        # Drive back to origin before generating new path
        origin = np.array([0.0, 0.0])
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            current_pos = self.pose[:2]
            dist = np.linalg.norm(current_pos - origin)

            if dist < 0.2:
                break  # close enough

            # simple proportional control
            error = origin - current_pos
            angle_to_goal = math.atan2(error[1], error[0])
            heading_error = yaw_error(angle_to_goal, self.pose[2])

            cmd = Twist()
            cmd.linear.x = 2
            cmd.angular.z = 1.5 * heading_error
            self.pub.publish(cmd)
            
            rate.sleep()

        # stop robot
        self.pub.publish(Twist())
        rospy.sleep(1.0)  # allow settling

        self.pose = np.zeros(5, dtype=np.float32)       # [x, y, theta, wL, wR]
        self.prevPose = np.zeros(5, dtype=np.float32)   # [x, y, theta, wL, wR]
        self.actions = np.zeros(2)
        self.weights = np.zeros(4)
        self.gains = np.zeros(3)
        self.delta_steering = 0.0
        self.episode_steps = 0
        self.episode_reward = 0
        self.t_current = 0.0
        self.prevDeltaSteering = 0.0
        self.prevVelocity = 0.0
        self.last_closest_idx = 0
        
        self.velFiltered = 0.0
        self.desiredPoseIdx = 0
        
        # Reset steering wheel reward params
        self.delta_action = 0.0
        self.delta_previous = 0.0

        if self.use_seed and self.seed_val is not None:
            self.seed_val += 1
            random.seed(self.seed_val)

        # Generate path
        if self.use_dubins:
            dubins_paths = generate_dubins(
                start_angle=self.pose[2],
                length=self.path_length_dubins,
                bin=self.radiusOfCBin,
                interval1=self.interval1,
                interval2=self.interval2
            )
            wps = get_dubins_waypoints(dubins_paths, spacing=self.waypoint_spacing)
            self.path = wps[:, :self.num_path_points]
        else:
            self.path = generate_waypoints(
                num_points=self.num_path_points,
                radius=self.waypoint_spacing
            )

        # Reset timer
        self.t_current = 0.0

        # Build distance array for path
        self.path_length = np.zeros(self.path.shape[1])
        for i in range(1, self.path.shape[1]):
            self.path_length[i] = self.path_length[i-1] + \
                distance(self.path[:,i], self.path[:,i-1])
        self.path_length_current_time = self.path_length / self.desiredVelocity


        # Initial observation & counters
        self.episode_steps = 0
        obs = self.observe()
        self.episode_reward = 0
        self.lookahead_points = np.zeros((2, self.num_episode_steps))

        # Clear logs
        self.F_trac.clear()
        self.F_roll.clear()
        self.F_drag.clear()
        self.lambda_.clear()
        self.F_f.clear()
        self.F_r.clear()
        self.alpha_f.clear()
        self.alpha_r.clear()
        self.desired_torque.clear()
        self.steer_angle.clear()
        self.poses.clear()
        self.sideSlipRatio.clear()
        self.weights_overall.clear()
        self.gains_overall.clear()
        self.velSetPoint_RL.clear()
        self.avgVel_RL.clear()
        self.smoothVel_RL.clear()

        self.previous_states[:] = 0.0
        self.previous_input[:] = 0.0
        self.weights_fb[:] = 0.0
        self.gains_fb[:] = 0.0


        self.num_episodes += 1

        # dynamic plotting 
        if self.dynamicPlot:
            fig, self.ax = plot_init()

        # additional tracking arrays
        self.obs_path_array = np.zeros((2, self.observation_points))
        self.previous_states = np.zeros((self.history_length, 5), dtype=np.float32)
        self.previous_input = np.zeros((self.history_length, 2), dtype=np.float32)
        self.weights_fb = np.zeros((1, 4), dtype=np.float32)
        self.gains_fb = np.zeros((1, 3), dtype=np.float32)

        #### Random Friction Surface ####
        # self.friction_surface = random.choice([0.1, 0.4, 0.7, 1.0])
        self.friction_surface = 1.0

        return self.observe()
    
    def reward(self):
        # Locate target waypoint
        idx    = self.desiredPoseIdx
        x_des, y_des = self.path[:,idx]

        # compute current distance error
        dx, dy = self.pose[0]-x_des, self.pose[1]-y_des
        dist   = math.hypot(dx, dy)

        # compute delta distance from last step
        prev   = getattr(self, 'prev_dist_err', None)
        if prev is None:
            delta = 0.0
        else:
            delta = prev - dist   # positive if get closer

        self.prev_dist_err = dist
        self.smooth()
        self.smoothVel_RL.append(self.velFiltered)
        # reward is simply deleta distance -- encourage getting closer
        return delta
    
    # def reward(self):
    #     """
    #     Compute the reward based on the current state of the environment. 
    #     Prioritize the reward for closeness to the desired trajectory (cte + distance error) and steering.
    #     """
    #     reward_pose = 0.0
    #     reward_percentage = 0.0
    #     reward_percentage_torq = 0.0
        
    #     # desired pose for reference (from the global path)
    #     desired_pose = np.array([self.path[0][self.desiredPoseIdx],
    #                              self.path[1][self.desiredPoseIdx]])
        
    #     if self.t_current > 0.0:
    #         # reward for closeness to the desired trajectory (cte + distance error)
    #         reward_pose = -1 * (self.cte + self.distError)
    #         reward_pose = reward_pose / self.scalingFactor
            
    #         # penalize large changes in steering (if the difference exceeds pi/4)
    #         if abs(self.prevDeltaSteering - self.delta_steering) > math.pi/4:
    #             reward_percentage = -0.01 * abs(self.prevDeltaSteering - self.delta_steering)

    #         # smooth the velocity and append to the smoothed velocity log.
    #         self.smooth()
    #         self.smoothVel_RL.append(self.velFiltered)
            
    #         # compute forward velocity from the 5D state (omegaₗ and omegaᵣ)
    #         v_actual = 0.5 * self.model.r * (self.pose[3] + self.pose[4])
            
    #         # penalize large changes in velocity compared to the smoothed value.
    #         if abs(self.velFiltered - v_actual) > 0.1:
    #             reward_percentage_torq = -0.01 * abs(self.velFiltered - v_actual)

    #     reward = (10 * reward_pose) + reward_percentage + reward_percentage_torq
    #     return reward



    
    def smooth(self):
        """
        Smooth the velocity by blending the previously filtered velocity
        """
        # Weight factor between 0 and 1
        weight = 0.94
        # Compute the actual forward velocity from both wheel speeds:
        v_actual = 0.5 * self.model.r * (self.pose[3] + self.pose[4])
        # Smooth the value by blending the previously filtered velocity
        # with the current computed velocity.
        smoothed_val = self.velFiltered * weight + (1 - weight) * v_actual
        # Update the filtered velocity.
        self.velFiltered = smoothed_val


    def compute_inputs(self, action):
        """
        Compute the inputs for the differential drive model based on the current state and the desired trajectory.
        The inputs are computed using two PID controllers for velocity and yaw rate.
        
        """
        tau_L_ff = float(action[0])
        tau_R_ff = float(action[1])
        
        # 1) grab raw in [0,1]
        raw_gains = action[6:9]                   # [raw_kp_vel, raw_kp_omega, raw_kp_psi]

        # 2) scale + offset → [init - half_range, init + half_range]
        kp_vel, kp_omega, kp_psi = raw_gains * self.gains_scale + self.gains_offset

        # now kp_* ≥ init - half_range; and since raw_gains≥0, kp_*≥init-half_range≥1.0 in your example
        # if you still want to floor at zero:
        kp_vel   = max(kp_vel,   0.0)
        kp_omega = max(kp_omega, 0.0)
        kp_psi   = max(kp_psi,   0.0)

        # 3) update your PIDs
        self.vel_pid.kp   = kp_vel
        self.omega_pid.kp = kp_omega
        self.psi_kp       = kp_psi
        
        # Current linear / angular velocity 
        wL, wR = self.pose[3], self.pose[4]                             # wheel speeds [rad s‑1]
        v_actual = 0.5 * self.model.r * (wL + wR)                       # forward speed  [m s‑1]
        omega_actual = self.model.r / self.model.W * (wR - wL)          # body‑yaw rate  [rad s‑1]

        # Desired yaw rate from look‑aheadd 
        x, y, theta = self.pose[:3]

        # choose a look‑ahead point K steps ahead on the local path
        K = min(10, self.obs_path_array.shape[1] - 1)
        look_x, look_y= self.obs_path_array[:, K]

        # heading to the look‑ahead point
        heading_des = math.atan2(look_y - y, look_x - x)

        # wrap into (‑pi, pi]
        heading_err = (heading_des - theta + math.pi) % (2*math.pi) - math.pi

        # simple proportional mapping   (omega_desired = k * heading_err)
        omega_desired = self.psi_kp * heading_err                           # [rad s‑1]

        # Desired forward speed
        v_desired = self.desiredVelocity                                # constant  (≈ 9 m/s right now)

        # PID controllers
        v_error = v_desired - v_actual
        omega_error   = omega_desired - omega_actual

        tau             = self.vel_pid.computeControl(v_error)          # base torque
        delta_tau       = self.omega_pid.computeControl(omega_error)    # differential

        # Compose wheel torques
        tau_L           = tau - delta_tau
        tau_R           = tau + delta_tau

        tau_max         = 75.0 # [N-m]
        tau_L = tau_L_ff + (tau - delta_tau)
        tau_R = tau_R_ff + (tau + delta_tau)

        return tau_L, tau_R, heading_err, tau, delta_tau, kp_vel, kp_omega, kp_psi

    def step(self, action):
        """
        Take a step in the environment based on the given action.
        The function computes the new state of the environment, the reward, and whether the episode is done.
        """
        
        # print("Action: ", action)
        # Compute left/right torques
        tau_L, tau_R, headingError, _, _, vel_kp, omega_kp, psi_kp = self.compute_inputs(action)
        self.delta_steering = headingError                              # for logging "steering angle"
        self.actions = np.array([tau_L, tau_R], dtype=np.float32)
        self.weights = np.array([action[2], action[3], action[4], action[5]])  
        self.gains = np.array([vel_kp, omega_kp, psi_kp], dtype=np.float32)

        # Call differential drive step
        (next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r)  = self.model.step_dynamic_torque_scipy_rk(
                    state=self.pose,
                    control=(tau_L, tau_R),
                    t_init=self.t_current,
                    dt=self.timestep,
                    t_bound=self.t_current+self.timestep
                )

        self.pose = next_state
        
        self.current_vel = 0.5*self.model.r*(self.pose[3]+self.pose[4])
        self.poses.append(self.pose.copy())

        # Log 
        self.F_trac.append(f_trac)
        self.F_roll.append(f_roll)
        self.F_drag.append(f_drag)
        self.lambda_.append(slip_ratio)
        self.F_f.append(F_f)
        self.F_r.append(F_r)
        self.alpha_f.append(alpha_f*180/math.pi)
        self.alpha_r.append(alpha_r*180/math.pi)

        self.sideSlipRatio.append(0.0)  # mock
        avg_torque = 0.5*(tau_L + tau_R)
        self.desired_torque.append(avg_torque)
        
        self.steer_angle.append(self.delta_steering*180/math.pi)
        # print("Action: ", self.actions[0], self.actions[1])
        # print("Weights: ", self.weights[0], self.weights[1], self.weights[2], self.weights[3])
        self.weights_overall.append(self.weights.copy())
        self.gains_overall.append(self.gains.copy())
        # self.velSetPoint_RL.append(velSetPoint)
    
        self.episode_steps += 1
        self.t_current += self.timestep

        reward = self.reward()
        self.prevPose = self.pose
        self.prevDeltaSteering = self.delta_steering
        self.prevVelocity = self.current_vel
        self.episode_reward += reward

        if(self.dynamicPlot and self.episode_steps % 10 == 0):
            self.plot_pose()

        v  = 0.5 * self.model.r * (self.pose[3] + self.pose[4])  # estimate velocity
        wz = self.model.r / self.model.W * (self.pose[4] - self.pose[3])  # estimate yaw rate

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = wz
        self.pub.publish(cmd)
        
        done = False 

        # 1) time limit
        if self.t_current >= self.finalSimTime:
            done = True
            reward = -100.0
            print("Time limit reached, friction surface: ", self.friction_surface,
                  " episode rwd: ", self.episode_reward)
            if self.evaluate:
                self.plot()
            return self.observe(), reward, done, {}

        # 2) success if we got within 7 of the last path point
        last_x = self.path[0][-1]
        last_y = self.path[1][-1]
        if self.getL2norm(self.pose[0], self.pose[1], last_x, last_y) < 7.0:
            done = True
            reward = +1.0
            print("Reached the end of the path in :", self.t_current,
                  "seconds friction surface: ", self.friction_surface,
                  " episode rwd: ", self.episode_reward)
            if self.evaluate:
                self.plot()
            return self.observe(), reward, done, {}

        # 3) out of bounds if > 10000 away
        if self.getL2norm(self.pose[0], self.pose[1], last_x, last_y) > 10000.0:
            done = True
            reward = -1000.0
            print("Current time:", self.t_current)
            print("Out of bounds")
            print("Pose", self.pose)
            print("Last 15 poses", self.poses[-150:])
            # CAUTION: do NOT exit the entire Python process. Just return done
            # If you truly want to forcibly kill the process, do "exit()"
            # But that stops training entirely, so typically we skip that:
            # exit()
            return self.observe(), reward, done, {}

        obs = self.observe()
        return obs, reward, done, {}

    def getL2norm(self, poseX, poseY, finalPointX, finalPointY):
        """
        Compute the L2 norm between the current pose and the final point.
        """
        return math.sqrt((poseX - finalPointX)**2 + (poseY - finalPointY)**2)


    def plot(self):
        """
        Static Multi-Figure Plots
        """
        poses = np.array(self.poses)  # shape (N,5)

        # Figure 1
        plt.figure(figsize=(10, 8))

        # # velocity => 0.5*r*(wL+wR)
        plt.subplot(3,1,1)
        plt.title("Long Velocity (mph)")
        wL = poses[:,3]
        wR = poses[:,4]
        v_array = 0.5*self.model.r*(wL+wR)
        plt.plot(v_array*2.24, label='vehicle velocity')
        plt.plot(np.array(self.smoothVel_RL)*2.24, label="smoothed vehicle velocity")
        plt.plot(np.ones(len(v_array))*self.desiredVelocity*2.24, label="velocity set point")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        
        # 2) steering angle
        plt.subplot(3,1,2)
        plt.title("Cross Track Error (m)")
        plt.plot(self.ctes, label="cross track error (m)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        
        plt.subplot(3,1,3)
        plt.title("Heading (deg)")
        plt.plot(self.steer_angle, label='heading angle (deg)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("fig1_velocity_steering.png")

        # Figure 2
        plt.figure(figsize=(10, 8))
        plt.title("Fig 2: Trajectory and Global Path")
        plt.plot(poses[:, 0], poses[:, 1], label="trajectory of vehicle")
        plt.plot(self.path[0, :], self.path[1, :], label="global path")
        plt.legend()
        plt.grid()
        plt.savefig("fig2_trajectory_path.png")

        # Figure 4: Weights
        weights_ = np.array(self.weights_overall)
        if len(weights_) > 0:
            plt.figure(figsize=(10, 8))
            plt.suptitle("Fig 3: Feedback Weights")
            plt.subplot(4,1,1)
            plt.plot(weights_[:,0], label='weight 1')
            plt.grid()
            plt.legend()
            plt.subplot(4,1,2)
            plt.plot(weights_[:,1], label='weight 2')
            plt.grid()
            plt.legend()
            plt.subplot(4,1,3)
            plt.plot(weights_[:,2], label='weight 3')
            plt.grid()
            plt.legend()
            plt.subplot(4,1,4)
            plt.plot(weights_[:,3], label='weight 4')
            plt.grid()
            plt.legend()
            plt.savefig("fig3_feedback_weights.png")


        # Figure 4: Weights
        gains_ = np.array(self.gains_overall)
        if len(weights_) > 0:
            plt.figure(figsize=(10, 8))
            plt.suptitle("Fig 3: Feedback Gains")
            plt.subplot(3,1,1)
            plt.plot(gains_[:,0], label='velocity kp')
            plt.grid()
            plt.legend()
            plt.subplot(3,1,2)
            plt.plot(gains_[:,1], label='yaw rate kp')
            plt.grid()
            plt.legend()
            plt.subplot(3,1,3)
            plt.plot(gains_[:,2], label='heading kp')
            plt.grid()
            plt.legend()
            plt.savefig("fig4_feedback_gains.png")
            
        # Figure 5: 
        plt.figure(figsize=(10, 8))
        plt.suptitle("Fig 5: Vehicle States and Commands")
        # 1) velocity in mph
        plt.subplot(6,1,1)
        plt.plot(v_array*2.24, label="v_longitudinal (mph)")
        plt.plot(np.ones(len(v_array))*self.desiredVelocity*2.24, label="velocity set point (mph)")
        plt.legend()
        plt.grid()

        # 2) "omega_wheel" => wR?
        plt.subplot(6,1,2)
        # plt.plot(wR, label="omega_wheel (right) rad/s")
        # plt.plot(wL, label="omega_wheel (left) rad/s")
        plt.plot((wR+wL)/2, label="omega_wheel (rad/s)")
        plt.legend()
        plt.grid()

        # 3) "v_lateral" => 0 in diff drive
        plt.subplot(6,1,3)
        plt.plot(np.zeros(len(v_array)), label="v_lateral (0 for diff_drive)")
        plt.legend()
        plt.grid()

        # 4) "omega_body" => (r/W)*(wR-wL)
        plt.subplot(6,1,4)
        r, W = self.model.r, self.model.W
        omega_body = (r/W)*(wR - wL)
        plt.plot(omega_body, label="omega body (yaw rate) (rad/s)")
        plt.legend()
        plt.grid()

        # 5) "Torque"
        plt.subplot(6,1,5)
        plt.plot(self.desired_torque, label="average torque (N-m)")
        plt.legend()
        plt.grid()

        # 6) "Steering Angle"
        plt.subplot(6,1,6)
        plt.plot(self.steer_angle, label="steering angle (deg)")
        plt.legend()
        plt.grid()

        # # 7) x
        # plt.subplot(9,1,7)
        # plt.plot(poses[:,0], label="x position (m)")
        # plt.legend()
        # plt.grid()

        # # 8) y
        # plt.subplot(9,1,8)
        # plt.plot(poses[:,1], label="y position (m)")
        # plt.legend()
        # plt.grid()

        # 9) sideSlipRatio
        # plt.subplot(9,1,9)
        # if len(self.sideSlipRatio) == len(v_array):
        #     plt.plot(np.array(self.sideSlipRatio)*180/math.pi, label="side slip angle (0 for diff_drive)")
        # else:
        #     plt.plot(np.zeros(len(v_array)), label="side slip angle (0 for diff_drive)")
        # plt.legend()
        # plt.grid()
        plt.savefig("fig5_states_and_commands.png")
        
        
        # Figure 5: 4x2 subplots for f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r
        plt.figure(figsize=(10, 8))
        plt.suptitle("Fig 6: Tire Forces and Dynamics")

        N = len(v_array)
        # Convert logs to arrays
        f_trac_arr   = np.array(self.F_trac)   if len(self.F_trac)   == N else np.zeros(N)
        f_roll_arr   = np.array(self.F_roll)   if len(self.F_roll)   == N else np.zeros(N)
        f_drag_arr   = np.array(self.F_drag)   if len(self.F_drag)   == N else np.zeros(N)
        slip_ratio_a = np.array(self.lambda_)  if len(self.lambda_)  == N else np.zeros(N)
        F_f_arr      = np.array(self.F_f)      if len(self.F_f)      == N else np.zeros(N)
        F_r_arr      = np.array(self.F_r)      if len(self.F_r)      == N else np.zeros(N)
        alpha_f_arr  = np.array(self.alpha_f)  if len(self.alpha_f)  == N else np.zeros(N)
        alpha_r_arr  = np.array(self.alpha_r)  if len(self.alpha_r)  == N else np.zeros(N)

        plt.subplot(4,2,1)
        plt.plot(f_trac_arr, label="f_trac (N)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,2)
        plt.plot(f_roll_arr, label="f_roll (N)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,3)
        plt.plot(f_drag_arr, label="f_drag (N)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,4)
        plt.plot(slip_ratio_a, label="long slip ratio")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,5)
        plt.plot(F_f_arr, label="F_f (N)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,6)
        plt.plot(F_r_arr, label="F_r (N)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,7)
        plt.plot(alpha_f_arr, label="alpha_f (deg)")
        plt.legend()
        plt.grid()

        plt.subplot(4,2,8)
        plt.plot(alpha_r_arr, label="alpha_r (deg)")
        plt.legend()
        plt.grid()
        plt.savefig("fig6_tire_forces_dynamics.png")


        plt.show()

    def plot_pose(self):
        """
        Dynamic plotting each step
        """
        x = self.pose[0]
        y = self.pose[1]
        yaw = self.pose[2]
        
        #### Debugging ####
        # if self.closest_idx_plot is not None:
        #     cpt = self.path[:, self.closest_idx_plot]
        #     self.ax.plot(cpt[0], cpt[1], 'yx', ms=9, label='closest wp')
        
        # if self.desiredPoseIdx is not None:
        #     dpt = self.path[:, self.desiredPoseIdx]
        #     self.ax.plot(dpt[0], dpt[1], 'yx', ms=9, label='desired wp')
            
        line1 ,= self.ax.plot(self.path[0, :], self.path[1, :], color='red', label='global reference path')

        line2 ,= self.ax.plot(self.obs_path_array[0, :], self.obs_path_array[1, :], color='black', label='lookahead observation')

        line3 ,= self.ax.plot(x, y, marker='.', color='blue', label='vehicle current position')

        line4 ,= self.ax.plot(self.path[0, self.desiredPoseIdx], self.path[1, self.desiredPoseIdx], marker='.', color='green', label='target waypoint (lookahead)')

        self.ax.set_xlabel('meters')
        self.ax.set_ylabel('meters')

        self.ax.legend(
            ['global reference path', 'lookahead observation', 'vehicle current position', 'target waypoint (lookahead)'],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.4),
            ncol=2,
            frameon=False
        )        
        plt.pause(0.1)
