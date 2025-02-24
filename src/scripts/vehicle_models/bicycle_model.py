#!/usr/bin/env python3

import math
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

import unittest

class BicycleModel:
    def __init__(self, center_to_front, center_to_rear, mass=1000, inertia=1600,
            tire_radius=0.3, cornering_stiffness_front=4e4,
            cornering_stiffness_rear=4e4, longitudinal_stiffness=3e4,
            plot=False):
        # vehicle parameters
        self.m = mass # kg
        self.iz = inertia # kg m^2
        self.lf = center_to_front # meters
        self.lr = center_to_rear # meters

        # resistance parameters
        # aerodynamic drag
        self.c0 = 4.1e-3 # N
        self.c1 = 6.6e-5 # N / (m/s)
        rho = 1.225
        A = 2.32
        cd = 0.36
        self.c2 = 0.5 * rho * A * cd # N / (m/s)^2
        # rolling resistance
        self.crr = 0.1 # rolling resistance coefficient
        self.fz = self.m * 9.80665 # vertical force on tires, N

        # wheel and tire parameters
        self.cf = cornering_stiffness_front # N/rad
        self.cr = cornering_stiffness_rear # N/rad
        self.cl = longitudinal_stiffness # longitudinal stiffness, N
        self.alpha_max = 4 / 180 * math.pi # max slip angle, rad
        self.slip_ratio_max = 0.2 # max slip ratio
        self.f_lat_max = self.cf * math.pi / 180 # max lateral tire force, N
        self.f_long_max = self.f_lat_max # max longitudinal tire force, N
        self.tr = tire_radius # meters
        self.iw = 0.5 # wheel inertia, kg m^2

        self.plot = plot
        if plot:
            self.plot_init()

    @staticmethod
    def limit_yaw(yaw):
        if yaw > math.pi:
            yaw -= 2 * math.pi
        elif yaw < -math.pi:
            yaw += 2 * math.pi
        return yaw

    def curvature_to_steer_angle(self, curvature):
        delta = math.atan(curvature * (self.lf + self.lr))
        return delta
    
    '''
    state is x, y, theta
    '''
    def step_kinematic(self, state, action, timestep=0.1, plot=False):
        v = action[0] # velocity
        delta = action[1] # steer angle

        # sideslip angle
        beta = math.atan(self.lr * math.tan(delta) / (self.lf + self.lr))

        # kinematics
        x_dot = v * math.cos(state[2] + beta)
        y_dot = v * math.sin(state[2] + beta)
        theta_dot = v * math.cos(beta) * math.tan(delta) / (self.lf + self.lr)
        state_dot = np.array([x_dot, y_dot, theta_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state, delta)

        return next_state

    '''
    state is v_long
    '''
    def longitudinal_dynamics(self, state, a_long, v_lat, omega,
            timestep=0.1):
        v_long_dot = (self.m * a_long - (self.c0 + self.c1 * state[0] + self.c2
            * state[0] ** 2)) / self.m + omega * v_lat
        state_dot = np.array([v_long_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        return next_state

    def tire_slip_ratio(self, T, v_long, omega_w):
        slip_ratio = 0
        if T > 0:
            slip_ratio = (omega_w * self.tr - v_long) /  (omega_w * self.tr)
        elif T < 0:
            slip_ratio = (omega_w * self.tr - v_long) / v_long
        if slip_ratio > self.slip_ratio_max:
            slip_ratio = self.slip_ratio_max
        elif slip_ratio < -self.slip_ratio_max:
            slip_ratio = -self.slip_ratio_max
        return slip_ratio

    def tractive_force(self, slip_ratio):
        force = self.cl * slip_ratio
        return force

    '''
    state is v_long, omega_w
    '''
    def longitudinal_dynamics_(self, state, T, v_lat, omega, timestep=0.1):
        slip_ratio = self.tire_slip_ratio(T, state[0], state[1])

        f_trac = self.tractive_force(slip_ratio)
        f_roll = self.crr * self.fz
        f_drag = self.c0 + self.c1 * state[0] + self.c2 * state[0] ** 2
        
        v_long_dot = (f_trac - f_roll - f_drag) / m + omega * v_lat
        omega_w_dot = (T - self.tr * f_trac) / self.iw

        state_dot = np.array([v_long_dot, omega_w_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        return next_state

    '''
    Calculate tire slip angles
    '''
    def tire_slip_angles(self, v_long, v_lat, omega, delta):
        # front tire
        v_ax = v_long
        v_ay = v_lat + omega * self.lf
        alpha_f = -delta + v_ay / (v_ax + 1e-3) # small angle approximation
        if alpha_f > self.alpha_max:
            alpha_f = self.alpha_max
        elif alpha_f < -self.alpha_max:
            alpha_f = -self.alpha_max

        # rear tire
        v_bx = v_long
        v_by = (v_lat - omega * self.lr)
        alpha_r = v_by / (v_bx + 1e-3) # small angle approximation
        if alpha_r > self.alpha_max:
            alpha_r = self.alpha_max
        elif alpha_r < -self.alpha_max:
            alpha_r = -self.alpha_max

        return alpha_f, alpha_r

    '''
    Calculate tire forces
    '''
    def tire_lateral_forces(self, alpha_f, alpha_r):
        # front tire
        F_f = self.cf * -alpha_f
        if F_f > self.f_lat_max:
            F_f = self.f_lat_max
        elif F_f < -self.f_lat_max:
            F_f = -self.f_lat_max

        # rear tire
        F_r = self.cr * -alpha_r
        if F_r > self.f_lat_max:
            F_r = self.f_lat_max
        elif F_r < -self.f_lat_max:
            F_r = -self.f_lat_max

        return F_f, F_r

    '''
    state is v_lat, omega
    '''
    def lateral_dynamics(self, state, delta, v_long, timestep=0.1):
        alpha_f, alpha_r = self.tire_slip_angles(v_long, state[0], state[1],
                delta)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        v_lat_dot = (F_f + F_r) / self.m - state[1] * v_long
        omega_dot = (self.lf * F_f - self.lr * F_r) / self.iz
        
        state_dot = np.array([v_lat_dot, omega_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        return next_state

    '''
    state is x, y, theta, v_long, v_lat, omega
    '''
    def step_dynamic(self, state, action, timestep=0.1, plot=False):
        a_long = action[0] # longitudinal acceleration
        delta = action[1] # steer angle

        next_state = np.zeros(state.shape)

        next_state[3:4] = self.longitudinal_dynamics(state[3:4], a_long,
                state[4], state[5], timestep=timestep)

        next_state[4:] = self.lateral_dynamics(state[4:], delta, state[3],
                timestep=timestep)

        x_dot = state[3] * math.cos(state[2]) - state[4] * math.sin(state[2])
        y_dot = state[3] * math.sin(state[2]) + state[4] * math.cos(state[2])
        theta_dot = state[5]
        kinematics = np.array([x_dot, y_dot, theta_dot])
        next_state[:3] = state[:3] + kinematics * timestep

        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state[:3], delta)

        return next_state
    
    '''
    state is x, y, theta, v_long, omega_w, v_lat, omega
    '''
    def step_dynamic_long(self, state, action, timestep=0.1, plot=False):
        T = action[0] # wheel torque
        delta = action[1] # steer angle

        next_state = np.zeros(state.shape)

        next_state[3:5] = self.longitudinal_dynamics(state[3:5], T, state[4],
                state[5], timestep=timestep)

        next_state[5:] = self.lateral_dynamics(state[5:], delta, state[3],
                timestep=timestep)

        x_dot = state[3] * math.cos(state[2]) - state[5] * math.sin(state[2])
        y_dot = state[3] * math.sin(state[2]) + state[5] * math.cos(state[2])
        theta_dot = state[6]
        kinematics = np.array([x_dot, y_dot, theta_dot])
        next_state[:3] = state[:3] + kinematics * timestep

        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state[:3], delta)

        return next_state

    '''
    Lateral dynamics without small angle approximation
    '''
    def step_dynamic_wsa(self, state, action, timestep=0.1, plot=False):
        v_long = action[0]
        delta = action[1]

        x = state[0]
        y = state[1]
        v_lat = state[2]
        theta = state[3]
        omega = state[4]

        # forces
        fr = -self.cr * math.atan((v_lat - omega * self.lr) / v_long)
        ff = -self.cf * math.atan((-v_long * math.sin(delta) + (v_lat + omega *
            self.lf) * math.cos(delta)) / (v_long * math.cos(delta) + (v_lat +
                omega * self.lf) * math.sin(delta)))

        # dynamics
        x_dot = v_long * math.cos(theta) - v_lat * math.sin(theta)
        y_dot = v_long * math.sin(theta) + v_lat * math.cos(theta)
        v_lat_dot = (fr + ff * math.cos(delta)) / self.m - omega * v_long
        theta_dot = omega
        omega_dot = (-self.lr * fr + self.lf * ff * math.cos(delta)) / self.iz
        state_dot = np.array([x_dot, y_dot, v_lat_dot, theta_dot, omega_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        next_state[3] = self.limit_yaw(next_state[3])

        if self.plot:
            pose = [next_state[0], next_state[1], next_state[3]]
            self.plot_pose(pose, delta)

        return next_state
 
    def plot_init(self):
        plt.ion # interactive on
        fig, self.ax = plt.subplots()
        self.ax.set_title('Trajectory')
        self.ax.set_xlabel('meters')
        self.ax.set_ylabel('meters')
        self.ax.set_aspect('equal')
        plt.pause(0.1)

    def plot_pose(self, pose, steer_angle):
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        #print((yaw + steer_angle) * 180 / math.pi)

        # vehicle axles
        xr = x - self.lr * math.cos(yaw)
        yr = y - self.lr * math.sin(yaw)
        xf = x + self.lf * math.cos(yaw)
        yf = y + self.lf * math.sin(yaw)

        self.ax.scatter(x, y, color='black') # body center
        self.ax.plot([xr, xf], [yr, yf], color='blue') # body
        self.ax.arrow(xf, yf, math.cos(yaw + steer_angle),
                math.sin(yaw + steer_angle), color='green') # front wheel vector
        plt.pause(0.1)

class Test(unittest.TestCase):
    def test_kinematic(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True)
        pose = np.zeros(3)
        action = [10.0, math.pi / 18]
        for i in range(100):
            pose = bm.step_kinematic(pose, action)

    def test_dynamic(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True)
        state = np.zeros(6)
        state[3] = 5
        state[4] = 0
        state[5] = 0
        action = [0, math.pi / 6]
        for i in range(100):
            state = bm.step_dynamic(state, action)
            #print(state)

    def test_dynamic_wsa(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True)
        state = np.zeros(5)
        action = [10.0, math.pi / 6]
        for i in range(100):
            state = bm.step_dynamic_wsa(state, action)
            #print(state)

    def test_dynamic_long(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True)
        state= np.zeros(6)
        state[3] = 5
        action = [0, math.pi / 6]
        for i in range(100):
            state = bm.step_dynamic(state, action)

    def test_curvature_to_steer_angle(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0)
        ks = [0, math.inf, -math.inf, 1/5, 1/20]
        for k in ks:
            print('curvature: %f, steer angle: %f' %(k, bm.curvature_to_steer_angle(k)))

if __name__ == '__main__':
    unittest.main()
