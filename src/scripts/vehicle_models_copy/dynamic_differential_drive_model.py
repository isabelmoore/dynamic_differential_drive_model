

#!/usr/bin/env python3

import math
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import scipy.integrate as integrate
import unittest

class BicycleModel:
    def __init__(self, center_to_front, center_to_rear, mass=1000, inertia=1600,
            tire_radius=0.3, cornering_stiffness_front=8e4,
            cornering_stiffness_rear=8e4, longitudinal_stiffness=20e3,
            plot=False, controlObjects=None, label=None):
        # vehicle parameters
        self.m = mass # kg
        self.iz = inertia # kg m^2
        self.lf = center_to_front # meters
        self.lr = center_to_rear # meters
        self.lambda_friction = 1.0  # Default value for lambda_friction

        self.tire_radius = tire_radius
        # resistance parameters
        # aerodynamic drag
        self.c0 = 4.1e-3 # N
        self.c1 = 6.6e-5 # N / (m/s)
        rho = 1.225
        A = 2.32
        cd = 0.36
        self.c2 = 0.5 * rho * A * cd # N / (m/s)^2
        # rolling resistance
        self.crr = 0.001#8.44e-3 # rolling resistance coefficient
        self.fz = self.m * 9.80665 # vertical force on tires, N

        # wheel and tire parameters
        self.track_width = 1.537 # meters of jeep
        self.cf = cornering_stiffness_front # N/rad
        self.cr = cornering_stiffness_rear # N/rad
        self.cl = longitudinal_stiffness # longitudinal stiffness, N
        self.alpha_max = 6 / 180 * math.pi # max slip angle, rad
        self.slip_ratio_max = 0.2 # max slip ratio
        self.f_lat_max = self.cf * self.alpha_max # max lateral tire force, N
        self.f_long_max = self.fz # max longitudinal tire force, N
        self.tr = tire_radius # meters
        self.iw = 0.5 # wheel inertia, kg m^2
        self.max_steer_angle = math.pi/3 # max steering angle, rad


        # self.T_PID = controlObjects[0] #torque object
        # self.delta_PPC = controlObjects[1] #Steer angle object


        self.plot = plot
        self.label = label  # This will be used to label each test case plot
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
        v_left = action[0]
        v_right = action[1]

        v_long = (v_left + v_right) / 2  # average of both wheels for forward vel
        omega_z = (v_right - v_left) / self.track_width  # angular vel (turning rate)

        x_dot = v_long * math.cos(state[2])
        y_dot = v_long * math.sin(state[2])
        theta_dot = omega_z

        next_state = np.array([
            state[0] + x_dot * timestep,  # x pos
            state[1] + y_dot * timestep,  # y pos
            state[2] + theta_dot * timestep  # orientation (theta)
        ])

        next_state[2] = self.limit_yaw(next_state[2])  # Ensure yaw stays in [-pi, pi]

        if self.plot:
            self.plot_pose(next_state, None)

        return next_state


    '''
    state is v_long
    '''
    def longitudinal_dynamics(self, state, T_left, T_right, timestep=0.1):
        # vels from torques
        omega_left = T_left / self.iw
        omega_right = T_right / self.iw

        v_left = omega_left * self.tire_radius
        v_right = omega_right * self.tire_radius
        v_long = (v_left + v_right) / 2

        next_state = state + v_long * timestep
        return next_state


    def tire_slip_ratio(self, T, v_long, omega_w):
        # if T > 1:
        #     slip_ratio = (omega_w * self.tr - v_long) /  (omega_w * self.tr)
        # elif T < 1:
        #     slip_ratio = (omega_w * self.tr - v_long) / v_long
        slip_ratio = ((omega_w * self.tr) - v_long) / max(abs(v_long),0.25)
        # if slip_ratio > self.slip_ratio_max:
        #     slip_ratio = self.slip_ratio_max
        # elif slip_ratio < -self.slip_ratio_max:
        #     slip_ratio = -self.slip_ratio_max
        return slip_ratio

    def tractive_force(self, slip_ratio):
        # force = self.cl * slip_ratio
        k=0.105
        c=1
        A=0.1047
        # force = (self.f_long_max*(slip_ratio/k))/(np.power((c**2+np.power(slip_ratio/k,8)),1/7))
        force = (self.f_long_max*(slip_ratio/k))/(c**2+(slip_ratio/k)**2)**(1/2)
        # force =  (slip_ratio * self.cl)/(A + abs(slip_ratio))
        # if force > self.f_long_max:
        #     force = self.f_long_max
        # elif force < -self.f_long_max:
        #     force = -self.f_long_max
        return force*self.lambda_friction

    '''
    state is v_long, omega_w
    '''
    def longitudinal_dyanmics_torque(self, state, T_left, T_right, v_lat, omega, timestep=0.001):
        omega_left = T_left / self.iw
        omega_right = T_right / self.iw

        v_left = omega_left * self.tr
        v_right = omega_right * self.tr
        v_long = (v_left + v_right) / 2  # forward vel

        slip_ratio = self.tire_slip_ratio((T_left + T_right) / 2, v_long, omega)

        f_trac = self.tractive_force(slip_ratio) # tractive force

        f_roll = 0.0  # rolling resistance (self.crr * self.fz )
        f_drag = 0.0  # drag force (self.c0 + self.c1 * v_long + self.c2 * v_long ** 2)

        v_long_dot = ((f_trac - f_roll - f_drag) / self.m) + (omega * v_lat) # longitudinal acceleration
        omega_w_dot = ((T_left + T_right) / 2 - self.tr * f_trac) / self.iw # wheel angular vel change

        state_dot = np.array([v_long_dot, omega_w_dot])
        next_state = state + state_dot * timestep

        return next_state, f_trac, f_roll, f_drag, slip_ratio


    """
    lateral dyanmics with torque runge-kutta v_lat, omega
    """
    def lateral_dynamics_torque_rk(self,episode_steps,ppc,pose, state, delta, v_long, timestep=0.1):
            
        # alpha_f, alpha_r = self.tire_slip_angles(v_long, state[0], state[1],
        #         delta)
        # F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        # v_lat_dot = (F_f + F_r) / self.m - state[1] * v_long
        # omega_dot = (self.lf * F_f - self.lr * F_r) / self.iz

        k1_v_lat_dot,_,_,_,_ = self.calc_v_lat_dot(state[0], state[1], v_long, delta)
        curvature, _,_ = ppc.run(pose+ (k1_v_lat_dot*timestep/2),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k2_v_lat_dot,_,_,_,_ = self.calc_v_lat_dot(state[0] + (k1_v_lat_dot * timestep / 2), 
                                           state[1] + (k1_v_lat_dot*timestep/2), v_long 
                                           + (k1_v_lat_dot*timestep/2), delta)
        curvature, _,_ = ppc.run(pose+ (k2_v_lat_dot*timestep/2),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k3_v_lat_dot,_,_,_,_ = self.calc_v_lat_dot(state[0] + (k2_v_lat_dot * timestep / 2),
                                             state[1] + (k2_v_lat_dot*timestep/2), v_long
                                                + (k2_v_lat_dot*timestep/2), delta)
        curvature, _,_ = ppc.run(pose+ (k3_v_lat_dot*timestep),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k4_v_lat_dot, F_f, F_r, alpha_f, alpha_r = self.calc_v_lat_dot(state[0] + (k3_v_lat_dot * timestep),
                                            state[1] + (k3_v_lat_dot*timestep), v_long
                                                + (k3_v_lat_dot*timestep), delta)
        
        k1_omega_dot = self.calc_omega_dot(v_long, state[1], state[0], delta)
        curvature, _,_ = ppc.run(pose + (k1_omega_dot*timestep/2),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k2_omega_dot = self.calc_omega_dot(v_long, state[1] + (k1_omega_dot*timestep/2), state[0] + (k1_omega_dot*timestep/2), delta)
        curvature, _,_ = ppc.run(pose+ (k2_omega_dot*timestep/2),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k3_omega_dot = self.calc_omega_dot(v_long, state[1] + (k2_omega_dot*timestep/2), state[0] + (k2_omega_dot*timestep/2), delta)
        curvature, _,_ = ppc.run(pose+ (k3_omega_dot*timestep),
                targetIndex=episode_steps)
        delta = math.atan(curvature * (1.0+1.0))
        k4_omega_dot = self.calc_omega_dot(v_long, state[1] + (k3_omega_dot*timestep), state[0] + (k3_omega_dot*timestep), delta)
        

        v_lat_dot = state[0] + (timestep/6) * (k1_v_lat_dot + 2*k2_v_lat_dot + 2*k3_v_lat_dot + k4_v_lat_dot)
        omega_dot = state[1] + (timestep/6) * (k1_omega_dot + 2*k2_omega_dot + 2*k3_omega_dot + k4_omega_dot)

        next_state = np.array([v_lat_dot, omega_dot])

        # # next state
        # next_state = state + state_dot * timestep

        return next_state, F_f, F_r, alpha_f, alpha_r
    
    """
    omega_dot for Runge-Kutta method in differential drive
    """
    def calc_omega_dot(self, v_long, omega, v_lat, T_left, T_right):
        alpha_f, alpha_r = self.tire_slip_angles(v_long, v_lat, omega)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        # yaw rate (omega_dot) based on torque difference
        yaw_torque = (T_right - T_left) / self.track_width  # yaw torque from left and right wheel torque
        omega_dot = (self.lf * F_f - self.lr * F_r + yaw_torque) / self.iz
        return omega_dot


    """
    v_lat_dot for Runge-Kutta method in differential drive
    """
    def calc_v_lat_dot(self, v_lat, omega, v_long, T_left, T_right):
        alpha_f, alpha_r = self.tire_slip_angles(v_long, v_lat, omega)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        # lateral vel change (v_lat_dot)
        v_lat_dot = (F_f + F_r) / self.m - omega * v_long
        return v_lat_dot, F_f, F_r, alpha_f, alpha_r


    """
    State is x,y,theta - Runge-Kutta method
    """
    def kinematics_torque_rk(self, state, timestep=0.001):
        v_left = state[3]  # vel of left wheel
        v_right = state[5]  # vel of right wheel
        
        v_long = (v_left + v_right) / 2
        omega_z = (v_right - v_left) / self.track_width   # angular vel 

        # runge-Kutta integration for x, y, and theta (yaw)
        k1_x, k1_y, k1_theta = self.calc_xy_dot(state[2], v_long, omega_z)
        k2_x, k2_y, k2_theta = self.calc_xy_dot(state[2] + (k1_theta * timestep / 2), 
                                                v_long, omega_z)
        k3_x, k3_y, k3_theta = self.calc_xy_dot(state[2] + (k2_theta * timestep / 2),
                                                v_long, omega_z)
        k4_x, k4_y, k4_theta = self.calc_xy_dot(state[2] + (k3_theta * timestep), 
                                                v_long, omega_z)

        # x, y, and theta (yaw)
        x_dot = state[0] + (timestep / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_dot = state[1] + (timestep / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        theta_dot = state[2] + (timestep / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

        next_state = np.array([x_dot, y_dot, theta_dot])
        return next_state

    """
    x_dot,y_dot,theta_dot for Runge-Kutta method
    """
    def calc_xy_dot(self,theta,v_long,v_lat,omega):
        x_dot = v_long * math.cos(theta) - v_lat * math.sin(theta)
        y_dot = v_long * math.sin(theta) + v_lat * math.cos(theta)
        theta_dot = omega
        return x_dot,y_dot,theta_dot


    '''
    state is v_long, omega_w - Runge-Kutta method
    '''
    def longitudinal_dyanmics_torque_rk(self, tc, velSetpoint, state, T, v_lat, omega, timestep=0.001):
        # slip_ratio = self.tire_slip_ratio(T, state[0], state[1])

        # f_trac = self.tractive_force(slip_ratio)
        # f_roll = self.crr * self.fz
        # f_drag = self.c0 + self.c1 * state[0] + self.c2 * state[0] ** 2
        
        # v_long_dot = ((f_trac - f_roll - f_drag) / self.m) + (omega * v_lat)
        # omega_w_dot = (T - (self.tr * f_trac)) / self.iw

        k1,_,_,_,_ = self.calc_v_long_dot(tc.computeControl(velSetpoint - state[0]), state[0], state[1], v_lat, omega)
        k2,_,_,_,_ = self.calc_v_long_dot(tc.computeControl(velSetpoint - (state[0]+(k1*timestep/2))), state[0] + 
                                          (k1 * timestep / 2), state[1] + (k1*timestep/2), v_lat + (k1*timestep/2), omega + (k1*timestep/2))
        k3,_,_,_,_ = self.calc_v_long_dot(tc.computeControl(velSetpoint - (state[0]+(k2*timestep/2))), state[0] + (k2 * timestep / 2), 
                                          state[1] + (k2*timestep/2), v_lat + (k2*timestep/2), omega + (k2*timestep/2))
        k4, f_trac, f_roll, f_drag, slip_ratio = self.calc_v_long_dot(tc.computeControl(velSetpoint - (state[0]+(k3*timestep))), 
                                                                      state[0] + (k3 * timestep), state[1] + (k3*timestep)
                                                                      , v_lat + (k3*timestep), omega + (k3*timestep))
        v_long_dot = state[0] + (timestep/6) * (k1 + 2*k2 + 2*k3 + k4)

        k1_omega_w = self.calc_omega_w_dot(tc.computeControl(velSetpoint - (state[0])), state[0], state[1])
        k2_omega_w = self.calc_omega_w_dot(tc.computeControl(velSetpoint - (state[0]+(k1_omega_w*timestep/2))), 
                                           state[0] + (k1_omega_w * timestep / 2), state[1] + (k1_omega_w*timestep/2))
        k3_omega_w = self.calc_omega_w_dot(tc.computeControl(velSetpoint - (state[0]+(k2_omega_w*timestep/2))), 
                                           state[0] + (k2_omega_w * timestep / 2), state[1] + (k2_omega_w*timestep/2))
        k4_omega_w = self.calc_omega_w_dot(tc.computeControl(velSetpoint - (state[0]+(k3_omega_w*timestep))), 
                                           state[0] + (k3_omega_w * timestep), state[1] + (k3_omega_w*timestep))
        omega_w_dot = state[1] + (timestep/6) * (k1_omega_w + 2*k2_omega_w + 2*k3_omega_w + k4_omega_w)

        next_state = np.array([v_long_dot, omega_w_dot])



        # state_dot = np.array([v_long_dot, omega_w_dot])

        # next state
        # next_state = state + state_dot * timestep

        return next_state, f_trac, f_roll, f_drag, slip_ratio


    """
    omega_w_dot for Runge-Kutta method
    """
    def calc_omega_w_dot(self, T, v_long, omega_w):
        slip_ratio = self.tire_slip_ratio(T, v_long, omega_w)
        f_trac = self.tractive_force(slip_ratio)
        omega_w_dot = (T - (self.tr * f_trac)) / self.iw
        return omega_w_dot

    """
    v_long_dot for Runge-Kutta method
    """    
    def calc_v_long_dot(self, T, v_long, omega_w, v_lat, omega):
        slip_ratio = self.tire_slip_ratio(T, v_long, omega_w)
        f_trac = self.tractive_force(slip_ratio)
        f_roll = self.crr * self.fz
        f_drag = self.c0 + self.c1 * v_long + self.c2 * v_long ** 2
        v_long_dot = ((f_trac - f_roll - f_drag) / self.m) + (omega * v_lat)
        return v_long_dot, f_trac, f_roll, f_drag, slip_ratio

    '''
    tire slip angles for differential drive
    '''
    def tire_slip_angles(self, v_long, v_lat, omega):
        # front tire 
        v_ax = v_long
        v_ay = v_lat + omega * self.lf
        
        # slip angle for front and rear axles
        alpha_f = math.atan2(v_ay, v_ax)  # skip angle at front - assuming symmetry
        
        # rear tire
        v_bx = v_long
        v_by = v_lat - omega * self.lr
        alpha_r = math.atan2(v_by, v_bx)  # slip angle at rear

        return alpha_f, alpha_r


    '''
    tire forces
    '''
    def tire_lateral_forces(self, alpha_f, alpha_r):
        # front tire
        # F_f = self.cf * -alpha_f
        # if F_f > self.f_lat_max:
        #     F_f = self.f_lat_max
        # elif F_f < -self.f_lat_max:
        #     F_f = -self.f_lat_max
        k=0.105
        c=1
        A=0.1047
        #Front tire
        F_f = (self.f_lat_max*(-alpha_f/k))/(c**2+(-alpha_f/k)**2)**(1/2)
        # F_f =  (- alpha_f * self.cf)/(A + abs(-alpha_f))
        # Rear tire
        F_r = (self.f_lat_max*(-alpha_r/k))/(c**2+(-alpha_r/k)**2)**(1/2)
        # F_r =  (- alpha_r * self.cr)/(A + abs(-alpha_r))
        # F_r = (self.f_lat_max*(-alpha_r/k))/(np.power((c**2+np.power(-alpha_r/k,8)),1/7))


        # # rear tire
        # F_r = self.cr * -alpha_r
        # if F_r > self.f_lat_max:
        #     F_r = self.f_lat_max
        # elif F_r < -self.f_lat_max:
        #     F_r = -self.f_lat_max

        return F_f*self.lambda_friction, F_r*self.lambda_friction

    '''
    state is v_lat, omega
    '''
    def lateral_dynamics(self, state, T_left, T_right, v_long, timestep=0.1):
        # If no torque is applied, lateral dynamics should return zero changes
        if T_left == 0.0 and T_right == 0.0:
            state_dot = np.array([0.0, 0.0])
            return state, 0.0, 0.0, 0.0, 0.0

        v_left = T_left / self.iw
        v_right = T_right / self.iw

        v_long = (v_left + v_right) / 2
        omega_z = (v_right - v_left) / self.track_width

        if len(state) > 1:
            v_lat = state[1]  
        else:
            v_lat = 0.0  # default if it's not in state

        v_lat_dot = -v_lat * v_long
        omega_dot = omega_z

        state_dot = np.array([v_lat_dot, omega_dot])

        next_state = state + state_dot * timestep   # next state by integrating over time

        return next_state, 0.0, 0.0, 0.0, 0.0   # placeholder values for forces (since not computed here)

    '''
    state is x, y, theta, v_long, v_lat, omega
    '''
    def step_dynamic(self, state, action, timestep=0.1):
        T_left = action[0]  # torque on left wheel
        T_right = action[1]  # torque on right wheel
        
        # wheel angular vels based on torques
        omega_left = T_left / self.iw  
        omega_right = T_right / self.iw  
        
        v_left = omega_left * self.tire_radius
        v_right = omega_right * self.tire_radius
        v_long = (v_left + v_right) / 2
        omega_z = (v_right - v_left) / self.track_width # angular vel (yaw rate)

        # x, y, and theta (orientation)
        x_dot = v_long * math.cos(state[2])
        y_dot = v_long * math.sin(state[2])
        theta_dot = omega_z

        next_state = np.array([
            state[0] + x_dot * timestep,  # x pos
            state[1] + y_dot * timestep,  # y pos
            state[2] + theta_dot * timestep  # theta (orientation)
        ])
        
        next_state[2] = self.limit_yaw(next_state[2])  # Keep yaw between -pi and pi
        if self.plot:
            self.plot_pose(next_state, None)
        return next_state

        

    """
    step dynamic with Runge-Kutta method (handling torque for left and right wheels)
    """
    def step_dynamic_torque_rk(self, episode_steps, ppc, tc, velSetPoint, state, action, timestep=0.1, plot=False):
        T_left = action[0]  # torque applied to left wheel
        T_right = action[1]  # torque applied to right wheel

        # RK method
        """K1"""
        self.calcIntermediateValues(state, T_left, T_right)  # Pass T_left and T_right as arguments
        k1 = self.diff_eq(0, state, T_left, T_right)
        
        """K2"""
        self.calcIntermediateValues(state + (k1 * timestep / 2), T_left, T_right)
        k2 = self.diff_eq(0, state + (k1 * timestep / 2), T_left, T_right)

        """K3"""
        self.calcIntermediateValues(state + (k2 * timestep / 2), T_left, T_right)
        k3 = self.diff_eq(0, state + (k2 * timestep / 2), T_left, T_right)

        """K4"""
        self.calcIntermediateValues(state + (k3 * timestep), T_left, T_right)
        k4 = self.diff_eq(0, state + (k3 * timestep), T_left, T_right)

        """next state"""
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * timestep / 6

        next_state[2] = self.limit_yaw(next_state[2])  # Make sure yaw stays within bounds

        if self.plot:
            self.plot_pose(next_state[:3], None) 

        return next_state, self.f_trac, self.f_roll, self.f_drag, self.lambda_, self.F_f, self.F_r, self.alpha_f, self.alpha_r


    def step_dynamic_torque_scipy_rk(self, state, action, t_init=0, timestep=0.1, plot=False, t_bound=10, return_full_solution=False, lambda_friction=1.0):
        T_left = action[0]  # torque applied to left wheel
        T_right = action[1]  # torque applied to right wheel
        self.lambda_friction = lambda_friction

        x = state[0]
        y = state[1]
        theta = state[2]
        v_x = state[3]
        omega_w = state[4]
        v_y = state[5]
        omega = state[6]

        init_state = np.array([x, y, theta, v_x, omega_w, v_y, omega])

        # diff_eq function gets T_left and T_right as arguments
        solution = integrate.RK45(lambda t, X: self.diff_eq(t, X, T_left, T_right), t0=t_init, y0=init_state, t_bound=t_bound, vectorized=True)

        t_values = []
        y_values = []
        
        while solution.status != 'finished':
            solution.step()
            t_values.append(solution.t)
            y_values.append(solution.y)

        next_state = y_values[-1]

        if self.plot:
            self.plot_pose(next_state[:3], None)

        if return_full_solution:
            return t_values, y_values
        else:
            return next_state, self.f_trac, self.f_roll, self.f_drag, self.lambda_, self.F_f, self.F_r, self.alpha_f, self.alpha_r


    """
    Differential equation in vectorized form
    """
    def diff_eq(self, t, X, T_left, T_right):
        x =       X[0]
        y =       X[1]
        theta =   X[2]
        v_x =     X[3]
        omega_w = X[4]
        v_y =     X[5]
        omega =   X[6]

        self.calcIntermediateValues(X, T_left, T_right)

        x_dot = v_x * math.cos(theta) - v_y * math.sin(theta)
        y_dot = v_x * math.sin(theta) + v_y * math.cos(theta)
        theta_dot = omega
        vx_dot = ((self.f_trac - self.f_roll - self.f_drag) / self.m) + (v_y * omega)
        omega_w_dot = ((T_left + T_right) / 2 / self.iw) - ((self.tr * (self.f_trac - self.f_roll)) / self.iw)
        vy_dot = ((self.F_f + self.F_r) / self.m) - (v_x * omega)
        omega_dot = ((self.lf * self.F_f) - (self.lr * self.F_r)) / self.iz

        next_state = np.array([
            x_dot[0] if isinstance(x_dot, np.ndarray) else x_dot, 
            y_dot[0] if isinstance(y_dot, np.ndarray) else y_dot, 
            theta_dot[0] if isinstance(theta_dot, np.ndarray) else theta_dot, 
            vx_dot[0] if isinstance(vx_dot, np.ndarray) else vx_dot, 
            omega_w_dot[0] if isinstance(omega_w_dot, np.ndarray) else omega_w_dot, 
            vy_dot[0] if isinstance(vy_dot, np.ndarray) else vy_dot, 
            omega_dot if isinstance(omega_dot, np.float64) else omega_dot
        ], dtype=np.float64)

        return next_state


        
    """
    Function to intermediate values
    """
    def calcIntermediateValues(self, init_state, T_left, T_right):
        x = init_state[0]
        y = init_state[1]
        theta = init_state[2]
        v_x = init_state[3]
        omega_w = init_state[4]
        v_y = init_state[5]
        omega = init_state[6]

        sgn_vx = np.sign(v_x)

        # with assumption torques for left and right affect traction force differently
        T_avg = (T_left + T_right) / 2
        self.lambda_ = self.tire_slip_ratio(T_avg, v_x, omega_w)  # average torque
        self.f_trac = self.tractive_force(self.lambda_)
        self.f_roll = sgn_vx * self.crr * self.fz
        self.f_drag = (sgn_vx * self.c0) + (self.c1 * v_x) + (self.c2 * v_x**2 * sgn_vx)
        self.alpha_f, self.alpha_r = self.tire_slip_angles(v_x, v_y, omega)
        self.F_f, self.F_r = self.tire_lateral_forces(self.alpha_f, self.alpha_r)

    
    '''
    state is x, y, theta, v_long, omega_w, v_lat, omega
    '''
    def step_dynamic_long(self, state, action, timestep=0.1, plot=False):
        T_left = action[0]  # torque on left wheel
        T_right = action[1]  # torque on right wheel

        next_state = np.zeros(state.shape)

        #  longitudinal dynamics - on left and right torques
        next_state[3:5], f_trac, f_roll, f_drag, slip_ratio = self.longitudinal_dyanmics_torque(
            state[3:5], T_left, T_right, state[4], state[5], timestep=timestep)

        #  lateral dynamics- returning lateral vel and yaw rate (omega_dot)
        lateral_state, F_f, F_r, alpha_f, alpha_r = self.lateral_dynamics(
            state[5:], T_left, T_right, state[3], timestep=timestep)

        next_state[5] = lateral_state[0]  # lateral vel
        omega_dot = lateral_state[1]  # yaw rate (omega_dot)

        # kinematic updates
        x_dot = state[3] * math.cos(state[2]) - state[5] * math.sin(state[2])
        y_dot = state[3] * math.sin(state[2]) + state[5] * math.cos(state[2])
        
        # yaw rate (theta_dot) difference between left and right torques
        theta_dot = omega_dot 

        kinematics = np.array([x_dot, y_dot, theta_dot])
        next_state[:3] = state[:3] + kinematics * timestep

        #  theta (yaw angle) stays within bounds
        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state[:3], None)  # No steering angle needed for plotting

        return next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r

    '''
    Lateral dynamics without small angle approximation
    '''
    def step_dynamic_wsa(self, state, action, timestep=0.1, plot=False):
        v_left = action[0]  # Left wheel vel
        v_right = action[1]  # Right wheel vel

        v_long = (v_left + v_right) / 2
        omega_z = (v_right - v_left) / self.track_width  

        x = state[0]
        y = state[1]
        theta = state[3]  # orientation (yaw)
        
        x_dot = v_long * math.cos(theta)
        y_dot = v_long * math.sin(theta)
        theta_dot = omega_z  # usin omega_z for yaw rate

        # state without lateral dynamics
        next_state = np.array([
            x + x_dot * timestep,  # x pos
            y + y_dot * timestep,  # y pos
            0,  # ignoring lateral vel for differential drive
            self.limit_yaw(theta + theta_dot * timestep),  # yaw (theta)
            omega_z  # omega (yaw rate)
        ])

        if self.plot:
            pose = [next_state[0], next_state[1], next_state[3]]
            self.plot_pose(pose, None)

        return next_state


 
    def plot_init(self):
        plt.ion()  
        fig, self.ax = plt.subplots()
        self.ax.set_title(f'Trajectory - {self.label}')  
        self.ax.set_xlabel('Meters (X)')
        self.ax.set_ylabel('Meters (Y)')
        self.ax.set_aspect('equal')
        plt.pause(0.1)

    def plot_pose(self, pose, steer_angle=None):
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        # vehicle axles
        xr = x - self.lr * math.cos(yaw)
        yr = y - self.lr * math.sin(yaw)
        xf = x + self.lf * math.cos(yaw)
        yf = y + self.lf * math.sin(yaw)

        self.ax.scatter(x, y, color='black')  
        self.ax.plot([xr, xf], [yr, yf], color='blue') 

        # if no steer_angle is provided, no plot steering vector
        if steer_angle is not None:
            self.ax.arrow(xf, yf, math.cos(yaw + steer_angle),
                        math.sin(yaw + steer_angle), color='green')  # front wheel vector

        handles, labels = self.ax.get_legend_handles_labels()
        if not labels or self.label not in labels:
            self.ax.legend(handles, labels, loc='upper right')

        plt.pause(0.1)

class Test(unittest.TestCase):
    def test_kinematic(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True, label="Kinematic")
        pose = np.zeros(3)  # initial pose
        action = [5.0, 7.0]  # Left and right wheel vels
        for i in range(100):
            pose = bm.step_kinematic(pose, action)
            # print(f"Step {i+1}: Pose = {pose}")

    def test_dynamic(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True, label="Dynamic")
        state = np.zeros(6)  # State: x, y, theta, v_long, v_lat, omega
        state[3] = 5  # initialize forward vel
        action = [50.0, 60.0]  # Left and right wheel torques
        for i in range(100):
            state = bm.step_dynamic(state, action)
            # print(f"Step {i+1}: State = {state}")

    def test_dynamic_long(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True, label="Dynamic Long")
        state = np.zeros(6)  # State: x, y, theta, v_long, v_lat, omega
        state[3] = 5  # initialize forward vel
        action = [50.0, 60.0]  # Small difference for constant turn
        for i in range(100):
            state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r = bm.step_dynamic_long(state, action)
            # print(f"Step {i+1}: State = {state}, Slip Ratio = {slip_ratio}, Tractive Force = {f_trac}")

    def test_dynamic_wsa(self):
        bm = BicycleModel(center_to_front=1.0, center_to_rear=1.0, plot=True, label="Dynamic WSA")
        state = np.zeros(5)  # State: x, y, v_lat, theta, omega
        action = [10.0, 12.0]  # Left and right wheel vels
        for i in range(100):
            state = bm.step_dynamic_wsa(state, action)
            print(f"Step {i+1}: State = {state}")

    def test_curvature_to_steer_angle(self):
        # no need to test curvature to steer angle in differential drive - since this function has become irrelevant
        pass

if __name__ == '__main__':
    unittest.main()
