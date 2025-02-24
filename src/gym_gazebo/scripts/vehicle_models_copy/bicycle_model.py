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
            plot=False,controlObjects=None):
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
        self.crr = 0.001#8.44e-3 # rolling resistance coefficient
        self.fz = self.m * 9.80665 # vertical force on tires, N

        # wheel and tire parameters
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


        # self.T_PID = controlObjects[0] #Torque object
        # self.delta_PPC = controlObjects[1] #Steer angle object


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
    def longitudinal_dyanmics_torque(self, state, T, v_lat, omega, timestep=0.001):
        slip_ratio = self.tire_slip_ratio(T, state[0], state[1])
        f_trac = 0.0
        f_roll = 0.0
        f_drag = 0.0

        if(T!=0.0):
            f_trac = self.tractive_force(slip_ratio)
            f_roll = 0.0#self.crr * self.fz
            # f_drag = self.c0 + self.c1 * state[0] + self.c2 * state[0] ** 2
            
            v_long_dot = ((f_trac - f_roll - f_drag) / self.m) + (omega * v_lat)
            omega_w_dot = (T - (self.tr * f_trac)) / self.iw
        else:
            v_long_dot = 0.0
            omega_w_dot = 0.0

        state_dot = np.array([v_long_dot, omega_w_dot])

        # calculate next state
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

        # # calculate next state
        # next_state = state + state_dot * timestep

        return next_state, F_f, F_r, alpha_f, alpha_r
    

    """
    Calculate omega_dot for Runge-Kutta method
    """
    def calc_omega_dot(self, v_long, omega, v_lat, delta):
        alpha_f, alpha_r = self.tire_slip_angles(v_long, v_lat, omega,
                delta)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        omega_dot = (self.lf * F_f - self.lr * F_r) / self.iz
        return omega_dot
    

    """
    Calculate v_lat_dot for Runge-Kutta method
    """
    def calc_v_lat_dot(self, v_lat, omega, v_long, delta):
        alpha_f, alpha_r = self.tire_slip_angles(v_long, v_lat, omega,
                delta)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        v_lat_dot = (F_f + F_r) / self.m - omega * v_long
        return v_lat_dot, F_f, F_r, alpha_f, alpha_r


    """
    State is x,y,theta - Runge-Kutta method
    """
    def kinematics_torque_rk(self,state,timestep = 0.001):

        # x_dot = state[3] * math.cos(state[2]) - state[5] * math.sin(state[2])
        # y_dot = state[3] * math.sin(state[2]) + state[5] * math.cos(state[2])
        # theta_dot = state[6]

        k1_x,k1_y,k1_theta = self.calc_xy_dot(state[2], state[3], state[5],state[6])
        k2_x,k2_y,k2_theta = self.calc_xy_dot( state[2] + (k1_theta * timestep / 2), 
                                              state[3] + (k1_x * timestep / 2), state[5] + 
                                              (k1_y * timestep / 2), state[6] + (k1_theta * timestep / 2))
        k3_x,k3_y,k3_theta = self.calc_xy_dot( state[2] + (k2_theta * timestep / 2),
                                                state[3] + (k2_x * timestep / 2), state[5] +
                                                (k2_y * timestep / 2), state[6] + (k2_theta * timestep / 2))
        k4_x,k4_y,k4_theta = self.calc_xy_dot( state[2] + (k3_theta * timestep),
                                                state[3] + (k3_x * timestep), state[5] +
                                                (k3_y * timestep), state[6] + (k3_theta * timestep))
        
        x_dot = state[0] + (timestep/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_dot = state[1] + (timestep/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        theta_dot = state[2] + (timestep/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)

        next_state = np.array([x_dot, y_dot, theta_dot])

        return next_state


    """
    Calculate x_dot,y_dot,theta_dot for Runge-Kutta method
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

        # calculate next state
        # next_state = state + state_dot * timestep

        return next_state, f_trac, f_roll, f_drag, slip_ratio


    """
    Calculate omega_w_dot for Runge-Kutta method
    """
    def calc_omega_w_dot(self, T, v_long, omega_w):
        slip_ratio = self.tire_slip_ratio(T, v_long, omega_w)
        f_trac = self.tractive_force(slip_ratio)
        omega_w_dot = (T - (self.tr * f_trac)) / self.iw
        return omega_w_dot

    """
    Calculate v_long_dot for Runge-Kutta method
    """    
    def calc_v_long_dot(self, T, v_long, omega_w, v_lat, omega):
        slip_ratio = self.tire_slip_ratio(T, v_long, omega_w)
        f_trac = self.tractive_force(slip_ratio)
        f_roll = self.crr * self.fz
        f_drag = self.c0 + self.c1 * v_long + self.c2 * v_long ** 2
        v_long_dot = ((f_trac - f_roll - f_drag) / self.m) + (omega * v_lat)
        return v_long_dot, f_trac, f_roll, f_drag, slip_ratio

    '''
    Calculate tire slip angles
    '''
    def tire_slip_angles(self, v_long, v_lat, omega, delta):
        # front tire
        v_ax = v_long
        v_ay = v_lat + omega * self.lf
        # alpha_f = -delta + (v_ay / (v_ax + 1e-3)) # small angle approximation
        alpha_f = -delta + math.atan2(v_ay , (v_ax)) 
        # alpha_f = math.atan2(-delta + v_ay , (v_ax )) 
        # if alpha_f > self.alpha_max:
        #     alpha_f = self.alpha_max
        # elif alpha_f < -self.alpha_max:
        #     alpha_f = -self.alpha_max

        # rear tire
        v_bx = v_long
        v_by = (v_lat - omega * self.lr)
        # alpha_r = v_by / (v_bx + 1e-3) # small angle approximation
        alpha_r = math.atan2(v_by, (v_bx))
        # if alpha_r > self.alpha_max:
        #     alpha_r = self.alpha_max
        # elif alpha_r < -self.alpha_max:
        #     alpha_r = -self.alpha_max

        return alpha_f, alpha_r

    '''
    Calculate tire forces
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
    def lateral_dynamics(self, state, delta,T, v_long, timestep=0.1):

        if(T==0.0):
            state_dot = np.array([0.0,0.0])
            F_f = 0.0
            F_r = 0.0
            alpha_f = 0.0
            alpha_r = 0.0
            return state_dot, F_f, F_r, alpha_f, alpha_r
        alpha_f, alpha_r = self.tire_slip_angles(v_long, state[0], state[1],
                delta)
        F_f, F_r = self.tire_lateral_forces(alpha_f, alpha_r)

        v_lat_dot = (F_f + F_r) / self.m - state[1] * v_long
        omega_dot = (self.lf * F_f - self.lr * F_r) / self.iz
        
        state_dot = np.array([v_lat_dot, omega_dot])

        # calculate next state
        next_state = state + state_dot * timestep

        return next_state, F_f, F_r, alpha_f, alpha_r

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
    

    """
    step dynamic with Runge-Kutta method
    """
    def step_dynamic_torque_rk(self,episode_steps, ppc, tc, velSetPoint,state, action, timestep=0.1, plot=False):
        self.T = action[0] # wheel torque
        self.delta = action[1] # steer angle
        x = state[0]
        y = state[1]
        theta = state[2]
        v_x = state[3]
        omega_w = state[4]
        v_y = state[5]
        omega = state[6]

        init_state = np.zeros(state.shape) #x_{n}
        init_state = [x,
                      y,
                      theta,
                      v_x,
                      omega_w,
                      v_y,
                      omega] #x_{n}

        # print("Initial state: ", init_state)
        # print("Action: ", action)
        next_state = np.zeros(state.shape)
        k1 = np.zeros(state.shape)
        k2 = np.zeros(state.shape)
        k3 = np.zeros(state.shape)
        k4 = np.zeros(state.shape)
        time_t = 0.0
        """
        Start RK
        """
        """K1"""
        #Calculate Intermediate Values for K1
        self.calcIntermediateValues(init_state,action)
        #Calculate K1
        k1 = self.diff_eq(time_t,init_state)
        """K2"""
        self.calcIntermediateValues(init_state + (k1 * timestep/2),action)
        #Calculate K2
        k2 = self.diff_eq(time_t,init_state + (k1 * timestep/2))
        """K3"""
        self.calcIntermediateValues(init_state + (k2 * timestep/2),action)
        #Calculate K3
        k3 = self.diff_eq(time_t,init_state + (k2 * timestep/2))
        """K4"""
        self.calcIntermediateValues(init_state + (k3 * timestep),action)
        #Calculate K4
        k4 = self.diff_eq(time_t,init_state + (k3 * timestep))

        """Calculate next state"""
        next_state = init_state + (k1 + 2*k2 + 2*k3 + k4) * timestep/6

        """Calculate next state with euler method"""
        # next_state = init_state + (k1 * timestep)

        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state[:3], self.delta)

        return next_state, self.f_trac, self.f_roll, self.f_drag, self.lambda_, self.F_f, self.F_r, self.alpha_f, self.alpha_r
    

    def step_dynamic_torque_scipy_rk(self,state,action,t_init=0,timestep=0.1,plot=False,t_bound=10,return_full_solution=False,lambda_friction=1.0):
        self.T = action[0]# Wheel torque
        self.delta = action[1]# Steer angle
        self.lambda_friction = lambda_friction
        
        x = state[0]
        y = state[1]
        theta = state[2]
        v_x = state[3]
        omega_w = state[4]
        v_y = state[5]
        omega = state[6]

        init_state = np.zeros(state.shape) #x_{n}
        init_state = [x,
                      y,
                      theta,
                      v_x,
                      omega_w,
                      v_y,
                      omega] #x_{n}

        next_state = np.zeros(state.shape)
        solution = integrate.RK45(self.diff_eq,t0=t_init,y0=init_state,t_bound=t_bound,vectorized=True)
        t_values = []
        y_values = []
        while solution.status !='finished':
            solution.step()
            t_values.append(solution.t)
            y_values.append(solution.y)

        
        next_state = y_values[-1]

        if self.plot:
            self.plot_pose(next_state[:3], self.delta)

        if return_full_solution:
            return t_values,y_values
        else:
            return next_state, self.f_trac, self.f_roll, self.f_drag, self.lambda_, self.F_f, self.F_r, self.alpha_f, self.alpha_r
    
    """
    Differential equation in vectorized form
    """
    def diff_eq(self, t, X):
        x = X[0]
        y = X[1]
        theta = X[2]
        v_x = X[3]
        omega_w = X[4]
        v_y = X[5]
        omega = X[6]

        self.calcIntermediateValues(X)

        # Calculate derivatives
        x_dot = float(v_x * math.cos(theta) - v_y * math.sin(theta))
        y_dot = float(v_x * math.sin(theta) + v_y * math.cos(theta))
        theta_dot = float(omega)
        vx_dot = float(((self.f_trac - self.f_roll - self.f_drag) / self.m) + (v_y * omega))
        omega_w_dot = float((self.T / self.iw) - ((self.tr * (self.f_trac - self.f_roll)) / self.iw))
        vy_dot = float(((self.F_f + self.F_r) / self.m) - (v_x * omega))
        omega_dot = float(((self.lf * self.F_f) - (self.lr * self.F_r)) / self.iz)

        # Ensure all variables are scalars and create the next state array
        next_state = np.array([x_dot, y_dot, theta_dot, vx_dot, omega_w_dot, vy_dot, omega_dot], dtype=np.float64)

        return next_state


        

        
    """
    Function to calculate intermediate values
    """
    def calcIntermediateValues(self,init_state):
        x = init_state[0]
        y = init_state[1]
        theta = init_state[2]
        v_x = init_state[3]
        omega_w = init_state[4]
        v_y = init_state[5]
        omega = init_state[6]

        sgn_vx = np.sign(v_x)
        self.lambda_ = self.tire_slip_ratio(self.T,v_x, omega_w)
        # print("slip ratio: ", self.lambda_)
        self.f_trac = self.tractive_force(self.lambda_)
        # print("tractive force: ", self.f_trac)
        self.f_roll =  sgn_vx * self.crr * self.fz
        self.f_drag = (sgn_vx * self.c0) + (self.c1 * v_x) + (self.c2 * v_x**2 * sgn_vx)
        self.alpha_f, self.alpha_r = self.tire_slip_angles(v_x,v_y,omega,self.delta)
        self.F_f,self.F_r = self.tire_lateral_forces(self.alpha_f, self.alpha_r)

    
    '''
    state is x, y, theta, v_long, omega_w, v_lat, omega
    '''
    def step_dynamic_long(self, state, action, timestep=0.1, plot=False):
        T = action[0] # wheel torque
        delta = action[1] # steer angle

        next_state = np.zeros(state.shape)

        next_state[3:5], f_trac, f_roll, f_drag, slip_ratio = self.longitudinal_dyanmics_torque(state[3:5], T, state[4],
                state[5], timestep=timestep)

        next_state[5:], F_f, F_r, alpha_f, alpha_r = self.lateral_dynamics(state[5:], delta,T, state[3],
                timestep=timestep)

        x_dot = state[3] * math.cos(state[2]) - state[5] * math.sin(state[2])
        y_dot = state[3] * math.sin(state[2]) + state[5] * math.cos(state[2])
        theta_dot = state[6]
        kinematics = np.array([x_dot, y_dot, theta_dot])
        next_state[:3] = state[:3] + kinematics * timestep

        next_state[2] = self.limit_yaw(next_state[2])

        if self.plot:
            self.plot_pose(next_state[:3], delta)

        return next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r

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
