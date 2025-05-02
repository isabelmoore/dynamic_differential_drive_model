#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import unittest

class DifferentialDriveModel:
    """
    Example differential-drive robot model with:
      State: [x, y, theta, omega_L, omega_R]
      Inputs: [tau_L, tau_R]   (left and right wheel torques)

    The code structure is similar to the old BicycleModel, including:
      - c0, c1, c2 for aerodynamic drag
      - crr for rolling resistance
      - Plotting helpers to visualize the trajectory
      - A step_kinematic update if you only want pure kinematics
      - A step_dynamic_torque_scipy_rk for a more advanced torque-based update
        that returns 9 items (state + forces + slip angles).
    """
    
    #############################################################################
    # Jeep Parameters
    #############################################################################

    # def __init__(self, 
    #             center_to_front=1.0, 
    #             center_to_rear=1.0, 
    #             mass=1000, 
    #             inertia=1600,
    #             tire_radius=0.3, 
    #             cornering_stiffness_front=8e4,
    #             cornering_stiffness_rear=8e4, 
    #             longitudinal_stiffness=20e3,
    #             plot=False, 
    #             controlObjects=None, 
    #             title=None):

    #     # Vehicle geometry
    #     self.lf = center_to_front       # distance from CoM to front axle
    #     self.lr = center_to_rear        # distance from CoM to rear axle
    #     self.W = self.lf + self.lr      # total wheelbase
    #     self.r = tire_radius            # wheel radius

    #     # Mass properties
    #     self.m = mass                   # mass in kg
    #     self.Iz = inertia               # moment of inertia about z-axis
    #     self.Iw = 0.5                   # wheel inertia (default)

    #     # Resistance parameters
    #     self.c0 = 4.1e-3                # constant drag [N]
    #     self.c1 = 6.6e-5                # linear drag [N/(m/s)]
    #     rho = 1.225                    # air density [kg/m^3]
    #     A = 2.32                       # frontal area [m^2]
    #     cd = 0.36                      # drag coefficient
    #     self.c2 = 0.5 * rho * A * cd   # quadratic drag [N/(m/s)^2]
    #     self.crr = 0.001               # rolling resistance coefficient
    #     self.fz = self.m * 9.80665     # normal force

    #     # Tire stiffness and limits
    #     self.cf = cornering_stiffness_front
    #     self.cr = cornering_stiffness_rear
    #     self.cl = longitudinal_stiffness
    #     self.alpha_max = 6 / 180 * math.pi        # max slip angle (rad)
    #     self.slip_ratio_max = 0.2                 # max slip ratio
    #     self.f_lat_max = self.cf * self.alpha_max # max lateral tire force
    #     self.f_long_max = self.fz                 # max longitudinal force
    #     self.max_steer_angle = math.pi / 3

    #     # Control objects (optional)
    #     # if controlObjects:
    #     #     self.T_PID = controlObjects[0]
    #     #     self.delta_PPC = controlObjects[1]

    #     # Plotting
    #     self.plot = plot
    #     if self.plot:
    #         self.plot_init(title)

    #     # Dynamic inputs and states
    #     self.tau_L = 0.0
    #     self.tau_R = 0.0
    #     self.lambda_friction_L = 1.0
    #     self.lambda_friction_R = 1.0

    #     # Intermediate forces
    #     self.f_drag = 0.0
    #     self.f_roll = 0.0
    #     self.f_trac_L = 0.0
    #     self.f_trac_R = 0.0

    #############################################################################
    # Jackal Parameters
    #############################################################################
    def __init__(
        self,
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
        title=None,
        lambda_friction = 1.0
    ):
        # Vehicle geometry
        self.lf = center_to_front                   # [m] distance from CoM to front axle
        self.lr = center_to_rear                    # [m] distance from CoM to rear axle
        self.W = self.lf + self.lr                  # [m] total wheelbase
        self.r = tire_radius                        # [m] wheel radius

        # Mass properties
        self.m = mass                               # [kg] mass
        self.Iz = inertia                           # [kg·m^2] moment of inertia about z-axis
        self.Iw = 0.5                               # [kg·m^2] wheel inertia 

        # approximate rolling & aerodynamic parameters scaled for smaller UGV
        self.c0 = 1e-3                              # [N]
        self.c1 = 1e-5                              # [N/(m/s)]
        rho = 1.225                                 # [kg/m^3] air density
        A = 0.06                                    # [m^2] small frontal area estimate (0.3 x 0.2)
        cd = 0.8                                    # guess for drag coefficient
        self.c2 = 0.5 * rho * A * cd                # [N/(m/s)^2]
        self.crr = 0.02                             # [unitless] rolling resistance coefficient, smaller for slower speeds
        self.fz = self.m * 9.80665                  # [N] normal force on tires

        # Tire stiffness and limits
        self.cf = cornering_stiffness_front
        self.cr = cornering_stiffness_rear
        self.cl = longitudinal_stiffness
        self.alpha_max = 6 / 180 * math.pi          # [rad] max slip angle
        self.slip_ratio_max = 0.2                   # max slip ratio
        self.f_lat_max = self.cf * self.alpha_max   # max lateral tire force
        self.f_long_max = self.fz                   # max longitudinal force
        self.max_steer_angle = math.pi / 3

        # Control objects (optional)
        # if controlObjects:
        #     self.T_PID = controlObjects[0]
        #     self.delta_PPC = controlObjects[1]

        # Plotting
        self.plot = plot
        if self.plot:
            self.plot_init(title)

        # Dynamic inputs and states
        self.tau_L = 0.0
        self.tau_R = 0.0
        self.lambda_friction_L = 1.0
        self.lambda_friction_R = 1.0

        # Intermediate forces
        self.f_drag = 0.0
        self.f_roll = 0.0
        self.f_trac_L = 0.0
        self.f_trac_R = 0.0

        self.friction_coefficient = lambda_friction
        
    @staticmethod
    def limit_yaw(yaw):
        """Wrap heading angle to [-pi, pi]."""
        if yaw > math.pi:
            yaw -= 2.0 * math.pi
        elif yaw < -math.pi:
            yaw += 2.0 * math.pi
        return yaw

    def plot_init(self, title):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_aspect("equal")
        plt.pause(0.1)

    def plot_pose(self, state):
        """
        Plot the robot pose (center point + short arrow for heading).
        """
        x, y, theta, _, _ = state
        self.ax.plot(x, y, 'k.')  # center dot
        arrow_length = 0.3
        self.ax.arrow(x, y,
                      arrow_length * math.cos(theta),
                      arrow_length * math.sin(theta),
                      head_width=0.05, color='r')
        plt.pause(0.001)

    #####################################
    # Kinematic step: if you only have wheel speeds
    #####################################
    def step_kinematic(self, state, control, dt=0.1):
        """
        Kinematic update (no torque, no inertia):
        state   = [x, y, theta, omega_L, omega_R]
        control = [omega_L, omega_R] (desired wheel speeds)
        """
        x, y, theta, _, _ = state
        omega_L_des, omega_R_des = control

        # Kinematic model
        v = 0.5 * self.r * (omega_L_des + omega_R_des)
        w = (self.r / self.W) * (omega_R_des - omega_L_des)

        # Integrate
        x_next = x + v * math.cos(theta) * dt
        y_next = y + v * math.sin(theta) * dt
        theta_next = self.limit_yaw(theta + w * dt)

        next_state = np.array([x_next, y_next, theta_next,
                               omega_L_des, omega_R_des])
        if self.plot:
            self.plot_pose(next_state)

        return next_state

    #####################################
    # Dynamic approach: torque -> wheel acceleration
    # plus drag/rolling friction, integrated with SciPy RK
    #####################################
    def step_dynamic_torque_scipy_rk(self, state, control,
                                     t_init=0.0,
                                     dt=0.1,
                                     t_bound=10.0,
                                     lambda_friction_L=1.0,
                                     lambda_friction_R=1.0,
                                     return_full_solution=False):
        """
        Similar to your old bicycle code's step_dynamic_torque_scipy_rk, but now
        returns 9 values: (next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r).

        That way, your environment code can do:
            (next_state, f_trac, f_roll, f_drag, slip_ratio,
             F_f, F_r, alpha_f, alpha_r) = self.model.step_dynamic_torque_scipy_rk(...)
        exactly as in the bicycle model.
        """
        self.tau_L = control[0]
        self.tau_R = control[1]
        self.lambda_friction_L = lambda_friction_L
        self.lambda_friction_R = lambda_friction_R

        # Unpack current state
        x0, y0, theta0, wL0, wR0 = state
        init_state = np.array([x0, y0, theta0, wL0, wR0])

        # Create an RK45 integrator
        solver = integrate.RK45(
            fun=self.diff_eq,
            t0=t_init,
            y0=init_state,
            t_bound=t_init + dt,
            vectorized=True
        )

        t_vals = []
        y_vals = []

        while solver.status != 'finished':
            solver.step()
            t_vals.append(solver.t)
            y_vals.append(solver.y)

        next_state = y_vals[-1]  # final integrated state
        next_state[2] = self.limit_yaw(next_state[2])  # wrap heading

        if self.plot:
            self.plot_pose(next_state)

        # We'll define left & right traction forces from torque/radius
        f_trac_L = self.tau_L / self.r
        f_trac_R = self.tau_R / self.r
        f_trac = 0.5*(f_trac_L + f_trac_R)  # average

        # slip_ratio, F_f, F_r, alpha_f, alpha_r => placeholders
        slip_ratio = 0.0
        F_f = 0.0
        F_r = 0.0
        alpha_f = 0.0
        alpha_r = 0.0

        # Return either the entire time-series or just the final state + 8 extras
        if return_full_solution:
            return t_vals, y_vals
        else:
            return (
                next_state,  # shape [5], [x, y, theta, wL, wR]
                f_trac,      # average traction
                self.f_roll, # from diff_eq
                self.f_drag, # from diff_eq
                slip_ratio,
                F_f,
                F_r,
                alpha_f,
                alpha_r
            )

    def diff_eq(self, t, X):
        """
        The ODE right-hand side for the diff-drive:
        X = [x, y, theta, wL, wR]
        returns [x_dot, y_dot, theta_dot, wL_dot, wR_dot]
        """
        x, y, theta, wL, wR = X

        # 1) Robot linear velocity & yaw rate
        v = 0.5 * self.r * (wL + wR)
        w = (self.r / self.W) * (wR - wL)

        # 2) Drag/rolling
        sgn_v = np.sign(v)
        # total drag ~ c0 + c1*v + c2*v^2
        self.f_drag = sgn_v * self.c0 + self.c1 * v + self.c2 * (v**2)*sgn_v
        # rolling friction
        self.f_roll = self.crr * self.fz

        # split the friction half per wheel
        friction_per_wheel = 0.5*(self.f_drag + self.f_roll)

        # 3) Wheel dynamics
        wL_dot = (self.tau_L - friction_per_wheel*self.r)/self.Iw
        wR_dot = (self.tau_R - friction_per_wheel*self.r)/self.Iw

        # 4) Robot chassis kinematics
        x_dot = v*math.cos(theta)
        y_dot = v*math.sin(theta)
        theta_dot = w

        return np.array([x_dot, y_dot, theta_dot, wL_dot, wR_dot], dtype=float)


class TestDiffDrive(unittest.TestCase):
    def test_kinematic(self):
        dd = DifferentialDriveModel(plot=True, title="Kinematic Trajectory")
        state = np.zeros(5)  # [x, y, theta, wL, wR]
        control = [2.0, 2.0]  # same speeds => straight line
        for i in range(200):
            state = dd.step_kinematic(state, control, dt=0.1)

    def test_dynamic_left(self):
        dd = DifferentialDriveModel(plot=True, title="Dynamic Trajectory")
        state = np.zeros(5)  # [x, y, theta, wL, wR]
        control = [1.80, 2.0]  # left torque=1, right torque=2 => arcs
        t0 = 0.0
        for i in range(200):
            state_tuple = dd.step_dynamic_torque_scipy_rk(
                state, control, t_init=t0, dt=0.1
            )
            # state_tuple is now (next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r)
            # We only need next_state[0] for continuing
            state = state_tuple[0]
            t0 += 0.1

    def test_dynamic_right(self):
        dd = DifferentialDriveModel(plot=True, title="Dynamic Trajectory")
        state = np.zeros(5)  # [x, y, theta, wL, wR]
        control = [2, 1.8]  # left torque=1, right torque=2 => arcs
        t0 = 0.0
        for i in range(200):
            state_tuple = dd.step_dynamic_torque_scipy_rk(
                state, control, t_init=t0, dt=0.1
            )
            # state_tuple is now (next_state, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r)
            # We only need next_state[0] for continuing
            state = state_tuple[0]
            t0 += 0.1

if __name__ == "__main__":
    unittest.main()
