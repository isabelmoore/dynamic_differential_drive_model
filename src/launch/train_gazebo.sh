#!/bin/bash

# Use sim time
rosparam set use_sim_time True

# Load ROS parameters
roslaunch gym_gazebo params.launch action:=train use_model:=False

# Run train script
rosrun gym_gazebo main.py
