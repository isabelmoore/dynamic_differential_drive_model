#!/bin/bash

# Load ROS parameters
roslaunch gym_gazebo params.launch action:=evaluate use_model:=False

# Run train script
rosrun gym_gazebo main.py
