#!/bin/bash

# Load ROS parameters
roslaunch gym_gazebo params.launch action:=evaluate_ppc use_model:=True

# Run train script
rosrun gym_gazebo main.py
