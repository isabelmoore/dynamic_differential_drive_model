#!/bin/bash

# Load ROS parameters
roslaunch gym_gazebo params.launch action:=train

# Run train script
rosrun gym_gazebo main.py
