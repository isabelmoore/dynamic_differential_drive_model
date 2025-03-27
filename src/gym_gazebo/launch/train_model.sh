#!/bin/bash

# Load ROS parameters
roslaunch gym_gazebo params.launch action:=train model_dir:=/home/root/models

# Run train script
rosrun gym_gazebo main.py
