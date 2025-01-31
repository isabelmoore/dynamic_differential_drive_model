# Use the official ROS Noetic base image
FROM ros:noetic

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    ros-noetic-catkin \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /ros_ws

# Copy your ROS workspace into the container
COPY . /ros_ws/src/gym_gazebo

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source ROS environment setup
RUN echo "source /ros_ws/devel/setup.bash" >> ~/.bashrc

# Default entrypoint (override as needed)
CMD ["/bin/bash"]

RUN apt-get update && apt-get install -y ros-noetic-jackal-gazebo
