FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# Args for User
ARG UNAME=user # If not specified it will run as root user. 
ARG UID=1000
ARG GID=1000

# Ensure that installs are non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        sudo \
        iputils-ping \
        udev \
        usbutils \
        net-tools \
        wget \
        iproute2 \
        curl \
        nano \
        git \
        lsb-release

# setup sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    
# install dependencies for building ros packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    python3-pip \
    python3-catkin-tools\
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*    

# Install ros packages
RUN apt-get update && apt-get install -y \
    ros-noetic-catkin \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-jackal-simulator \
    ros-noetic-jackal-desktop \
    ros-noetic-jackal-navigation \
    && rm -rf /var/lib/apt/lists/*

# ros-noetic-jackal-gazebo\    
# Install python3 packages
#RUN python3 -m pip install --upgrade pip
RUN pip3 install \
    pyYAML\
    scikit-build\
    cmake\
    scipy\
    tensorflow\
    rospkg==1.5.1\
    numpy==1.24.4\
    gym\
    gym-notices\
    gymnasium\
    matplotlib==3.7.5\
    tensorboard==2.14.0\
    stable-baselines3==1.5.0\
    torch==2.4.1\
    dubins

RUN apt-get update && apt-get install -y bash

# Install Matplot GUI backends
RUN apt-get update
RUN apt-get install -y python3.8-tk
RUN apt-get install -y python3-pil python3-pil.imagetk



RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

# Get Workspace Dependencies
RUN mkdir -p ~/home/staff/s/saminmoosavi/gym_gazebo_docker/src
COPY src /home/staff/s/saminmoosavi/gym_gazebo_docker/src
RUN cd ~/home/staff/s/saminmoosavi/gym_gazebo_docker && \
    sudo apt update &&\
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y


# Create user
#RUN groupadd -g $GID $UNAME
#RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# Allow the user to run sudo without a password
#RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
#RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers > /dev/null

#USER $UNAME

RUN echo 'source ~/.bashrc' >> ~/.bash_profile
      
# Copy entrypoint
COPY docker/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
