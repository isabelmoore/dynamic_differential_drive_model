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
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Add ROS repo
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
    lsb-release \
    build-essential \
    cmake \
    python3-dev \
    python3-distutils \
    bash \
    python3-pip \
    python3-pil \
    python3-pil.imagetk \
    python3.8-tk \
    && rm -rf /var/lib/apt/lists/*

# install core ROS tools via pip (to avoid dpkg conflicts)
RUN pip3 install \
    catkin_pkg \
    rospkg \
    rosdep \
    rosdistro \
    vcstools \
    rosinstall \
    catkin_tools

# setup ROS repo
RUN apt-get update && apt-get install -y curl lsb-release gnupg2 && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# bootstrap rosdep
RUN rosdep init && \
    rosdep update --rosdistro $ROS_DISTRO

# install ROS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy


# Copy and install pinned requirements
# COPY requirements.txt ~/src/gym_gazebo/scripts/requirements.txt
RUN conda create -y -n venv_gazebo python=3.8 && \
    conda install -y -n venv_gazebo pip

RUN /opt/conda/bin/conda run -n venv_gazebo pip install \
        defusedxml \
        absl-py==2.1.0 \
        astunparse==1.6.3 \
        cachetools==5.5.2 \
        catkin-pkg==1.0.0 \
        certifi==2025.1.31 \
        charset-normalizer==3.4.1 \
        cloudpickle==1.6.0 \
        cmake==3.31.6 \
        contourpy==1.1.1 \
        cycler==0.12.1 \
        distro==1.9.0 \
        docutils==0.20.1 \
        dubins==1.0.1 \
        Farama-Notifications==0.0.4 \
        filelock==3.16.1 \
        flatbuffers==2.0.7 \
        fonttools==4.56.0 \
        fsspec==2025.2.0 \
        future==1.0.0 \
        gast==0.4.0 \
        git-filter-repo==2.47.0 \
        google-auth==2.38.0 \
        google-auth-oauthlib==1.0.0 \
        google-pasta==0.2.0 \
        grpcio==1.70.0 \
        gym==0.17.3 \
        gym-notices==0.0.8 \
        h5py==3.11.0 \
        idna==3.10 \
        importlib_metadata==8.5.0 \
        importlib_resources==6.4.5 \
        Jinja2==3.1.5 \
        keras==2.7.0 \
        Keras-Preprocessing==1.1.2 \
        kiwisolver==1.4.7 \
        libclang==18.1.1 \
        Markdown==3.7 \
        MarkupSafe==2.1.5 \
        matplotlib==3.7.5 \
        mpmath==1.3.0 \
        networkx==3.1 \
        numpy==1.24.4 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.20.5 \
        nvidia-nvjitlink-cu12==12.8.61 \
        nvidia-nvtx-cu12==12.1.105 \
        oauthlib==3.2.2 \
        opencv-python==4.11.0.86 \
        opt_einsum==3.4.0 \
        packaging==24.2 \
        pandas==2.0.3 \
        pillow==10.4.0 \
        protobuf==3.19.6 \
        pyasn1==0.6.1 \
        pyasn1_modules==0.4.1 \
        pyglet==1.5.0 \
        pyparsing==3.1.4 \
        python-dateutil==2.9.0.post0 \
        pytz==2025.1 \
        PyYAML==6.0.2 \
        requests==2.32.3 \
        requests-oauthlib==2.0.0 \
        rospkg==1.5.1 \
        rsa==4.9 \
        scikit-build==0.18.1 \
        scipy==1.10.1 \
        six==1.17.0 \
        stable-baselines3==1.4.0 \
        sympy==1.13.3 \
        tensorboard==2.14.0 \
        tensorboard-data-server==0.7.2 \
        tensorflow==2.7.4 \
        tensorflow-estimator==2.7.0 \
        tensorflow-io-gcs-filesystem==0.34.0 \
        termcolor==2.4.0 \
        tomli==2.2.1 \
        torch==2.4.1 \
        triton==3.0.0 \
        typing_extensions==4.12.2 \
        tzdata==2025.1 \
        urllib3==2.2.3 \
        Werkzeug==3.0.6 \
        wrapt==1.17.2 \
        zipp==3.20.2 

RUN echo "conda activate venv_gazebo" >> ~/.bashrc

# Install ros packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    python3-distutils \
    ros-noetic-catkin \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-jackal-simulator \
    ros-noetic-jackal-desktop \
    ros-noetic-jackal-navigation \
    && rm -rf /var/lib/apt/lists/*




RUN apt-get update && apt-get install -y bash

# Install Matplot GUI backends
RUN apt-get update
RUN apt-get install -y python3.8-tk
RUN apt-get install -y python3-pil python3-pil.imagetk



RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

# Get Workspace Dependencies
RUN mkdir -p ~/home/wizard/gym_gazebo_docker/src
COPY src /home/wizard/gym_gazebo_docker/src
RUN cd ~/home/wizard/gym_gazebo_docker && \
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
RUN echo 'source docker/.bashrc' >> ~/.bash_profile

# Copy entrypoint
COPY docker/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
