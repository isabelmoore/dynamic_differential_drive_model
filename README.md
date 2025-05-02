# Gym Gazebo Docker Set Up

## Command Line Instructions

To set up the Gym Gazebo Docker environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/gym_gazebo_docker.git
    cd gym_gazebo_docker
    ```

2. Run the Docker container:
    ```bash
    docker compose build
    ```

3. Build and source:
    ```bash
    sc
    build
    ```

4. Run Gazebo Simulation:
    ```bash
    sim
    ```

## Testing Simulation

1. Circle:
    ```bash
    rostopic pub /cmd_vel geometry_msgs/Twist \
    "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" -r 10
    ```


2. Using `I-J-K-L` commands to manually move:
    ```bash
    sudo apt-get update
    sudo apt-get install ros-noetic-teleop-twist-keyboard
    rosrun teleop_twist_keyboard teleop_twist_keyboard.py
    ```


## Running and evaluate:
1. Activating Conda Environment:
    ```bash
    source /opt/conda/etc/profile.d/conda.sh
    conda activate venv_gazebo
    ```
2. In one terminal, run:
    ```bash
    roslaunch jackal_gazebo empty_world.launch
    ```

3. In another terminal, run `evaluate_model.py`
    ```bash
    jackal
    source /opt/conda/etc/profile.d/conda.sh
    conda activate venv_gazebo
    cd src/gym_gazebo/scripts
    python evaluate_model.py
    ```

4. In short:
    ```bash
    jackal
    source docker/.bashrc
    sc
    source /opt/conda/etc/profile.d/conda.sh
    conda activate venv_gazebo
    sim
roslaunch jackal_gazebo spawn_jackal.launch
roslaunch gazebo_ros empty_world.launch \
  world_name:="$(find gym_gazebo)/worlds/empty_plugin.world" \
  paused:=false
roslaunch jackal_gazebo spawn_jackal.launch

    jackal
    source /opt/conda/etc/profile.d/conda.sh
    conda activate venv_gazebo
    cd src/gym_gazebo/scripts
    python evaluate_model.py
    ```