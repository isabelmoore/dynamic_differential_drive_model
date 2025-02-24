# OpenAI (Stable Baselines 3) and Gazebo Integration

## LENS Lab Computer Instructions
- Install Docker.io: `dzdo apt-get install -y docker.io`
- Status for Docker: `dzdo systemctl status docker`
- Starting Docker Daemon: `dzdo systemctl start docker`
  - If this fails, run `dzdo systemctl restart docker`
- Building Docker image: `dzdo docker compose build`
- Run the Container: `dzdo docker compose -f ~/afcagc_gym_gazebo/docker-compose.yml run --rm ros_noetic`
- In another terminal, run: `dzdo docker ps` to check the running containers. Yours should be there as from the previous step.
- Then, run the following (ensure`<container_name>` matches the listed container name under `NAMES`):
  - `dzdo docker exec -it <container_name> /bin/bash` 
- Source the Noetic environment: `source /opt/ros/noetic/setup.bash`
- Run: `roscore`
- Within the first terminal, source: `source /opt/ros/noetic/setup.bash`
- Build the workspace: `catkin_make`
  - If you have permission issues in building, run the following with `<user>` as you TAMU username:
    ```
    sudo chown -R <user>:<user> ~/gym_gazebo_docker
    sudo chmod -R 777 ~/gym_gazebo_docker
    ```
- Run: `source devel/setup.bash`
- Train the model: `src/gym_gazebo/launch/train_model.sh`


## Instructions
- Python Virtual Environment (Optional)
  - Create Python virtual environment: `python3 -m venv env`
  - Activate Python virtual environment: `source env/bin/activate`
  - Upgrade pip: `python3 -m pip install --upgrade pip`
  - Install Python packages: `python3 -m pip install -r requirements.txt`
  - Deactivate Python virtual environment: `deactivate`
- Build Gazebo Step Plugin
  - Create build directory: `mkdir plugins/build`
  - Build: `cd plugins/build`, `cmake ../`, `make`
  - Add build path to Gazebo plugin path environment variable: `export
    GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:$(pwd)`
- Start Gazebo simulation: `roslaunch gym_gazebo empty_world.launch
  world_name:=$(rospack find gym_gazebo)/worlds/empty_plugin.world` or start
  ROS: `roscore`
- Update parameters file
- Train with Gazebo: `launch/train_gazebo.sh`
- Train with model: `launch/train_model.sh`
- Start Tensorboard: `tensorboard --logdir logs`
- Evaluate: `launch/evaluate.sh`


