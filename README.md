# Gym Gazebo Docker Set Up

## Building and Sourcing Environment

To set up the Gym Gazebo Docker environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone git@github.com:tamu-edu/afcagc_gym_gazebo.git --branch docker-ddd
    mv afcagc_gym_gazebo gym_gazebo_docker
    cd gym_gazebo_docker
    ```
2. Build submodules for Jackal
    ```bash
    git submodule init
    git submodule update
    ```

3. Build the Docker container: 
    ```bash
    docker-compose build
    ```
    If on the Lambda computers, follow these steps; otherwise skip.
    ```bash
    dzdo systemctl start docker    # start docker
    dzdo systemctl status docker    # ensure docker is running
    dzdo docker-compose build
    ```


4. Run the container:
    ```bash
    docker compose -f ~/gym_gazebo_docker/docker-compose.yml run --rm ros_noetic
    ```
5. Within the container, build and source: 
    ```bash
    cd gym_gazebo_docker/
    source docker/.bashrc
    sc
    build
    source /opt/conda/etc/profile.d/conda.sh
    conda activate venv_gazebo
    ```

# Necessary Alterations to Jackal Packages

### Ground Truth Plugin:
1. To implement the Ground Truth ROS topic, locate file:
`gym_gazebo_docker/src/jackal/jackal_description/urdf/jackal.gazebo`

2. Add the following code, near the other plugins (around line `39`):
    ```xml
    <!-- 2) Ground‐truth plugin -->
    <gazebo>
        <plugin filename="libgazebo_ros_p3d.so" name="ground_truth_plugin">
        <modelName>jackal</modelName>
        <bodyName>base_link</bodyName>
        <topicName>ground_truth/state</topicName>
        <updateRate>100.0</updateRate>
        <gaussianNoise>0.0</gaussianNoise>
        <frameName>world</frameName>
        </plugin>
    </gazebo>
    ```

### Friction Coefficient:
To alter the friction oefficient, alter lines `55-56` in file:

`gym_gazebo_docker/src/jackal/jackal_description/urdf/jackal.urdf.xacro`

For more information:
https://classic.gazebosim.org/tutorials?tut=friction


## Running and Evaluating:
Ensure you have built and source your environment for each terminal, following the previous step. 

1. Run Gazebo Simulation:
    ```bash
    sim
    ```
    If on the Lambda computers and do not see the simulation opening, exit the container and run the following commands; otherwise skip:
    ```bash 
    xhost +local:root

    docker-compose -f ~/gym_gazebo_docker/afcagc_gym_gazebo/docker-compose.yml run \
    --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ros_noetic
    ```

2. (Optional) Testing the simulation:   
    In another terminal, you can run the following commands:
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


3. Evaluating the model:
    ```bash
    cd src/gym_gazebo/scripts
    python evaluate_model.py
    ```
