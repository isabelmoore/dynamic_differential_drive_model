import numpy as np
import matplotlib.pyplot as plt
import vehicle_models_copy.bicycle_model as bmd
from path_following.pure_pursuit import PurePursuit
import math
import random
import time
import dubins
from pid import PID
import pdb

def start():
    """Initialize the vehicle"""
    pose = np.zeros(7)
    # pose[3]=10*0.45
    # pose[4]=pose[3]/0.3
    path_length = 250#10
    interval1 = path_length/20
    interval2 = interval1*2
    fig,ax = plot_init()

    dynamic_plot = True

    """Generate Dubins Path"""
    radiusOfCBin = 2
    waypoint_spacing = 1e-1
    num_path_points = int(path_length / waypoint_spacing)
    # random.seed(42)
    dubins_paths = generate_dubins(start_angle=pose[2],
        length=path_length,bin=radiusOfCBin,interval1=interval1,interval2=interval2)
    path = get_dubins_waypoints(dubins_paths,
        spacing=waypoint_spacing)
    path = path[:, :num_path_points]
    rollingFrictionCoeff = np.ones(path.shape[1])
    #multiply the rolling friction coefficient by 0.01 
    #for the first 20% of the path
    rollingFrictionCoeff[:int(0.2*path.shape[1])] = 0.01
    #multiply the rolling friction coefficient by 0.01 
    #for the next 20% of the path
    rollingFrictionCoeff[int(0.2*path.shape[1]):int(0.4*path.shape[1])] = 0.012
    #multiply the rolling friction coefficient by 0.01
    #for the next 20% of the path
    rollingFrictionCoeff[int(0.4*path.shape[1]):int(0.6*path.shape[1])] = 0.009
    #multiply the rolling friction coefficient by 0.01
    #for the next 20% of the path
    rollingFrictionCoeff[int(0.6*path.shape[1]):int(0.8*path.shape[1])] = 0.01
    #multiply the rolling friction coefficient by 0.01
    #for the next 20% of the path
    rollingFrictionCoeff[int(0.8*path.shape[1]):] = 0.015


        
    #Generate the length vector of the path
    path_length = np.zeros(path.shape[1])
    for i in range(1,path.shape[1]):
        path_length[i] = path_length[i-1] + math.sqrt((path[0,i]-path[0,i-1])**2 + (path[1,i]-path[1,i-1])**2)
    print("path length",path_length[-1])
    """Initialize the Pure Pursuit Controller"""
    ppc =PurePursuit(path, spacing=waypoint_spacing)
    #Torque Controller PID
    # tc = PID(kp=1000, ki=0.0, kd=0.0, satLower=-10000.0, satUpper=10000.0)#Original torque controller
    tc = PID(kp=2000, ki=0.0, kd=0.0, satLower=-5880.0, satUpper=5880.0)#Torque controller for velocity control
    episode_steps = 0
    """Initialize the Action Lookahead and Velocity"""
    v_mph = 10
    action = [5,v_mph*0.45]
    """Initialize the Vehicle Model"""
    model = bmd.BicycleModel(center_to_front=1.0,center_to_rear=1.0, plot=False)
    lookahead,velocitySetPoint = scale_action(action)
    path_length_current_time = path_length/velocitySetPoint
    poses= []
    actions = []
    f_trac_vals=[]
    f_roll_vals=[]
    f_drag_vals=[]
    slip_ratio_vals=[]
    F_f_vals=[]
    F_r_vals=[]
    alpha_f_vals=[]
    alpha_r_vals=[]
    lookahead_points_ = []
    side_slip_ratio = []
    t_current = 0.0
    obs_distance = 10.0#Distance of observation

    finalSimTime = 1000.0
    timestep = 1e-1 #Timestep for input discretization
    reward = 0.0
    rew_pose_ = 0.0
    rew_steering_ = 0.0
    rew_slip_ratio_ = 0.0
    for i in range(0,int(finalSimTime/timestep)):

        if(getL2norm(pose[0],pose[1],path[0][-1],path[1][-1])<0.5):
            print("Reward",reward,rew_pose_,rew_steering_,rew_slip_ratio_)
            print("Reached the end of the path")
            plot_all(poses,path,actions,f_trac_vals,f_roll_vals,f_drag_vals,slip_ratio_vals,F_f_vals,F_r_vals,alpha_f_vals,alpha_r_vals,lookahead_points_,velocitySetPoint,side_slip_ratio,t_current)
            exit()

        """Compute inputs to the vehicle model"""
        desiredTorque,desiredDelta = compute_inputs(tc,ppc,pose,desiredVel=velocitySetPoint,desiredLookahead=lookahead)

        """Call scipy RK45 to update the pose"""
        pose, f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r = model.step_dynamic_torque_scipy_rk(pose, (desiredTorque, desiredDelta), 
                                                                                                                  timestep=timestep,t_init=t_current,t_bound=t_current+timestep)
        # print("slip_ratio",slip_ratio)
        poses.append(pose)
        actions.append((desiredTorque,desiredDelta))
        """Append f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r"""
        f_trac_vals.append(f_trac)
        f_roll_vals.append(f_roll)
        f_drag_vals.append(f_drag)
        slip_ratio_vals.append(slip_ratio)
        F_f_vals.append(F_f)
        F_r_vals.append(F_r)
        alpha_f_vals.append(alpha_f)
        alpha_r_vals.append(alpha_r)
        side_slip_ratio.append(math.atan2(pose[5],pose[3]))

        """Calculate the reward for thr robotic vehicle - pass the last two actions"""
        rew_pose, rew_steering, rew_slip_ratio, desiredIdx = get_reward(path_length_current_time=path_length_current_time,path=path,t_current=t_current,pose = pose,action=actions)
        reward += (rew_pose)
        rew_pose_ += rew_pose
        rew_steering_ += rew_steering
        rew_slip_ratio_ += rew_slip_ratio
        episode_steps += 1
        t_current += timestep

        """Plot the pose"""
        if(dynamic_plot and episode_steps%25==0):
            pose_ = np.array([pose[0],pose[1],pose[2]])
            plot_pose(fig,ax,pose_,desiredDelta,path,desiredIdx,obs_distance,waypoint_spacing)


        if(i>=int(finalSimTime/timestep)-1):
            plot_all(poses,path,actions,f_trac_vals,f_roll_vals,f_drag_vals,slip_ratio_vals,F_f_vals,F_r_vals,alpha_f_vals,alpha_r_vals,lookahead_points_,velocitySetPoint,side_slip_ratio,t_current)
            exit()

    """Call runge kutta to update the pose"""
    # t_values,y_values = model.step_dynamic_torque_scipy_rk(pose, (desiredTorque, delta),timestep=0.1,t_init=0.0,t_bound=10,return_full_solution=True)

    # plot_pose_full_solution(y_values,path)

def get_difference_reward(quantity):
    """Get the difference in the quantity"""
    return np.abs(quantity[-1]-quantity[-2])


def get_reward(path_length_current_time,path,t_current,pose,action):
    """Get the reward for the current state"""
    reward_pose = 0.0
    reward_steering = 0.0
    reward_torque = 0.0
    reward_slip_ratio = 0.0
    desiredPoseIdx = np.argmin(np.abs(path_length_current_time - t_current))
    desiredPose = np.array([path[0][desiredPoseIdx],path[1][desiredPoseIdx],0])
    if(t_current>0.0):
        reward_pose = -1*getL2norm(pose[0],pose[1],desiredPose[0],desiredPose[1])/t_current
        reward_steering = -1*get_difference_reward(np.array(action)[:,1])/t_current
        reward_torque = -1*get_difference_reward(np.array(action)[:,0])/t_current
    reward_slip_ratio = -1*math.atan2(pose[5],pose[3])

    return reward_pose, reward_steering, reward_slip_ratio, desiredPoseIdx


def getL2norm(poseX,poseY,finalPointX,finalPointY):
    """Get the L2 norm of the pose and the final point"""
    return math.sqrt((poseX-finalPointX)**2 + (poseY-finalPointY)**2)

# [x,y,theta,v_x,omega_wheel,v_y,omega_body] = [0,0,0,0,0,0,0]
def plot_all(poses,path,actions,f_trac,f_roll,f_drag,slip_ratio,F_f,F_r,alpha_f,alpha_r,lookahead_points_,velocitySetPoint,side_slip_ratio,SimTime):
    print("\n\n\nFinal Sim Time",SimTime)
    poses = np.array(poses)
    # print(poses[:,5],poses[:,6])
    "create subplots of all poses"
    plt.figure()

    plt.subplot(9,1,1)
    plt.plot(poses[:,3]*2.24,label="v_longitudinal in mpH")
    #add the velocity set point
    plt.plot(np.ones(len(poses[:,3]))*velocitySetPoint*2.24,label="velocity set point in mpH")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,2)
    plt.plot(poses[:,4],label="omega_wheel")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,3)
    plt.plot(poses[:,5]*2.24,label="v_lateral in mpH")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,4)
    plt.plot(poses[:,6],label="omega_body")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,5)
    plt.plot(np.array(actions)[:,0],label="Torque")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,6)
    plt.plot(np.array(actions)[:,1],label="Steering Angle")
    plt.legend()
    plt.grid()
    
    plt.subplot(9,1,7)
    plt.plot(poses[:,0],label="x")
    plt.legend()
    plt.grid()

    plt.subplot(9,1,8)
    plt.plot(poses[:,1],label="y")
    plt.legend()
    plt.grid()
    
    plt.subplot(9,1,9)
    plt.plot(np.array(side_slip_ratio)*180/math.pi,label="side slip angle")
    plt.legend()
    plt.grid()

    """Plot v_lateral and omega_body"""
    plt.figure()
    plt.plot(poses[:,5],label="v_lateral")
    plt.plot(poses[:,6],label="omega_body")
    plt.legend()
    plt.grid()

    
    # plt.show()
    """plot f_trac, f_roll, f_drag, slip_ratio, F_f, F_r, alpha_f, alpha_r as subplots"""
    plt.figure()
    plt.subplot(4,2,1)
    plt.plot(np.array(f_trac),label="f_trac")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,2)
    plt.plot(np.array(f_roll),label="f_roll")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,3)
    plt.plot(np.array(f_drag),label="f_drag")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,4)
    plt.plot(np.array(slip_ratio),label="slip_ratio")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,5)
    plt.plot(np.array(F_f),label="F_f")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,6)
    plt.plot(np.array(F_r),label="F_r")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,7)
    plt.plot(np.array(alpha_f),label="alpha_f")
    plt.legend()
    plt.grid()

    plt.subplot(4,2,8)
    plt.plot(np.array(alpha_r),label="alpha_r")
    plt.legend()
    plt.grid()


    """plot the longitudinal velocity and the wheel velocity"""
    # plt.figure()
    # plt.plot(poses[:,3],label="v_longitudinal")
    # plt.plot(poses[:,4],label="omega_wheel")
    # plt.legend()
    # plt.show()

    # """plot the lateral velocity and the yaw rate"""
    # plt.figure()
    # plt.plot(poses[:,5],label="v_lateral")
    # plt.plot(poses[:,6],label="omega_body")
    # plt.legend()
    # plt.show()

    "plot the trajectory"


    plt.figure()
    plt.plot(poses[:,0],poses[:,1],label="Trajectory vehicle (x,y))")
    plt.plot(path[0,:],path[1,:],label="Global Path - given")
    #plot the lookahead points
    lookahead_points_ = np.array(lookahead_points_)
    # plt.plot(lookahead_points_[:,0],lookahead_points_[:,1],'.',label="lookahead points")
    plt.legend()
    plt.grid()
    plt.show()


def plot_init():
    plt.ion # interactive on
    fig, ax = plt.subplots()
    ax.set_title('Trajectory')
    ax.set_xlabel('meters')
    ax.set_ylabel('meters')
    ax.set_aspect('equal')
    plt.pause(0.1)
    return fig, ax


def plot_pose(fig,ax,pose,steer_angle,path,desiredIdx,obs_distance,waypoint_spacing):
    x = pose[0]
    y = pose[1]
    yaw = pose[2]
    lr = 1.0
    lf = 1.0
    #print((yaw + steer_angle) * 180 / math.pi)

    # vehicle axles
    xr = x - lr * math.cos(yaw)
    yr = y - lr * math.sin(yaw)
    xf = x + lf * math.cos(yaw)
    yf = y + lf * math.sin(yaw)

    obs_path_array = get_observation(path,desiredIdx,obs_distance,waypoint_spacing)

    ax.plot(path[0, :], path[1, :], color='red') # path
    ax.scatter(x, y,marker = '.' ,color='blue') # body center
    ax.scatter(path[0,desiredIdx],path[1,desiredIdx],marker = '.',color='green') # desired point
    #ax.plot(path[0,desiredIdx:desiredIdx+1000],path[1,desiredIdx:desiredIdx+1000],color='black') # desired point
    ax.plot(obs_path_array[0,:],obs_path_array[1,:],color='black') # desired point
    # ax.plot([xr, xf], [yr, yf], color='blue') # body
    # ax.arrow(xf, yf, math.cos(yaw + steer_angle),
            # math.sin(yaw + steer_angle), color='green') # front wheel vector
    plt.pause(0.1)


def get_observation(path,desiredIdx,obs_distance,waypoint_spacing):
    """Get the next 10 meters of the path"""

    if(desiredIdx+(obs_distance/waypoint_spacing)<=len(path[0,:])):
        obs_path_array = np.zeros((2,int(obs_distance/waypoint_spacing)))
        obs_path_array[0,:] = path[0,desiredIdx:int(desiredIdx+(obs_distance/waypoint_spacing))]
        obs_path_array[1,:] = path[1,desiredIdx:int(desiredIdx+(obs_distance/waypoint_spacing))]
    else:
        #Set all the values to the last value of the path
        obs_path_array = np.zeros((2,int(obs_distance/waypoint_spacing)))
        obs_path_array[0,:] = path[0][-1]
        obs_path_array[1,:] = path[1][-1]       

        obs_path_array[0,:len(path[0,desiredIdx:])] = path[0,desiredIdx:]
        obs_path_array[1,:len(path[1,desiredIdx:])] = path[1,desiredIdx:]
    
    return obs_path_array




        
def scale_action(action):
    normalized_action_scale = [1.0, 1.0]
    normalized_action_offset = [0.0, 0.0]
    lookahead = normalized_action_scale[0] * action[0] + \
            normalized_action_offset[0] # [0.0, 2.5]
    
    velocity = normalized_action_scale[1] * action[1] + \
            normalized_action_offset[1]
    # velocity = action[1]
    # lookahead = action[0]

    return lookahead,velocity

def get_dubins_waypoints(dubins_paths, spacing=0.001):
    x = []
    y = []
    for path in dubins_paths:
        poses, _ = path.sample_many(spacing)
        for pose in poses:
            x.append(pose[0])
            y.append(pose[1])

    return np.array([x, y])


def generate_dubins(start_angle=0, length=50,bin = 1,interval1=20,interval2=10):
    count=0
    def choose_end(start,count):
        end = [0, 0, 0]
        #for i in range(2):
        #    if(random.uniform(0,100)<=10):
        #        delta = +1*random.uniform(length/10, length/5)#(100,200) 10%
        #    else:
        #        delta =  +1*random.uniform(length/10, length/5)#(100,200) 90%
        if(count == 0):
            end[2] = random.choice([-math.pi/4,math.pi/4])
        else:
            end[2] = random.uniform(-math.pi/3,(math.pi/3))
        r = random.uniform(interval1,interval2)
        #print("r",r)
        end[0] = start[0] + r*math.cos(end[2])
        end[1] = start[1] + r*math.sin(end[2])
        return tuple(end)

    p0 = (0, 0, start_angle)
    p1 = choose_end(p0,count)
    count = count + 1

    if(bin == 1):
        radius = (interval2 - interval1)/1 #50
    elif(bin == 2):
        radius = (interval2 - interval1)/2 #50
    elif(bin == 3):
        radius = (interval2 - interval1)/4 #50
    elif(bin == 4):
        radius = (interval2 - interval1)/8 #50
    elif(bin == 5):
        radius = (interval2 - interval1)/16 #50
    elif(bin == 6):
        radius = (interval2 - interval1)/32 #50

    print("radius",radius)
    # radius = (interval2 - interval1)/2 #50
    current_length = 0
    paths = []
    while current_length < length:
        paths.append(dubins.shortest_path(p0, p1, radius))
        current_length += paths[-1].path_length()
        # Update start and end poses
        p0 = p1
        p1 = choose_end(p0,count)
        count = count + 1

    return paths

def compute_inputs(tc,ppc,pose,desiredVel = 2.0,desiredLookahead = 0.5):
    
    v_x = pose[3]
    desiredTorque = tc.computeControl(desiredVel - v_x)
    ppc.setLookahead(desiredLookahead)
    curvature, lookaheadPoint, closestDistance = ppc.run(pose[:3])
    delta = math.atan(curvature*(1+1))
    delta = np.clip(delta,-math.pi/3,math.pi/3)

    return desiredTorque,delta


if __name__=="__main__":
    start()