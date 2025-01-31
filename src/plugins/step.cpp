#include "gazebo/gazebo.hh"
#include "gazebo/common/Plugin.hh"
#include "gazebo/msgs/msgs.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/transport/transport.hh"
#include "ignition/math/Pose3.hh"

#include <thread>
#include <vector>
#include <chrono>

#include "ros/ros.h"
#include "ros/console.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"

#include <gym_gazebo/Step.h>

namespace gazebo
{
  class StepPlugin : public WorldPlugin
  {
    public:
      void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
      {
        this->world = _parent;
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&StepPlugin::OnUpdate, this));

        if (!ros::isInitialized())
        {
          int argc = 0;
          char** argv = NULL;
          ros::init(argc, argv, "gazebo_client",
              ros::init_options::NoSigintHandler);
        }
        this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
        this->service = this->rosNode->advertiseService("/gazebo/step",
            &StepPlugin::StepService, this);

        // Read SDF parameters
        this->sdfModel = _sdf->Get<std::string>("model");
        this->sdfController = _sdf->Get<std::string>("controller");
        std::string method = _sdf->Get<std::string>("method");
        if (method == "instant")
        {
          this->method = Method::INSTANT;
        }
        else if (method == "motor")
        {
          this->method = Method::MOTOR;
        }
        else if (method == "pid")
        {
          this->method = Method::PID;
        }
        else
        {
          ROS_WARN("Invalid value for SDF parameter 'method'. Using motor.");
          this->method = Method::MOTOR;
        }
 
        this->initialized = false;
        ROS_INFO("Loaded Gazebo Step plugin");
      }

      void Initialize()
      {
        // Read ROS parameters
        std::vector<std::string> leftWheels, rightWheels;
        double wheelRadius, wheelSeparation;
        double wheelRadiusMultiplier, wheelSeparationMultiplier;
        this->rosNode->getParam("/" + this->sdfController + "/left_wheel",
            leftWheels);
        this->rosNode->getParam("/" + this->sdfController + "/right_wheel",
            rightWheels);
        this->rosNode->getParam("/" + this->sdfController + "/wheel_radius",
            wheelRadius);
        this->rosNode->getParam("/" + this->sdfController + "/wheel_separation",
            wheelSeparation);
        this->rosNode->getParam("/" + this->sdfController +
            "/wheel_radius_multiplier", wheelRadiusMultiplier);
        this->rosNode->getParam("/" + this->sdfController +
            "/wheel_separation_multiplier", wheelSeparationMultiplier);

        // Calculate and store effective wheel radius, wheel separation
        this->wheelRadius = wheelRadiusMultiplier * wheelRadius;
        this->wheelSeparation = wheelSeparationMultiplier * wheelSeparation;
        
        // Store model pointer
        this->model = this->world->ModelByName(this->sdfModel);
        
        // Store scoped names
        for (std::string name : leftWheels)
        {
          this->leftWheels.push_back(
              this->model->GetJoint(name)->GetScopedName());
        }
        for (std::string name : rightWheels)
        {
          this->rightWheels.push_back(
              this->model->GetJoint(name)->GetScopedName());
        }

        // Create joint controllers
        this->controller.reset(new physics::JointController(this->model));
        for (std::string name : this->leftWheels)
        {
          this->controller->AddJoint(this->model->GetJoint(name));
          // PID(p, i, d, imax, imin, cmdMax, cmdMin)
          this->controller->SetVelocityPID(name,
              common::PID(1, 0, 0, 0, 0, -100.0, 100.0));
        }
        for (std::string name : this->rightWheels)
        {
          this->controller->AddJoint(this->model->GetJoint(name));
          this->controller->SetVelocityPID(name,
              common::PID(1, 0, 0, 0, 0, -100.0, 100.0));
        }

        this->initialized = true;
        ROS_INFO("Gazebo Step plugin initialized");
      }

      bool StepService(gym_gazebo::Step::Request &req,
          gym_gazebo::Step::Response &res)
      {
        if (!this->initialized) this->Initialize();

        // Calculate steps
        double stepSize = this->world->Physics()->GetMaxStepSize();
        ROS_DEBUG("step size: %f", stepSize);
        uint32_t steps = req.seconds / stepSize;
        ROS_DEBUG("steps: %d", steps);
        
        // Calculate joint velocities
        double lx = req.command.linear.x;
        double az = req.command.angular.z;
        this->wheelVelocityLeft = (lx - az * this->wheelSeparation / 2.0) /
          this->wheelRadius;
        this->wheelVelocityRight = (lx + az * this->wheelSeparation / 2.0) /
          this->wheelRadius;
        ROS_DEBUG("wheel velocity left: %f, right: %f",
            this->wheelVelocityLeft, this->wheelVelocityRight);
        
        // Set joint velocities now (MOTOR, PID) or on update (INSTANT)
        if (this->method == Method::MOTOR)
        {
          this->SetJointVelocities(Method::MOTOR);
        }
        else if (this->method == Method::PID)
        {
          this->SetJointVelocities(Method::PID);
        }
        
        // Take steps
        this->world->Step(steps);
        ROS_DEBUG("Simulated %d steps", steps);

        // Get odometry
        ignition::math::Pose3<double> pose = this->model->WorldPose();
        ignition::math::Vector3<double> position = pose.Pos();
        ignition::math::Quaternion<double> orientation = pose.Rot();
        ignition::math::Vector3<double> linearVelocity =
          this->model->RelativeLinearVel();
        ignition::math::Vector3<double> angularVelocity =
          this->model->RelativeAngularVel();
        res.odometry.pose.pose.position.x = position.X();
        res.odometry.pose.pose.position.y = position.Y();
        res.odometry.pose.pose.position.z = position.Z();
        res.odometry.pose.pose.orientation.x = orientation.X();
        res.odometry.pose.pose.orientation.y = orientation.Y();
        res.odometry.pose.pose.orientation.z = orientation.Z();
        res.odometry.pose.pose.orientation.w = orientation.W();
        res.odometry.twist.twist.linear.x = linearVelocity.X();
        res.odometry.twist.twist.linear.y = linearVelocity.Y();
        res.odometry.twist.twist.linear.z = linearVelocity.Z();
        res.odometry.twist.twist.angular.x = angularVelocity.X();
        res.odometry.twist.twist.angular.y = angularVelocity.Y();
        res.odometry.twist.twist.angular.z = angularVelocity.Z();
        res.odometry.header.frame_id = "world";
        res.odometry.child_frame_id = "base_link";

        // Reset joint velocities to zero here?
        
        return true;
      }

      void OnUpdate()
      {
        static uint32_t updates = 0;
        updates++;
        ROS_DEBUG("updates: %d", updates);

        // Reset joint velocities
        if (this->method == Method::INSTANT)
        {
          this->SetJointVelocities(Method::INSTANT);
        }
        else if (this->method == Method::PID)
        {
          this->controller->Update(); // PID
          //this->PrintJointControllerInfo();
        }
      }

    private:
      enum class Method {INSTANT, MOTOR, PID};

      void SetJointVelocity(Method method, std::string name, double velocity)
      {
        switch (method)
        {
          case Method::INSTANT:
            this->model->GetJoint(name)->SetVelocity(0, velocity);
            break;
          case Method::MOTOR:
            {
              this->model->GetJoint(name)->SetParam("fmax", 0, 100.0);
              this->model->GetJoint(name)->SetParam("vel", 0, velocity);
              break;
            }
          case Method::PID:
            this->controller->SetVelocityTarget(name, velocity);
            break;
        }
      }

      /*
       * See tutorial: http://gazebosim.org/tutorials?tut=set_velocity&cat=.
       */
      void SetJointVelocities(Method method)
      {
        for (std::string name : this->leftWheels)
        {
          this->SetJointVelocity(method, name, this->wheelVelocityLeft);
        }
        for (std::string name : this->rightWheels)
        {
          this->SetJointVelocity(method, name, this->wheelVelocityRight);
        }
      }

      void PrintJointControllerInfo()
      {
        //std::map<std::string, common::PID> pids =
        //  this->model->GetJointController()->GetVelocityPIDs();
        std::map<std::string, common::PID> pids =
          this->controller->GetVelocityPIDs();
        std::map<std::string, common::PID>::iterator it;
        double pe, ie, de;
        for (it = pids.begin(); it != pids.end(); it++)
        {
          it->second.GetErrors(pe, ie, de);
          std::cout << it->first
                    << ": P = "
                    << it->second.GetPGain()
                    << ", I = "
                    << it->second.GetIGain()
                    << ", D = "
                    << it->second.GetDGain()
                    << ", Imax = "
                    << it->second.GetIMax()
                    << ", Imin = "
                    << it->second.GetIMin()
                    << ", cmd = "
                    << it->second.GetCmd()
                    << ", cmdMax = "
                    << it->second.GetCmdMax()
                    << ", cmdMin = "
                    << it->second.GetCmdMin()
                    << ", error = "
                    << pe
                    << std::endl;
        }
      }

      void PrintJointInfo(std::string jointName)
      {
        double value;
        physics::JointPtr joint = this->model->GetJoint(jointName);
        
        value = joint->GetEffortLimit(0);
        std::cout << "effort limit 0: " << value << std::endl;
        value = joint->GetVelocityLimit(0);
        std::cout << "velocity limit 0: " << value << std::endl;
        value = joint->UpperLimit();
        std::cout << "upper limit: " << value << std::endl;
      }

      // Gazebo
      physics::WorldPtr world;
      physics::ModelPtr model;
      physics::JointControllerPtr controller;
      event::ConnectionPtr updateConnection;
      
      // ROS
      std::unique_ptr<ros::NodeHandle> rosNode;
      ros::Subscriber rosSub;
      ros::ServiceServer service;
      
      std::vector<std::string> leftWheels, rightWheels;
      double wheelRadius, wheelSeparation;
      double wheelVelocityLeft, wheelVelocityRight;
      bool initialized;
      std::string sdfModel, sdfController;
      Method method;
  };

  GZ_REGISTER_WORLD_PLUGIN(StepPlugin);
}
