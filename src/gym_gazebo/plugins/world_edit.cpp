#include <sdf/sdf.hh>
#include <ignition/math/Pose3.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo
{
  class WorldEdit : public WorldPlugin
  {
    public:
      void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
      {
        // Create transport node
        transport::NodePtr node(new transport::Node());

        // Initialize node with world name
        node->Init(_parent->Name());

        // Create a publisher on the ~/physics topic
        transport::PublisherPtr physicsPub =
          node->Advertise<msgs::Physics>("~/physics");

        msgs::Physics physicsMsg;
        physicsMsg.set_type(msgs::Physics::ODE);

        // set step time
        physicsMsg.set_max_step_size(0.01);

        // change gravity
        msgs::Set(physicsMsg.mutable_gravity(), ignition::math::Vector3d(0.01,
              0.0, 0.1));

        physicsPub->Publish(physicsMsg);
      }
  };

  // Register plugin with simulator
  GZ_REGISTER_WORLD_PLUGIN(WorldEdit)
}
