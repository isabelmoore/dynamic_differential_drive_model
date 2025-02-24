#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class ModelPush : public ModelPlugin
  {
    public:
      void Load(physics::ModelPtr _parent, sdf::ElementPtr)
      {
        // Store pointer to the model
        this->model = _parent;

        // Listen to update event. Event is broadcast every simulation iteration
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&ModelPush::OnUpdate, this));
      }

      // Called by the world update start event
      void OnUpdate()
      {
        // Apply a small linear velocity to the model
        this->model->SetLinearVel(ignition::math::Vector3d(0.3, 0, 0));
      }

    private:
      // pointer to the model
      physics::ModelPtr model;

      // pointer to update event connection
      event::ConnectionPtr updateConnection;
  };

  // Register plugin with simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelPush);
}
