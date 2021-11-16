package us.ihmc.avatar.initialSetup;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import us.ihmc.euclid.geometry.interfaces.Pose3DBasics;
import us.ihmc.euclid.geometry.interfaces.Pose3DReadOnly;
import us.ihmc.euclid.referenceFrame.tools.ReferenceFrameTools;
import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.Vector3D;
import us.ihmc.euclid.tuple4D.Quaternion;
import us.ihmc.mecano.multiBodySystem.interfaces.FloatingJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.JointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.OneDoFJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.RigidBodyBasics;
import us.ihmc.mecano.spatial.interfaces.FixedFrameTwistBasics;
import us.ihmc.mecano.tools.MultiBodySystemTools;
import us.ihmc.robotModels.FullHumanoidRobotModel;
import us.ihmc.robotics.partNames.ArmJointName;
import us.ihmc.robotics.partNames.HumanoidJointNameMap;
import us.ihmc.robotics.partNames.LegJointName;
import us.ihmc.robotics.partNames.NeckJointName;
import us.ihmc.robotics.partNames.SpineJointName;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.scs2.definition.robot.RobotDefinition;
import us.ihmc.simulationConstructionSetTools.util.HumanoidFloatingRootJointRobot;
import us.ihmc.simulationconstructionset.OneDegreeOfFreedomJoint;

public abstract class HumanoidRobotInitialSetup implements RobotInitialSetup<HumanoidFloatingRootJointRobot>
{
   protected double initialYaw = 0.0;
   protected double initialGroundHeight = 0.0;
   protected final Vector3D additionalOffset = new Vector3D();

   protected final Point3D rootJointPosition = new Point3D();
   protected final Quaternion rootJointOrientation = new Quaternion();
   protected final Vector3D rootJointAngularVelocityInBody = new Vector3D();
   protected final Vector3D rootJointLinearVelocityInWorld = new Vector3D();
   protected final Map<String, Double> jointPositions = new HashMap<>();
   protected final HumanoidJointNameMap jointMap;

   public HumanoidRobotInitialSetup(HumanoidJointNameMap jointMap)
   {
      this.jointMap = jointMap;
   }

   public List<Double> getInitialJointAngles() // Implement for kinematics sim & start pose
   {
      return null;
   }

   public Pose3DReadOnly getInitialPelvisPose() // Implement for kinematics sim & start pose
   {
      return null;
   }

   public void setJoint(RobotSide robotSide, LegJointName legJointName, double q)
   {
      String jointName = jointMap.getLegJointName(robotSide, legJointName);
      if (jointName != null)
         setJoint(jointName, q);
   }

   public void setJoint(RobotSide robotSide, ArmJointName armJointName, double q)
   {
      String jointName = jointMap.getArmJointName(robotSide, armJointName);
      if (jointName != null)
         setJoint(jointName, q);
   }

   public void setJoint(SpineJointName spineJointName, double q)
   {
      String jointName = jointMap.getSpineJointName(spineJointName);
      if (jointName != null)
         setJoint(jointName, q);
   }

   public void setJoint(NeckJointName neckJointName, double q)
   {
      String jointName = jointMap.getNeckJointName(neckJointName);
      if (jointName != null)
         setJoint(jointName, q);
   }

   public void setJoint(String jointName, double q)
   {
      jointPositions.put(jointName, q);
   }

   public void adjustRootJointHeightUsingLowestSole(RobotDefinition robotDefinition)
   {
      RigidBodyBasics rootBody = robotDefinition.newInstance(ReferenceFrameTools.constructARootFrame("temp"));
      initializeRobot(rootBody, false);
      rootBody.updateFramesRecursively();

      double minSoleHeight = Double.POSITIVE_INFINITY;
      RigidBodyTransform temp = new RigidBodyTransform();

      for (RobotSide robotSide : RobotSide.values)
      {
         RigidBodyBasics foot = MultiBodySystemTools.findRigidBody(rootBody, jointMap.getFootName(robotSide));
         if (foot == null)
            continue;
         temp.set(foot.getParentJoint().getFrameAfterJoint().getTransformToRoot());
         temp.multiply(jointMap.getSoleToParentFrameTransform(robotSide));
         minSoleHeight = Math.min(temp.getTranslationZ(), minSoleHeight);
      }

      if (Double.isFinite(minSoleHeight))
         rootJointPosition.setZ(-minSoleHeight);
   }

   @Override
   public void initializeRobot(HumanoidFloatingRootJointRobot robot)
   {
      for (OneDegreeOfFreedomJoint joint : robot.getOneDegreeOfFreedomJoints())
      {
         Double jointPosition = getJointPosition(joint.getName());

         if (jointPosition != null)
         {
            joint.setQ(jointPosition);
         }
      }

      robot.getRootJoint().getPosition().set(rootJointPosition);
      robot.getRootJoint().getPosition().add(additionalOffset);
      robot.getRootJoint().getPosition().addZ(initialGroundHeight);
      robot.getRootJoint().setOrientation(rootJointOrientation);
      robot.getRootJoint().getOrientation().prependYawRotation(initialYaw);
      robot.update();
   }

   @Override
   public void initializeFullRobotModel(FullHumanoidRobotModel fullRobotModel)
   {
      initializeRobot(fullRobotModel.getElevator());
   }

   public void initializeRobot(RigidBodyBasics rootBody)
   {
      initializeRobot(rootBody, true);
   }

   private void initializeRobot(RigidBodyBasics rootBody, boolean applyRootJointPose)
   {
      for (JointBasics joint : rootBody.childrenSubtreeIterable())
      {
         if (joint instanceof OneDoFJointBasics)
         {
            Double jointPosition = getJointPosition(joint.getName());

            if (jointPosition != null)
            {
               ((OneDoFJointBasics) joint).setQ(jointPosition);
            }
         }
      }

      if (applyRootJointPose && rootBody.getChildrenJoints().size() == 1)
      {
         JointBasics rootJoint = rootBody.getChildrenJoints().get(0);

         if (rootJoint instanceof FloatingJointBasics)
         {
            FloatingJointBasics floatingJoint = (FloatingJointBasics) rootJoint;
            Pose3DBasics jointPose = floatingJoint.getJointPose();
            FixedFrameTwistBasics jointTwist = floatingJoint.getJointTwist();
            jointPose.set(rootJointPosition, rootJointOrientation);
            jointTwist.getAngularPart().set(rootJointAngularVelocityInBody);
            jointPose.getOrientation().inverseTransform(rootJointLinearVelocityInWorld, jointTwist.getLinearPart());
         }
      }
   }

   @Override
   public void setInitialYaw(double yaw)
   {
      initialYaw = yaw;
   }

   @Override
   public double getInitialYaw()
   {
      return initialYaw;
   }

   @Override
   public void setInitialGroundHeight(double groundHeight)
   {
      initialGroundHeight = groundHeight;
   }

   @Override
   public double getInitialGroundHeight()
   {
      return initialGroundHeight;
   }

   @Override
   public void setOffset(Vector3D additionalOffset)
   {
      this.additionalOffset.set(additionalOffset);
   }

   @Override
   public void getOffset(Vector3D offsetToPack)
   {
      offsetToPack.set(additionalOffset);
   }

   public Double getJointPosition(String jointName)
   {
      return jointPositions.get(jointName);
   }

   public Map<String, Double> getJointPositions()
   {
      return jointPositions;
   }

   public Point3D getRootJointPosition()
   {
      return rootJointPosition;
   }

   public Quaternion getRootJointOrientation()
   {
      return rootJointOrientation;
   }

   public Vector3D getRootJointAngularVelocityInBody()
   {
      return rootJointAngularVelocityInBody;
   }

   public Vector3D getRootJointLinearVelocityInWorld()
   {
      return rootJointLinearVelocityInWorld;
   }
}
