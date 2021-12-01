package us.ihmc.robotModels.description;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.lang3.mutable.MutableObject;

import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.euclid.tuple3D.Vector3D;
import us.ihmc.graphicsDescription.Graphics3DObject;
import us.ihmc.graphicsDescription.appearance.AppearanceDefinition;
import us.ihmc.graphicsDescription.instructions.Graphics3DPrimitiveInstruction;
import us.ihmc.log.LogTools;
import us.ihmc.robotics.graphics.VisualDefinitionConverter;
import us.ihmc.robotics.robotDescription.CameraSensorDescription;
import us.ihmc.robotics.robotDescription.CollisionMeshDescription;
import us.ihmc.robotics.robotDescription.ExternalForcePointDescription;
import us.ihmc.robotics.robotDescription.FloatingJointDescription;
import us.ihmc.robotics.robotDescription.ForceSensorDescription;
import us.ihmc.robotics.robotDescription.GraphicsObjectsHolder;
import us.ihmc.robotics.robotDescription.GroundContactPointDescription;
import us.ihmc.robotics.robotDescription.IMUSensorDescription;
import us.ihmc.robotics.robotDescription.JointDescription;
import us.ihmc.robotics.robotDescription.KinematicPointDescription;
import us.ihmc.robotics.robotDescription.LidarSensorDescription;
import us.ihmc.robotics.robotDescription.LinkDescription;
import us.ihmc.robotics.robotDescription.LinkGraphicsDescription;
import us.ihmc.robotics.robotDescription.LoopClosurePinConstraintDescription;
import us.ihmc.robotics.robotDescription.OneDoFJointDescription;
import us.ihmc.robotics.robotDescription.PinJointDescription;
import us.ihmc.robotics.robotDescription.RobotDescription;
import us.ihmc.robotics.robotDescription.SliderJointDescription;
import us.ihmc.scs2.definition.collision.CollisionShapeDefinition;
import us.ihmc.scs2.definition.geometry.GeometryDefinition;
import us.ihmc.scs2.definition.robot.CameraSensorDefinition;
import us.ihmc.scs2.definition.robot.ExternalWrenchPointDefinition;
import us.ihmc.scs2.definition.robot.GroundContactPointDefinition;
import us.ihmc.scs2.definition.robot.IMUSensorDefinition;
import us.ihmc.scs2.definition.robot.JointDefinition;
import us.ihmc.scs2.definition.robot.KinematicPointDefinition;
import us.ihmc.scs2.definition.robot.LidarSensorDefinition;
import us.ihmc.scs2.definition.robot.LoopClosureDefinition;
import us.ihmc.scs2.definition.robot.OneDoFJointDefinition;
import us.ihmc.scs2.definition.robot.PrismaticJointDefinition;
import us.ihmc.scs2.definition.robot.RevoluteJointDefinition;
import us.ihmc.scs2.definition.robot.RigidBodyDefinition;
import us.ihmc.scs2.definition.robot.RobotDefinition;
import us.ihmc.scs2.definition.robot.SixDoFJointDefinition;
import us.ihmc.scs2.definition.robot.WrenchSensorDefinition;
import us.ihmc.scs2.definition.visual.MaterialDefinition;
import us.ihmc.scs2.definition.visual.VisualDefinition;

public class RobotDefinitionConverter
{
   public static GraphicsObjectsHolder toGraphicsObjectsHolder(RobotDefinition robotDefinition)
   {
      Map<String, Graphics3DObject> cache = new HashMap<>();
      return jointName ->
      {
         return cache.computeIfAbsent(jointName, name ->
         {
            JointDefinition jointDefinition = robotDefinition.getJointDefinition(jointName);
            return RobotDefinitionConverter.toLinkGraphicsDescription(jointDefinition.getSuccessor().getVisualDefinitions());
         });
      };
   }

   public static RobotDescription toRobotDescription(RobotDefinition robotDefinition)
   {
      RobotDescription robotDescription = new RobotDescription(robotDefinition.getName());

      for (JointDefinition rootJointDefinition : robotDefinition.getRootJointDefinitions())
         createAndAddJointsRecursive(robotDescription, rootJointDefinition);

      createAndAddLoopClosureJoints(robotDescription, robotDefinition);

      // Fixed the inner joints of four bar joints: The 2 first inner joints needs to be connected to the four-bar parent joint.
      for (JointDefinition jointDefinition : robotDefinition.getAllJoints())
      {
         if (!(jointDefinition instanceof InvertedFourBarJointDefinition))
            continue;

         InvertedFourBarJointDefinition fourBarDefinition = (InvertedFourBarJointDefinition) jointDefinition;
         InvertedFourBarJointDescription fourBarDescription = (InvertedFourBarJointDescription) robotDescription.getJointDescription(fourBarDefinition.getName());

         for (JointDefinition loopJointDefinition : fourBarDefinition.getFourBarJoints())
         {
            if (loopJointDefinition.isLoopClosure())
               continue;

            if (loopJointDefinition.getParentJoint() == fourBarDefinition.getParentJoint())
            {
               PinJointDescription loopJointDescription = Arrays.stream(fourBarDescription.getFourBarJoints())
                                                                .filter(j -> j.getName().equals(loopJointDefinition.getName())).findFirst().get();
               loopJointDescription.setParentJoint(fourBarDescription.getParentJoint());
            }
         }
      }

      for (String jointName : robotDefinition.getNameOfJointsToIgnore())
         robotDescription.getJointDescription(jointName).setIsDynamic(false);

      return robotDescription;
   }

   public static void createAndAddJointsRecursive(RobotDescription robotDescription, JointDefinition rootJointToCopy)
   {
      JointDescription rootJointDescription = toJointDescription(rootJointToCopy);
      robotDescription.addRootJoint(rootJointDescription);

      LinkDescription linkDescription = toLinkDescription(rootJointToCopy.getSuccessor());
      rootJointDescription.setLink(linkDescription);

      for (JointDefinition childJointToCopy : rootJointToCopy.getSuccessor().getChildrenJoints())
         RobotDefinitionConverter.createAndAddJointsRecursive(rootJointDescription, childJointToCopy);
   }

   public static void createAndAddJointsRecursive(JointDescription parentJoint, JointDefinition jointToCopy)
   {
      if (jointToCopy.isLoopClosure())
         return; // We address the loop closures in a second pass.

      JointDescription jointDescription = toJointDescription(jointToCopy);
      parentJoint.addJoint(jointDescription);

      LinkDescription linkDescription = toLinkDescription(jointToCopy.getSuccessor());
      jointDescription.setLink(linkDescription);

      for (JointDefinition childJointToCopy : jointToCopy.getSuccessor().getChildrenJoints())
         createAndAddJointsRecursive(jointDescription, childJointToCopy);
   }

   private static void createAndAddLoopClosureJoints(RobotDescription robotDescription, RobotDefinition robotDefinition)
   {
      for (JointDefinition jointDefinition : robotDefinition.getAllJoints())
      {
         if (!jointDefinition.isLoopClosure())
            continue;

         if (!(jointDefinition instanceof RevoluteJointDefinition))
            throw new UnsupportedOperationException("Only supports revolute loop closures.");

         RevoluteJointDefinition revoluteJointDefinition = (RevoluteJointDefinition) jointDefinition;
         LoopClosureDefinition loopClosureDefinition = revoluteJointDefinition.getLoopClosureDefinition();

         String name = revoluteJointDefinition.getName();
         Vector3D offsetFromParentJoint = jointDefinition.getTransformToParent().getTranslation();
         Vector3D offsetFromLinkParentJoint = loopClosureDefinition.getTransformToSuccessorParent().getTranslation();
         Vector3D axis = revoluteJointDefinition.getAxis();
         LoopClosurePinConstraintDescription loopClosurePinConstraintDescription = new LoopClosurePinConstraintDescription(name,
                                                                                                                           offsetFromParentJoint,
                                                                                                                           offsetFromLinkParentJoint,
                                                                                                                           axis);
         loopClosurePinConstraintDescription.setGains(loopClosureDefinition.getKpSoftConstraint(), loopClosureDefinition.getKdSoftConstraint());
         robotDescription.getJointDescription(revoluteJointDefinition.getParentJoint().getName()).addConstraint(loopClosurePinConstraintDescription);
         loopClosurePinConstraintDescription.setLink(robotDescription.getJointDescription(revoluteJointDefinition.getSuccessor().getParentJoint().getName())
                                                                     .getLink());
      }
   }

   public static LinkDescription toLinkDescription(RigidBodyDefinition source)
   {
      LinkDescription output = new LinkDescription(source.getName());
      output.setMass(source.getMass());
      output.setCenterOfMassOffset(source.getCenterOfMassOffset());
      output.setMomentOfInertia(source.getMomentOfInertia());
      output.setLinkGraphics(RobotDefinitionConverter.toLinkGraphicsDescription(source.getVisualDefinitions()));
      output.getCollisionMeshes().addAll(RobotDefinitionConverter.toCollisionMeshDescriptions(source.getCollisionShapeDefinitions()));
      return output;
   }

   public static JointDescription toJointDescription(JointDefinition source)
   {
      if (source instanceof SixDoFJointDefinition)
         return RobotDefinitionConverter.toFloatingJointDescription((SixDoFJointDefinition) source);
      if (source instanceof RevoluteJointDefinition)
         return RobotDefinitionConverter.toPinJointDescription((RevoluteJointDefinition) source);
      if (source instanceof PrismaticJointDefinition)
         return RobotDefinitionConverter.toSliderJointDescription((PrismaticJointDefinition) source);
      if (source instanceof InvertedFourBarJointDefinition)
         return RobotDefinitionConverter.toInvertedFourBarJointDescription((InvertedFourBarJointDefinition) source);
      return null;
   }

   public static FloatingJointDescription toFloatingJointDescription(SixDoFJointDefinition source)
   {
      FloatingJointDescription output = new FloatingJointDescription(source.getName());
      RobotDefinitionConverter.copyJointProperties(source, output);
      return output;
   }

   public static PinJointDescription toPinJointDescription(RevoluteJointDefinition source)
   {
      PinJointDescription output = new PinJointDescription(source.getName(), source.getTransformToParent().getTranslation(), source.getAxis());
      RobotDefinitionConverter.copyOneDoFJointProperties(source, output);
      return output;
   }

   public static SliderJointDescription toSliderJointDescription(PrismaticJointDefinition source)
   {
      SliderJointDescription output = new SliderJointDescription(source.getName(), source.getTransformToParent().getTranslation(), source.getAxis());
      RobotDefinitionConverter.copyOneDoFJointProperties(source, output);
      return output;
   }

   public static InvertedFourBarJointDescription toInvertedFourBarJointDescription(InvertedFourBarJointDefinition source)
   {
      Map<String, JointDefinition> nameToJointDefinitionMap = Arrays.stream(source.getFourBarJoints())
                                                                    .collect(Collectors.toMap(JointDefinition::getName, Function.identity()));

      InvertedFourBarJointDescription output = new InvertedFourBarJointDescription(source.getName());

      RevoluteJointDefinition loopClosureJointDefinition = Arrays.stream(source.getFourBarJoints()).filter(j -> j.isLoopClosure()).findFirst().get();
      LoopClosureDefinition loopClosureInfo = loopClosureJointDefinition.getLoopClosureDefinition();
      MutableObject<LinkDescription> loopClosureLink = new MutableObject<>();

      output.setFourBarJoints(Arrays.stream(source.getFourBarJoints()).filter(j -> !j.isLoopClosure()).map(j ->
      {
         PinJointDescription jointDescription = toPinJointDescription(j);
         LinkDescription linkDescription = toLinkDescription(j.getSuccessor());
         if (linkDescription.getName().equals(loopClosureJointDefinition.getSuccessor().getName()))
            loopClosureLink.setValue(linkDescription);
         jointDescription.setLink(linkDescription);
         return jointDescription;
      }).toArray(PinJointDescription[]::new));

      String name = loopClosureJointDefinition.getName();
      Vector3D offsetFromParentJoint = loopClosureJointDefinition.getTransformToParent().getTranslation();
      Vector3D offsetFromLinkParentJoint = loopClosureInfo.getTransformToSuccessorParent().getTranslation();
      Vector3D axis = loopClosureJointDefinition.getAxis();
      LoopClosurePinConstraintDescription fourBarClosure = new LoopClosurePinConstraintDescription(name,
                                                                                                   offsetFromParentJoint,
                                                                                                   offsetFromLinkParentJoint,
                                                                                                   axis);
      fourBarClosure.setGains(loopClosureInfo.getKpSoftConstraint(), loopClosureInfo.getKdSoftConstraint());
      fourBarClosure.setLink(loopClosureLink.getValue());
      output.setFourBarClosure(fourBarClosure);

      Map<String, JointDescription> nameToJointDescriptionMap = Arrays.stream(output.getFourBarJoints())
                                                                      .collect(Collectors.toMap(JointDescription::getName, Function.identity()));

      // Connect joints
      for (JointDescription jointDescription : output.getFourBarJoints())
      {
         String parentJointName = nameToJointDefinitionMap.get(jointDescription.getName()).getParentJoint().getName();
         if (!parentJointName.equals(source.getParentJoint().getName())) // Can't connect these yet
            nameToJointDescriptionMap.get(parentJointName).addJoint(jointDescription);
      }

      { // Loop closure
         String parentJointName = loopClosureJointDefinition.getParentJoint().getName();
         nameToJointDescriptionMap.get(parentJointName).addConstraint(fourBarClosure);
      }

      return output;
   }

   private static void copyOneDoFJointProperties(OneDoFJointDefinition source, OneDoFJointDescription destination)
   {
      RobotDefinitionConverter.copyJointProperties(source, destination);
      destination.setLimitStops(source.getPositionLowerLimit(), source.getPositionUpperLimit(), source.getKpSoftLimitStop(), source.getKdSoftLimitStop());
      destination.setVelocityLimits(source.getVelocityUpperLimit(), Double.NaN); // TODO Should maybe add this parameter to the definition
      destination.setEffortLimit(source.getEffortUpperLimit());
      destination.setStiction(source.getStiction());
      destination.setDamping(source.getDamping());
   }

   private static void copyJointProperties(JointDefinition source, JointDescription destination)
   {
      if (!source.getTransformToParent().getRotation().isZeroOrientation())
         LogTools.warn("Ignoring non-zero rotation for transform of joint: {}.", source.getName());
      destination.setOffsetFromParentJoint(source.getTransformToParent().getTranslation());
      source.getSensorDefinitions(IMUSensorDefinition.class).forEach(imu -> destination.addIMUSensor(RobotDefinitionConverter.toIMUSensorDescription(imu)));
      source.getSensorDefinitions(WrenchSensorDefinition.class)
            .forEach(forceSensor -> destination.addForceSensor(RobotDefinitionConverter.toForceSensorDescription(forceSensor)));
      source.getSensorDefinitions(LidarSensorDefinition.class)
            .forEach(lidar -> destination.addLidarSensor(RobotDefinitionConverter.toLidarSensorDescription(lidar)));
      source.getSensorDefinitions(CameraSensorDefinition.class)
            .forEach(camera -> destination.addCameraSensor(RobotDefinitionConverter.toCameraSensorDescription(camera)));

      source.getKinematicPointDefinitions().forEach(kp -> destination.addKinematicPoint(RobotDefinitionConverter.toKinematicPointDescription(kp)));
      source.getExternalWrenchPointDefinitions()
            .forEach(efp -> destination.addExternalForcePoint(RobotDefinitionConverter.toExternalForcePointDescription(efp)));
      source.getGroundContactPointDefinitions()
            .forEach(gcp -> destination.addGroundContactPoint(RobotDefinitionConverter.toGroundContactPointDescription(gcp)));

      // TODO This should probably also live in the definition
      destination.setIsDynamic(true);
   }

   public static IMUSensorDescription toIMUSensorDescription(IMUSensorDefinition source)
   {
      IMUSensorDescription output = new IMUSensorDescription(source.getName(), new RigidBodyTransform(source.getTransformToJoint()));
      output.setAccelerationNoiseMean(source.getAccelerationNoiseMean());
      output.setAccelerationNoiseStandardDeviation(source.getAccelerationNoiseStandardDeviation());
      output.setAccelerationBiasMean(source.getAccelerationBiasMean());
      output.setAccelerationBiasStandardDeviation(source.getAccelerationBiasStandardDeviation());

      output.setAngularVelocityNoiseMean(source.getAngularVelocityNoiseMean());
      output.setAngularVelocityNoiseStandardDeviation(source.getAngularVelocityNoiseStandardDeviation());
      output.setAngularVelocityBiasMean(source.getAngularVelocityBiasMean());
      output.setAngularVelocityBiasStandardDeviation(source.getAngularVelocityBiasStandardDeviation());
      return output;
   }

   public static ForceSensorDescription toForceSensorDescription(WrenchSensorDefinition source)
   {
      ForceSensorDescription output = new ForceSensorDescription(source.getName(), new RigidBodyTransform(source.getTransformToJoint()));
      output.setUseGroundContactPoints(true);
      output.setUseShapeCollision(false);
      return output;
   }

   public static LidarSensorDescription toLidarSensorDescription(LidarSensorDefinition source)
   {
      LidarSensorDescription output = new LidarSensorDescription(source.getName(), source.getTransformToJoint());
      output.setSweepYawMin(source.getSweepYawMin());
      output.setSweepYawMax(source.getSweepYawMax());
      output.setHeightPitchMin(source.getHeightPitchMin());
      output.setHeightPitchMax(source.getHeightPitchMax());
      output.setMinRange(source.getMinRange());
      output.setMaxRange(source.getMaxRange());
      output.setPointsPerSweep(source.getPointsPerSweep());
      output.setScanHeight(source.getScanHeight());
      return output;
   }

   public static CameraSensorDescription toCameraSensorDescription(CameraSensorDefinition source)
   {
      CameraSensorDescription output = new CameraSensorDescription(source.getName(), source.getTransformToJoint());
      // TODO Check if the transform has to be modified as in toCameraSensorDefinition(...)
      output.setFieldOfView(source.getFieldOfView());
      output.setClipNear(source.getClipNear());
      output.setClipFar(source.getClipFar());
      output.setImageWidth(source.getImageWidth());
      output.setImageHeight(source.getImageHeight());
      return output;
   }

   public static KinematicPointDescription toKinematicPointDescription(KinematicPointDefinition source)
   {
      return new KinematicPointDescription(source.getName(), source.getTransformToParent().getTranslation());
   }

   public static ExternalForcePointDescription toExternalForcePointDescription(ExternalWrenchPointDefinition source)
   {
      return new ExternalForcePointDescription(source.getName(), source.getTransformToParent().getTranslation());
   }

   public static GroundContactPointDescription toGroundContactPointDescription(GroundContactPointDefinition source)
   {
      return new GroundContactPointDescription(source.getName(), new Vector3D(source.getTransformToParent().getTranslation()), source.getGroupIdentifier());
   }

   public static LinkGraphicsDescription toLinkGraphicsDescription(Collection<? extends VisualDefinition> source)
   {
      LinkGraphicsDescription output = new LinkGraphicsDescription();
      output.combine(VisualDefinitionConverter.toGraphics3DObject(source));
      return output;
   }

   public static Graphics3DObject toGraphics3DObject(Collection<? extends VisualDefinition> source)
   {
      return VisualDefinitionConverter.toGraphics3DObject(source);
   }

   public static Graphics3DObject toGraphics3DObject(VisualDefinition source)
   {
      return VisualDefinitionConverter.toGraphics3DObject(source);
   }

   public static List<Graphics3DPrimitiveInstruction> toGraphics3DPrimitiveInstruction(GeometryDefinition source)
   {
      return VisualDefinitionConverter.toGraphics3DPrimitiveInstruction(source);
   }

   public static AppearanceDefinition toAppearanceDefinition(MaterialDefinition source)
   {
      return VisualDefinitionConverter.toAppearanceDefinition(source);
   }

   public static List<CollisionMeshDescription> toCollisionMeshDescriptions(Collection<? extends CollisionShapeDefinition> source)
   {
      return source.stream().map(RobotDefinitionConverter::toCollisionMeshDescription).filter(Objects::nonNull).collect(Collectors.toList());
   }

   public static CollisionMeshDescription toCollisionMeshDescription(CollisionShapeDefinition source)
   {
      return null; // TODO implement me
   }
}
