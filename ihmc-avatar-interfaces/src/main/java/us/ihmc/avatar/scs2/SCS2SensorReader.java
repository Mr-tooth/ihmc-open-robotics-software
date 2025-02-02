package us.ihmc.avatar.scs2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ejml.data.DMatrixRMaj;

import us.ihmc.commons.Conversions;
import us.ihmc.euclid.referenceFrame.tools.ReferenceFrameTools;
import us.ihmc.mecano.multiBodySystem.CrossFourBarJoint;
import us.ihmc.mecano.multiBodySystem.interfaces.CrossFourBarJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.FloatingJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.JointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.JointReadOnly;
import us.ihmc.mecano.multiBodySystem.interfaces.OneDoFJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.OneDoFJointReadOnly;
import us.ihmc.mecano.multiBodySystem.interfaces.RigidBodyBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.SixDoFJointBasics;
import us.ihmc.mecano.multiBodySystem.interfaces.SixDoFJointReadOnly;
import us.ihmc.robotics.sensors.ForceSensorData;
import us.ihmc.robotics.sensors.ForceSensorDataHolder;
import us.ihmc.robotics.sensors.ForceSensorDataHolderReadOnly;
import us.ihmc.robotics.sensors.ForceSensorDefinition;
import us.ihmc.robotics.sensors.IMUDefinition;
import us.ihmc.scs2.simulation.robot.controller.SimControllerInput;
import us.ihmc.scs2.simulation.robot.multiBodySystem.interfaces.SimJointReadOnly;
import us.ihmc.scs2.simulation.robot.multiBodySystem.interfaces.SimRigidBodyReadOnly;
import us.ihmc.scs2.simulation.robot.sensors.SimIMUSensor;
import us.ihmc.scs2.simulation.robot.sensors.SimWrenchSensor;
import us.ihmc.sensorProcessing.imu.IMUSensor;
import us.ihmc.sensorProcessing.sensorProcessors.OneDoFJointStateReadOnly;
import us.ihmc.sensorProcessing.sensorProcessors.SensorOutputMapReadOnly;
import us.ihmc.sensorProcessing.sensorProcessors.SensorProcessing;
import us.ihmc.sensorProcessing.simulatedSensors.SensorDataContext;
import us.ihmc.sensorProcessing.simulatedSensors.SensorReader;
import us.ihmc.sensorProcessing.simulatedSensors.StateEstimatorSensorDefinitions;
import us.ihmc.sensorProcessing.stateEstimation.IMUSensorReadOnly;
import us.ihmc.sensorProcessing.stateEstimation.SensorProcessingConfiguration;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoLong;

public class SCS2SensorReader implements SensorReader
{
   private final StateEstimatorSensorDefinitions stateEstimatorSensorDefinitions = new StateEstimatorSensorDefinitions();

   private final SimControllerInput controllerInput;

   private final List<SimIMUSensor> simIMUSensors = new ArrayList<>();
   private final List<IMUSensor> controllerIMUSensors;
   private final List<IMUDefinition> imuSensorDefinitions = stateEstimatorSensorDefinitions.getIMUSensorDefinitions();

   private final List<SimWrenchSensor> simWrenchSensors = new ArrayList<>();
   private final List<ForceSensorData> controllerWrenchSensors;
   private final List<ForceSensorDefinition> wrenchSensorDefinitions = stateEstimatorSensorDefinitions.getForceSensorDefinitions();
   private ForceSensorDataHolder forceSensorOutputs = null;

   private final List<JointSensorReader> jointSensorReaders = new ArrayList<>();
   private final List<OneDoFJointStateReadOnly> jointSensorOutputList = new ArrayList<>();
   private final Map<OneDoFJointBasics, OneDoFJointStateReadOnly> jointToSensorOutputMap = new HashMap<>();

   private final YoRegistry registry = new YoRegistry(getClass().getSimpleName());
   private final YoLong timestamp;
   private final YoLong controllerTimestamp;
   private final YoLong sensorHeadPPSTimetamp;

   private final RigidBodyBasics rootBody;
   private final List<JointBasics> controllerJointList;
   private final List<JointReadOnly> simJointList;

   private final SensorProcessing sensorProcessing;
   private final SensorOutputMapReadOnly perfectSensorOutputMap;

   private final boolean usePerfectSensor;

   public static SCS2SensorReader newSensorReader(SimControllerInput controllerInput,
                                                  FloatingJointBasics rootJoint,
                                                  IMUDefinition[] imuDefinitions,
                                                  ForceSensorDataHolder forceSensorDataHolder,
                                                  SensorProcessingConfiguration sensorProcessingConfiguration)
   {
      return new SCS2SensorReader(controllerInput, rootJoint, imuDefinitions, forceSensorDataHolder, sensorProcessingConfiguration, false);
   }

   public static SCS2SensorReader newPerfectSensorReader(SimControllerInput controllerInput,
                                                         FloatingJointBasics rootJoint,
                                                         ForceSensorDataHolder forceSensorDataHolderToUpdate)
   {
      return new SCS2SensorReader(controllerInput, rootJoint, null, forceSensorDataHolderToUpdate, null, true);
   }

   private SCS2SensorReader(SimControllerInput controllerInput,
                            FloatingJointBasics rootJoint,
                            IMUDefinition[] imuDefinitions,
                            ForceSensorDataHolder forceSensorDataHolder,
                            SensorProcessingConfiguration sensorProcessingConfiguration,
                            boolean usePerfectSensor)
   {
      this.controllerInput = controllerInput;
      this.usePerfectSensor = usePerfectSensor;

      rootBody = rootJoint.getPredecessor();
      simJointList = new ArrayList<>(controllerInput.getInput().getAllJoints());
      controllerJointList = new ArrayList<>(rootJoint.subtreeList());

      for (JointBasics joint : rootJoint.subtreeIterable())
      {
         if (joint instanceof OneDoFJointBasics)
         {
            OneDoFJointBasics oneDoFJoint = (OneDoFJointBasics) joint;
            OneDoFJointStateReadOnly jointSensorOutput = OneDoFJointStateReadOnly.createFromOneDoFJoint(oneDoFJoint, true);
            jointSensorOutputList.add(jointSensorOutput);
            jointToSensorOutputMap.put(oneDoFJoint, jointSensorOutput);
            stateEstimatorSensorDefinitions.addJointSensorDefinition(oneDoFJoint);
         }
      }

      if (usePerfectSensor)
      {
         sensorProcessing = null;

         controllerIMUSensors = new ArrayList<>();
         controllerWrenchSensors = new ArrayList<>();
         timestamp = new YoLong("timestamp", registry);
         controllerTimestamp = new YoLong("controllerTimestamp", registry);
         sensorHeadPPSTimetamp = new YoLong("sensorHeadPPSTimetamp", registry);
         addWrenchSensors(forceSensorDataHolder);
         perfectSensorOutputMap = new SensorOutputMapReadOnly()
         {
            @Override
            public long getWallTime()
            {
               return timestamp.getLongValue();
            }

            @Override
            public long getMonotonicTime()
            {
               return controllerTimestamp.getLongValue();
            }

            @Override
            public long getSyncTimestamp()
            {
               return sensorHeadPPSTimetamp.getLongValue();
            }

            @Override
            public OneDoFJointStateReadOnly getOneDoFJointOutput(OneDoFJointBasics oneDoFJoint)
            {
               return jointToSensorOutputMap.get(oneDoFJoint);
            }

            @Override
            public List<? extends OneDoFJointStateReadOnly> getOneDoFJointOutputs()
            {
               return jointSensorOutputList;
            }

            @Override
            public List<? extends IMUSensorReadOnly> getIMUOutputs()
            {
               return controllerIMUSensors;
            }

            @Override
            public ForceSensorDataHolderReadOnly getForceSensorOutputs()
            {
               return forceSensorOutputs;
            }
         };
      }
      else
      {
         for (IMUDefinition imuDefinition : imuDefinitions)
            addIMUSensor(imuDefinition);
         addWrenchSensors(forceSensorDataHolder);
         sensorProcessing = new SensorProcessing(stateEstimatorSensorDefinitions, sensorProcessingConfiguration, registry);
         simJointList.remove(0);
         controllerJointList.remove(0);
         controllerIMUSensors = null;
         controllerWrenchSensors = null;
         timestamp = null;
         controllerTimestamp = null;
         sensorHeadPPSTimetamp = null;
         perfectSensorOutputMap = null;
      }

      for (JointBasics controllerJoint : controllerJointList)
      {
         SimJointReadOnly simJoint = controllerInput.getInput().findJoint(controllerJoint.getName());

         if (controllerJoint instanceof SixDoFJointBasics)
         {
            if (usePerfectSensor)
            {
               jointSensorReaders.add(new SixDoFJointSensorReader((SixDoFJointBasics) controllerJoint, (SixDoFJointReadOnly) simJoint));
            }
         }
         else if (controllerJoint instanceof CrossFourBarJointBasics)
         {
            CrossFourBarJointBasics controllerCrossFourBarJoint = (CrossFourBarJointBasics) controllerJoint;

            if (controllerInput.getInput().findJoint(controllerCrossFourBarJoint.getName()) != null)
            {
               jointSensorReaders.add(new OneDoFJointSensorReader((OneDoFJointBasics) controllerJoint, (OneDoFJointReadOnly) simJoint, sensorProcessing));
            }
            else
            {
               OneDoFJointReadOnly[] simJoints = new OneDoFJointReadOnly[4];
               simJoints[0] = (OneDoFJointReadOnly) controllerInput.getInput().findJoint(controllerCrossFourBarJoint.getJointA().getName());
               simJoints[1] = (OneDoFJointReadOnly) controllerInput.getInput().findJoint(controllerCrossFourBarJoint.getJointB().getName());
               simJoints[2] = (OneDoFJointReadOnly) controllerInput.getInput().findJoint(controllerCrossFourBarJoint.getJointC().getName());
               simJoints[3] = (OneDoFJointReadOnly) controllerInput.getInput().findJoint(controllerCrossFourBarJoint.getJointD().getName());
               jointSensorReaders.add(new CrossFourBarJointSensorReader(controllerCrossFourBarJoint, simJoints, sensorProcessing));
            }
         }
         else if (controllerJoint instanceof OneDoFJointBasics)
         {
            jointSensorReaders.add(new OneDoFJointSensorReader((OneDoFJointBasics) controllerJoint, (OneDoFJointReadOnly) simJoint, sensorProcessing));
         }
      }
   }

   private void addIMUSensor(IMUDefinition definition)
   {
      SimRigidBodyReadOnly rigidBody = controllerInput.getInput().findRigidBody(definition.getRigidBody().getName());
      List<SimIMUSensor> sensorList = rigidBody.getParentJoint().getAuxialiryData().getIMUSensors();
      try
      {
         simIMUSensors.add(sensorList.stream().filter(candidate -> candidate.getName().equals(definition.getName())).findFirst().get());
         if (usePerfectSensor)
            controllerIMUSensors.add(new IMUSensor(definition, null));
         stateEstimatorSensorDefinitions.addIMUSensorDefinition(definition);
      }
      catch (Exception e)
      {
         throw new RuntimeException("Troublesome sensor: " + definition.getName() + ", sensor list: " + sensorList, e);
      }
   }

   private void addWrenchSensors(ForceSensorDataHolder forceSensorDataHolderToUpdate)
   {
      for (ForceSensorDefinition definition : forceSensorDataHolderToUpdate.getForceSensorDefinitions())
      {
         addWrenchSensor(definition, forceSensorDataHolderToUpdate.get(definition));
      }
      forceSensorOutputs = forceSensorDataHolderToUpdate;
   }

   private void addWrenchSensor(ForceSensorDefinition definition, ForceSensorData sensorDataToUpdate)
   {
      simWrenchSensors.add(controllerInput.getInput().findRigidBody(definition.getRigidBody().getName()).getParentJoint().getAuxialiryData().getWrenchSensors()
                                          .stream().filter(candidate -> candidate.getName().equals(definition.getSensorName())).findFirst().get());
      if (usePerfectSensor)
         controllerWrenchSensors.add(sensorDataToUpdate);
      stateEstimatorSensorDefinitions.addForceSensorDefinition(definition);
   }

   private final DMatrixRMaj wrenchMatrix = new DMatrixRMaj(6, 1);

   @Override
   public void initialize()
   {
      if (!usePerfectSensor)
         sensorProcessing.initialize();
   }

   @Override
   public long read(SensorDataContext sensorDataContext)
   {
      long timestamp = Conversions.secondsToNanoseconds(controllerInput.getTime());

      if (usePerfectSensor)
      {
         this.timestamp.set(timestamp);
         controllerTimestamp.set(timestamp);
         sensorHeadPPSTimetamp.set(timestamp);
      }

      for (int i = 0; i < jointSensorReaders.size(); i++)
      {
         jointSensorReaders.get(i).read();
      }

      if (usePerfectSensor)
      {
         rootBody.updateFramesRecursively();
         for (int i = 0; i < imuSensorDefinitions.size(); i++)
         {
            IMUSensor controllerSensor = controllerIMUSensors.get(i);
            SimIMUSensor simSensor = simIMUSensors.get(i);
            controllerSensor.setOrientationMeasurement(simSensor.getOrientationFiltered());
            controllerSensor.setAngularVelocityMeasurement(simSensor.getAngularVelocityFiltered());
            controllerSensor.setLinearAccelerationMeasurement(simSensor.getLinearAccelerationFiltered());
         }

         for (int i = 0; i < wrenchSensorDefinitions.size(); i++)
         {
            ForceSensorData controllerData = controllerWrenchSensors.get(i);
            SimWrenchSensor simWrench = simWrenchSensors.get(i);
            controllerData.setWrench(simWrench.getWrenchFiltered().getAngularPart(), simWrench.getWrenchFiltered().getLinearPart());
         }
      }
      else
      {
         for (int i = 0; i < imuSensorDefinitions.size(); i++)
         {
            IMUDefinition definition = imuSensorDefinitions.get(i);
            SimIMUSensor simSensor = simIMUSensors.get(i);
            sensorProcessing.setOrientationSensorValue(definition, simSensor.getOrientationFiltered());
            sensorProcessing.setAngularVelocitySensorValue(definition, simSensor.getAngularVelocityFiltered());
            sensorProcessing.setLinearAccelerationSensorValue(definition, simSensor.getLinearAccelerationFiltered());
         }

         for (int i = 0; i < wrenchSensorDefinitions.size(); i++)
         {
            ForceSensorDefinition definition = wrenchSensorDefinitions.get(i);
            SimWrenchSensor simWrench = simWrenchSensors.get(i);
            simWrench.getWrench().get(wrenchMatrix);
            sensorProcessing.setForceSensorValue(definition, wrenchMatrix);
         }

         sensorProcessing.startComputation(timestamp, timestamp, timestamp);
      }

      return timestamp;
   }

   public StateEstimatorSensorDefinitions getStateEstimatorSensorDefinitions()
   {
      return stateEstimatorSensorDefinitions;
   }

   @Override
   public SensorOutputMapReadOnly getProcessedSensorOutputMap()
   {
      return usePerfectSensor ? perfectSensorOutputMap : sensorProcessing;
   }

   @Override
   public SensorOutputMapReadOnly getRawSensorOutputMap()
   {
      return usePerfectSensor ? perfectSensorOutputMap : sensorProcessing.getRawSensorOutputMap();
   }

   public YoRegistry getRegistry()
   {
      return registry;
   }

   private static interface JointSensorReader
   {
      void read();
   }

   private static class SixDoFJointSensorReader implements JointSensorReader
   {
      private final SixDoFJointBasics controllerJoint;
      private final SixDoFJointReadOnly simJoint;

      public SixDoFJointSensorReader(SixDoFJointBasics controllerJoint, SixDoFJointReadOnly simJoint)
      {
         this.controllerJoint = controllerJoint;
         this.simJoint = simJoint;
      }

      @Override
      public void read()
      {
         controllerJoint.setJointConfiguration(simJoint);
         controllerJoint.setJointTwist(simJoint);
         controllerJoint.setJointAcceleration(simJoint);
         controllerJoint.setJointWrench(simJoint);
      }
   }

   private static class OneDoFJointSensorReader implements JointSensorReader
   {
      private final OneDoFJointBasics controllerJoint;
      private final OneDoFJointReadOnly simJoint;
      private final SensorProcessing sensorProcessing;

      public OneDoFJointSensorReader(OneDoFJointBasics controllerJoint, OneDoFJointReadOnly simJoint, SensorProcessing sensorProcessing)
      {
         this.controllerJoint = controllerJoint;
         this.simJoint = simJoint;
         this.sensorProcessing = sensorProcessing;
      }

      @Override
      public void read()
      {
         if (sensorProcessing == null)
         {
            controllerJoint.setJointConfiguration(simJoint);
            controllerJoint.setJointTwist(simJoint);
            controllerJoint.setJointAcceleration(simJoint);
            controllerJoint.setJointWrench(simJoint);
         }
         else
         {
            sensorProcessing.setJointPositionSensorValue(controllerJoint, simJoint.getQ());
            sensorProcessing.setJointVelocitySensorValue(controllerJoint, simJoint.getQd());
            sensorProcessing.setJointAccelerationSensorValue(controllerJoint, simJoint.getQdd());
            sensorProcessing.setJointTauSensorValue(controllerJoint, simJoint.getTau());
         }
      }
   }

   private static class CrossFourBarJointSensorReader implements JointSensorReader
   {
      private final CrossFourBarJointBasics controllerJoint;
      private final OneDoFJointReadOnly[] simJoints;
      private final SensorProcessing sensorProcessing;

      private final CrossFourBarJoint localFourBarJoint;
      private final int[] activeJointIndices;

      public CrossFourBarJointSensorReader(CrossFourBarJointBasics controllerJoint, OneDoFJointReadOnly[] simJoints, SensorProcessing sensorProcessing)
      {
         this.controllerJoint = controllerJoint;
         this.simJoints = simJoints;
         this.sensorProcessing = sensorProcessing;

         localFourBarJoint = CrossFourBarJoint.cloneCrossFourBarJoint(controllerJoint, ReferenceFrameTools.constructARootFrame("dummy"), "dummy");
         if (controllerJoint.getJointA().isLoopClosure() || controllerJoint.getJointD().isLoopClosure())
            activeJointIndices = new int[] {1, 2};
         else
            activeJointIndices = new int[] {0, 3};
      }

      @Override
      public void read()
      {
         double q = simJoints[activeJointIndices[0]].getQ() + simJoints[activeJointIndices[1]].getQ();
         double qd = simJoints[activeJointIndices[0]].getQd() + simJoints[activeJointIndices[1]].getQd();
         double qdd = 0.0; // TODO simJoints[activeJointIndices[0]].getQdd() + simJoints[activeJointIndices[1]].getQdd();
         localFourBarJoint.setQ(q);
         localFourBarJoint.setQd(qd);
         localFourBarJoint.setQdd(qdd);
         localFourBarJoint.updateFrame();
         DMatrixRMaj loopJacobian = localFourBarJoint.getFourBarFunction().getLoopJacobian();
         double tau = loopJacobian.get(activeJointIndices[0]) * simJoints[activeJointIndices[0]].getTau()
               + loopJacobian.get(activeJointIndices[1]) * simJoints[activeJointIndices[1]].getTau();
         int actuatedJointIndex = localFourBarJoint.getFourBarFunction().getActuatedJointIndex();

         if (actuatedJointIndex == 0 || actuatedJointIndex == 3)
            tau /= (loopJacobian.get(0) + loopJacobian.get(3));
         else
            tau /= (loopJacobian.get(1) + loopJacobian.get(2));

         if (sensorProcessing == null)
         {
            controllerJoint.setQ(q);
            controllerJoint.setQd(qd);
            controllerJoint.setQdd(qdd);
            controllerJoint.setTau(tau);
         }
         else
         {
            sensorProcessing.setJointPositionSensorValue(controllerJoint, q);
            sensorProcessing.setJointVelocitySensorValue(controllerJoint, qd);
            sensorProcessing.setJointAccelerationSensorValue(controllerJoint, qdd);
            sensorProcessing.setJointTauSensorValue(controllerJoint, tau);
         }
      }
   }
}
