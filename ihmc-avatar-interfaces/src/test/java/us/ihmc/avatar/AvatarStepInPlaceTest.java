package us.ihmc.avatar;

import static us.ihmc.robotics.Assert.assertFalse;
import static us.ihmc.robotics.Assert.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import controller_msgs.msg.dds.FootstepDataListMessage;
import controller_msgs.msg.dds.FootstepDataMessage;
import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.avatar.initialSetup.OffsetAndYawRobotInitialSetup;
import us.ihmc.avatar.testTools.DRCSimulationTestHelper;
import us.ihmc.commonWalkingControlModules.controlModules.foot.FootControlModule.ConstraintType;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.highLevelStates.WalkingHighLevelHumanoidController;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.highLevelStates.walkingController.states.WalkingStateEnum;
import us.ihmc.commons.PrintTools;
import us.ihmc.commons.thread.ThreadTools;
import us.ihmc.euclid.referenceFrame.FramePoint3D;
import us.ihmc.euclid.referenceFrame.FrameQuaternion;
import us.ihmc.euclid.referenceFrame.FrameVector3D;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.Vector3D;
import us.ihmc.humanoidRobotics.communication.packets.HumanoidMessageTools;
import us.ihmc.mecano.frames.MovingReferenceFrame;
import us.ihmc.robotModels.FullHumanoidRobotModel;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.robotics.robotSide.SideDependentList;
import us.ihmc.robotics.stateMachine.core.StateTransitionCondition;
import us.ihmc.simulationConstructionSetTools.util.environments.FlatGroundEnvironment;
import us.ihmc.simulationToolkit.controllers.PushRobotController;
import us.ihmc.simulationconstructionset.SimulationConstructionSet;
import us.ihmc.simulationconstructionset.util.simulationRunner.BlockingSimulationRunner.SimulationExceededMaximumTimeException;
import us.ihmc.simulationconstructionset.util.simulationTesting.SimulationTestingParameters;
import us.ihmc.tools.MemoryTools;
import us.ihmc.yoVariables.variable.YoBoolean;
import us.ihmc.yoVariables.variable.YoEnum;

public abstract class AvatarStepInPlaceTest implements MultiRobotTestInterface
{
   private SimulationTestingParameters simulationTestingParameters = SimulationTestingParameters.createFromSystemProperties();
   private DRCSimulationTestHelper drcSimulationTestHelper;
   private DRCRobotModel robotModel;
   private FullHumanoidRobotModel fullRobotModel;

   private PushRobotController pushRobotController;
   private SideDependentList<StateTransitionCondition> singleSupportStartConditions = new SideDependentList<>();

   protected int getNumberOfSteps()
   {
      return 1;
   }

   protected double getStepLength()
   {
      return 0.0;
   }

   protected double getStepWidth()
   {
      return 0.08;
   }

   protected double getForcePointOffsetZInChestFrame()
   {
      return 0.3;
   }

   protected OffsetAndYawRobotInitialSetup getStartingLocation()
   {
      return new OffsetAndYawRobotInitialSetup();
   }

   @BeforeEach
   public void setup()
   {
      MemoryTools.printCurrentMemoryUsageAndReturnUsedMemoryInMB(getClass().getSimpleName() + " before test.");

      FlatGroundEnvironment flatGround = new FlatGroundEnvironment();
      String className = getClass().getSimpleName();

      robotModel = getRobotModel();
      fullRobotModel = robotModel.createFullRobotModel();

      PrintTools.debug("simulationTestingParameters.getKeepSCSUp " + simulationTestingParameters.getKeepSCSUp());
      drcSimulationTestHelper = new DRCSimulationTestHelper(simulationTestingParameters, getRobotModel());
      drcSimulationTestHelper.setTestEnvironment(flatGround);
      drcSimulationTestHelper.setStartingLocation(getStartingLocation());
      drcSimulationTestHelper.createSimulation(className);

      double z = getForcePointOffsetZInChestFrame();

      pushRobotController = new PushRobotController(drcSimulationTestHelper.getRobot(),
                                                    fullRobotModel.getPelvis().getParentJoint().getName(),
                                                    new Vector3D(0, 0, z));
      SimulationConstructionSet scs = drcSimulationTestHelper.getSimulationConstructionSet();
      scs.addYoGraphic(pushRobotController.getForceVisualizer());

      for (RobotSide robotSide : RobotSide.values)
      {
         String sidePrefix = robotSide.getCamelCaseNameForStartOfExpression();
         @SuppressWarnings("unchecked")
         final YoEnum<ConstraintType> footConstraintType = (YoEnum<ConstraintType>) scs.findVariable(sidePrefix + "FootControlModule",
                                                                                                    sidePrefix + "FootCurrentState");
         @SuppressWarnings("unchecked")
         final YoEnum<WalkingStateEnum> walkingState = (YoEnum<WalkingStateEnum>) scs.findVariable(WalkingHighLevelHumanoidController.class.getSimpleName(),
                                                                                                  "walkingCurrentState");
         singleSupportStartConditions.put(robotSide, new SingleSupportStartCondition(footConstraintType));
      }
   }

   @AfterEach
   public void tearDown()
   {
      if (simulationTestingParameters.getKeepSCSUp())
      {
         ThreadTools.sleepForever();
      }

      // Do this here in case a test fails. That way the memory will be recycled.
      if (drcSimulationTestHelper != null)
      {
         drcSimulationTestHelper.destroySimulation();
         drcSimulationTestHelper = null;
      }

      simulationTestingParameters = null;
      MemoryTools.printCurrentMemoryUsageAndReturnUsedMemoryInMB(getClass().getSimpleName() + " after test.");
   }

   @Test
   public void testStepInPlace() throws SimulationExceededMaximumTimeException
   {
      setupCameraSideView();

      drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(0.5);

      FootstepDataListMessage footMessage = new FootstepDataListMessage();

      MovingReferenceFrame stepFrame = drcSimulationTestHelper.getControllerFullRobotModel().getSoleFrame(RobotSide.LEFT);

      FramePoint3D footLocation = new FramePoint3D(stepFrame);
      FrameQuaternion footOrientation = new FrameQuaternion(stepFrame);
      footLocation.changeFrame(ReferenceFrame.getWorldFrame());
      footOrientation.changeFrame(ReferenceFrame.getWorldFrame());

      footMessage.getFootstepDataList().add().set(HumanoidMessageTools.createFootstepDataMessage(RobotSide.LEFT, footLocation, footOrientation));
      footMessage.setAreFootstepsAdjustable(true);

      double initialTransfer = robotModel.getWalkingControllerParameters().getDefaultInitialTransferTime();
      double transfer = robotModel.getWalkingControllerParameters().getDefaultTransferTime();
      double swing = robotModel.getWalkingControllerParameters().getDefaultSwingTime();
      int steps = footMessage.getFootstepDataList().size();

      drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(1.0);
      drcSimulationTestHelper.publishToController(footMessage);
      double simulationTime = initialTransfer + (transfer + swing) * steps + 1.0;

      assertTrue(drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(simulationTime));
   }

   @Test
   public void testStepInPlaceWithPush() throws SimulationExceededMaximumTimeException
   {
      setupCameraSideView();

      drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(0.5);

      FootstepDataListMessage footMessage = new FootstepDataListMessage();

      MovingReferenceFrame stepFrame = drcSimulationTestHelper.getControllerFullRobotModel().getSoleFrame(RobotSide.LEFT);

      FramePoint3D footLocation = new FramePoint3D(stepFrame);
      FrameQuaternion footOrientation = new FrameQuaternion(stepFrame);
      footLocation.changeFrame(ReferenceFrame.getWorldFrame());
      footOrientation.changeFrame(ReferenceFrame.getWorldFrame());

      FootstepDataMessage footstep = HumanoidMessageTools.createFootstepDataMessage(RobotSide.LEFT, footLocation, footOrientation);

      footMessage.getFootstepDataList().add().set(footstep);
      footMessage.setAreFootstepsAdjustable(true);

      double initialTransfer = robotModel.getWalkingControllerParameters().getDefaultInitialTransferTime();
      double transfer = robotModel.getWalkingControllerParameters().getDefaultTransferTime();
      double swing = robotModel.getWalkingControllerParameters().getDefaultSwingTime();
      int steps = footMessage.getFootstepDataList().size();

      drcSimulationTestHelper.publishToController(footMessage);
      double simulationTime = initialTransfer + (transfer + swing) * steps + 1.0;

      FrameVector3D forceDirection = new FrameVector3D(stepFrame, new Vector3D(0.0, 1.0, 0.0));
      forceDirection.changeFrame(ReferenceFrame.getWorldFrame());

      double icpErrorDeadband = robotModel.getWalkingControllerParameters().getICPOptimizationParameters().getMinICPErrorForStepAdjustment();
      double desiredICPError = icpErrorDeadband * 0.7;
      double omega = robotModel.getWalkingControllerParameters().getOmega0();
      double desiredVelocityError = desiredICPError * omega;
      double mass = fullRobotModel.getTotalMass();
      double pushDuration = 0.05;

      double magnitude = getPushForceScaler() * mass * desiredVelocityError / pushDuration;
      PrintTools.info("Push 1 magnitude = " + magnitude);
      pushRobotController.applyForceDelayed(singleSupportStartConditions.get(RobotSide.LEFT), 0.0, forceDirection, magnitude, pushDuration);

      YoBoolean leftFootstepWasAdjusted = (YoBoolean) drcSimulationTestHelper.getYoVariable("leftFootSwingFootstepWasAdjusted");

      double dt = 0.05;
      for (int i = 0; i < simulationTime / dt; i++)
      {
         assertTrue(drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(dt));
         assertFalse(leftFootstepWasAdjusted.getBooleanValue());
      }

      assertTrue(drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(0.5));

      drcSimulationTestHelper.publishToController(footMessage);

      desiredICPError = icpErrorDeadband * 1.15;
      desiredVelocityError = desiredICPError * omega;

      magnitude = getPushForceScaler() * mass * desiredVelocityError / pushDuration;
      PrintTools.info("Push 2 magnitude = " + magnitude);
      pushRobotController.applyForceDelayed(singleSupportStartConditions.get(RobotSide.LEFT), 0.0, forceDirection, magnitude, pushDuration);

      assertTrue(drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(initialTransfer));
      for (int i = 0; i < (simulationTime - initialTransfer) / dt; i++)
      {
         assertTrue(drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(dt));
         if (i > 2 && singleSupportStartConditions.get(RobotSide.LEFT).testCondition(Double.NaN))
            assertTrue("Footstep wasn't adjusted, when it should have been", leftFootstepWasAdjusted.getBooleanValue());
      }

      drcSimulationTestHelper.simulateAndBlockAndCatchExceptions(0.5);
   }

   /**
    * Overwrite this if the force required to trigger a step adjustment is higher. This can happen for
    * example is step adjustment phase in is used.
    *
    * @return multiplier for the push force.
    */
   public double getPushForceScaler()
   {
      return 1.0;
   }

   private void setupCameraSideView()
   {
      Point3D cameraFix = new Point3D(0.0, 0.0, 1.0);
      Point3D cameraPosition = new Point3D(0.0, 10.0, 1.0);
      drcSimulationTestHelper.setupCameraForUnitTest(cameraFix, cameraPosition);
   }

   private class SingleSupportStartCondition implements StateTransitionCondition
   {
      private final YoEnum<ConstraintType> footConstraintType;

      public SingleSupportStartCondition(YoEnum<ConstraintType> footConstraintType)
      {
         this.footConstraintType = footConstraintType;
      }

      @Override
      public boolean testCondition(double time)
      {
         return footConstraintType.getEnumValue() == ConstraintType.SWING;
      }
   }
}
