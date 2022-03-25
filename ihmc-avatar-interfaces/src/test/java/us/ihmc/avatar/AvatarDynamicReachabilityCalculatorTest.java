package us.ihmc.avatar;

import static us.ihmc.robotics.Assert.assertTrue;

import java.io.InputStream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.avatar.testTools.scs2.SCS2AvatarTestingSimulation;
import us.ihmc.avatar.testTools.scs2.SCS2AvatarTestingSimulationFactory;
import us.ihmc.commonWalkingControlModules.controlModules.foot.FootControlModule.ConstraintType;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.highLevelStates.walkingController.states.WalkingStateEnum;
import us.ihmc.commons.thread.ThreadTools;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.robotModels.FullHumanoidRobotModel;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.simulationConstructionSetTools.bambooTools.BambooTools;
import us.ihmc.simulationConstructionSetTools.util.environments.FlatGroundEnvironment;
import us.ihmc.simulationconstructionset.util.simulationTesting.SimulationTestingParameters;
import us.ihmc.tools.MemoryTools;
import us.ihmc.yoVariables.variable.YoEnum;

public abstract class AvatarDynamicReachabilityCalculatorTest
{
   private static final SimulationTestingParameters simulationTestingParameters = SimulationTestingParameters.createFromSystemProperties();

   private SCS2AvatarTestingSimulation simulationTestHelper;

   private static String shortScript = "scripts/ExerciseAndJUnitScripts/dynamicReachabilityForwardShort.xml";
   private static String mediumScript = "scripts/ExerciseAndJUnitScripts/dynamicReachabilityForwardMedium.xml";
   private static String longScript = "scripts/ExerciseAndJUnitScripts/dynamicReachabilityForwardLong.xml";

   private static double simulationTime = 10.0;

   private double swingTime, transferTime;
   private double totalMass;

   @BeforeEach
   public void showMemoryUsageBeforeTest()
   {
      MemoryTools.printCurrentMemoryUsageAndReturnUsedMemoryInMB(getClass().getSimpleName() + " before test.");
      BambooTools.reportTestStartedMessage(simulationTestingParameters.getShowWindows());
   }

   @AfterEach
   public void destroySimulationAndRecycleMemory()
   {
      // Do this here in case a test fails. That way the memory will be recycled.
      if (simulationTestHelper != null)
      {
         simulationTestHelper.finishTest();
         simulationTestHelper = null;
      }

      MemoryTools.printCurrentMemoryUsageAndReturnUsedMemoryInMB(getClass().getSimpleName() + " after test.");
      BambooTools.reportTestFinishedMessage(simulationTestingParameters.getShowWindows());
   }

   protected abstract DRCRobotModel getRobotModel();

   @Disabled
   @Test
   public void testForwardWalkingShort()
   {
      setupTest(shortScript);

      boolean success = simulationTestHelper.simulateAndWait(simulationTime);

      assertTrue(success);
   }

   @Disabled
   @Test
   public void testForwardWalkingMedium()
   {
      setupTest(mediumScript);

      boolean success = simulationTestHelper.simulateAndWait(simulationTime);

      assertTrue(success);
   }

   @Disabled
   @Test
   public void testForwardWalkingLong()
   {
      setupTest(longScript);

      boolean success = simulationTestHelper.simulateAndWait(simulationTime);

      assertTrue(success);
   }

   private void setupTest(String scriptName)
   {
      this.setupTest(scriptName, ReferenceFrame.getWorldFrame());
   }

   private void setupTest(String scriptName, ReferenceFrame yawReferenceFrame)
   {
      FlatGroundEnvironment flatGround = new FlatGroundEnvironment();
      simulationTestHelper = SCS2AvatarTestingSimulationFactory.createDefaultTestSimulation(getRobotModel(), flatGround, simulationTestingParameters);
      simulationTestHelper.start();
      FullHumanoidRobotModel fullRobotModel = getRobotModel().createFullRobotModel();
      totalMass = fullRobotModel.getTotalMass();

      if (scriptName != null && !scriptName.isEmpty())
      {
         simulationTestHelper.simulateAndWait(0.001);
         InputStream scriptInputStream = getClass().getClassLoader().getResourceAsStream(scriptName);
         if (yawReferenceFrame != null)
         {
            simulationTestHelper.loadScriptFile(scriptInputStream, yawReferenceFrame);
         }
         else
         {
            simulationTestHelper.loadScriptFile(scriptInputStream, ReferenceFrame.getWorldFrame());
         }
      }

      for (RobotSide robotSide : RobotSide.values)
      {
         String sidePrefix = robotSide.getCamelCaseNameForStartOfExpression();
         String footPrefix = sidePrefix + "Foot";
         @SuppressWarnings("unchecked")
         final YoEnum<ConstraintType> footConstraintType = (YoEnum<ConstraintType>) simulationTestHelper.findVariable(sidePrefix + "FootControlModule",
                                                                                                                      footPrefix + "State");
         @SuppressWarnings("unchecked")
         final YoEnum<WalkingStateEnum> walkingState = (YoEnum<WalkingStateEnum>) simulationTestHelper.findVariable("WalkingHighLevelHumanoidController",
                                                                                                                    "walkingState");
      }

      setupCamera();
      swingTime = getRobotModel().getWalkingControllerParameters().getDefaultSwingTime();
      transferTime = getRobotModel().getWalkingControllerParameters().getDefaultTransferTime();
      ThreadTools.sleep(1000);
   }

   private void setupCamera()
   {
      Point3D cameraFix = new Point3D(0.0, 0.0, 0.89);
      Point3D cameraPosition = new Point3D(10.0, 2.0, 1.37);
      simulationTestHelper.setCamera(cameraFix, cameraPosition);
   }
}
