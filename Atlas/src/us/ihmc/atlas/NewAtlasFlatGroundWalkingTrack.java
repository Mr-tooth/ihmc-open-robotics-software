package us.ihmc.atlas;

import com.martiansoftware.jsap.JSAPException;
import us.ihmc.atlas.highLevelHumanoidControl.NewAtlasMomentumControllerFactory;
import us.ihmc.avatar.NewDRCFlatGroundWalkingTrack;
import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.avatar.drcRobot.RobotTarget;
import us.ihmc.avatar.initialSetup.DRCGuiInitialSetup;
import us.ihmc.avatar.initialSetup.DRCRobotInitialSetup;
import us.ihmc.avatar.initialSetup.DRCSCSInitialSetup;
import us.ihmc.commonWalkingControlModules.configurations.ICPWithTimeFreezingPlannerParameters;
import us.ihmc.commonWalkingControlModules.configurations.WalkingControllerParameters;
import us.ihmc.commonWalkingControlModules.desiredHeadingAndVelocity.HeadingAndVelocityEvaluationScriptParameters;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.factories.AbstractMomentumBasedControllerFactory;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.factories.ContactableBodiesFactory;
import us.ihmc.commonWalkingControlModules.highLevelHumanoidControl.factories.WalkingProvider;
import us.ihmc.humanoidRobotics.communication.packets.dataobjects.NewHighLevelControllerStates;
import us.ihmc.jMonkeyEngineToolkit.GroundProfile3D;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.robotics.robotSide.SideDependentList;
import us.ihmc.simulationToolkit.controllers.OscillateFeetPerturber;
import us.ihmc.simulationconstructionset.HumanoidFloatingRootJointRobot;
import us.ihmc.simulationconstructionset.SimulationConstructionSet;
import us.ihmc.simulationconstructionset.util.ground.BumpyGroundProfile;
import us.ihmc.simulationconstructionset.util.ground.FlatGroundProfile;

public class NewAtlasFlatGroundWalkingTrack extends NewDRCFlatGroundWalkingTrack
{
   private static final DRCRobotModel defaultModelForGraphicSelector = new AtlasRobotModel(AtlasRobotVersion.ATLAS_UNPLUGGED_V5_NO_HANDS, RobotTarget.SCS, false);

   private static final boolean USE_BUMPY_GROUND = false;
   private static final boolean USE_FEET_PERTURBER = false;

   public NewAtlasFlatGroundWalkingTrack(DRCRobotInitialSetup<HumanoidFloatingRootJointRobot> robotInitialSetup, DRCGuiInitialSetup guiInitialSetup, DRCSCSInitialSetup scsInitialSetup,
                                       boolean useVelocityAndHeadingScript, boolean cheatWithGroundHeightAtForFootstep, DRCRobotModel model)
   {
      super(robotInitialSetup, guiInitialSetup, scsInitialSetup, useVelocityAndHeadingScript, cheatWithGroundHeightAtForFootstep, model,
           WalkingProvider.VELOCITY_HEADING_COMPONENT, new HeadingAndVelocityEvaluationScriptParameters()); // should always be committed as VELOCITY_HEADING_COMPONENT
   }

   public static void main(String[] args) throws JSAPException
   {

      DRCRobotModel model = null;
      model = AtlasRobotModelFactory.selectSimulationModelFromFlag(args);

      if (model == null)
         model = AtlasRobotModelFactory.selectModelFromGraphicSelector(defaultModelForGraphicSelector);

      if (model == null)
         throw new RuntimeException("No robot model selected");

      DRCGuiInitialSetup guiInitialSetup = new DRCGuiInitialSetup(true, false);

      final double groundHeight = 0.0;
      GroundProfile3D groundProfile;
      if (USE_BUMPY_GROUND)
      {
         groundProfile = createBumpyGroundProfile();
      }
      else
      {
         groundProfile = new FlatGroundProfile(groundHeight);
      }

      DRCSCSInitialSetup scsInitialSetup = new DRCSCSInitialSetup(groundProfile, model.getSimulateDT());
      scsInitialSetup.setDrawGroundProfile(true);
      scsInitialSetup.setInitializeEstimatorToActual(true);

      double initialYaw = 0.3;
      DRCRobotInitialSetup<HumanoidFloatingRootJointRobot> robotInitialSetup = model.getDefaultRobotInitialSetup(groundHeight, initialYaw);

      boolean useVelocityAndHeadingScript = true;
      boolean cheatWithGroundHeightAtForFootstep = false;

      NewAtlasFlatGroundWalkingTrack drcFlatGroundWalkingTrack = new NewAtlasFlatGroundWalkingTrack(robotInitialSetup, guiInitialSetup, scsInitialSetup,
                                                                                                useVelocityAndHeadingScript, cheatWithGroundHeightAtForFootstep, model);

      if (USE_FEET_PERTURBER)
         createOscillateFeetPerturber(drcFlatGroundWalkingTrack);
   }

   @Override
   public AbstractMomentumBasedControllerFactory getControllerFactory(ContactableBodiesFactory contactableBodiesFactory,
                                                                      SideDependentList<String> footForceSensorNames,
                                                                      SideDependentList<String> footContactSensorNames,
                                                                      SideDependentList<String> wristSensorNames,
                                                                      WalkingControllerParameters walkingControllerParameters,
                                                                      ICPWithTimeFreezingPlannerParameters capturePointPlannerParameters,
                                                                      NewHighLevelControllerStates initialControllerState,
                                                                      NewHighLevelControllerStates fallbackControllerState)
   {
      return new NewAtlasMomentumControllerFactory(contactableBodiesFactory, footForceSensorNames, footContactSensorNames, wristSensorNames,
                                                   walkingControllerParameters, capturePointPlannerParameters, initialControllerState, fallbackControllerState);
   }

   private static void createOscillateFeetPerturber(NewDRCFlatGroundWalkingTrack drcFlatGroundWalkingTrack)
   {
      SimulationConstructionSet simulationConstructionSet = drcFlatGroundWalkingTrack.getSimulationConstructionSet();
      HumanoidFloatingRootJointRobot robot = drcFlatGroundWalkingTrack.getAvatarSimulation().getHumanoidFloatingRootJointRobot();

      int ticksPerPerturbation = 10;
      OscillateFeetPerturber oscillateFeetPerturber = new OscillateFeetPerturber(robot, simulationConstructionSet.getDT() * ((double) ticksPerPerturbation));
      oscillateFeetPerturber.setTranslationMagnitude(new double[] { 0.01, 0.015, 0.005 });
      oscillateFeetPerturber.setRotationMagnitudeYawPitchRoll(new double[] { 0.017, 0.012, 0.011 });

      oscillateFeetPerturber.setTranslationFrequencyHz(RobotSide.LEFT, new double[] { 0.0, 0, 3.3 });
      oscillateFeetPerturber.setTranslationFrequencyHz(RobotSide.RIGHT, new double[] { 0.0, 0, 1.3 });

      oscillateFeetPerturber.setRotationFrequencyHzYawPitchRoll(RobotSide.LEFT, new double[] { 0.0, 0, 7.3 });
      oscillateFeetPerturber.setRotationFrequencyHzYawPitchRoll(RobotSide.RIGHT, new double[] { 0., 0, 1.11 });

      robot.setController(oscillateFeetPerturber, ticksPerPerturbation);
   }

   private static BumpyGroundProfile createBumpyGroundProfile()
   {
      double xAmp1 = 0.05, xFreq1 = 0.5, xAmp2 = 0.01, xFreq2 = 0.5;
      double yAmp1 = 0.01, yFreq1 = 0.07, yAmp2 = 0.05, yFreq2 = 0.37;
      BumpyGroundProfile groundProfile = new BumpyGroundProfile(xAmp1, xFreq1, xAmp2, xFreq2, yAmp1, yFreq1, yAmp2, yFreq2);
      return groundProfile;
   }
}
