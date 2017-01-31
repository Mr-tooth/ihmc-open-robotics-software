package us.ihmc.atlas.StepAdjustmentVisualizers;

import us.ihmc.atlas.AtlasRobotModel;
import us.ihmc.atlas.AtlasRobotVersion;
import us.ihmc.atlas.parameters.AtlasWalkingControllerParameters;
import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.commonWalkingControlModules.configurations.WalkingControllerParameters;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.robotics.stateMachines.conditionBasedStateMachine.StateTransitionCondition;

import javax.vecmath.Vector3d;

public class AtlasStepAdjustmentDemo
{
   private static StepScriptType stepScriptType = StepScriptType.FORWARD_FAST;
   private static TestType testType = TestType.ADJUSTMENT_ONLY;
   private static PushDirection pushDirection = PushDirection.OUTWARD;

   private static String forwardFastScript = "scripts/stepAdjustment_forwardWalkingFast.xml";
   private static String forwardSlowScript = "scripts/stepAdjustment_forwardWalkingSlow.xml";
   private static String stationaryFastScript = "scripts/stepAdjustment_stationaryWalkingFast.xml";
   private static String stationarySlowScript = "scripts/stepAdjustment_stationaryWalkingSlow.xml";

   private static double simulationTime = 10.0;

   public AtlasStepAdjustmentDemo()
   {
   }

   public static void main(String[] args)
   {
      AtlasRobotModel robotModel = new AtlasRobotModel(AtlasRobotVersion.ATLAS_UNPLUGGED_V5_NO_HANDS, DRCRobotModel.RobotTarget.SCS, false)
      {
         @Override
         public WalkingControllerParameters getWalkingControllerParameters()
         {
            return new AtlasWalkingControllerParameters(RobotTarget.SCS, getJointMap())
            {
               @Override
               public boolean allowDisturbanceRecoveryBySpeedingUpSwing()
               {
                  switch (testType)
                  {
                  case FEEDBACK_ONLY:
                  case ADJUSTMENT_ONLY:
                     return false;
                  default:
                     return true;
                  }
               }

               @Override
               public boolean useOptimizationBasedICPController()
               {
                  switch (testType)
                  {
                  case FEEDBACK_ONLY:
                  case SPEED_UP_ONLY:
                     return false;
                  default:
                     return true;
                  }
               }

               @Override
               public double getMinimumSwingTimeForDisturbanceRecovery()
               {
                  switch(stepScriptType)
                  {
                  case FORWARD_FAST:
                     return 0.5;
                  default: // FORWARD_SLOW
                     return 0.6;
                  }
               }
            };
         }
      };

      String script;
      Vector3d forceDirection;
      double percentWeight;

      switch(pushDirection)
      {
      case FORWARD:
         forceDirection = new Vector3d(1.0, 0.0, 0.0);

         switch (stepScriptType)
         {
         case FORWARD_SLOW:
            script = forwardSlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.1;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.49;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.34;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.56;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.44;
               break;
            }
            break;
         case STATIONARY_FAST:
            script = stationaryFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.92;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 1.63;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.3;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.47;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.47;
               break;
            }
            break;
         case STATIONARY_SLOW:
            script = stationarySlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 0.75;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.55;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.32;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.48;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.47;
               break;
            }
            break;
         default: // FORWARD_FAST
            script = forwardFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.65;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 1.0;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.45;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.62;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.39;
               break;
            }
            break;
         }
         break;
      case BACKWARD:
         forceDirection = new Vector3d(-1.0, 0.0, 0.0);

         switch (stepScriptType)
         {
         case FORWARD_SLOW:
            script = forwardSlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 0.83;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.62;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.4;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.67;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.66;
               break;
            }
            break;
         case STATIONARY_FAST:
            script = stationaryFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.22;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 1.04;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.31;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.48;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.49;
               break;
            }
            break;
         case STATIONARY_SLOW:
            script = stationarySlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 0.71;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.5;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.3; //// TODO: 1/30/17
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.48;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.48;
               break;
            }
            break;
         default: // FORWARD_FAST
            script = forwardFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.85;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 1.7;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.25;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.87;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.87;
               break;
            }
            break;
         }
         break;
      default: // OUTWARD
         forceDirection = new Vector3d(0.0, -1.0, 0.0);

         switch (stepScriptType)
         {
         case FORWARD_SLOW:
            script = forwardSlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 0.98;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.27;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.2;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.47;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.25;
               break;
            }
            break;
         case STATIONARY_FAST:
            script = stationaryFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.08;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.86;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.16;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.33;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.26;
               break;
            }
            break;
         case STATIONARY_SLOW:
            script = stationarySlowScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 0.83;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.33;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.3; //// TODO: 1/30/17
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.52;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.25;
               break;
            }
            break;
         default: // FORWARD_FAST
            script = forwardFastScript;

            switch(testType)
            {
            case BIG_ADJUSTMENT:
               percentWeight = 1.41;
               break;
            case ADJUSTMENT_ONLY:
               percentWeight = 0.95;
               break;
            case NO_ADJUSTMENT:
               percentWeight = 0.2;
               break;
            case SPEED_UP_ONLY:
               percentWeight = 0.36;
               break;
            default: // doesn't allow speed up or step adjustment
               percentWeight = 0.22;
               break;
            }
            break;
         }
         break;
      }


      StepAdjustmentDemoHelper stepAdjustmentDemo = new StepAdjustmentDemoHelper(robotModel, script);


      // push parameters:

      double swingTime = stepAdjustmentDemo.getSwingTime();
      double delay = 0.5 * swingTime;
      double magnitude = percentWeight * stepAdjustmentDemo.getTotalMass() * 9.81;
      double duration = 0.1 * swingTime;

      // push timing:
      StateTransitionCondition pushCondition = stepAdjustmentDemo.getSingleSupportStartCondition(RobotSide.RIGHT);
      stepAdjustmentDemo.applyForceDelayed(pushCondition, delay, forceDirection, magnitude, duration);
      boolean success = stepAdjustmentDemo.simulateAndBlockAndCatchExceptions(simulationTime);
   }
}