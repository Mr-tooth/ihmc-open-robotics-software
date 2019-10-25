package us.ihmc.footstepPlanning.postProcessing;

import controller_msgs.msg.dds.FootstepDataMessage;
import controller_msgs.msg.dds.FootstepPlanningRequestPacket;
import controller_msgs.msg.dds.FootstepPlanningToolboxOutputStatus;
import us.ihmc.footstepPlanning.postProcessing.parameters.FootstepPostProcessingParametersReadOnly;
import us.ihmc.commonWalkingControlModules.configurations.ICPPlannerParameters;
import us.ihmc.commons.InterpolationTools;
import us.ihmc.euclid.referenceFrame.FramePose3D;

import java.util.List;

public class StepSplitFractionPostProcessingElement implements FootstepPlanPostProcessingElement
{
   private final FootstepPostProcessingParametersReadOnly parameters;
   private final ICPPlannerParameters walkingControllerParameters;

   public StepSplitFractionPostProcessingElement(FootstepPostProcessingParametersReadOnly parameters,
                                                 ICPPlannerParameters walkingControllerParameters)
   {
      this.parameters = parameters;
      this.walkingControllerParameters = walkingControllerParameters;
   }

   /** {@inheritDoc} **/
   @Override
   public boolean isActive()
   {
      return parameters.splitFractionProcessingEnabled();
   }

   /** {@inheritDoc} **/
   @Override
   public FootstepPlanningToolboxOutputStatus postProcessFootstepPlan(FootstepPlanningRequestPacket request, FootstepPlanningToolboxOutputStatus outputStatus)
   {
      FootstepPlanningToolboxOutputStatus processedOutput = new FootstepPlanningToolboxOutputStatus(outputStatus);

      FramePose3D stanceFootPose = new FramePose3D();
      stanceFootPose.setPosition(request.getStanceFootPositionInWorld());
      stanceFootPose.setOrientation(request.getStanceFootOrientationInWorld());

      FramePose3D nextFootPose = new FramePose3D();

      double defaultTransferSplitFraction = walkingControllerParameters.getTransferSplitFraction();
      double defaultWeightDistribution = 0.5;

      List<FootstepDataMessage> footstepDataMessageList = processedOutput.getFootstepDataList().getFootstepDataList();
      for (int stepNumber = 0; stepNumber < footstepDataMessageList.size(); stepNumber++)
      {
         if (stepNumber > 0)
         {
            stanceFootPose.setPosition(footstepDataMessageList.get(stepNumber - 1).getLocation());
            stanceFootPose.setOrientation(footstepDataMessageList.get(stepNumber - 1).getOrientation());
         }

         nextFootPose.setPosition(footstepDataMessageList.get(stepNumber).getLocation());
         nextFootPose.setOrientation(footstepDataMessageList.get(stepNumber).getOrientation());

         // This step is a big step down.
         double stepDownHeight = nextFootPose.getZ() - stanceFootPose.getZ();

         if (nextFootPose.getZ() - stanceFootPose.getZ() < -parameters.getStepHeightForLargeStepDown())
         {
            double alpha = Math.min(1.0, Math.abs(stepDownHeight) / parameters.getLargestStepDownHeight());
            double transferSplitFraction = InterpolationTools.linearInterpolate(defaultTransferSplitFraction,
                                                                                parameters.getTransferSplitFractionAtFullDepth(), alpha);
            double transferWeightDistribution = InterpolationTools.linearInterpolate(defaultWeightDistribution,
                                                                                     parameters.getTransferWeightDistributionAtFullDepth(), alpha);

            if (stepNumber == footstepDataMessageList.size() - 1)
            { // this is the last step
               processedOutput.getFootstepDataList().setFinalTransferSplitFraction(transferSplitFraction);
               processedOutput.getFootstepDataList().setFinalTransferWeightDistribution(transferWeightDistribution);
            }
            else
            {
               footstepDataMessageList.get(stepNumber + 1).setTransferSplitFraction(transferSplitFraction);
               footstepDataMessageList.get(stepNumber + 1).setTransferWeightDistribution(transferWeightDistribution);
            }
         }

      }

      return processedOutput;
   }

   /** {@inheritDoc} **/
   @Override
   public PostProcessingEnum getElementName()
   {
      return PostProcessingEnum.STEP_SPLIT_FRACTIONS;
   }
}
