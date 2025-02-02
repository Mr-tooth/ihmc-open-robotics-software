package us.ihmc.behaviors.lookAndStep;

import java.util.UUID;

import controller_msgs.msg.dds.FootstepDataListMessage;
import controller_msgs.msg.dds.RobotConfigurationData;
import controller_msgs.msg.dds.WalkingStatusMessage;
import us.ihmc.behaviors.tools.walkingController.ControllerStatusTracker;
import us.ihmc.commons.Conversions;
import us.ihmc.commons.thread.ThreadTools;
import us.ihmc.commons.thread.TypedNotification;
import us.ihmc.communication.packets.ExecutionMode;
import us.ihmc.tools.Timer;
import us.ihmc.tools.TimerSnapshotWithExpiration;
import us.ihmc.footstepPlanning.*;
import us.ihmc.footstepPlanning.graphSearch.parameters.FootstepPlannerParametersReadOnly;
import us.ihmc.footstepPlanning.swing.SwingPlannerParametersReadOnly;
import us.ihmc.avatar.drcRobot.ROS2SyncedRobotModel;
import us.ihmc.behaviors.tools.footstepPlanner.MinimalFootstep;
import us.ihmc.behaviors.tools.interfaces.RobotWalkRequester;
import us.ihmc.behaviors.tools.interfaces.StatusLogger;
import us.ihmc.behaviors.tools.interfaces.UIPublisher;
import us.ihmc.tools.thread.MissingThreadTools;
import us.ihmc.tools.thread.ResettableExceptionHandlingExecutorService;

import static us.ihmc.behaviors.lookAndStep.LookAndStepBehaviorAPI.LastCommandedFootsteps;

public class LookAndStepSteppingTask
{
   protected StatusLogger statusLogger;
   protected UIPublisher uiPublisher;
   protected LookAndStepBehaviorParametersReadOnly lookAndStepParameters;
   protected FootstepPlannerParametersReadOnly footstepPlannerParameters;
   protected SwingPlannerParametersReadOnly swingPlannerParameters;

   protected RobotWalkRequester robotWalkRequester;
   protected Runnable doneWaitingForSwingOutput;

   protected FootstepPlan footstepPlan;
   protected ROS2SyncedRobotModel syncedRobot;
   protected TimerSnapshotWithExpiration robotDataReceptionTimerSnaphot;
   protected long previousStepMessageId = 0L;
   protected LookAndStepImminentStanceTracker imminentStanceTracker;
   protected ControllerStatusTracker controllerStatusTracker;
   private final Timer timerSincePlanWasSent = new Timer();

   protected final TypedInput<RobotConfigurationData> robotConfigurationData = new TypedInput<>();

   public static class LookAndStepStepping extends LookAndStepSteppingTask
   {
      private ResettableExceptionHandlingExecutorService executor;
      private final TypedInput<FootstepPlan> footstepPlanInput = new TypedInput<>();
      private BehaviorTaskSuppressor suppressor;

      public void initialize(LookAndStepBehavior lookAndStep)
      {
         controllerStatusTracker = lookAndStep.controllerStatusTracker;
         imminentStanceTracker = lookAndStep.imminentStanceTracker;
         statusLogger = lookAndStep.statusLogger;
         syncedRobot = lookAndStep.robotInterface.newSyncedRobot();
         lookAndStepParameters = lookAndStep.lookAndStepParameters;
         footstepPlannerParameters = lookAndStep.footstepPlannerParameters;
         swingPlannerParameters = lookAndStep.swingPlannerParameters;
         uiPublisher = lookAndStep.helper::publish;
         robotWalkRequester = lookAndStep.robotInterface::requestWalk;
         doneWaitingForSwingOutput = () ->
         {
            if (!lookAndStep.isBeingReset.get())
            {
               lookAndStep.behaviorStateReference.set(LookAndStepBehavior.State.FOOTSTEP_PLANNING);
               lookAndStep.bodyPathLocalization.acceptSwingSleepComplete();
            }
         };

         executor = MissingThreadTools.newSingleThreadExecutor(getClass().getSimpleName(), true, 1);
         footstepPlanInput.addCallback(data -> executor.clearQueueAndExecute(this::evaluateAndRun));

         suppressor = new BehaviorTaskSuppressor(statusLogger, "Stepping task");
         suppressor.addCondition("Not in robot motion state", () -> !lookAndStep.behaviorStateReference.get().equals(LookAndStepBehavior.State.STEPPING));
         suppressor.addCondition(() -> "Footstep plan not OK: numberOfSteps = " + (footstepPlan == null ? null : footstepPlan.getNumberOfSteps())
                                       + ". Planning again...", () -> !(footstepPlan != null && footstepPlan.getNumberOfSteps() > 0), doneWaitingForSwingOutput);
         suppressor.addCondition("Robot disconnected", () -> !robotDataReceptionTimerSnaphot.isRunning());
         suppressor.addCondition("Robot not in walking state", () -> !lookAndStep.controllerStatusTracker.isInWalkingState());
      }

      public void reset()
      {
         executor.interruptAndReset();
         previousStepMessageId = 0L;
      }

      public void acceptFootstepPlan(FootstepPlan footstepPlan)
      {
         footstepPlanInput.set(footstepPlan);
      }

      private void evaluateAndRun()
      {
         footstepPlan = footstepPlanInput.getLatest();
         syncedRobot.update();
         robotDataReceptionTimerSnaphot = syncedRobot.getDataReceptionTimerSnapshot()
                                                     .withExpiration(lookAndStepParameters.getRobotConfigurationDataExpiration());

         if (suppressor.evaulateShouldAccept())
         {
            performTask();
         }
      }
   }

   protected void performTask()
   {
      FootstepDataListMessage footstepDataListMessage = new FootstepDataListMessage();
      footstepDataListMessage.setOffsetFootstepsHeightWithExecutionError(true);
      FootstepDataMessageConverter.appendPlanToMessage(footstepPlan, footstepDataListMessage);
      // TODO: Add combo to look and step UI to chose which steps to visualize
      uiPublisher.publishToUI(LastCommandedFootsteps, MinimalFootstep.convertFootstepDataListMessage(footstepDataListMessage, "Look and Step Last Commanded"));

      ExecutionMode executionMode;
      if (lookAndStepParameters.getMaxStepsToSendToController() > 1)
      {
         executionMode = ExecutionMode.OVERRIDE; // ALPHA. Seems to not work on real robot.
      }
      else
      {
         executionMode = previousStepMessageId == 0L ? ExecutionMode.OVERRIDE : ExecutionMode.QUEUE;
      }
      imminentStanceTracker.addCommandedFootsteps(footstepPlan, executionMode);

      footstepDataListMessage.getQueueingProperties().setExecutionMode(executionMode.toByte());
      long messageId = UUID.randomUUID().getLeastSignificantBits();
      footstepDataListMessage.getQueueingProperties().setMessageId(messageId);
      footstepDataListMessage.getQueueingProperties().setPreviousMessageId(previousStepMessageId);
      previousStepMessageId = messageId;
      statusLogger.warn("Requesting walk {}ing {} step plan starting with {} foot.",
                        executionMode.name(),
                        footstepPlan.getNumberOfSteps(),
                        footstepPlan.getFootstep(0).getRobotSide().name());
      TypedNotification<WalkingStatusMessage> walkingStatusNotification = robotWalkRequester.requestWalk(footstepDataListMessage);
      timerSincePlanWasSent.reset();

      ThreadTools.startAsDaemon(() -> robotWalkingThread(walkingStatusNotification), "RobotWalking");
      waitForPartOfSwing(lookAndStepParameters.getSwingDuration());
   }

   private void waitForPartOfSwing(double swingDuration)
   {
      double estimatedRobotTimeWhenPlanWasSent = getEstimatedRobotTime();
      double percentSwingToWait = lookAndStepParameters.getPercentSwingToWait();
      double waitDuration = swingDuration * percentSwingToWait;
      double maxDurationToWait = 10.0;
      double robotTimeToStopWaitingRegardless = estimatedRobotTimeWhenPlanWasSent + maxDurationToWait;
      statusLogger.info("Waiting up to {} s for commanded step to start...", maxDurationToWait);

      boolean stepStartTimeRecorded = false;
      double robotTimeInWhichStepStarted = Double.NaN;
      double robotTimeToStopWaiting = Double.NaN;
      while (true)
      {
         double moreRobustRobotTime = getMoreRobustRobotTime(estimatedRobotTimeWhenPlanWasSent);
         // FIXME: What if the queue size was larger? Need to know when the step we sent is started
         boolean stepHasStarted = imminentStanceTracker.getStepsStartedSinceCommanded() > 0;
         boolean haveWaitedMaxDuration = moreRobustRobotTime >= robotTimeToStopWaitingRegardless;
         boolean robotIsNotWalkingAnymoreForSomeReason = !controllerStatusTracker.isWalking();
         boolean stepCompletedEarly = imminentStanceTracker.getStepsCompletedSinceCommanded() > 0;

         // Part 1: Wait for the step to start with a timeout
         if (haveWaitedMaxDuration)
         {
            statusLogger.info("Waited max duration of {} s. Done waiting.", maxDurationToWait);
            break;
         }

         // Part 2: The commanded step has started
         if (stepHasStarted)
         {
            if (!stepStartTimeRecorded)
            {
               stepStartTimeRecorded = true;
               robotTimeInWhichStepStarted = getMoreRobustRobotTime(estimatedRobotTimeWhenPlanWasSent);
               robotTimeToStopWaiting = moreRobustRobotTime + waitDuration;
               statusLogger.info("Waiting {} s for {} % of swing...", waitDuration, percentSwingToWait * 100.0);
            }

            if (robotIsNotWalkingAnymoreForSomeReason)
            {
               statusLogger.info("Robot not walking anymore {} s after step started for some reason. Done waiting.",
                                 moreRobustRobotTime - robotTimeInWhichStepStarted);
               break;
            }
            else if (stepCompletedEarly)
            {
               statusLogger.info("Step completed {} s early. Done waiting.", robotTimeToStopWaiting - moreRobustRobotTime);
               break;
            }
            else if (moreRobustRobotTime >= robotTimeToStopWaiting)
            {
               statusLogger.info("{} % of swing complete! Done waiting.", (moreRobustRobotTime - estimatedRobotTimeWhenPlanWasSent) / swingDuration);
               break;
            }
         }

         ThreadTools.sleepSeconds(0.01); // Prevent free spinning
      }

      doneWaitingForSwingOutput.run();
   }

   private double getMoreRobustRobotTime(double estimatedRobotTimeWhenPlanWasSent)
   {
      return Math.max(getEstimatedRobotTime(), estimatedRobotTimeWhenPlanWasSent + timerSincePlanWasSent.getElapsedTime());
   }

   private double getEstimatedRobotTime()
   {
      return Conversions.nanosecondsToSeconds(syncedRobot.getTimestamp()) + syncedRobot.getDataReceptionTimerSnapshot().getTimePassedSinceReset();
   }

   private void robotWalkingThread(TypedNotification<WalkingStatusMessage> walkingStatusNotification)
   {
      statusLogger.debug("Waiting for robot walking...");
      walkingStatusNotification.blockingPoll();
      statusLogger.debug("Robot walk complete.");
   }
}
