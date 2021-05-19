package us.ihmc.gdx.ui.graphics;

import com.badlogic.gdx.graphics.g3d.Renderable;
import com.badlogic.gdx.graphics.g3d.RenderableProvider;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.Pool;
import javafx.scene.transform.Translate;
import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.avatar.drcRobot.RemoteSyncedRobotModel;
import us.ihmc.commons.exception.DefaultExceptionHandler;
import us.ihmc.euclid.referenceFrame.interfaces.FramePose3DReadOnly;
import us.ihmc.gdx.FocusBasedGDXCamera;
import us.ihmc.humanoidRobotics.frames.HumanoidReferenceFrames;
import us.ihmc.ros2.ROS2NodeInterface;
import us.ihmc.tools.thread.Activator;
import us.ihmc.tools.thread.ExceptionHandlingThreadScheduler;

public class GDXRemoteRobotVisualizer implements RenderableProvider
{
   private final DRCRobotModel robotModel;
   private final RemoteSyncedRobotModel syncedRobot;
   private final ExceptionHandlingThreadScheduler scheduler;

   private final GDXRobotModelGraphic robotModelGraphic = new GDXRobotModelGraphic();
   private Activator robotLoadedActivator = new Activator();
   private boolean trackRobot = false;
   private FocusBasedGDXCamera cameraForOptionalTracking;
   private Translate robotTranslate;
   private Translate savedCameraTranslate;
   private boolean hasPrepended = false;

   public GDXRemoteRobotVisualizer(DRCRobotModel robotModel, ROS2NodeInterface ros2Node)
   {
      this.robotModel = robotModel;
      syncedRobot = new RemoteSyncedRobotModel(robotModel, ros2Node);

      scheduler = new ExceptionHandlingThreadScheduler(getClass().getSimpleName(), DefaultExceptionHandler.MESSAGE_AND_STACKTRACE, 1, true);
      // this doesn't work because you have to load GDX models in the same thread as the context or something
//      scheduler.scheduleOnce(() -> loadRobotModelAndGraphics(robotModel.getRobotDescription()));

      robotModelGraphic.loadRobotModelAndGraphics(robotModel.getRobotDescription(), syncedRobot.getFullRobotModel().getElevator());
      robotLoadedActivator.activate();
   }

   public void create()
   {
   }

   public void setTrackRobot(FocusBasedGDXCamera camera, boolean trackRobot)
   {
      this.trackRobot = trackRobot;
      cameraForOptionalTracking = camera;
      if (!hasPrepended)
      {
         hasPrepended = true;
         robotTranslate = new Translate();
//         cameraForOptionalTracking.prependTransform(robotTranslate);
      }
      else if (trackRobot)
      {
//         cameraForOptionalTracking.getTranslate().setX(savedCameraTranslate.getX());
//         cameraForOptionalTracking.getTranslate().setY(savedCameraTranslate.getY());
//         cameraForOptionalTracking.getTranslate().setZ(savedCameraTranslate.getZ());
      }
      else // !trackRobot
      {
//         savedCameraTranslate = cameraForOptionalTracking.getTranslate().clone();
      }
   }

   public void update()
   {
      if (robotLoadedActivator.poll())
      {
         syncedRobot.update();

         robotModelGraphic.getGraphicsRobot().update();
         robotModelGraphic.getRobotRootNode().update();

         if (trackRobot)
         {
            FramePose3DReadOnly walkingFrame = syncedRobot.getFramePoseReadOnly(HumanoidReferenceFrames::getMidFeetUnderPelvisFrame);

            robotTranslate.setX(walkingFrame.getPosition().getX());
            robotTranslate.setY(walkingFrame.getPosition().getY());
            robotTranslate.setZ(walkingFrame.getPosition().getZ());
         }
      }
   }

   @Override
   public void getRenderables(Array<Renderable> renderables, Pool<Renderable> pool)
   {
      if (robotLoadedActivator.poll())
      {
         robotModelGraphic.getRenderables(renderables, pool);
      }
   }

   public void destroy()
   {
      scheduler.shutdownNow();
      robotModelGraphic.destroy();
   }
}
