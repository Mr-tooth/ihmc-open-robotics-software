package us.ihmc.commonWalkingControlModules.controlModules.pelvisOrientation;

import us.ihmc.commonWalkingControlModules.controlModules.head.PelvisOrientationPacket;
import us.ihmc.utilities.math.geometry.FramePose;
import us.ihmc.utilities.math.geometry.ReferenceFrame;
import us.ihmc.utilities.net.ObjectConsumer;


/**
 * User: Matt
 * Date: 2/18/13
 */
public class DesiredPelvisPoseProvider implements ObjectConsumer<PelvisOrientationPacket>
{

   private FramePose desiredPelvisPose;
   private Boolean hasNewPose;

   public DesiredPelvisPoseProvider()
   {
      desiredPelvisPose = new FramePose(ReferenceFrame.getWorldFrame());
      hasNewPose = false;
   }

   public synchronized boolean checkForNewPose()
   {
      return hasNewPose;
   }

   public synchronized FramePose getDesiredPelvisPose()
   {
      hasNewPose = false;
      
      return desiredPelvisPose;
   }

   public void consumeObject(PelvisOrientationPacket object)
   {
      hasNewPose = true;
      desiredPelvisPose = new FramePose(ReferenceFrame.getWorldFrame(), object.getPoint(), object.getQuaternion());
   }
}
