package us.ihmc.commonWalkingControlModules.instantaneousCapturePoint;

import us.ihmc.commonWalkingControlModules.desiredFootStep.TransferToAndNextFootstepsData;
import us.ihmc.utilities.math.geometry.FramePoint2d;

public interface FinalDesiredICPCalculator
{
   public abstract void initialize(TransferToAndNextFootstepsData transferToAndNextFootstepsData);
   public abstract FramePoint2d getFinalDesiredICP();
}
