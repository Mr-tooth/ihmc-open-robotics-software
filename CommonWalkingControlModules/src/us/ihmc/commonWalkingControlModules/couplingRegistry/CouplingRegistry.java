package us.ihmc.commonWalkingControlModules.couplingRegistry;

import us.ihmc.commonWalkingControlModules.RobotSide;
import us.ihmc.commonWalkingControlModules.bipedSupportPolygons.BipedSupportPolygons;
import us.ihmc.commonWalkingControlModules.desiredFootStep.Footstep;
import us.ihmc.utilities.math.geometry.FrameConvexPolygon2d;
import us.ihmc.utilities.math.geometry.FramePoint;
import us.ihmc.utilities.math.geometry.FrameVector2d;
import us.ihmc.utilities.math.geometry.ReferenceFrame;
import us.ihmc.utilities.screwTheory.Wrench;

import com.yobotics.simulationconstructionset.BooleanYoVariable;
import com.yobotics.simulationconstructionset.DoubleYoVariable;

public interface CouplingRegistry
{
   public abstract void setSingleSupportDuration(double singleSupportDuration);

   public abstract double getSingleSupportDuration();


   public abstract void setDoubleSupportDuration(double doubleSupportDuration);

   public abstract double getDoubleSupportDuration();


   public abstract void setDesiredVelocity(FrameVector2d desiredVelocity);

   public abstract FrameVector2d getDesiredVelocity();


   public abstract void setCaptureRegion(FrameConvexPolygon2d captureRegion);

   public abstract FrameConvexPolygon2d getCaptureRegion();


   public abstract void setCapturePoint(FramePoint capturePoint);

   public abstract FramePoint getCapturePointInFrame(ReferenceFrame desiredFrame);


   public abstract void setDesiredFootstep(Footstep desiredStepLocation);

   public abstract Footstep getDesiredFootstep();


   public abstract void setSupportLeg(RobotSide supportLeg);

   public abstract RobotSide getSupportLeg();


   public abstract void setEstimatedSwingTimeRemaining(double estimatedSwingTimeRemaining);

   public abstract double getEstimatedSwingTimeRemaining();


   public abstract void setBipedSupportPolygons(BipedSupportPolygons bipedSupportPolygons);

   public abstract BipedSupportPolygons getBipedSupportPolygons();


   public void setForceHindOnToes(BooleanYoVariable forceHindOnToes);

   public boolean getForceHindOnToes();


   public void setUpperBodyWrench(Wrench upperBodyWrench);

   public Wrench getUpperBodyWrench();




}
