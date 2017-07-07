package us.ihmc.robotics.math.filters;

import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.tuple2D.Vector2D;
import us.ihmc.yoVariables.registry.YoVariableRegistry;
import us.ihmc.yoVariables.variable.YoDouble;
import us.ihmc.robotics.geometry.FrameTuple2D;
import us.ihmc.robotics.math.frames.YoFramePoint;
import us.ihmc.robotics.math.frames.YoFrameTuple2d;
import us.ihmc.robotics.math.frames.YoFrameVariableNameTools;
import us.ihmc.robotics.math.frames.YoFrameVector2d;

public class FilteredVelocityYoFrameVector2d extends YoFrameVector2d
{
   private final FilteredVelocityYoVariable xDot, yDot;

   public static FilteredVelocityYoFrameVector2d createFilteredVelocityYoFrameVector2d(String namePrefix, String nameSuffix, YoDouble alpha, double dt, YoVariableRegistry registry, YoFrameTuple2d<?, ?> yoFrameVectorToDifferentiate)
   {
      return createFilteredVelocityYoFrameVector2d(namePrefix, nameSuffix, "", alpha, dt, registry, yoFrameVectorToDifferentiate);
   }

   public static FilteredVelocityYoFrameVector2d createFilteredVelocityYoFrameVector2d(String namePrefix, String nameSuffix, String description, YoDouble alpha, double dt, YoVariableRegistry registry, YoFrameTuple2d<?, ?> yoFrameVectorToDifferentiate)
   {
      FilteredVelocityYoVariable xDot = new FilteredVelocityYoVariable(YoFrameVariableNameTools.createXName(namePrefix, nameSuffix), description, alpha, yoFrameVectorToDifferentiate.getYoX(), dt, registry);
      FilteredVelocityYoVariable yDot = new FilteredVelocityYoVariable(YoFrameVariableNameTools.createYName(namePrefix, nameSuffix), description, alpha, yoFrameVectorToDifferentiate.getYoY(), dt, registry);

      FilteredVelocityYoFrameVector2d ret = new FilteredVelocityYoFrameVector2d(xDot, yDot, alpha, dt, registry, yoFrameVectorToDifferentiate.getReferenceFrame());

      return ret;
   }

   public static FilteredVelocityYoFrameVector2d createFilteredVelocityYoFrameVector2d(String namePrefix, String nameSuffix, YoDouble alpha, double dt, YoVariableRegistry registry, ReferenceFrame referenceFrame)
   {
      FilteredVelocityYoVariable xDot = new FilteredVelocityYoVariable(YoFrameVariableNameTools.createXName(namePrefix, nameSuffix), "", alpha, dt, registry);
      FilteredVelocityYoVariable yDot = new FilteredVelocityYoVariable(YoFrameVariableNameTools.createYName(namePrefix, nameSuffix), "", alpha, dt, registry);

      FilteredVelocityYoFrameVector2d ret = new FilteredVelocityYoFrameVector2d(xDot, yDot, alpha, dt, registry, referenceFrame);

      return ret;
   }

   private FilteredVelocityYoFrameVector2d(FilteredVelocityYoVariable xDot, FilteredVelocityYoVariable yDot, YoDouble alpha, double dt, YoVariableRegistry registry, ReferenceFrame referenceFrame)
   {
      super(xDot, yDot, referenceFrame);

      this.xDot = xDot;
      this.yDot = yDot;
   }

   private FilteredVelocityYoFrameVector2d(FilteredVelocityYoVariable xDot, FilteredVelocityYoVariable yDot, YoDouble alpha, double dt, YoVariableRegistry registry, YoFramePoint yoFramePointToDifferentiate)
   {
      super(xDot, yDot, yoFramePointToDifferentiate.getReferenceFrame());

      this.xDot = xDot;
      this.yDot = yDot;
   }

   public void update()
   {
      xDot.update();
      yDot.update();
   }

   public void update(Vector2D vector)
   {
      xDot.update(vector.getX());
      yDot.update(vector.getY());
   }

   public void update(FrameTuple2D<?, ?> tupleToDifferentiate)
   {
      checkReferenceFrameMatch(tupleToDifferentiate);
      xDot.update(tupleToDifferentiate.getX());
      yDot.update(tupleToDifferentiate.getY());
   }

   public void update(YoFrameVector2d vector)
   {
      checkReferenceFrameMatch(vector.getReferenceFrame());
      xDot.update(vector.getX());
      yDot.update(vector.getY());
   }

   public void reset()
   {
      xDot.reset();
      yDot.reset();
   }
}
