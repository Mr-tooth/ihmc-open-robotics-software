package us.ihmc.gdx.ui.gizmo;

import us.ihmc.euclid.geometry.interfaces.Line3DReadOnly;
import us.ihmc.euclid.transform.interfaces.RigidBodyTransformReadOnly;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;

import java.util.function.Function;

public class StepCheckIsPointInsideAlgorithm
{
   private final SphereRayIntersection boundingSphereIntersection = new SphereRayIntersection();
   private final Point3D interpolatedPoint = new Point3D();

   public void setup(double radius, RigidBodyTransformReadOnly transform)
   {
      boundingSphereIntersection.setup(radius, transform);
   }

   public void setup(double radius, Point3DReadOnly offset, RigidBodyTransformReadOnly transform)
   {
      boundingSphereIntersection.setup(radius, offset, transform);
   }

   public void setup(double radius, Point3DReadOnly positionInWorld)
   {
      boundingSphereIntersection.setup(radius, positionInWorld);
   }

   public double intersect(Line3DReadOnly pickRay, int resolution, Function<Point3DReadOnly, Boolean> isPointInside)
   {
      return intersect(pickRay, resolution, isPointInside, interpolatedPoint, true);
   }

   public double intersect(Line3DReadOnly pickRay,
                           int resolution,
                           Function<Point3DReadOnly, Boolean> isPointInside,
                           Point3D intersectionToPack,
                           boolean calculateDistance)
   {
      if (boundingSphereIntersection.intersect(pickRay))
      {
         for (int i = 0; i < resolution; i++)
         {
            intersectionToPack.interpolate(boundingSphereIntersection.getFirstIntersectionToPack(),
                                           boundingSphereIntersection.getSecondIntersectionToPack(),
                                           i / (double) resolution);
            if (isPointInside.apply(intersectionToPack))
            {
               return calculateDistance ? intersectionToPack.distance(pickRay.getPoint()) : 0.0;
            }
         }
      }

      return Double.NaN;
   }

   public Point3D getClosestIntersection()
   {
      return interpolatedPoint;
   }
}
