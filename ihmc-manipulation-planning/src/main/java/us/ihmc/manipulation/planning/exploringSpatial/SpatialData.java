package us.ihmc.manipulation.planning.exploringSpatial;

import java.util.ArrayList;
import java.util.List;

import us.ihmc.commons.PrintTools;
import us.ihmc.euclid.matrix.RotationMatrix;
import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.robotics.geometry.AngleTools;

/**
 * This is to hold spatial of the rigid bodies.
 * This is packed with time onto SpatialNode that can be depicted by SpatialNodePlotter.
 */
public class SpatialData
{
   private final List<String> exploringRigidBodyNames;
   private final List<RigidBodyTransform> rigidBodySpatials;

   public SpatialData()
   {
      exploringRigidBodyNames = new ArrayList<String>();
      rigidBodySpatials = new ArrayList<RigidBodyTransform>();
   }

   public SpatialData(SpatialData other)
   {
      this();
      for (int i = 0; i < other.rigidBodySpatials.size(); i++)
         rigidBodySpatials.add(new RigidBodyTransform(other.rigidBodySpatials.get(i)));
      exploringRigidBodyNames.addAll(other.exploringRigidBodyNames);
   }

   public void addSpatial(String exploringRigidBodyName, RigidBodyTransform transform)
   {
      this.exploringRigidBodyNames.add(exploringRigidBodyName);
      this.rigidBodySpatials.add(new RigidBodyTransform(transform));
   }

   public void interpolate(SpatialData dataOne, SpatialData dataTwo, double alpha)
   {
      for (int i = 0; i < rigidBodySpatials.size(); i++)
      {
         RigidBodyTransform transformOne = dataOne.getRigidBodySpatials().get(i);
         RigidBodyTransform transformTwo = dataTwo.getRigidBodySpatials().get(i);

         Point3D interpolatedPoint = new Point3D();
         interpolatedPoint.interpolate(transformOne.getTranslationVector(), transformTwo.getTranslationVector(), alpha);

         RotationMatrix interpolatedRotationMatrix = new RotationMatrix();
         interpolatedRotationMatrix.interpolate(transformOne.getRotationMatrix(), transformTwo.getRotationMatrix(), alpha);

         rigidBodySpatials.get(i).setTranslation(interpolatedPoint);
         rigidBodySpatials.get(i).setRotation(interpolatedRotationMatrix);
      }
   }

   public double getPositionDistance(SpatialData other)
   {
      double distance = 0.0;

      for (int i = 0; i < rigidBodySpatials.size(); i++)
      {
         if (!getRigidBodyName(i).equals(other.getRigidBodyName(i)))
            PrintTools.warn("other spatial data has different order");

         RigidBodyTransform transformOther = other.getRigidBodySpatials().get(i);

         Point3D position = new Point3D(rigidBodySpatials.get(i).getTranslationVector());

         distance = distance + position.distance(new Point3D(transformOther.getTranslationVector()));
      }

      return distance;
   }

   public double getOrientationDistance(SpatialData other)
   {
      double distance = 0.0;

      for (int i = 0; i < rigidBodySpatials.size(); i++)
      {
         double orientationDistance;

         if (rigidBodySpatials.get(i).getRotationMatrix().equals(other.getRigidBodySpatials().get(i).getRotationMatrix()))
            orientationDistance = 0.0;
         else
            orientationDistance = rigidBodySpatials.get(i).getRotationMatrix().distance(other.getRigidBodySpatials().get(i).getRotationMatrix());

         orientationDistance = AngleTools.trimAngleMinusPiToPi(orientationDistance);
         orientationDistance = Math.abs(orientationDistance);

         distance = distance + orientationDistance;
      }

      return distance;
   }

   public double getMaximumPositionDistance(SpatialData other)
   {
      double distance = Double.NEGATIVE_INFINITY;

      for (int i = 0; i < rigidBodySpatials.size(); i++)
      {
         RigidBodyTransform transformOther = other.getRigidBodySpatials().get(i);

         Point3D position = new Point3D(rigidBodySpatials.get(i).getTranslationVector());

         double positionDistance = position.distance(new Point3D(transformOther.getTranslationVector()));

         if (distance < positionDistance)
            distance = positionDistance;
      }

      return distance;
   }

   public double getMaximumOrientationDistance(SpatialData other)
   {
      double distance = Double.NEGATIVE_INFINITY;

      for (int i = 0; i < rigidBodySpatials.size(); i++)
      {
         double orientationDistance = rigidBodySpatials.get(i).getRotationMatrix().distance(other.getRigidBodySpatials().get(i).getRotationMatrix());
         orientationDistance = AngleTools.trimAngleMinusPiToPi(orientationDistance);
         orientationDistance = Math.abs(orientationDistance);
         if (distance < orientationDistance)
            distance = orientationDistance;
      }

      return distance;
   }

   public void replaceSpatialData(String rigidBodyName, RigidBodyTransform transform)
   {
      for (int i = 0; i < exploringRigidBodyNames.size(); i++)
      {
         if (getRigidBodyName(i) == rigidBodyName)
         {
            rigidBodySpatials.set(i, new RigidBodyTransform(transform));
         }
      }
   }

   public String getRigidBodyName(int i)
   {
      return exploringRigidBodyNames.get(i);
   }

   public List<RigidBodyTransform> getRigidBodySpatials()
   {
      return rigidBodySpatials;
   }

   public int getNumberOfExploringRigidBodies()
   {
      return exploringRigidBodyNames.size();
   }
}