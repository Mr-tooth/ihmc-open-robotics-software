package us.ihmc.commonWalkingControlModules.dynamicPlanning.comPlanning;

import org.junit.jupiter.api.Test;
import us.ihmc.commons.MathTools;
import us.ihmc.commons.RandomNumbers;
import us.ihmc.euclid.referenceFrame.FramePoint3D;
import us.ihmc.euclid.referenceFrame.FrameVector3D;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.referenceFrame.interfaces.FramePoint3DReadOnly;
import us.ihmc.euclid.referenceFrame.tools.EuclidFrameRandomTools;
import us.ihmc.euclid.referenceFrame.tools.EuclidFrameTestTools;
import us.ihmc.euclid.tools.EuclidCoreTestTools;
import us.ihmc.euclid.tuple3D.Point3D;

import java.util.Random;

public class DCMToolsTest
{
   @Test
   public void testLinearLongCalculation()
   {
      Random random = new Random(1738L);

      FramePoint3D initialVRP = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 1.0);
      FramePoint3D finalVRP = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 1.0);
      FramePoint3D initialDCM = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 1.0);

      double duration = RandomNumbers.nextDouble(random, 0.0, 2.0);
      double time = RandomNumbers.nextDouble(random, 0.0, duration);
      double omega = RandomNumbers.nextDouble(random, 1.0, 10.0);

      FramePoint3D returnedDCM = new FramePoint3D();
      DCMTrajectoryTestTools.computeDCMUsingLinearVRP(omega, time, duration, initialDCM, initialVRP, finalVRP, returnedDCM);

      double sigma = time / duration + 1.0 / (omega * duration);
      double sigma0 = 1.0 / (omega * duration);
      FramePoint3D v = new FramePoint3D();
      FramePoint3D v0 = new FramePoint3D();
      v.interpolate(initialVRP, finalVRP, sigma);
      v0.interpolate(initialVRP, finalVRP, sigma0);

      FramePoint3D expectedDCM = new FramePoint3D();
      expectedDCM.set(initialDCM);
      expectedDCM.sub(v0);
      expectedDCM.scale(Math.exp(omega * time));
      expectedDCM.add(v);

      EuclidFrameTestTools.assertFramePoint3DGeometricallyEquals(expectedDCM, returnedDCM, 1e-5);
      EuclidFrameTestTools
            .assertFramePoint3DGeometricallyEquals(getLinearDCMAlternateCalculation(omega, time, duration, initialDCM, initialVRP, finalVRP), returnedDCM,
                                                   1e-5);

      // test final DCM
      FramePoint3D finalDCM = new FramePoint3D();
      DCMTrajectoryTestTools.computeDCMUsingLinearVRP(omega, duration, duration, initialDCM, initialVRP, finalVRP, finalDCM);
      sigma = duration / duration + 1.0 / (omega * duration);
      sigma0 = 1.0 / (omega * duration);
      v = new FramePoint3D();
      v0 = new FramePoint3D();
      v.interpolate(initialVRP, finalVRP, sigma);
      v0.interpolate(initialVRP, finalVRP, sigma0);

      expectedDCM = new FramePoint3D();
      expectedDCM.set(initialDCM);
      expectedDCM.sub(v0);
      expectedDCM.scale(Math.exp(omega * duration));
      expectedDCM.add(v);

      EuclidFrameTestTools.assertFramePoint3DGeometricallyEquals(expectedDCM, finalDCM, 1e-5 * expectedDCM.distance(new Point3D()));

      // test recursing back from final
      FramePoint3D returnedInitialDCM = new FramePoint3D();
      DCMTrajectoryTestTools.computeDCMUsingLinearVRP(omega, -duration, -duration, finalDCM, finalVRP, initialVRP, returnedInitialDCM);

      EuclidFrameTestTools.assertFramePoint3DGeometricallyEquals(initialDCM, returnedInitialDCM, 1e-5);

   }

   @Test
   public void testLinearNoVRPMotion()
   {
      Random random = new Random(1738L);

      for (int i = 0; i < 1000; i++)
      {
         FramePoint3D desiredVRP = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 10.0);
         FramePoint3D initialDCM = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 10.0);

         double duration = RandomNumbers.nextDouble(random, 0.0, 10.0);
         double time = RandomNumbers.nextDouble(random, 0.0, duration);
         double omega = RandomNumbers.nextDouble(random, 1.0, 10.0);

         FramePoint3D expectedDCM = new FramePoint3D();
         FramePoint3D returnedDCM = new FramePoint3D();
         DCMTrajectoryTestTools.computeDCMUsingConstantVRP(omega, time, initialDCM, desiredVRP, expectedDCM);
         DCMTrajectoryTestTools.computeDCMUsingLinearVRP(omega, time, duration, initialDCM, desiredVRP, desiredVRP, returnedDCM);

         EuclidFrameTestTools.assertFramePoint3DGeometricallyEquals(expectedDCM, returnedDCM, 1e-5);
      }
   }

   @Test
   public void testDCMCalculations()
   {
      Random random = new Random(1738L);

      for (int i = 0; i < 1000; i++)
      {
         FramePoint3D desiredCoMPosition = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 10.0);
         FrameVector3D desiredCoMVelocity = EuclidFrameRandomTools.nextFrameVector3D(random, ReferenceFrame.getWorldFrame(), -10.0, 10.0);
         FrameVector3D desiredCoMAcceleration = EuclidFrameRandomTools.nextFrameVector3D(random, ReferenceFrame.getWorldFrame(), -10.0, 10.0);

         double omega = RandomNumbers.nextDouble(random, 1.0, 10.0);
         FramePoint3D expectedDCM = new FramePoint3D();
         FramePoint3D expectedVRP = new FramePoint3D();
         FramePoint3D expectedVRP2 = new FramePoint3D();
         FrameVector3D expectedDCMVelocity = new FrameVector3D();
         FramePoint3D returnedDCM = new FramePoint3D();
         FramePoint3D returnedVRP = new FramePoint3D();
         FramePoint3D returnedVRP2 = new FramePoint3D();
         FrameVector3D returnedDCMVelocity = new FrameVector3D();

         expectedDCM.scaleAdd(1.0 / omega, desiredCoMVelocity, desiredCoMPosition);
         expectedDCMVelocity.scaleAdd(1.0 / omega, desiredCoMAcceleration, desiredCoMVelocity);
         expectedVRP.scaleAdd(-1.0 / MathTools.square(omega), desiredCoMAcceleration, desiredCoMPosition);
         expectedVRP2.scaleAdd(-1.0 / omega, expectedDCMVelocity, expectedDCM);

         DCMTools.computeDesiredDCMPosition(desiredCoMPosition, desiredCoMVelocity, omega, returnedDCM);
         DCMTools.computeDesiredDCMVelocity(desiredCoMVelocity, desiredCoMAcceleration, omega, returnedDCMVelocity);
         DCMTools.computeDesiredVRPPosition(returnedDCM, returnedDCMVelocity, omega, returnedVRP);
         DCMTools.computeDesiredVRPPositionFromCoM(desiredCoMPosition, desiredCoMAcceleration, omega, returnedVRP2);

         EuclidCoreTestTools.assertPoint3DGeometricallyEquals(expectedDCM, returnedDCM, 1e-5);
         EuclidCoreTestTools.assertVector3DGeometricallyEquals(expectedDCMVelocity, returnedDCMVelocity, 1e-5);
         EuclidCoreTestTools.assertPoint3DGeometricallyEquals(expectedVRP, expectedVRP2, 1e-5);
         EuclidCoreTestTools.assertPoint3DGeometricallyEquals(expectedVRP, returnedVRP2, 1e-5);
         EuclidCoreTestTools.assertPoint3DGeometricallyEquals(expectedVRP, returnedVRP, 1e-5);
      }
   }

   private static FramePoint3DReadOnly getLinearDCMAlternateCalculation(double omega, double time, double duration, FramePoint3DReadOnly initialDCM,
                                                                        FramePoint3DReadOnly initialVRP, FramePoint3DReadOnly finalVRP)
   {
      FramePoint3D dcm = new FramePoint3D();

      double sigma = time / duration + 1.0 / (omega * duration);
      double sigma0 = 1.0 / (omega * duration);
      double exponential = Math.exp(omega * time);

      dcm.scaleAdd(1.0 - sigma - exponential + exponential * sigma0, initialVRP, dcm);
      dcm.scaleAdd(sigma - exponential * sigma0, finalVRP, dcm);
      dcm.scaleAdd(exponential, initialDCM, dcm);

      return dcm;
   }

}
