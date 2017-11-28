package us.ihmc.robotics.linearAlgebra;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.interfaces.decomposition.QRPDecomposition;
import org.ejml.interfaces.decomposition.SingularValueDecomposition;
import org.junit.Test;
import us.ihmc.commons.PrintTools;
import us.ihmc.commons.RandomNumbers;
import us.ihmc.continuousIntegration.ContinuousIntegrationAnnotations;
import us.ihmc.robotics.time.ExecutionTimer;
import us.ihmc.yoVariables.registry.YoVariableRegistry;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class NullspaceCalculatorTimingTest
{
   @ContinuousIntegrationAnnotations.ContinuousIntegrationTest(estimatedDuration = 0.0)
   @Test(timeout = 30000)
   public void testRemoveNullspaceComponent()
   {
      YoVariableRegistry registry = new YoVariableRegistry(getClass().getSimpleName());
      ExecutionTimer svdTimer = new ExecutionTimer("svdTimer", registry);
      ExecutionTimer leastSquaresTimer = new ExecutionTimer("leastSquarestTimer", registry);
      ExecutionTimer qrTimer = new ExecutionTimer("qrTimer", registry);
      ExecutionTimer svdDecomposerTimer = new ExecutionTimer("svdDecomposerTimer", registry);
      ExecutionTimer qrDecomposerTimer = new ExecutionTimer("qrDecomposerTimer", registry);
      ExecutionTimer qrpDecomposerTimer = new ExecutionTimer("qrpDecomposerTimer", registry);


      NullspaceCalculator svdCalculator = new SVDNullspaceCalculator(100, true);
      NullspaceCalculator qrCalculator = new QRNullspaceCalculator(100);
      NullspaceCalculator leastSquaresCalculator = new DampedLeastSquaresNullspaceCalculator(100, 0.0, registry);

      SingularValueDecomposition<DenseMatrix64F> svdDecomposer = DecompositionFactory.svd(10, 10, false, true, false);
      QRDecomposition<DenseMatrix64F> qrDecomposer = DecompositionFactory.qr(10, 10);
      QRPDecomposition<DenseMatrix64F> qrpDecomposer = DecompositionFactory.qrp(10, 10);

      for (int i = 0; i < 10000; i++)
      {
         Random random = new Random(10L);
         double[] Jvalues = RandomNumbers.nextDoubleArray(random, 200, 10.0);
         double[] Avalues = RandomNumbers.nextDoubleArray(random, 100, 10.0);
         DenseMatrix64F J = new DenseMatrix64F(10, 20, false, Jvalues);
         DenseMatrix64F A = new DenseMatrix64F(5, 20, false, Avalues);

         DenseMatrix64F A_projected1 = new DenseMatrix64F(5, 20, false, Avalues);
         DenseMatrix64F A_projected2 = new DenseMatrix64F(5, 20, false, Avalues);
         DenseMatrix64F A_projected3 = new DenseMatrix64F(5, 20, false, Avalues);

         svdDecomposerTimer.startMeasurement();
         svdDecomposer.decompose(J);
         svdDecomposerTimer.stopMeasurement();

         qrDecomposerTimer.startMeasurement();
         qrDecomposer.decompose(J);
         qrDecomposerTimer.stopMeasurement();

         qrpDecomposerTimer.startMeasurement();
         qrpDecomposer.decompose(J);
         qrpDecomposerTimer.stopMeasurement();

         svdTimer.startMeasurement();
         svdCalculator.projectOntoNullspace(A, J, A_projected1);
         svdTimer.stopMeasurement();

         leastSquaresTimer.startMeasurement();
         leastSquaresCalculator.projectOntoNullspace(A, J, A_projected2);
         leastSquaresTimer.stopMeasurement();

         qrTimer.startMeasurement();
         qrCalculator.projectOntoNullspace(A, J, A_projected3);
         qrTimer.stopMeasurement();


         // check the solutions are equal
         for (int j = 0; j < A_projected1.getNumElements(); j++)
         {
            assertEquals(A_projected1.get(j), A_projected2.get(j), 1e-5);
            assertEquals(A_projected1.get(j), A_projected3.get(j), 1e-5);
         }
      }

      PrintTools.info("SVD average time : " + svdTimer.getAverageTime());
      PrintTools.info("Least Squares average time : " + leastSquaresTimer.getAverageTime());
      PrintTools.info("QR average time : " + qrTimer.getAverageTime());
      PrintTools.info("SVD decompose average time : " + svdDecomposerTimer.getAverageTime());
      PrintTools.info("QR decompose average time : " + qrDecomposerTimer.getAverageTime());
      PrintTools.info("QRP decompose average time : " + qrpDecomposerTimer.getAverageTime());
   }
}
