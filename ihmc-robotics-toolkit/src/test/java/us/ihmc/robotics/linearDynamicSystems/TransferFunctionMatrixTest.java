package us.ihmc.robotics.linearDynamicSystems;

import static us.ihmc.robotics.Assert.*;

import org.junit.jupiter.api.*;

import Jama.Matrix;
import us.ihmc.robotics.dataStructures.ObsoletePolynomial;

public class TransferFunctionMatrixTest
{
   private TransferFunction simpleDecayOne, secondOrderResponseOne;
   private TransferFunction simpleDecayTwo, secondOrderResponseTwo;
   private TransferFunctionMatrix transferFunctionMatrix;

   @BeforeEach
   public void setUp() throws Exception
   {
      secondOrderResponseOne = TransferFunction.constructSecondOrderTransferFunction(1.0, 10.0, 0.3);
      simpleDecayOne = new TransferFunction(new double[] {1.0}, new double[] {1.0, -3.0});
      secondOrderResponseTwo = TransferFunction.constructSecondOrderTransferFunction(2.0, 20.0, 2.0);
      simpleDecayTwo = new TransferFunction(new double[] {1.0}, new double[] {1.0, -10.0});

      TransferFunction[][] transferFunctions = new TransferFunction[][]
      {
         {simpleDecayOne, secondOrderResponseOne}, {simpleDecayTwo, secondOrderResponseTwo}
      };

      transferFunctionMatrix = new TransferFunctionMatrix(transferFunctions);
   }

   @AfterEach
   public void tearDown() throws Exception
   {
      simpleDecayOne = null;
      secondOrderResponseOne = null;
      simpleDecayTwo = null;
      secondOrderResponseTwo = null;

      transferFunctionMatrix = null;
   }

	@Test
   public void testGet()
   {
      verifyEpsilonEqual(simpleDecayOne, transferFunctionMatrix.get(0, 0), 1e-7);
   }

   private void verifyEpsilonEqual(TransferFunction expectedTransferFunction, TransferFunction actualTransferFunction, double epsilon)
   {
      verifyEpsilonEqual(expectedTransferFunction.getNumeratorCoefficients(), actualTransferFunction.getNumeratorCoefficients(), epsilon);
      verifyEpsilonEqual(expectedTransferFunction.getDenominatorCoefficients(), actualTransferFunction.getDenominatorCoefficients(), epsilon);
   }

   private void verifyEpsilonEqual(double[] expected, double[] actual, double epsilon)
   {
      assertEquals(expected.length, actual.length);

//    if (expected.length != actual.length)
//       fail();

      for (int i = 0; i < expected.length; i++)
      {
         assertEquals(expected[i], actual[i], epsilon);
      }
   }

	@Test
   public void testPremultiply()
   {
      Matrix matrixC = new Matrix(new double[][]
      {
         {1.0, 0.0}, {0.0, 1.0}
      });
      TransferFunctionMatrix newTransferFunctionMatrix = transferFunctionMatrix.preMultiply(matrixC);
      assertTrue(newTransferFunctionMatrix.epsilonEquals(transferFunctionMatrix, 1e-7));

      matrixC = new Matrix(new double[][]
      {
         {1.0, 0.0}, {0.0, 0.0}
      });
      newTransferFunctionMatrix = transferFunctionMatrix.preMultiply(matrixC);

      TransferFunction[][] transferFunctions = new TransferFunction[][]
      {
         {simpleDecayOne, secondOrderResponseOne}, {TransferFunction.constructZeroTransferFunction(), TransferFunction.constructZeroTransferFunction()}
      };

      transferFunctionMatrix = new TransferFunctionMatrix(transferFunctions);
      assertTrue(newTransferFunctionMatrix.epsilonEquals(transferFunctionMatrix, 1e-7));
   }

	@Test
   public void testPreMultiplyException()
   {
      Assertions.assertThrows(RuntimeException.class, () -> {
      int rows = transferFunctionMatrix.getRows();
      int columns = transferFunctionMatrix.getColumns();
      Matrix testMatrix = Matrix.random(rows, columns - 1);

      transferFunctionMatrix.preMultiply(testMatrix);
      });
   }

	@Test
   public void testTimes()
   {
      Matrix matrixC = new Matrix(new double[][]
      {
         {1.0, 0.0}, {0.0, 1.0}
      });
      TransferFunctionMatrix newTransferFunctionMatrix = transferFunctionMatrix.times(matrixC);
      assertTrue(newTransferFunctionMatrix.epsilonEquals(transferFunctionMatrix, 1e-7));

      matrixC = new Matrix(new double[][]
      {
         {1.0, 0.0}, {0.0, 0.0}
      });
      newTransferFunctionMatrix = transferFunctionMatrix.times(matrixC);

      TransferFunction[][] transferFunctions = new TransferFunction[][]
      {
         {simpleDecayOne, TransferFunction.constructZeroTransferFunction()}, {simpleDecayTwo, TransferFunction.constructZeroTransferFunction()}
      };

      transferFunctionMatrix = new TransferFunctionMatrix(transferFunctions);
      assertTrue(newTransferFunctionMatrix.epsilonEquals(transferFunctionMatrix, 1e-7));
   }

	@Test
   public void testTimesException()
   {
      Assertions.assertThrows(RuntimeException.class, () -> {
      int rows = transferFunctionMatrix.getRows();
      int columns = transferFunctionMatrix.getColumns();
      Matrix testMatrix = Matrix.random(rows - 1, columns);

      transferFunctionMatrix.times(testMatrix);
      });
   }

	@Test
   public void testPlusDouble()
   {
      Matrix testMatrix = Matrix.random(transferFunctionMatrix.getRows(), transferFunctionMatrix.getRows());
      TransferFunctionMatrix result = transferFunctionMatrix.plus(testMatrix);

      for (int m = 0; m < testMatrix.getRowDimension(); m++)
      {
         for (int n = 0; n < testMatrix.getColumnDimension(); n++)
         {
            TransferFunction resultTF = result.get(m, n);
            TransferFunction originalTF = transferFunctionMatrix.get(m, n);
            ObsoletePolynomial originalNumeratorCopy = new ObsoletePolynomial(originalTF.getNumeratorCoefficients());
            ObsoletePolynomial originalDenominatorCopy = new ObsoletePolynomial(originalTF.getDenominatorCoefficients());


            /*
             *  Different way of calculating the sum of the originalTF and the scalar a:
             * originalTF = p1/p2; resultTF = originalTF + a = p1/p2 + (a*p2)/p2 = (p1+a*p2)/p2
             * resultTF = productTF * originalTF = productTF * p1/p2 = (p1+a*p2)/p2
             * productTF = (p1+a*p2)/p1
             * Note that this is not valid when p1 == 0.0
             */
            double matrixValue = testMatrix.get(m, n);
            if (originalNumeratorCopy.equalsZero())
            {    // TODO: Add a way such that this too is tested.
               assertTrue(resultTF.getNumeratorCoefficients()[0] == matrixValue);
               assertEquals(1, resultTF.getNumeratorCoefficients().length);
            }
            else
            {
               TransferFunction productTF = new TransferFunction(originalDenominatorCopy.times(matrixValue).plus(originalNumeratorCopy), originalNumeratorCopy);

               TransferFunction expectedTF = productTF.times(originalTF);
               assertTrue(expectedTF.epsilonEquals(resultTF, 1e-5));
            }

         }
      }
   }

	@Test
   public void testPlusDoubleException()
   {
      Assertions.assertThrows(RuntimeException.class, () -> {
      Matrix testMatrix = new Matrix(new double[][]
      {
         {2.0, 5.0}
      });
      transferFunctionMatrix.plus(testMatrix);
      });
   }

	@Test
   public void testPlusTransferFunctionDouble()
   {
      TransferFunction[][] transferFunctions = new TransferFunction[][]
      {
         {secondOrderResponseTwo, simpleDecayTwo}, {secondOrderResponseOne, simpleDecayOne}
      };
      TransferFunctionMatrix testMatrix = new TransferFunctionMatrix(transferFunctions);

      TransferFunctionMatrix result = transferFunctionMatrix.plus(testMatrix);

      for (int m = 0; m < testMatrix.getRows(); m++)
      {
         for (int n = 0; n < testMatrix.getColumns(); n++)
         {
            TransferFunction resultTF = result.get(m, n);
            TransferFunction originalTFOne = transferFunctionMatrix.get(m, n);
            ObsoletePolynomial originalNumeratorOneCopy = new ObsoletePolynomial(originalTFOne.getNumeratorCoefficients());
            ObsoletePolynomial originalDenominatorOneCopy = new ObsoletePolynomial(originalTFOne.getDenominatorCoefficients());

            TransferFunction originalTFTwo = testMatrix.get(m, n);
            ObsoletePolynomial originalNumeratorTwoCopy = new ObsoletePolynomial(originalTFTwo.getNumeratorCoefficients());
            ObsoletePolynomial originalDenominatorTwoCopy = new ObsoletePolynomial(originalTFTwo.getDenominatorCoefficients());

            /*
             * Different way of calculating the sum of TF1 and TF2:
             *             TF1 = N1/D1; TF2 = N2/D2; result = N1/D1 + N2/D2 = (N1*D2 + N2*D1)/(D1*D2)
             *             result = productTF * TF1 => productTF = (N1*D2 + N2*D1)/(N1*D2) = 1 + (N2*D1)/(N1*D2)
             *             Note that this is not applicable when N1 == 0.0
             */
            if (originalNumeratorOneCopy.equalsZero())
            {    // TODO: Add a way such that this too is tested.
               assertTrue(originalTFTwo.epsilonEquals(resultTF, 1e-5));
            }
            else
            {
               TransferFunction productTF = new TransferFunction(originalNumeratorTwoCopy.times(originalDenominatorOneCopy),
                                               originalNumeratorOneCopy.times(originalDenominatorTwoCopy));
               productTF = productTF.plus(1.0);

               TransferFunction expectedTF = productTF.times(originalTFOne);
               assertTrue(expectedTF.epsilonEquals(resultTF, 1e-5));
            }

         }
      }
   }

	@Test
   public void testPlusTransferFunctionException()
   {
      Assertions.assertThrows(RuntimeException.class, () -> {
      TransferFunctionMatrix testMatrix = new TransferFunctionMatrix(new TransferFunction[][]
      {
         {secondOrderResponseOne, secondOrderResponseTwo}
      });
      transferFunctionMatrix.plus(testMatrix);
      });
   }

}
