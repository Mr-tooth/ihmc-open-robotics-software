package us.ihmc.robotics;

import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Test;
import us.ihmc.commons.RandomNumbers;

import java.util.Random;

import static us.ihmc.robotics.Assert.assertEquals;

public class MatrixMissingToolsTest
{
   @Test
   public void testSetDiagonalValues()
   {
      int iters = 100;
      DMatrixRMaj matrixToSet = new DMatrixRMaj(4, 7);
      Random random = new Random(1738L);
      for (int i = 0; i < iters; i++)
      {
         matrixToSet.setData(RandomNumbers.nextDoubleArray(random, 4 * 7, 100));
         DMatrixRMaj originalMatrix = new DMatrixRMaj(matrixToSet);
         double value = RandomNumbers.nextDouble(random, 10.0);
         MatrixMissingTools.setDiagonalValues(matrixToSet, value, 1, 3);

         for (int row = 0; row < 4; row++)
         {
            for (int col = 0; col < 7; col++)
            {
               if (row == 1 && col == 3)
                  assertEquals(value, matrixToSet.get(row, col), 1e-7);
               else if (row == 2 && col == 4)
                  assertEquals(value, matrixToSet.get(row, col), 1e-7);
               else if (row == 3 && col == 5)
                  assertEquals(value, matrixToSet.get(row, col), 1e-7);
               else
                  assertEquals("(" + row + ", " + col + ")", originalMatrix.get(row, col), matrixToSet.get(row, col), 1e-7);
            }

         }
      }
   }
}
