package com.yobotics.simulationconstructionset.matrixtest;

import static org.junit.Assert.*;
import java.util.Random;
import org.junit.Test;
import us.ihmc.utilities.math.MathTools;
import com.mathworks.jama.*;


public class MatrixTest
{
   int rowDimension = 40;
   int columnDimension = 30;

   Matrix matrix;
   Matrix matrixInverted;


   @Test
   public void testInvert()
   {
      System.out.println("");
      System.out.println("Matrix Inversion test");
      fillMatrix(rowDimension, rowDimension);
      displayMatrix(matrix, "matrix");

      Matrix matrixMultiplicationResult = new Matrix(rowDimension, rowDimension);

      // Invert Matrix
      matrixInverted = matrix.inverse();    // with jama
      displayMatrix(matrixInverted, "matrixInverted");

      // Matrix * Matrix inverted * Matrix should be equal to Matrix
      matrixMultiplicationResult = matrix.times(matrixInverted).times(matrix);    // with jama
      displayMatrix(matrixMultiplicationResult, "matrixMultiplicationResult");

      boolean numberAreCloseEnough = false;

      for (int i = 0; i < matrixMultiplicationResult.getRowDimension(); i++)
      {
         for (int j = 0; j < matrixMultiplicationResult.getColumnDimension(); j++)
         {
            // We lost precision during the multiplication so result can not be exactly the same
            if ((matrixMultiplicationResult.get(i, j) * 1.001 > matrix.get(i, j)) && (matrixMultiplicationResult.get(i, j) * 0.999 < matrix.get(i, j)))
               numberAreCloseEnough = true;
            else
               numberAreCloseEnough = false;

            assertFalse("Error at i = " + i + " j = " + j, !numberAreCloseEnough);
         }
      }

      assertTrue("Matrix Inversion works well", true);
   }


   @Test
   public void testPseudoInvert()
   {
      System.out.println("");
      System.out.println("Matrix Pseudo Inversion test");
      fillMatrix(rowDimension, columnDimension);
      displayMatrix(matrix, "matrix");

      Matrix matrixMultiplicationResult = new Matrix(rowDimension, columnDimension);

      // Invert Matrix
      matrixInverted = MathTools.pseudoinverse(matrix);
      displayMatrix(matrixInverted, "matrixInverted");

      // Matrix * Matrix inverted * Matrix should be equal to Matrix
      matrixMultiplicationResult = matrix.times(matrixInverted).times(matrix);    // with jama
      displayMatrix(matrixMultiplicationResult, "matrixMultiplicationResult");

      boolean numberAreCloseEnough = false;

      for (int i = 0; i < matrixMultiplicationResult.getRowDimension(); i++)
      {
         for (int j = 0; j < matrixMultiplicationResult.getColumnDimension(); j++)
         {
            // We lost precision during the multiplication so result can not be exactly the same
            if ((matrixMultiplicationResult.get(i, j) * 1.001 > matrix.get(i, j)) && (matrixMultiplicationResult.get(i, j) * 0.999 < matrix.get(i, j)))
               numberAreCloseEnough = true;
            else
               numberAreCloseEnough = false;

            assertFalse("Error at i = " + i + " j = " + j, !numberAreCloseEnough);
         }
      }

      assertTrue("Matrix Inversion works well", true);
   }

   public void fillMatrix(int row, int column)
   {
      matrix = new Matrix(row, column);
      matrixInverted = new Matrix(column, row);
      Random generator = new Random();
      for (int i = 0; i < matrix.getRowDimension(); i++)
      {
         for (int j = 0; j < matrix.getColumnDimension(); j++)
         {
            matrix.set(i, j, generator.nextDouble());
         }
      }
   }

   public void displayMatrix(Matrix matrix, String matrixName)
   {
      System.out.println("");
      System.out.println(matrixName);

      for (int i = 0; i < matrix.getRowDimension(); i++)
      {
         System.out.println("");

         for (int j = 0; j < matrix.getColumnDimension(); j++)
         {
            System.out.print(matrix.get(i, j) + " ");
         }
      }
   }

}
