package us.ihmc.robotics.linearProgramming;

import gnu.trove.list.array.TDoubleArrayList;
import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Test;

public class DictionaryFormLinearProgramSolverTest
{
   @Test
   public void testSimplexPhaseII_0()
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 3.0, 4.0, 2.0,
                                                            4.0, -2.0, 0.0, 0.0,
                                                            8.0, -1.0, 0.0, -2.0,
                                                            6.0, 0.0, -3.0, -1.0});
      dictionary.reshape(4, 4);

      solver.doSimplexPhaseII(dictionary);
      solver.printSolution();
   }

   @Test
   public void testSimplexPhaseII_1()
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, -2.0, 1.0,
                                                            0.0, -2.0, 1.0, -1.0,
                                                            0.0, -3.0, -1.0, -1.0,
                                                            0.0, 5.0, -3.0, 2.0});
      dictionary.reshape(4, 4);

      solver.doSimplexPhaseII(dictionary);
      solver.printSolution();
   }

   @Test
   public void testSimplexPhaseII_2()
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      DMatrixRMaj dictionary = new DMatrixRMaj(new double[]{0.0, 1.0, 2.1,
                                                            4.0, -1.0, 0.0,
                                                            2.0, 0.0, -1.0,
                                                            6.0, -1.0, -2.0});
      dictionary.reshape(4, 3);

      solver.doSimplexPhaseII(dictionary);
      solver.printSolution();
   }

   @Test
   public void testSimplexPhaseII_PiecewiseLinearQuarterCircle()
   {
      DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();

      int numberOfLines = 30;
      DMatrixRMaj dictionary = new DMatrixRMaj(numberOfLines + 1, 3);

      double optDirectionX1 = 1.0;
      double optDirectionX2 = 1.0;
      dictionary.set(0, 1, optDirectionX1);
      dictionary.set(0, 2, optDirectionX2);

      for (int i = 0; i < numberOfLines; i++)
      {
         double theta = 0.5 * Math.PI * i / (numberOfLines - 1);
         dictionary.set(i + 1, 0, 1.0);
         dictionary.set(i + 1, 1, -Math.cos(theta));
         dictionary.set(i + 1, 2, -Math.sin(theta));
      }

      solver.doSimplexPhaseII(dictionary);
      solver.printSolution();
   }
}
