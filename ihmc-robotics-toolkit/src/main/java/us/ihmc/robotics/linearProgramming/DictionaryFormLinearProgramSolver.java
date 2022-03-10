package us.ihmc.robotics.linearProgramming;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.math3.util.Precision;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.MatrixType;
import us.ihmc.commons.time.Stopwatch;
import us.ihmc.log.LogTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.robotics.linearAlgebra.careSolvers.MatrixToolsLocal;

import java.util.Arrays;

/**
 * Solves a dictionary form LP using the criss-cross or simplex methods.
 * Simplex implementation borrows from org.apache.commons.math3.optim.linear.SimplexSolver and doi.org/10.3929/ethz-b-000426221 (ch. 4)
 */
public class DictionaryFormLinearProgramSolver
{
   private static final boolean debug = false;
   private static final int maxVariables = 100;
   private static final int maxIterations = 1000;
   private static final int nullIndex = -1;
   private static final double epsilon = 1e-6;

   private DMatrixRMaj dictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);
   private DMatrixRMaj tempDictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);

   private final TIntArrayList basisIndices = new TIntArrayList(maxVariables);
   private final TIntArrayList nonBasisIndices = new TIntArrayList(maxVariables);
   private final TDoubleArrayList solution = new TDoubleArrayList(maxVariables);


   private final Stopwatch timer = new Stopwatch();
   private final SolverStatistics phase1Statistics = new SolverStatistics();
   private final SolverStatistics phase2Statistics = new SolverStatistics();

   enum Phase { PHASE_I, PHASE_II}

   class SolverStatistics
   {
      double solveTime;
      int iterations;
      boolean foundSolution;

      void clear()
      {
         solveTime = Double.NaN;
         iterations = 0;
         foundSolution = false;
      }
   }

   public void solveSimplex(DMatrixRMaj startingDictionary)
   {
      if (startingDictionary.getNumCols() > maxVariables)
      {
         throw new IllegalArgumentException("Simplex method has a maximum of " + maxVariables + " decision variables, " + startingDictionary.getNumCols() + " provided.");
      }

      phase1Statistics.clear();
      phase2Statistics.clear();

      /* Phase I: solve augmented LP */

      // setup augmented dictionary
      dictionary.reshape(startingDictionary.getNumRows(), 1 + startingDictionary.getNumCols());
      Arrays.fill(dictionary.getData(), 0.0);

      int auxiliaryColumn = dictionary.getNumCols() - 1;
      dictionary.set(0, auxiliaryColumn, -1.0);
      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         dictionary.set(i, auxiliaryColumn, 1.0);
      }

      MatrixTools.setMatrixBlock(dictionary, 0, 0, startingDictionary, 1, 0, startingDictionary.getNumRows() - 1, startingDictionary.getNumCols(), 1.0);

      // perform pivot to make primal feasible


      /* Phase II: optimize feasible dictionary */
      performSimplexPhase(dictionary, Phase.PHASE_II);
   }

   private void setupIndexLists(DMatrixRMaj dictionary)
   {
      basisIndices.reset();
      nonBasisIndices.reset();

      for (int i = 0; i < dictionary.getNumCols(); i++)
      {
         nonBasisIndices.add(i);
      }

      basisIndices.add(-1);
      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         basisIndices.add((i - 1) + nonBasisIndices.size());
      }
   }

   /* package private for testing */
   void performSimplexPhase(DMatrixRMaj startingDictionary, Phase phase)
   {
      SolverStatistics statistics = phase == Phase.PHASE_I ? phase1Statistics : phase2Statistics;

      timer.reset();
      dictionary.set(startingDictionary);
      tempDictionary.set(startingDictionary);
      setupIndexLists(startingDictionary);

      while (true)
      {
         if (maxIterations > 0 && statistics.iterations++ > maxIterations)
         {
            statistics.foundSolution = false;
            break;
         }

         if (isSimplexOptimal())
         {
            statistics.foundSolution = true;
            break;
         }

         int s = computeSimplexPivotColumn();

         int r = computeSimplexPivotRow(s);
         if (r == nullIndex)
         {
            statistics.foundSolution = false;
            break;
         }

         if (debug)
         {
            System.out.println("Dictionary:\n");
            System.out.println(dictionary);
            System.out.println("Pivoting on (" + r + "," + s + ")\n");
         }

         performPivot(r, s);
      }

      packSolution();
      statistics.solveTime = timer.lapElapsed();
   }

   private void packSolution()
   {
      solution.reset();
      int capacity = basisIndices.size() + nonBasisIndices.size();
      for (int i = 0; i < capacity; i++)
      {
         solution.add(0.0);
      }

      for (int i = 1; i < basisIndices.size(); i++)
      {
         solution.set(basisIndices.get(i) - 1, dictionary.get(i, 0));
      }
   }

   /* Checks optimality assuming feasibility, so only the objective row needs to be checked */
   private boolean isSimplexOptimal()
   {
      for (int j = 1; j < dictionary.getNumCols(); j++)
      {
         if (dictionary.get(0, j) > epsilon)
         {
            return false;
         }
      }

      return true;
   }

   // use Bland pivot rule, which finds the positive objective row entry with the lowest corresponding variable index
   private int computeSimplexPivotColumn()
   {
      int minimumEntryIndex = Integer.MAX_VALUE;
      int column = nullIndex;

      for (int j = 1; j < dictionary.getNumCols(); j++)
      {
         double entry = dictionary.get(0, j);
         if (entry < epsilon)
         {
            continue;
         }

         int index = nonBasisIndices.get(j);
         if (index < minimumEntryIndex)
         {
            minimumEntryIndex = index;
            column = j;
         }
      }

      return column;
   }

   private final TIntArrayList minRatioIndices = new TIntArrayList(maxVariables + 1);

   private int computeSimplexPivotRow(int column)
   {
      double minRatio = Double.MAX_VALUE;
      minRatioIndices.reset();

      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         double d_ig = dictionary.get(i, 0);
         double d_is = dictionary.get(i, column);

         if (d_is > -epsilon)
         {
            continue;
         }

         double ratio = Math.abs(d_ig / d_is);
         int cmp = Precision.compareTo(ratio, minRatio, epsilon);

         if (cmp == 0)
         {
            minRatioIndices.add(i);
         }
         else if (cmp < 0)
         {
            minRatioIndices.reset();
            minRatioIndices.add(i);
            minRatio = ratio;
         }
      }

      if (minRatioIndices.isEmpty())
      {
         return nullIndex;
      }
      else if (minRatioIndices.size() > 1)
      {
         // (from apache impl...)
         // apply Bland's rule to prevent cycling:
         //    take the row for which the corresponding basic variable has the smallest index

         int minRowIndex = Integer.MAX_VALUE;
         int minRow = nullIndex;

         for (int i = 0; i < minRatioIndices.size(); i++)
         {
            int variableIndex = basisIndices.get(minRatioIndices.get(i));
            if (variableIndex < minRowIndex)
            {
               minRowIndex = variableIndex;
               minRow = minRatioIndices.get(i);
            }
         }

         return minRow;
      }
      else
      {
         return minRatioIndices.get(0);
      }
   }

   // r = basisPivot
   // s = nonBasisPivot
   private void performPivot(int r, int s)
   {
      /* Pivot is performed on temp dictionary */
      for (int i = 0; i < dictionary.getNumRows(); i++)
      {
         for (int j = 0; j < dictionary.getNumCols(); j++)
         {
            if (i == r && j == s)
            {
               tempDictionary.set(i, j, 1.0 / dictionary.get(r, s));
            }
            else if (i == r)
            {
               tempDictionary.set(i, j, -dictionary.get(r, j) / (dictionary.get(r, s)));
            }
            else if (j == s)
            {
               tempDictionary.set(i, j, dictionary.get(i, s) / (dictionary.get(r, s)));
            }
            else
            {
               tempDictionary.set(i, j, dictionary.get(i, j) - dictionary.get(i, s) * dictionary.get(r, j) / (dictionary.get(r, s)));
            }
         }
      }

      /* Update index mapping */
      int originalBasisIndex = basisIndices.get(r);
      int originalNonBasisIndex = nonBasisIndices.get(s);
      basisIndices.set(r, originalNonBasisIndex);
      nonBasisIndices.set(s, originalBasisIndex);

      /* Swap dictionaries to avoid calling .set() */
      DMatrixRMaj previousDictionary = dictionary;
      dictionary = tempDictionary;
      tempDictionary = previousDictionary;
   }

   public TDoubleArrayList getSolution()
   {
      return solution;
   }

   public void printSolution()
   {
      LogTools.info("Solution stats");
      System.out.println("Status: " + status);
      System.out.println("Solver time (sec): " + solveTimeSeconds);
      System.out.println("Iterations: " + iterations);

      if (status != SolverStatus.OPTIMAL)
      {
         return;
      }

      System.out.println("Solution:");
      for (int i = 0; i < solution.size(); i++)
      {
         System.out.println("\t " + solution.get(i));
      }
   }

   public static void main(String[] args)
   {
//      System.out.println(Precision.compareTo(-1.0, 0d, epsilon));
//      System.out.println(Precision.compareTo(1.0, 0d, epsilon));
//      System.out.println(Precision.compareTo(-1e-10, 0d, epsilon));
//      System.out.println(Precision.compareTo(1e-10, 0d, epsilon));

      double[] x = new double[2];
      int i = 0;
      x[i++] = 1.0;
      x[i++] = 2.0;
      System.out.println(i);

      for (int j = 0; j < x.length; j++)
      {
         System.out.println(x[j]);
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////
   ///////////////////////// CRISS CROSS METHOD, IN PROGRESS ///////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////

   //   public void solveCrissCross(DMatrixRMaj startingDictionary)
   //   {
   //      dictionary.set(startingDictionary);
   //      tempDictionary.set(startingDictionary);
   //
   //      setupIndexLists(startingDictionary);
   //
   //      iterations = 0;
   //      status = SolverStatus.UNKNOWN;
   //      timer.reset();
   //
   //      while (true)
   //      {
   //         if (maxIterations > 0 && iterations++ > maxIterations)
   //         {
   //            status = SolverStatus.MAX_ITERATIONS;
   //            break;
   //         }
   //
   //         int candidateBasisPivot = findFirstNegativeColumnEntry(0);
   //         int candidateNonBasisPivot = findFirstPositiveRowEntry(0);
   //
   //         int basisPivot, nonBasisPivot;
   //         if (candidateBasisPivot == nullIndex && candidateNonBasisPivot == nullIndex)
   //         {
   //            status = SolverStatus.OPTIMAL;
   //            break;
   //         }
   //         else if (candidateBasisPivot != nullIndex && (candidateNonBasisPivot == nullIndex || candidateBasisPivot < candidateNonBasisPivot))
   //         {
   //            basisPivot = candidateBasisPivot;
   //            nonBasisPivot = findFirstPositiveRowEntry(basisPivot);
   //
   //            if (nonBasisPivot == nullIndex)
   //            {
   //               status = SolverStatus.INCONSISTENT;
   //               break;
   //            }
   //         }
   //         else
   //         {
   //            nonBasisPivot = candidateNonBasisPivot;
   //            basisPivot = findFirstNegativeColumnEntry(nonBasisPivot);
   //
   //            if (basisPivot == nullIndex)
   //            {
   //               status = SolverStatus.DUAL_INCONSISTENT;
   //               break;
   //            }
   //         }
   //
   //         if (debug)
   //         {
   //            System.out.println(dictionary);
   //            System.out.println("Pivoting on (" + basisPivot + "," + nonBasisPivot + ")\n");
   //         }
   //
   //         performPivot(basisPivot, nonBasisPivot);
   //      }
   //
   //      solveTimeSeconds1 = timer.totalElapsed();
   //      packSolution();
   //   }

//   private int findFirstNegativeColumnEntry(int column)
//   {
//      for (int i = 1; i < dictionary.getNumRows(); i++)
//      {
//         double d_ig = dictionary.get(i, column);
//         if (d_ig < -epsilon)
//         {
//            return i;
//         }
//      }
//
//      return nullIndex;
//   }
//
//   private int findFirstPositiveRowEntry(int row)
//   {
//      for (int j = 1; j < dictionary.getNumCols(); j++)
//      {
//         double d_fj = dictionary.get(row, j);
//         if (d_fj > epsilon)
//         {
//            return j;
//         }
//      }
//
//      return nullIndex;
//   }

}
