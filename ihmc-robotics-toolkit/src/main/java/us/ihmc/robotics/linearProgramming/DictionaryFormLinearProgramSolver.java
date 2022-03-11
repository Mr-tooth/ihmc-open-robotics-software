package us.ihmc.robotics.linearProgramming;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.math3.util.Precision;
import org.ejml.data.DMatrixRMaj;
import us.ihmc.commons.time.Stopwatch;
import us.ihmc.euclid.tools.EuclidCoreIOTools;
import us.ihmc.euclid.tools.EuclidCoreTools;
import us.ihmc.matrixlib.MatrixTools;

import java.util.Arrays;

/**
 * Solves a dictionary form LP using the criss-cross or simplex methods.
 * Simplex implementation borrows from org.apache.commons.math3.optim.linear.SimplexSolver and doi.org/10.3929/ethz-b-000426221 (ch. 4)
 */
public class DictionaryFormLinearProgramSolver
{
   private static final boolean debug = true;

   private static final int maxVariables = 200;
   private static final int maxIterations = 1000;
   private static final int nullMatrixIndex = -1;
   private static final int rhsVariableLexicalIndex = 0;
   private static final int auxVariableLexicalIndex = -1;
   private static final int objectiveLexicalIndex = -2;
   private static final int auxObjectiveLexicalIndex = -3;
   private static final double epsilon = 1e-6;

   private DMatrixRMaj dictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);
   private DMatrixRMaj tempDictionary = new DMatrixRMaj(maxVariables + 1, maxVariables + 1);

   private final TIntArrayList basisIndices = new TIntArrayList(maxVariables);
   private final TIntArrayList nonBasisIndices = new TIntArrayList(maxVariables);
   private final TDoubleArrayList solution = new TDoubleArrayList(maxVariables);

   private final Stopwatch timer = new Stopwatch();
   private final SolverStatistics phase1Statistics = new SolverStatistics();
   private final SolverStatistics phase2Statistics = new SolverStatistics();
   private final SolverStatistics crissCrossStatistics = new SolverStatistics();

   enum SimplexPhase
   {
      PHASE_I, PHASE_II;

      int objectiveSize()
      {
         return this == PHASE_I ? 2 : 1;
      }
   }

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

      @Override
      public String toString()
      {
         return "Solve time: " + solveTime + "\nIterations: " + iterations + "\nFound solution: " + foundSolution + "\n";
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
      dictionary.reshape(startingDictionary.getNumRows() + 1, 1 + startingDictionary.getNumCols());
      tempDictionary.reshape(startingDictionary.getNumRows() + 1, 1 + startingDictionary.getNumCols());
      setupSimplexIndexLists(dictionary);

      Arrays.fill(dictionary.getData(), 0.0);

      // TODO
      // - check if already feasible and if so skip phase I
      // - look more at logic setting up phase I dictionary

      int auxiliaryColumn = dictionary.getNumCols() - 1;
      dictionary.set(0, auxiliaryColumn, -1.0);
      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         dictionary.set(i, auxiliaryColumn, 1.0);
      }

      MatrixTools.setMatrixBlock(dictionary, 1, 0, startingDictionary, 0, 0, startingDictionary.getNumRows(), startingDictionary.getNumCols(), 1.0);
      if (debug)
      {
         printDictionary("Starting phase I dictionary");
      }

      // perform pivot to make primal feasible
      int r = computeInitialSimplexPhaseIPivotRow();
      performPivot(r, dictionary.getNumCols() - 1);

      if (debug)
      {
         System.out.println();
         printDictionary("Starting phase I dictionary after pivot");
      }

      // solve auxiliary problem
      performSimplexPhase(SimplexPhase.PHASE_I);

      if (!phase1Statistics.foundSolution || dictionary.get(0, 0) < -epsilon)
      {
         phase1Statistics.foundSolution = false;
         return;
      }

      boolean auxiliaryVariableInBasis = basisIndices.contains(auxVariableLexicalIndex);
      if (auxiliaryVariableInBasis)
      {
         int pivotColumn = findLargestMagnitudeObjectiveColumn();
         int pivotRow = basisIndices.indexOf(auxVariableLexicalIndex);
         performPivot(pivotRow, pivotColumn);
      }

      tempDictionary.reshape(dictionary.getNumRows() - 1, dictionary.getNumCols() - 1);
      MatrixTools.setMatrixBlock(tempDictionary, 0, 0, dictionary, 1, 0, tempDictionary.getNumRows(), tempDictionary.getNumCols(), 1.0);
      basisIndices.remove(auxObjectiveLexicalIndex);
      nonBasisIndices.remove(auxObjectiveLexicalIndex);

      /* Phase II: optimize feasible dictionary */
      performSimplexPhase(SimplexPhase.PHASE_II);

      packSolution();
   }

   void setupStandardIndexLists(DMatrixRMaj dictionary)
   {
      basisIndices.reset();
      nonBasisIndices.reset();

      nonBasisIndices.add(rhsVariableLexicalIndex);
      basisIndices.add(rhsVariableLexicalIndex);

      int lexicalIndex = 1;

      for (int i = 1; i < dictionary.getNumCols(); i++)
      {
         nonBasisIndices.add(lexicalIndex++);
      }

      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         basisIndices.add(lexicalIndex++);
      }
   }

   void setupSimplexIndexLists(DMatrixRMaj dictionary)
   {
      basisIndices.reset();
      nonBasisIndices.reset();

      int lexicalIndex = 1;

      nonBasisIndices.add(rhsVariableLexicalIndex);
      for (int i = 1; i < dictionary.getNumCols() - 1; i++)
      {
         nonBasisIndices.add(lexicalIndex++);
      }
      nonBasisIndices.add(auxVariableLexicalIndex);

      basisIndices.add(auxObjectiveLexicalIndex);
      basisIndices.add(objectiveLexicalIndex);
      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         basisIndices.add(lexicalIndex++);
      }
   }

   /* package private for testing */
   void performSimplexPhase(SimplexPhase phase)
   {
      SolverStatistics statistics = phase == SimplexPhase.PHASE_I ? phase1Statistics : phase2Statistics;

      timer.reset();
      tempDictionary.set(dictionary);

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
         int r = computeSimplexPivotRow(s, phase);

         if (r == nullMatrixIndex)
         {
            statistics.foundSolution = false;
            break;
         }

         if (debug)
         {
            System.out.println("Pivoting on (" + r + "," + s + ")\n");
         }

         performPivot(r, s);
      }

      statistics.solveTime = timer.lapElapsed();
   }

   private int computeInitialSimplexPhaseIPivotRow()
   {
      double minimumEntry = Double.MAX_VALUE;
      int minimumEntryRow = nullMatrixIndex;

      for (int i = SimplexPhase.PHASE_I.objectiveSize(); i < dictionary.getNumRows(); i++)
      {
         double entry = dictionary.get(i, 0);
         if (entry < minimumEntry)
         {
            minimumEntry = entry;
            minimumEntryRow = i;
         }
      }

      return minimumEntryRow;
   }

   private int findLargestMagnitudeObjectiveColumn()
   {
      double largestMagnitudeValue = 0.0;
      int column = nullMatrixIndex;

      for (int j = 1; j < dictionary.getNumCols(); j++)
      {
         double value = Math.abs(dictionary.get(0, j));
         if (value > largestMagnitudeValue)
         {
            largestMagnitudeValue = value;
            column = j;
         }
      }

      return column;
   }

   void packSolution()
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

   // use Bland pivot rule, which finds the positive objective row entry with the lowest corresponding variable (lexical) index
   private int computeSimplexPivotColumn()
   {
      int minimumEntryIndex = Integer.MAX_VALUE;
      int column = nullMatrixIndex;

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

   private int computeSimplexPivotRow(int column, SimplexPhase phase)
   {
      double minRatio = Double.MAX_VALUE;
      minRatioIndices.reset();

      for (int i = phase.objectiveSize(); i < dictionary.getNumRows(); i++)
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
         return nullMatrixIndex;
      }
      else if (minRatioIndices.size() > 1)
      {
         // (from apache impl...)
         // apply Bland's rule to prevent cycling:
         //    take the row for which the corresponding basic variable has the smallest index

         int minRowIndex = Integer.MAX_VALUE;
         int minRow = nullMatrixIndex;

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
      System.out.println("Solution:");
      for (int i = 0; i < solution.size(); i++)
      {
         System.out.println("\t " + solution.get(i));
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////// CRISS CROSS METHOD /////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////

   public void solveCrissCross(DMatrixRMaj startingDictionary)
   {
      crissCrossStatistics.clear();
      timer.reset();

      dictionary.set(startingDictionary);
      tempDictionary.set(startingDictionary);
      setupStandardIndexLists(startingDictionary);

      while (true)
      {
         if (maxIterations > 0 && crissCrossStatistics.iterations++ > maxIterations)
         {
            break;
         }

         int candidateBasisPivot = findNegativeColumnEntryWithBlandRule(0);
         int candidateNonBasisPivot = findPositiveRowEntryWithBlandRule(0);

         int basisPivot, nonBasisPivot;
         if (candidateBasisPivot == nullMatrixIndex && candidateNonBasisPivot == nullMatrixIndex)
         {
            crissCrossStatistics.foundSolution = true;
            break;
         }
         else if (candidateBasisPivot != nullMatrixIndex && (candidateNonBasisPivot == nullMatrixIndex
                                                             || basisIndices.get(candidateBasisPivot) < nonBasisIndices.get(candidateNonBasisPivot)))
         {
            basisPivot = candidateBasisPivot;
            nonBasisPivot = findPositiveRowEntryWithBlandRule(basisPivot);

            if (nonBasisPivot == nullMatrixIndex)
            {
               // inconsistent
               break;
            }
         }
         else
         {
            nonBasisPivot = candidateNonBasisPivot;
            basisPivot = findNegativeColumnEntryWithBlandRule(nonBasisPivot);

            if (basisPivot == nullMatrixIndex)
            {
               // dual inconsistent
               break;
            }
         }

         if (debug)
         {
            System.out.println(dictionary);
            System.out.println("Pivoting on (" + basisPivot + "," + nonBasisPivot + ")\n");
         }

         performPivot(basisPivot, nonBasisPivot);
      }

      crissCrossStatistics.solveTime = timer.totalElapsed();
      packSolution();
   }

   private int findNegativeColumnEntryWithBlandRule(int column)
   {
      int minLexicalIndex = Integer.MAX_VALUE;
      int row = nullMatrixIndex;

      for (int i = 1; i < dictionary.getNumRows(); i++)
      {
         double d_ig = dictionary.get(i, column);
         int lexicalIndex = basisIndices.get(i);

         if (d_ig < -epsilon && lexicalIndex < minLexicalIndex)
         {
            minLexicalIndex = lexicalIndex;
            row = i;
         }
      }

      return row;
   }

   private int findPositiveRowEntryWithBlandRule(int row)
   {
      int minLexicalIndex = Integer.MAX_VALUE;
      int column = nullMatrixIndex;

      for (int j = 1; j < dictionary.getNumCols(); j++)
      {
         double d_fj = dictionary.get(row, j);
         int lexicalIndex = nonBasisIndices.get(j);

         if (d_fj > epsilon && lexicalIndex < minLexicalIndex)
         {
            minLexicalIndex = lexicalIndex;
            column = j;
         }
      }

      return column;
   }

   public SolverStatistics getPhase1Statistics()
   {
      return phase1Statistics;
   }

   public SolverStatistics getPhase2Statistics()
   {
      return phase2Statistics;
   }

   public SolverStatistics getCrissCrossStatistics()
   {
      return crissCrossStatistics;
   }

   /* for testing */
   DMatrixRMaj getDictionary()
   {
      return dictionary;
   }

   private static final String entryFormat = EuclidCoreIOTools.getStringFormat(6, 3);
   private static final String entrySeparator = "\t\t";

   private void printDictionary(String label)
   {
      System.out.println(label);

      for (int row = -1; row < dictionary.getNumRows(); row++)
      {
         for (int column = -1; column < dictionary.getNumCols(); column++)
         {
            String entry = "";
            if (row == -1 && column == -1)
            {

            }
            else if (row == -1)
            {
               entry = formatIndex(nonBasisIndices.get(column)) + "\t";
            }
            else if (column == -1)
            {
               entry = formatIndex(basisIndices.get(row));
            }
            else
            {
               entry = String.format(entryFormat, dictionary.get(row, column));
            }

            System.out.print(entry + entrySeparator);
         }
         System.out.println();
      }
   }

   private static String formatIndex(int index)
   {
      if (index == rhsVariableLexicalIndex)
      {
         return "g";
      }
      else if (index == auxVariableLexicalIndex)
      {
         return "a";
      }
      else if (index == objectiveLexicalIndex)
      {
         return "f";
      }
      else if (index == auxObjectiveLexicalIndex)
      {
         return "f'";
      }
      else
      {
         return Integer.toString(index);
      }
   }
}

