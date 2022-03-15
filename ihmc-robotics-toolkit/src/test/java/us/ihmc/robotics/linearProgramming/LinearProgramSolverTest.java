package us.ihmc.robotics.linearProgramming;

import gnu.trove.list.array.TDoubleArrayList;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.linear.*;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import us.ihmc.euclid.tools.EuclidCoreRandomTools;
import us.ihmc.euclid.tools.EuclidCoreTools;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LinearProgramSolverTest
{

   @Test
   public void testAgainstApacheSolver_MaximumBoundingEllipses()
   {
      Random random = new Random(3920);
      int tests = 1000;

      for (int i = 0; i < tests; i++)
      {
         int dimensionality = 2 + random.nextInt(10);
         int constraints = 1 + random.nextInt(10);

         double rSq = 1.0 + 100.0 * random.nextDouble();
         double[] alphas = new double[dimensionality];
         for (int j = 0; j < alphas.length; j++)
         {
            alphas[j] = 1.0 + 30.0 * random.nextDouble();
         }

         double[] directionToMaximize = new double[dimensionality];
         for (int j = 0; j < dimensionality; j++)
         {
            directionToMaximize[j] = 0.1 + random.nextDouble();
         }

         List<double[]> gradients = new ArrayList<>();
         List<Double> constraintValues = new ArrayList<>();

         for (int j = 0; j < constraints; j++)
         {
            // compute initial point on curve
            double[] initialPoint = new double[dimensionality];
            double remainingPosValue = rSq;
            for (int k = 0; k < dimensionality - 1; k++)
            {
               double alphaXSq = EuclidCoreRandomTools.nextDouble(random, 0.0, remainingPosValue * 0.99 / alphas[k]);
               remainingPosValue -= alphaXSq;
               initialPoint[k] = Math.sqrt(alphaXSq / alphas[k]);
            }

            initialPoint[dimensionality - 1] = Math.sqrt(remainingPosValue / alphas[dimensionality - 1]);

            // compute gradient at this point
            double[] gradient = new double[dimensionality];
            for (int k = 0; k < dimensionality; k++)
            {
               gradient[k] = alphas[k] * initialPoint[k];
            }

            gradients.add(gradient);

            double bValue = 0.0;
            for (int k = 0; k < dimensionality; k++)
            {
               bValue += gradient[k] * initialPoint[k];
            }
            constraintValues.add(bValue);
         }

         // SOLVE WITH APACHE //
         SimplexSolver apacheSolver = new SimplexSolver();
         LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(directionToMaximize, 0.0);
         List<LinearConstraint> constraintList = new ArrayList<>();

         for (int j = 0; j < constraints; j++)
         {
            constraintList.add(new LinearConstraint(gradients.get(j), Relationship.LEQ, constraintValues.get(j)));
         }

         for (int j = 0; j < dimensionality; j++)
         {
            double[] nonNegativeConstraint = new double[dimensionality];
            nonNegativeConstraint[j] = 1.0;
            constraintList.add(new LinearConstraint(nonNegativeConstraint, Relationship.GEQ, 0.0));
         }

         PointValuePair apachePointValue = apacheSolver.optimize(new MaxIter(1000), objectiveFunction, new LinearConstraintSet(constraintList), GoalType.MAXIMIZE);
         double[] apacheSolution = apachePointValue.getPoint();

         // SOLVER WITH CUSTOM IMPL //
         DMatrixRMaj dictionary = new DMatrixRMaj(1 + constraints, 1 + dimensionality);
         for (int j = 0; j < dimensionality; j++)
         {
            dictionary.set(0, j + 1, directionToMaximize[j]);
         }

         for (int j = 0; j < constraints; j++)
         {
            dictionary.set(j + 1, 0, constraintValues.get(j));
            for (int k = 0; k < dimensionality; k++)
            {
               dictionary.set(j + 1, k + 1, - gradients.get(j)[k]);
            }
         }

         DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();
         solver.solveSimplex(dictionary);
         TDoubleArrayList customSolverSolution = solver.getSolution();

         for (int j = 0; j < apacheSolution.length; j++)
         {
            Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheSolution[j], customSolverSolution.get(j), 1e-5));
         }
      }
   }

   @Test
   public void testAgainstApacheSolver_MinimumBoundingEllipses()
   {
      Random random = new Random(3920);
      int tests = 1000;

      for (int i = 0; i < tests; i++)
      {
         int dimensionality = 2 + random.nextInt(10);
         int constraints = 1 + random.nextInt(10);

         double rSq = 1.0 + 100.0 * random.nextDouble();
         double[] alphas = new double[dimensionality];
         for (int j = 0; j < alphas.length; j++)
         {
            alphas[j] = 1.0 + 30.0 * random.nextDouble();
         }

         double[] directionToMinimize = new double[dimensionality];
         for (int j = 0; j < dimensionality; j++)
         {
            directionToMinimize[j] = 0.1 + random.nextDouble();
         }

         List<double[]> gradients = new ArrayList<>();
         List<Double> constraintValues = new ArrayList<>();

         for (int j = 0; j < constraints; j++)
         {
            // compute initial point on curve
            double[] initialPoint = new double[dimensionality];
            double remainingPosValue = rSq;
            for (int k = 0; k < dimensionality - 1; k++)
            {
               double alphaXSq = EuclidCoreRandomTools.nextDouble(random, 0.0, remainingPosValue * 0.99 / alphas[k]);
               remainingPosValue -= alphaXSq;
               initialPoint[k] = Math.sqrt(alphaXSq / alphas[k]);
            }

            initialPoint[dimensionality - 1] = Math.sqrt(remainingPosValue / alphas[dimensionality - 1]);

            // compute gradient at this point
            double[] gradient = new double[dimensionality];
            for (int k = 0; k < dimensionality; k++)
            {
               gradient[k] = alphas[k] * initialPoint[k];
            }

            gradients.add(gradient);

            double bValue = 0.0;
            for (int k = 0; k < dimensionality; k++)
            {
               bValue += gradient[k] * initialPoint[k];
            }
            constraintValues.add(bValue);
         }

         // SOLVE WITH APACHE //
         SimplexSolver apacheSolver = new SimplexSolver();
         LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(directionToMinimize, 0.0);
         List<LinearConstraint> constraintList = new ArrayList<>();

         for (int j = 0; j < constraints; j++)
         {
            constraintList.add(new LinearConstraint(gradients.get(j), Relationship.GEQ, constraintValues.get(j)));
         }

         for (int j = 0; j < dimensionality; j++)
         {
            double[] nonNegativeConstraint = new double[dimensionality];
            nonNegativeConstraint[j] = 1.0;
            constraintList.add(new LinearConstraint(nonNegativeConstraint, Relationship.GEQ, 0.0));
         }

         PointValuePair apachePointValue = apacheSolver.optimize(new MaxIter(1000), objectiveFunction, new LinearConstraintSet(constraintList), GoalType.MINIMIZE);
         double[] apacheSolution = apachePointValue.getPoint();

         // SOLVER WITH CUSTOM IMPL //
         DMatrixRMaj dictionary = new DMatrixRMaj(1 + constraints, 1 + dimensionality);
         for (int j = 0; j < dimensionality; j++)
         {
            dictionary.set(0, j + 1, -directionToMinimize[j]);
         }

         for (int j = 0; j < constraints; j++)
         {
            dictionary.set(j + 1, 0, - constraintValues.get(j));
            for (int k = 0; k < dimensionality; k++)
            {
               dictionary.set(j + 1, k + 1, gradients.get(j)[k]);
            }
         }

         DictionaryFormLinearProgramSolver solver = new DictionaryFormLinearProgramSolver();
         solver.solveSimplex(dictionary);
         TDoubleArrayList customSolverSolution = solver.getSolution();

         for (int j = 0; j < apacheSolution.length; j++)
         {
            Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheSolution[j], customSolverSolution.get(j), 1e-5));
         }
      }
   }
}
