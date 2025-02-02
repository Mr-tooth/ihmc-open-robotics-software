package us.ihmc.robotics.dataStructures;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.commons.MathTools;

/**
 * <p>
 * Polynomial Function with real coefficients. Immuatable.
 * </p>
 *
 * Has been replaced with {@link us.ihmc.robotics.math.trajectories.core.Polynomial}
 *
 * @author IHMC Biped Team
 * @version 1.0
 */
@Deprecated
public class ObsoletePolynomial
{
   private final double[] coefficients;
   private final double[] derivativeCoefficients;
   private final double[] doubleDerivativeCoefficients;
   private DMatrixRMaj constraintMatrix;
   private DMatrixRMaj constraintVector;
   private DMatrixRMaj coefficientVector;

   public ObsoletePolynomial(double constant)
   {
      this(new double[] {constant});
   }

   public ObsoletePolynomial(double coefficient1, double constant)
   {
      this(new double[] {coefficient1, constant});
   }

   public ObsoletePolynomial(double coefficient2, double coefficient1, double constant)
   {
      this(new double[] {coefficient2, coefficient1, constant});
   }

   public ObsoletePolynomial(double coefficient3, double coefficient2, double coefficient1, double constant)
   {
      this(new double[] {coefficient3, coefficient2, coefficient1, constant});
   }

   public ObsoletePolynomial(double[] coefficientsHighOrderFirst)
   {
      if ((coefficientsHighOrderFirst == null) || (coefficientsHighOrderFirst.length < 1))
         throw new RuntimeException("(coefficientsHighOrderFirst == null) || (coefficientsHighOrderFirst.length < 1)");

      int coefficientsToSkip = 0;
      if ((coefficientsHighOrderFirst.length > 1) && (Math.abs(coefficientsHighOrderFirst[0]) < 1e-7))
      {
         // Skip any zero coefficients, that is any whose coefficient is < 1e-7 times the max coefficient.
         double maxCoefficient = findMaxAbsoluteCoefficient(coefficientsHighOrderFirst);

         for (coefficientsToSkip = 0; coefficientsToSkip < coefficientsHighOrderFirst.length; coefficientsToSkip++)
         {
            if (Math.abs(coefficientsHighOrderFirst[coefficientsToSkip]) > 1e-7 * maxCoefficient)
            {
               break;
            }
         }
      }

      if (coefficientsHighOrderFirst.length - coefficientsToSkip == 0) // Skipped them all since was {0.0, 0.0, ..., 0.0};
      {
         coefficientsToSkip = coefficientsToSkip - 1;
      }

      this.coefficients = new double[coefficientsHighOrderFirst.length - coefficientsToSkip];

      for (int i = coefficientsToSkip; i < coefficientsHighOrderFirst.length; i++)
      {
         this.coefficients[i - coefficientsToSkip] = coefficientsHighOrderFirst[i];
      }

      if (this.coefficients.length == 1)
      {
         this.derivativeCoefficients = new double[] {0.0};
      }
      else
      {
         this.derivativeCoefficients = new double[this.coefficients.length - 1];
         int length = this.coefficients.length;
         for (int i = 0; i < this.coefficients.length - 1; i++)
         {
            this.derivativeCoefficients[i] = (length - i - 1) * this.coefficients[i];
         }
      }

      if (this.coefficients.length < 3)
      {
         this.doubleDerivativeCoefficients = new double[] {0.0};
      }
      else
      {
         this.doubleDerivativeCoefficients = new double[this.coefficients.length - 2];
         int length = this.coefficients.length;
         for (int i = 0; i < this.coefficients.length - 2; i++)
         {
            this.doubleDerivativeCoefficients[i] = (length - i - 1) * (length - i - 2) * this.coefficients[i];
         }
      }

      if (coefficients == null)
         throw new RuntimeException("(coefficients == null)");

      if (coefficients.length < 1)
      {
         System.err.println("coefficientsHighOrderFirst[0] = " + coefficientsHighOrderFirst[0]);
         System.err.println("coefficientsHighOrderFirst[1] = " + coefficientsHighOrderFirst[1]);

         throw new RuntimeException("(coefficients.length < 1)");

      }

   }

   private double findMaxAbsoluteCoefficient(double[] coefficients)
   {
      double maxCoefficient = 0.0;
      for (double coefficient : coefficients)
      {
         if (Math.abs(coefficient) > Math.abs(maxCoefficient))
         {
            maxCoefficient = Math.abs(coefficient);
         }
      }

      return maxCoefficient;
   }

   public static ObsoletePolynomial constructFromComplexPairRoot(ComplexNumber oneComplexRoot)
   {
      double a = oneComplexRoot.real();
      double b = oneComplexRoot.imag();

      return new ObsoletePolynomial(new double[] {1.0, -2.0 * a, a * a + b * b});
   }

   public static ObsoletePolynomial constructFromRealRoot(double realRoot)
   {
      return new ObsoletePolynomial(new double[] {1.0, -realRoot});
   }

   public static ObsoletePolynomial constructFromScaleFactorAndRoots(double scaleFactor, double[] realRoots, ComplexNumber[] complexRootPairs)
   {
      ObsoletePolynomial scalePolynomial = new ObsoletePolynomial(new double[] {scaleFactor});

      if (complexRootPairs == null)
         complexRootPairs = new ComplexNumber[] {};
      if (realRoots == null)
         realRoots = new double[] {};

      ObsoletePolynomial[] complexRootPolynomials = new ObsoletePolynomial[complexRootPairs.length];
      ObsoletePolynomial[] realRootPolynomials = new ObsoletePolynomial[realRoots.length];

      for (int i = 0; i < realRoots.length; i++)
      {
         realRootPolynomials[i] = ObsoletePolynomial.constructFromRealRoot(realRoots[i]);
      }

      for (int i = 0; i < complexRootPairs.length; i++)
      {
         complexRootPolynomials[i] = ObsoletePolynomial.constructFromComplexPairRoot(complexRootPairs[i]);
      }

      ObsoletePolynomial polynomialToReturn = scalePolynomial;

      for (ObsoletePolynomial polynomial : realRootPolynomials)
      {
         polynomialToReturn = polynomialToReturn.times(polynomial);
      }

      for (ObsoletePolynomial polynomial : complexRootPolynomials)
      {
         polynomialToReturn = polynomialToReturn.times(polynomial);
      }

      return polynomialToReturn;
   }

   public double evaluate(double input)
   {
      double x_n = 1.0;
      double ret = 0.0;

      for (int i = coefficients.length - 1; i >= 0; i--)
      {
         double coefficient = coefficients[i];
         ret = ret + coefficient * x_n;
         x_n = x_n * input;
      }

      return ret;
   }

   public double evaluateDerivative(double input)
   {
      double x_n = 1.0;
      double ret = 0.0;

      for (int i = coefficients.length - 2; i >= 0; i--)
      {
         double coefficient = derivativeCoefficients[i];
         ret = ret + coefficient * x_n;
         x_n = x_n * input;
      }

      return ret;
   }

   public double evaluateDoubleDerivative(double input)
   {
      double x_n = 1.0;
      double ret = 0.0;

      for (int i = coefficients.length - 3; i >= 0; i--)
      {
         double coefficient = doubleDerivativeCoefficients[i];
         ret = ret + coefficient * x_n;
         x_n = x_n * input;
      }

      return ret;
   }

   public ComplexNumber evaluate(ComplexNumber input)
   {
      ComplexNumber x_n = new ComplexNumber(1.0, 0.0);
      ComplexNumber ret = new ComplexNumber(0.0, 0.0);

      for (int i = coefficients.length - 1; i >= 0; i--)
      {
         double coefficient = coefficients[i];
         ret = ret.plus(x_n.times(coefficient));
         x_n = x_n.times(input);
      }

      return ret;
   }

   public int getOrder()
   {
      return coefficients.length - 1;
   }

   public double[] getCoefficients()
   {
      double[] ret = new double[coefficients.length];

      for (int i = 0; i < coefficients.length; i++)
      {
         ret[i] = coefficients[i];
      }

      return ret;
   }

   public ObsoletePolynomial times(double multiplier)
   {
      double[] coefficients = new double[this.coefficients.length];

      for (int cIndex = 0; cIndex < this.coefficients.length; cIndex++)
      {
         coefficients[cIndex] = this.coefficients[cIndex] * multiplier;
      }

      return new ObsoletePolynomial(coefficients);
   }

   public ObsoletePolynomial times(ObsoletePolynomial polynomialB)
   {
      // Do convolution on the coefficients:

      int order = this.getOrder() + polynomialB.getOrder();

      double[] coefficients = new double[order + 1];

      for (int cIndex = 0; cIndex <= order; cIndex++)
      {
         coefficients[cIndex] = 0.0;

         for (int aIndex = 0; aIndex <= cIndex; aIndex++)
         {
            int bIndex = cIndex - aIndex;

            if ((aIndex >= 0) && (bIndex >= 0) && (aIndex < this.coefficients.length) && (bIndex < polynomialB.coefficients.length))
            {
               coefficients[cIndex] += this.coefficients[aIndex] * polynomialB.coefficients[bIndex];
            }
         }
      }

      ObsoletePolynomial ret = new ObsoletePolynomial(coefficients);

      return ret;
   }

   public ObsoletePolynomial plus(ObsoletePolynomial polynomial)
   {
      int newOrder = this.getOrder();
      if (polynomial.getOrder() > newOrder)
         newOrder = polynomial.getOrder();

      // System.out.println("newOrder = " + newOrder);

      double[] newCoefficients = new double[newOrder + 1];

      for (int cIndex = 0; cIndex <= newOrder; cIndex++)
      {
         int thisIndex = this.getOrder() - cIndex;
         int otherIndex = polynomial.getOrder() - cIndex;
         int newIndex = newOrder - cIndex;

         newCoefficients[newIndex] = 0.0;
         if (thisIndex >= 0)
            newCoefficients[newIndex] += this.coefficients[thisIndex];
         if (otherIndex >= 0)
            newCoefficients[newIndex] += polynomial.coefficients[otherIndex];
      }

      return new ObsoletePolynomial(newCoefficients);
   }

   public String toString()
   {
      StringBuilder builder = new StringBuilder();

      for (int i = 0; i < coefficients.length - 1; i++)
      {
         builder.append(coefficients[i]);
         builder.append(" * x");
         int exponent = coefficients.length - i - 1;

         if (exponent > 1)
         {
            builder.append("^");
            builder.append(exponent);
         }

         builder.append(" + ");
      }

      builder.append(coefficients[coefficients.length - 1]);

      return builder.toString();
   }

   public boolean epsilonEquals(ObsoletePolynomial polynomial, double epsilon)
   {
      if (coefficients.length != polynomial.coefficients.length)
         return false;

      for (int i = 0; i < coefficients.length; i++)
      {
         if (Math.abs(coefficients[i] - polynomial.coefficients[i]) > epsilon)
            return false;
      }

      return true;
   }

   public void setQuintic(double x0, double x1, double y0, double yd0, double ydd0, double y1, double yd1, double ydd1)
   {
      MathTools.checkEquals(coefficients.length, 6);
      constraintMatrix = new DMatrixRMaj(new double[6][6]);
      constraintVector = new DMatrixRMaj(new double[6][1]);
      coefficientVector = new DMatrixRMaj(new double[6][1]);

      setPointConstraint(0, x0, y0);
      setDerivativeConstraint(1, x0, yd0);
      setDoubleDerivativeConstraint(2, x0, ydd0);

      setPointConstraint(3, x1, y1);
      setDerivativeConstraint(4, x1, yd1);
      setDoubleDerivativeConstraint(5, x1, ydd1);

      CommonOps_DDRM.solve(constraintMatrix, constraintVector, coefficientVector);
      setVariables();
   }

   public void setCubic(double x0, double x1, double y0, double yd0, double y1, double yd1)
   {
      MathTools.checkEquals(coefficients.length, 4);
      constraintMatrix = new DMatrixRMaj(new double[4][4]);
      constraintVector = new DMatrixRMaj(new double[4][1]);
      coefficientVector = new DMatrixRMaj(new double[4][1]);

      setPointConstraint(0, x0, y0);
      setDerivativeConstraint(1, x0, yd0);

      setPointConstraint(2, x1, y1);
      setDerivativeConstraint(3, x1, yd1);

      CommonOps_DDRM.solve(constraintMatrix, constraintVector, coefficientVector);
      setVariables();
   }

   private void setPointConstraint(int row, double xValue, double yValue)
   {
      double x_n = 1.0;

      for (int column = coefficients.length - 1; column >= 0; column--)
      {
         constraintMatrix.set(row, column, x_n);
         x_n *= xValue;
      }

      constraintVector.set(row, yValue);
   }

   private void setDerivativeConstraint(int row, double xValue, double yValue)
   {
      double x_n = 1.0;
      constraintMatrix.set(row, coefficients.length - 1, 0.0);

      for (int column = coefficients.length - 2; column >= 0; column--)
      {
         constraintMatrix.set(row, column, (coefficients.length - column - 1) * x_n);
         x_n *= xValue;
      }

      constraintVector.set(row, yValue);
   }

   private void setDoubleDerivativeConstraint(int row, double xValue, double yValue)
   {
      double x_n = 1.0;
      constraintMatrix.set(row, coefficients.length - 1, 0.0);
      constraintMatrix.set(row, coefficients.length - 2, 0.0);

      for (int column = coefficients.length - 3; column >= 0; column--)
      {
         constraintMatrix.set(row, column, (coefficients.length - column - 1) * (coefficients.length - column - 2) * x_n);
         x_n *= xValue;
      }

      constraintVector.set(row, yValue);
   }

   private void setVariables()
   {
      int length = coefficients.length;

      for (int row = 0; row < length; row++)
      {
         coefficients[row] = coefficientVector.get(row, 0);

         if (row < length - 1)
            derivativeCoefficients[row] = coefficientVector.get(row, 0);

         if (row < length - 2)
            doubleDerivativeCoefficients[row] = coefficientVector.get(row, 0);
      }

      for (int i = 0; i < length - 1; i++)
      {
         derivativeCoefficients[i] *= (length - i - 1);
      }

      for (int i = 0; i < length - 2; i++)
      {
         doubleDerivativeCoefficients[i] *= (length - i - 1) * (length - i - 2);
      }
   }

   public boolean equalsZero()
   {
      return (Math.abs(this.coefficients[0]) < 1e-15);
   }

   public double[] getDerivativeCoefficients()
   {
      return derivativeCoefficients.clone();
   }

   public double[] getDoubleDerivativeCoefficients()
   {
      return doubleDerivativeCoefficients.clone();
   }
}
