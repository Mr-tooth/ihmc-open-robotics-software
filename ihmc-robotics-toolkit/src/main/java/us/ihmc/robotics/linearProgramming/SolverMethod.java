package us.ihmc.robotics.linearProgramming;

public enum SolverMethod
{
   /* Faster solve method. Is more prone to edge cases */
   SIMPLEX,

   /* Slower but more robust method */
   CRISS_CROSS
}
