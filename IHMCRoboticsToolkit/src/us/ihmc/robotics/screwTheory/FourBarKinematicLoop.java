package us.ihmc.robotics.screwTheory;

import javax.vecmath.Vector3d;

import us.ihmc.robotics.Axis;
import us.ihmc.robotics.MathTools;
import us.ihmc.robotics.dataStructures.registry.YoVariableRegistry;
import us.ihmc.robotics.dataStructures.variable.DoubleYoVariable;
import us.ihmc.robotics.geometry.FramePoint;
import us.ihmc.robotics.geometry.FrameVector;
import us.ihmc.robotics.kinematics.fourbar.FourBarCalculatorFromFastRunner;
import us.ihmc.robotics.referenceFrames.ReferenceFrame;

public class FourBarKinematicLoop
{
   /*
    * Representation of the four bar with name correspondences.
    * This name convention matches the one used in the FourBarCalculator from fastRunner
    *   
    *              masterL
    *     master=A--------B
    *            |\      /|
    *            | \    / |
    *            |  \  /  |
    *            |   \/   |
    *            |   /\   |
    *            |  /  \  |
    *            | /    \ |
    *            |/      \|
    *            D--------C
    */
   private static final boolean DEBUG = true;

   private final static ReferenceFrame worldFrame = ReferenceFrame.getWorldFrame();
   private final String name;
   private final RevoluteJoint masterJointA;
   private final PassiveRevoluteJoint passiveJointB, passiveJointC, passiveJointD;
   private final FrameVector closurePointFromLastPassiveJointFrameVect;
   
   private final DoubleYoVariable masterJointQ;
   private final DoubleYoVariable masterJointQd;

   private final FramePoint masterJointAPosition = new FramePoint();
   private final FramePoint jointBPosition = new FramePoint();
   private final FramePoint jointCPosition = new FramePoint();
   private final FramePoint jointDPosition = new FramePoint();
   private double masterLinkAB, BC, CD, DA;
   private final FrameVector vectorBC = new FrameVector();
   private final FrameVector vectorCD = new FrameVector();
   private final FrameVector vectorDA = new FrameVector();
   private final FrameVector vectorAB = new FrameVector();

   private FourBarCalculatorFromFastRunner fourBarCalculator;
   
   private double[] interiorAnglesAtZeroConfiguration = new double[4];
   private double maxValidMasterJointAngle, minValidMasterJointAngle;
   
   private final FrameVector jointAxisInWorld;
   private final FrameVector masterAxis, jointBAxis, jointCAxis, jointDAxis;
   private final ReferenceFrame frameWithZAlongJointAxis;
   
   public FourBarKinematicLoop(String name, YoVariableRegistry registry, RevoluteJoint masterJointA, PassiveRevoluteJoint passiveJointB,
         PassiveRevoluteJoint passiveJointC, PassiveRevoluteJoint passiveJointD, Vector3d closurePointFromLastPassiveJointInWorld, boolean recomputeJointLimits)
   {
      this.name = name;
      this.masterJointA = masterJointA;
      this.passiveJointB = passiveJointB;
      this.passiveJointC = passiveJointC;
      this.passiveJointD = passiveJointD;
      closurePointFromLastPassiveJointFrameVect = new FrameVector(worldFrame, closurePointFromLastPassiveJointInWorld);
      
      masterJointQ = new DoubleYoVariable(name + "MasterJointQ", registry);
      masterJointQ.set(masterJointA.getQ());

      masterJointQd = new DoubleYoVariable(name + "MasterJointQd", registry);
      masterJointQd.set(masterJointA.getQd());
      
      masterAxis = masterJointA.getJointAxis();
      jointBAxis = passiveJointB.getJointAxis();
      jointCAxis = passiveJointC.getJointAxis();
      jointDAxis = passiveJointD.getJointAxis();

      jointAxisInWorld = new FrameVector();
      checkJointAxesAreParallelAndSetJointAxis();
      frameWithZAlongJointAxis = ReferenceFrame.constructReferenceFrameFromPointAndAxis(name + "FrameWithZAlongJointAxis", new FramePoint(), Axis.Z, jointAxisInWorld);
      checkCorrectJointOrder();

      initializeJointPositionsAndLinkVectors();
      masterLinkAB = getXYLengthInFrame(vectorAB, frameWithZAlongJointAxis);
      BC = getXYLengthInFrame(vectorBC, frameWithZAlongJointAxis);
      CD = getXYLengthInFrame(vectorCD, frameWithZAlongJointAxis);
      DA = getXYLengthInFrame(closurePointFromLastPassiveJointFrameVect, frameWithZAlongJointAxis);
      
      if (DEBUG)
      {
         System.out.println("\nLink length debugging: \n");
         System.out.println("masterLinkAB BC CD DA : " + masterLinkAB + ", " + BC + ", " + CD + ", " + DA);
      }

      verifyMasterJointLimits();
      setInteriorAngleOffsets();

      fourBarCalculator = new FourBarCalculatorFromFastRunner(DA, masterLinkAB, BC, CD);
      updateAnglesAndVelocities();

      if (DEBUG)
      {
         System.out.println("\nInitial joint angles debugging:\n\n" + "MasterQ: " + masterJointA.getQ() + "\njointBQ: " + passiveJointB.getQ() + "\njointCQ: "
               + passiveJointC.getQ() + "\njointDQ: " + passiveJointD.getQ() + "\n");
      }
   }

   private void checkJointAxesAreParallelAndSetJointAxis()
   {
      masterAxis.changeFrame(worldFrame);
      jointBAxis.changeFrame(worldFrame);
      jointCAxis.changeFrame(worldFrame);
      jointDAxis.changeFrame(worldFrame);

      // Both the exact same axis and a flipped axis are valid (eg: y and -y). So as long as the absolute value of the dot product is 1, the axis are parallel.      
      if (MathTools.epsilonEquals(Math.abs(masterAxis.dot(jointBAxis)), 1.0, 1.0e-7)
            && MathTools.epsilonEquals(Math.abs(masterAxis.dot(jointCAxis)), 1.0, 1.0e-7)
            && MathTools.epsilonEquals(Math.abs(masterAxis.dot(jointDAxis)), 1.0, 1.0e-7))
      {
         jointAxisInWorld.set(masterAxis);
      }
      else
      {
         throw new RuntimeException("All joints in the four bar must rotate around the same axis!");
      }
   }

   private void checkCorrectJointOrder()
   {
      if (masterJointA.getSuccessor() != passiveJointB.getPredecessor() || passiveJointB.getSuccessor() != passiveJointC.getPredecessor()
            || passiveJointC.getSuccessor() != passiveJointD.getPredecessor())
      {
         throw new RuntimeException("The joints that form the " + name + " four bar must be passed in clockwise or counterclockwise order");
      }

      if (DEBUG)
      {
         System.out.println("\nDebugging  check joint order:\n\nsuccessor \t predecessor\n" + masterJointA.getSuccessor() + "\t  "
               + passiveJointB.getPredecessor() + "\n" + passiveJointB.getSuccessor() + "\t  " + passiveJointC.getPredecessor() + "\n"
               + passiveJointC.getSuccessor() + "\t  " + passiveJointD.getPredecessor() + "\n");
      }
   }

   private void initializeJointPositionsAndLinkVectors()
   {
      jointBPosition.setToZero(passiveJointB.getFrameAfterJoint());
      jointCPosition.setToZero(passiveJointC.getFrameAfterJoint());
      jointDPosition.setToZero(passiveJointD.getFrameAfterJoint());
      masterJointAPosition.setToZero(masterJointA.getFrameAfterJoint());

      jointBPosition.changeFrame(worldFrame);
      jointCPosition.changeFrame(worldFrame);
      jointDPosition.changeFrame(worldFrame);
      masterJointAPosition.changeFrame(worldFrame);

      vectorBC.sub(jointCPosition, jointBPosition);
      vectorCD.sub(jointDPosition, jointCPosition);
      vectorDA.sub(masterJointAPosition, jointDPosition);
      vectorAB.sub(jointBPosition, masterJointAPosition);
   }

   private void setInteriorAngleOffsets()
   {
      vectorDA.normalize();
      vectorAB.normalize();
      vectorBC.normalize();
      vectorCD.normalize();
      
      vectorDA.changeFrame(worldFrame);
      vectorAB.changeFrame(worldFrame);
      vectorBC.changeFrame(worldFrame);
      vectorCD.changeFrame(worldFrame);

      interiorAnglesAtZeroConfiguration[0] = Math.PI - Math.acos(vectorDA.dot(vectorAB));
      interiorAnglesAtZeroConfiguration[1] = Math.PI - Math.acos(vectorAB.dot(vectorBC));
      interiorAnglesAtZeroConfiguration[2] = Math.PI - Math.acos(vectorBC.dot(vectorCD));
      interiorAnglesAtZeroConfiguration[3] = Math.PI - Math.acos(vectorCD.dot(vectorDA));

      if (DEBUG)
      {
         System.out.println("\nOffset angle debugging:\n");
         System.out.println("offset A = " + interiorAnglesAtZeroConfiguration[0]);
         System.out.println("offset B = " + interiorAnglesAtZeroConfiguration[1]);
         System.out.println("offset C = " + interiorAnglesAtZeroConfiguration[2]);
         System.out.println("offset D = " + interiorAnglesAtZeroConfiguration[3]);
      }
   }

   /**
    * Projects the link onto the plane of the four bar
    */
   private static double getXYLengthInFrame(FrameVector jointToJointVector, ReferenceFrame frameToProjectTo)
   {
      jointToJointVector.changeFrame(frameToProjectTo);
      return Math.sqrt(jointToJointVector.getX() * jointToJointVector.getX() + jointToJointVector.getY() * jointToJointVector.getY());
   }

   private void verifyMasterJointLimits() //TODO write a test for this
   {
      maxValidMasterJointAngle = Math.acos((masterLinkAB * masterLinkAB + DA * DA - (CD + BC) * (CD + BC)) / (2 * masterLinkAB * DA));
      minValidMasterJointAngle = Math.acos((DA * DA + (BC + masterLinkAB) * (BC + masterLinkAB) - CD * CD) / (2 * DA * (BC + masterLinkAB)));

      if (DEBUG)
      {
         System.out.println(" Max master joint angle: " + maxValidMasterJointAngle);
         System.out.println(" Min master joint angle: " + minValidMasterJointAngle);
      }

      // 1 - Angle limits not set
      if (masterJointA.getJointLimitLower() == Double.NEGATIVE_INFINITY || masterJointA.getJointLimitUpper() == Double.POSITIVE_INFINITY)
      {
         throw new RuntimeException("Must set the joint limits for the master joint of the " + name
               + " four bar.\nNote that for the given link lengths max angle is " + maxValidMasterJointAngle + "and min angle is" + minValidMasterJointAngle);
      }

      // 2 - Max angle limit is too large
      if (BC + CD > masterLinkAB + DA)
      {
         if (masterJointA.getJointLimitUpper() > maxValidMasterJointAngle)
         {
            throw new RuntimeException("The maximum valid joint angle for the master joint of the " + name + " four bar is " + maxValidMasterJointAngle
                  + " to avoid flipping, but was set to " + masterJointA.getJointLimitUpper());
         }
      }
      else if (masterJointA.getJointLimitUpper() > Math.PI)
      {
         throw new RuntimeException("The maximum valid joint angle for the master joint of the " + name + " four bar is " + maxValidMasterJointAngle
               + " to avoid flipping, but was set to " + masterJointA.getJointLimitUpper());
      }

      // 3 - Min angle limit is too small
      if (masterLinkAB + BC < DA + CD)
      {
         if (masterJointA.getJointLimitLower() < minValidMasterJointAngle)
         {
            throw new RuntimeException("The minimum valid joint angle for the master joint of the " + name + " four bar is " + minValidMasterJointAngle
                  + " to avoid flipping, but was set to " + masterJointA.getJointLimitLower());
         }
      }
      else if (masterJointA.getJointLimitLower() < 0.0)
      {
         throw new RuntimeException("The minimum valid joint angle for the master joint of the " + name + " four bar is " + minValidMasterJointAngle
               + " to avoid flipping, but was set to " + masterJointA.getJointLimitLower());
      }
   }

   public void updateAnglesAndVelocities()
   {
      fourBarCalculator.updateAnglesAndVelocitiesGivenAngleDAB(masterJointA.getQ() - interiorAnglesAtZeroConfiguration[0], masterJointA.getQd());
      passiveJointB.setQ(fourBarCalculator.getAngleABC() - interiorAnglesAtZeroConfiguration[1]);
      passiveJointC.setQ(fourBarCalculator.getAngleBCD() - interiorAnglesAtZeroConfiguration[2]);
      passiveJointD.setQ(fourBarCalculator.getAngleCDA() - interiorAnglesAtZeroConfiguration[3]);
      passiveJointB.setQd(fourBarCalculator.getAngleDtABC());
      passiveJointC.setQd(fourBarCalculator.getAngleDtBCD());
      passiveJointD.setQd(fourBarCalculator.getAngleDtCDA());
   }
}
