package us.ihmc.commonWalkingControlModules.modelPredictiveController.ioHandling;

import org.ejml.data.DMatrixRMaj;
import us.ihmc.commonWalkingControlModules.dynamicPlanning.comPlanning.MultipleCoMSegmentTrajectoryGenerator;
import us.ihmc.commonWalkingControlModules.modelPredictiveController.ContactPlaneProvider;
import us.ihmc.commonWalkingControlModules.modelPredictiveController.core.SE3MPCIndexHandler;
import us.ihmc.commons.lists.RecyclingArrayList;
import us.ihmc.euclid.axisAngle.AxisAngle;
import us.ihmc.euclid.axisAngle.interfaces.AxisAngleBasics;
import us.ihmc.euclid.matrix.Matrix3D;
import us.ihmc.euclid.matrix.interfaces.Matrix3DBasics;
import us.ihmc.euclid.referenceFrame.FrameQuaternion;
import us.ihmc.euclid.referenceFrame.FrameVector3D;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.referenceFrame.interfaces.*;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.robotics.math.trajectories.core.Polynomial3D;
import us.ihmc.robotics.math.trajectories.generators.MultipleWaypointsOrientationTrajectoryGenerator;
import us.ihmc.robotics.time.TimeIntervalProvider;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoDouble;

import java.util.List;

/**
 * This class is meant to handle the trajectory from the MPC module. It includes the trajectory for the full planning window, which is overwritten with the
 * solution for the planning window at the beginning.
 * <p>
 * It assembles all this solution into a continuous multi-segment trajectory for the center of mass, and a list of polynomials for the VRP.
 */
public class OrientationMPCTrajectoryHandler
{
   private final OrientationTrajectoryCalculator orientationInitializationCalculator;

   private final SE3MPCIndexHandler indexHandler;

   private final RecyclingArrayList<FrameOrientation3DBasics> desiredOrientation = new RecyclingArrayList<>(FrameQuaternion::new);
   private final RecyclingArrayList<FrameVector3DBasics> desiredAngularVelocity = new RecyclingArrayList<>(FrameVector3D::new);
   private final RecyclingArrayList<FrameVector3DBasics> desiredBodyAngularMomentaAboutPoint = new RecyclingArrayList<>(FrameVector3D::new);

   private final RecyclingArrayList<AxisAngleBasics> axisAngleErrorSolution = new RecyclingArrayList<>(AxisAngle::new);
   private final RecyclingArrayList<FrameVector3DBasics> bodyAngularMomentaAboutPointSolution = new RecyclingArrayList<>(FrameVector3D::new);

   private final RecyclingArrayList<FrameQuaternionBasics> orientationSolution = new RecyclingArrayList<>(FrameQuaternion::new);
   private final RecyclingArrayList<FrameVector3DBasics> angularVelocitySolution = new RecyclingArrayList<>(FrameVector3D::new);

   private final Matrix3DBasics momentOfInertiaMatrix = new Matrix3D();
   private final MultipleWaypointsOrientationTrajectoryGenerator orientationTrajectory;

   private final Polynomial3D internalAngularMomentumTrajectory = new Polynomial3D(3);
   private final YoDouble previewWindowEndTime;

   private final double mass;

   public OrientationMPCTrajectoryHandler(SE3MPCIndexHandler indexHandler, double mass, YoRegistry registry)
   {
      this.indexHandler = indexHandler;
      this.mass = mass;

      previewWindowEndTime = new YoDouble("orientationPreviewWindowEndTime", registry);
      internalAngularMomentumTrajectory.setConstant(new Point3D());

      orientationInitializationCalculator = new OrientationTrajectoryCalculator(registry);
      orientationTrajectory = new MultipleWaypointsOrientationTrajectoryGenerator("desiredCoMTrajectory", ReferenceFrame.getWorldFrame(), registry);
   }

   /**
    * Clears the CoM and VRP solution trajectories
    */
   public void clearTrajectory()
   {
      previewWindowEndTime.set(Double.NEGATIVE_INFINITY);
      orientationTrajectory.clear();
   }

   public void setInitialBodyOrientationState(FrameOrientation3DReadOnly bodyOrientation, FrameVector3DReadOnly bodyAngularVelocity)
   {
      orientationInitializationCalculator.setInitialBodyOrientation(bodyOrientation, bodyAngularVelocity);
   }

   public void solveForTrajectoryOutsidePreviewWindow(List<ContactPlaneProvider> fullContactSequence)
   {
      orientationInitializationCalculator.solveForTrajectory(fullContactSequence);

      removeInfoOutsidePreviewWindow();
      overwriteTrajectoryOutsidePreviewWindow();
   }

   public void computeDesiredTrajectory(double currentTimeInState)
   {
      desiredOrientation.clear();
      desiredAngularVelocity.clear();
      desiredBodyAngularMomentaAboutPoint.clear();

      int tickIndex = 0;
      for (int segment = 0; segment < indexHandler.getNumberOfSegments(); segment++)
      {
         for (int i = 0; i < indexHandler.getOrientationTicksInSegment(segment); i++)
         {
            currentTimeInState += indexHandler.getOrientationTickDuration(tickIndex);

            orientationTrajectory.compute(currentTimeInState);

            desiredOrientation.add().set(orientationTrajectory.getOrientation());
            desiredAngularVelocity.add().set(orientationTrajectory.getAngularVelocity());

            tickIndex++;
         }
      }
   }

   public void extractSolutionForPreviewWindow(DMatrixRMaj solutionCoefficients,
                                               MultipleCoMSegmentTrajectoryGenerator comTrajectorySolution,
                                               double currentTimeInState,
                                               double previewWindowDuration)
   {
      previewWindowEndTime.set(currentTimeInState + previewWindowDuration);
      extractSolutionVectors(solutionCoefficients);

      clearTrajectory();

      orientationSolution.clear();
      angularVelocitySolution.clear();

      for (int i = 0; i < axisAngleErrorSolution.size(); i++)
      {
         currentTimeInState += indexHandler.getOrientationTickDuration(i);

         FrameQuaternionBasics orientation = orientationSolution.add();
         orientation.set(desiredOrientation.get(i));
         orientation.append(axisAngleErrorSolution.get(i));

         internalAngularMomentumTrajectory.compute(currentTimeInState);
         comTrajectorySolution.compute(currentTimeInState);

         FrameVector3DBasics angularVelocity = angularVelocitySolution.add();
         angularVelocity.cross(comTrajectorySolution.getPosition(), comTrajectorySolution.getVelocity());
         angularVelocity.scaleAdd(-mass, internalAngularMomentumTrajectory.getPosition());

         orientation.inverseTransform(angularVelocity);
         momentOfInertiaMatrix.inverseTransform(angularVelocity);

         orientationTrajectory.appendWaypoint(currentTimeInState, orientation, angularVelocity);
      }

      overwriteTrajectoryOutsidePreviewWindow();
   }

   private void extractSolutionVectors(DMatrixRMaj solutionCoefficients)
   {
      int totalNumberOfTicks = 0;
      for (int i = 0; i < indexHandler.getNumberOfSegments(); i++)
         totalNumberOfTicks += indexHandler.getOrientationTicksInSegment(i);

      axisAngleErrorSolution.clear();
      bodyAngularMomentaAboutPointSolution.clear();

      for (int i = 0; i < totalNumberOfTicks; i++)
      {
         AxisAngleBasics axisAngleError = axisAngleErrorSolution.add();
         FrameVector3DBasics bodyAngularMomentumAboutPoint = bodyAngularMomentaAboutPointSolution.add();
         axisAngleError.setRotationVector(solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i), 0),
                                          solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i) + 1, 0),
                                          solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i) + 2, 0));
         bodyAngularMomentumAboutPoint.set(solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i) + 3, 0),
                                           solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i) + 4, 0),
                                           solutionCoefficients.get(indexHandler.getOrientationTickStartIndex(i) + 5, 0));
      }
   }

   private void removeInfoOutsidePreviewWindow()
   {
      while (orientationTrajectory.getCurrentNumberOfWaypoints() > 0 && orientationTrajectory.getLastWaypointTime() > previewWindowEndTime.getValue())
      {
         orientationTrajectory.removeWaypoint(orientationTrajectory.getCurrentNumberOfWaypoints() - 1);
      }
   }

   private void overwriteTrajectoryOutsidePreviewWindow()
   {
      MultipleWaypointsOrientationTrajectoryGenerator orientationTrajectoryOutsideWindow = orientationInitializationCalculator.getOrientationTrajectory();

      boolean hasTrajectoryAlready = orientationTrajectory.getCurrentNumberOfWaypoints() > 0;
      double existingEndTime = hasTrajectoryAlready ? orientationTrajectory.getLastWaypointTime() : 0.0;
      if (existingEndTime >= orientationTrajectoryOutsideWindow.getLastWaypointTime())
         return;

      int waypointIndexToAdd = getWaypointIndexAfterTime(existingEndTime + 1e-5, orientationInitializationCalculator.getOrientationTrajectory());
      if (waypointIndexToAdd == -1)
         return;

      for (; waypointIndexToAdd < orientationTrajectoryOutsideWindow.getCurrentNumberOfWaypoints(); waypointIndexToAdd++)
      {
         orientationTrajectory.appendWaypoint(orientationTrajectoryOutsideWindow.getWaypoint(waypointIndexToAdd));
      }

      orientationTrajectory.initialize();
   }

   private static int getWaypointIndexAfterTime(double time, MultipleWaypointsOrientationTrajectoryGenerator trajectory)
   {
      for (int i = 0; i < trajectory.getCurrentNumberOfWaypoints(); i++)
      {
         if (trajectory.getWaypoint(i).getTime() > time)
            return i;
      }

      return -1;
   }

   public void compute(double timeInPhase)
   {
      orientationTrajectory.compute(timeInPhase);
   }

   public void computeOutsidePreview(double timeInPhase)
   {
      orientationInitializationCalculator.compute(timeInPhase);
   }

   public FrameOrientation3DReadOnly getDesiredBodyOrientationOutsidePreview()
   {
      return orientationInitializationCalculator.getDesiredOrientation();
   }

   public FrameVector3DReadOnly getDesiredBodyVelocityOutsidePreview()
   {
      return orientationInitializationCalculator.getDesiredAngularVelocity();
   }


   public FrameOrientation3DReadOnly getDesiredBodyOrientation()
   {
      return orientationTrajectory.getOrientation();
   }

   public FrameVector3DReadOnly getDesiredAngularVelocity()
   {
      return orientationTrajectory.getAngularVelocity();
   }

   public FrameVector3DReadOnly getDesiredAngularAcceleration()
   {
      return orientationTrajectory.getAngularAcceleration();
   }
}
