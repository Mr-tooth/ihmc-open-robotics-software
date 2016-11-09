package us.ihmc.footstepPlanning.testTools;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import us.ihmc.footstepPlanning.FootstepPlanner;
import us.ihmc.footstepPlanning.FootstepPlanningResult;
import us.ihmc.graphics3DAdapter.graphics.Graphics3DObject;
import us.ihmc.graphics3DAdapter.graphics.appearances.AppearanceDefinition;
import us.ihmc.graphics3DAdapter.graphics.appearances.YoAppearance;
import us.ihmc.robotics.dataStructures.registry.YoVariableRegistry;
import us.ihmc.robotics.geometry.ConvexPolygon2d;
import us.ihmc.robotics.geometry.FramePose;
import us.ihmc.robotics.geometry.FrameVector;
import us.ihmc.robotics.geometry.PlanarRegionsList;
import us.ihmc.robotics.math.frames.YoFrameConvexPolygon2d;
import us.ihmc.robotics.math.frames.YoFramePoint;
import us.ihmc.robotics.math.frames.YoFramePose;
import us.ihmc.robotics.math.frames.YoFrameVector;
import us.ihmc.robotics.referenceFrames.PoseReferenceFrame;
import us.ihmc.robotics.referenceFrames.ReferenceFrame;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.robotics.robotSide.SideDependentList;
import us.ihmc.simulationconstructionset.Robot;
import us.ihmc.simulationconstructionset.SimulationConstructionSet;
import us.ihmc.simulationconstructionset.yoUtilities.graphics.YoGraphicPolygon;
import us.ihmc.simulationconstructionset.yoUtilities.graphics.YoGraphicPosition;
import us.ihmc.simulationconstructionset.yoUtilities.graphics.YoGraphicVector;
import us.ihmc.simulationconstructionset.yoUtilities.graphics.YoGraphicsListRegistry;
import us.ihmc.tools.thread.ThreadTools;

public class PlanningTestTools
{
   private static final ReferenceFrame worldFrame = ReferenceFrame.getWorldFrame();

   public static ConvexPolygon2d createDefaultFootPolygon()
   {
      double footLength = 0.2;
      double footWidth = 0.1;

      ConvexPolygon2d footPolygon = new ConvexPolygon2d();
      footPolygon.addVertex(footLength/2.0, footWidth/2.0);
      footPolygon.addVertex(footLength/2.0, -footWidth/2.0);
      footPolygon.addVertex(-footLength/2.0, footWidth/2.0);
      footPolygon.addVertex(-footLength/2.0, -footWidth/2.0);
      footPolygon.update();

      return footPolygon;
   }

   public static SideDependentList<ConvexPolygon2d> createDefaultFootPolygons()
   {
      SideDependentList<ConvexPolygon2d> footPolygons = new SideDependentList<>();
      for (RobotSide side : RobotSide.values)
         footPolygons.put(side, PlanningTestTools.createDefaultFootPolygon());
      return footPolygons;
   }

   public static void visualizeAndSleep(PlanarRegionsList planarRegionsList, List<FramePose> footseps, RobotSide firstStepSide,
         FramePose goalPose)
   {
      visualizeAndSleep(planarRegionsList, footseps, firstStepSide, goalPose, null, null);
   }

   public static void visualizeAndSleep(PlanarRegionsList planarRegionsList, List<FramePose> footseps, RobotSide firstStepSide)
   {
      visualizeAndSleep(planarRegionsList, footseps, firstStepSide, null, null, null);
   }

   public static void visualizeAndSleep(PlanarRegionsList planarRegionsList, List<FramePose> footseps, RobotSide firstStepSide,
         FramePose goalPose, YoVariableRegistry registry, YoGraphicsListRegistry graphicsListRegistry)
   {
      SimulationConstructionSet scs = new SimulationConstructionSet(new Robot("Dummy"));
      if (registry != null)
         scs.addYoVariableRegistry(registry);
      if (graphicsListRegistry != null)
         scs.addYoGraphicsListRegistry(graphicsListRegistry, true);

      Graphics3DObject graphics3DObject = new Graphics3DObject();
      graphics3DObject.addCoordinateSystem(0.3);
      if (planarRegionsList != null)
         graphics3DObject.addPlanarRegionsList(planarRegionsList, YoAppearance.Black());
      scs.addStaticLinkGraphics(graphics3DObject);

      YoVariableRegistry vizRegistry = new YoVariableRegistry("FootstepPlanningResult");
      YoGraphicsListRegistry vizGraphicsListRegistry = new YoGraphicsListRegistry();

      if (goalPose != null)
         addGoalViz(goalPose, vizRegistry, vizGraphicsListRegistry);

      if (footseps != null)
      {
         int i = 0;
         RobotSide stepSide = firstStepSide;
         YoFrameConvexPolygon2d yoDefaultFootPolygon = new YoFrameConvexPolygon2d("DefaultFootPolygon", worldFrame, 4, vizRegistry);
         yoDefaultFootPolygon.setConvexPolygon2d(createDefaultFootPolygon());
         for (FramePose footstepPose : footseps)
         {
            AppearanceDefinition appearance = stepSide == RobotSide.RIGHT ? YoAppearance.Green() : YoAppearance.Red();
            YoFramePose yoFootstepPose = new YoFramePose("footPose" + (i++), worldFrame, vizRegistry);
            yoFootstepPose.set(footstepPose);

            YoGraphicPolygon footstepViz = new YoGraphicPolygon("footstep" + (i++), yoDefaultFootPolygon, yoFootstepPose, 1.0, appearance);
            vizGraphicsListRegistry.registerYoGraphic("viz", footstepViz);
            stepSide = stepSide.getOppositeSide();
         }
      }

      scs.addYoVariableRegistry(vizRegistry);
      scs.addYoGraphicsListRegistry(vizGraphicsListRegistry, true);
      scs.startOnAThread();
      ThreadTools.sleepForever();
   }

   public static void addGoalViz(FramePose goalPose, YoVariableRegistry registry, YoGraphicsListRegistry graphicsListRegistry)
   {
      YoFramePoint yoGoal = new YoFramePoint("GoalPosition", worldFrame, registry);
      yoGoal.set(goalPose.getFramePointCopy());
      graphicsListRegistry.registerYoGraphic("viz", new YoGraphicPosition("GoalViz", yoGoal, 0.05, YoAppearance.White()));
      PoseReferenceFrame goalFrame = new PoseReferenceFrame("GoalFrame", goalPose);
      FrameVector goalOrientation = new FrameVector(goalFrame, 0.5, 0.0, 0.0);
      goalOrientation.changeFrame(worldFrame);
      YoFrameVector yoGoalOrientation = new YoFrameVector("GoalVector", worldFrame, registry);
      yoGoalOrientation.set(goalOrientation);
      graphicsListRegistry.registerYoGraphic("vizOrientation", new YoGraphicVector("GoalOrientationViz", yoGoal, yoGoalOrientation, 1.0, YoAppearance.White()));
   }

   public static ArrayList<FramePose> runPlanner(FootstepPlanner planner, FramePose initialStanceFootPose,
         RobotSide initialStanceSide, FramePose goalPose, PlanarRegionsList planarRegionsList)
   {
      planner.setInitialStanceFoot(initialStanceFootPose, initialStanceSide);
      planner.setGoalPose(goalPose);
      planner.setPlanarRegions(planarRegionsList);

      ArrayList<FramePose> footstepPlan = new ArrayList<>();
      FootstepPlanningResult result = planner.plan(footstepPlan);
      assertTrue("Planner was not able to provide valid result.", result.validForExecution());
      return footstepPlan;
   }

   public static boolean isGoalWithinFeet(FramePose goalPose, List<FramePose> footstepPlan)
   {
      int steps = footstepPlan.size();
      if (steps < 2)
         throw new RuntimeException("Did not get enough footsteps to check if goal is within feet.");

      FramePose lastFoostep = footstepPlan.get(steps - 1);
      FramePose secondLastFoostep = footstepPlan.get(steps - 2);
      FramePose achievedGoal = new FramePose();
      achievedGoal.interpolate(lastFoostep, secondLastFoostep, 0.5);

      if (achievedGoal.epsilonEquals(goalPose, 10E-2))
         return true;
      else
         return false;
   }
}
