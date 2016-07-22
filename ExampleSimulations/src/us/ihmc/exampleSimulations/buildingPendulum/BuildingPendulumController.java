package us.ihmc.exampleSimulations.buildingPendulum;

import us.ihmc.robotics.dataStructures.variable.BooleanYoVariable;
import us.ihmc.robotics.dataStructures.variable.EnumYoVariable;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.simulationconstructionset.robotController.SimpleRobotController;

public class BuildingPendulumController extends SimpleRobotController
{
   private final BuildingPendulumRobot robot;
   private final BooleanYoVariable atCenter = new BooleanYoVariable("AtCenter", registry);

   private RobotSide activeSide;
   private final EnumYoVariable<RobotSide> yoActiveSide = new EnumYoVariable<>("ActiveSide", registry, RobotSide.class);

   private double pendulumAngle;
   private double pendulumAngleSwitch;
   private double angularChange = 2*Math.asin(BuildingPendulumRobot.distance/(2*BuildingPendulumRobot.length));
   private double velocity = 0.0;


   public BuildingPendulumController(BuildingPendulumRobot robot)
   {
      this.robot = robot;

      if(robot.getPendulumAngle(RobotSide.LEFT)> robot.getPendulumAngle(RobotSide.RIGHT))
         activeSide = RobotSide.LEFT;
      else
         activeSide = RobotSide.RIGHT;

   }

   public void setPendulumAngles()
   {
      pendulumAngle = robot.getPendulumAngle(activeSide);
      pendulumAngleSwitch = robot.getSwitchAngle(activeSide);
   }

   public  void doControl()
   {
      setPendulumAngles();

      boolean atCenter;
      if (activeSide == RobotSide.LEFT)
         atCenter = pendulumAngle > pendulumAngleSwitch;
      else
         atCenter = pendulumAngle < pendulumAngleSwitch;

      if (atCenter)
      {
         // switch active side
         activeSide = activeSide.getOppositeSide();

         // compute and set the state of the new active side
         velocity = robot.getPendulumVelocity(activeSide.getOppositeSide()) * Math.cos(angularChange);
         robot.setPendulumAngle(activeSide, robot.getSwitchAngle(activeSide));
         robot.setPendulumVelocity(activeSide, velocity);
      }

      // set the inactive pendulum to stay at the switching position
      robot.setPendulumAngle(activeSide.getOppositeSide(), robot.getSwitchAngle(activeSide.getOppositeSide()));
      robot.setPendulumVelocity(activeSide.getOppositeSide(), 0.0);

      // set yoVariables for debugging in SCS
      this.atCenter.set(atCenter);
      yoActiveSide.set(activeSide);
   }
}
