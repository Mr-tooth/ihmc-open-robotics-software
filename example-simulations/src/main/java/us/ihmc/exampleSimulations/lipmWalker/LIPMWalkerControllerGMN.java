// GMN: TODO, DONE, Notes, answered & unanswered Q's:
//
// Questions:
// =========
// Q: What about CoM height servo, instead of body-height?
// Q: What would it mean to have a clock-less controller?
// Q: Try some AM compensation or Natural Posture control?
// Q: [FOR JERRY] How to make larger time window for interactive sim playback?
//    Q: [FOR JERRY] How to save above as reloaded config?
//
// TODO:
// ====
// + ELIMINATE DUPLICATE CODE BTW LIFT & SWING STATES
//   + Do we need a LIFT state?
// + I have some controller state that is not rewindable - need to fix
//   + YoVariables
//   + check out YoMatrix
// + Make human-readable var names
// + IMPROVE LOGGING SO CAN DEBUG BETTER
//   + Q: How well hitting desired TD?
//   + Q: How well hitting prescribed timing?
// + General cleanup
// + Predictive TDLO
//   + Add full pole-placement to TDLO: convert from continuous -> discrete
// + Better ground-speed matching at TD?
// + Get away from z-cubic-spline for SWING
// + Learn how to use Yo-variables
// + More learning of SCS plotting/video tools
// + Can we fix .getCurrentState() for these state machines?
//   + (get rid of inLiftState flag)
//
// DONE:
// ====
// + Access to matrices? Examples? 
// + Need Jacobians
// + Need to figure out how state machine works
// + Body-height servo
// + Body-pitch servo
// + Swing IK
// + TDLO implementation
// + Replace "robotSide" with just "side" everywhere
// + Check-in code
// + Tune-up swing tracking, both joints and TD (e.g. ground-speed matching)

 

package us.ihmc.exampleSimulations.lipmWalker;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;
import us.ihmc.log.LogTools;
import us.ihmc.robotics.math.frames.YoMatrix;
import us.ihmc.robotics.robotSide.RobotSide;
import us.ihmc.robotics.robotSide.SideDependentList;
import us.ihmc.robotics.stateMachine.core.State;
import us.ihmc.robotics.stateMachine.core.StateMachine;
import us.ihmc.robotics.stateMachine.core.StateTransitionCondition;
import us.ihmc.robotics.stateMachine.factories.StateMachineFactory;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoBoolean;
import us.ihmc.yoVariables.variable.YoDouble;
import us.ihmc.simulationconstructionset.util.RobotController;

public class LIPMWalkerControllerGMN implements RobotController
{
   private final LIPMWalkerRobot robot;
   private YoRegistry registry = new YoRegistry(getClass().getSimpleName());

   private enum States
   {
      // GMN: DOUBLE-SUPPORT ???
      //      A: "LIFT" is effectively double-support...
      SUPPORT, LIFT, SWING;
   }

   private final SideDependentList<StateMachine<States, State>> stateMachines;

   private final double g  = 9.81;
   private final double desiredBodyHeight = 0.8;
   private final double stepTime = 0.65;
   private final double lipTipFreq = Math.sqrt(g/desiredBodyHeight);
   private final double swingHeight = 0.12;
   private double kxTDLO, bxTDLO;
   
   private final YoDouble xTD = new YoDouble("xTD", registry);  // GMN: Need to restructure this...
   
   private final YoDouble leftLastTime = new YoDouble("leftLastTime",registry);
   private final YoDouble rightLastTime = new YoDouble("rightLastTime",registry);
   private final YoDouble leftx_d_foot = new YoDouble("leftx_d_foot",registry);
   private final YoDouble rightx_d_foot = new YoDouble("rightx_d_foot",registry);
   private final YoDouble lefttimeInStance = new YoDouble("lefttimeInState",registry);
   private final YoDouble righttimeInStance = new YoDouble("righttimeInState",registry);
   private final YoMatrix leftSwingCoeffs = new YoMatrix("leftSwingCoeffs",3,1,registry);
   private final YoMatrix rightSwingCoeffs = new YoMatrix("rightSwingCoeffs",3,1,registry);
   private final YoMatrix leftq_d = new YoMatrix("leftq_d",2,1,registry);
   private final YoMatrix rightq_d = new YoMatrix("rightq_d",2,1,registry);
   
   private final SideDependentList<YoDouble> lrLastTime = new SideDependentList<YoDouble>(leftLastTime,rightLastTime);
   private final SideDependentList<YoDouble> lrx_d_foot = new SideDependentList<YoDouble>(leftx_d_foot,rightx_d_foot);
   private final SideDependentList<YoDouble> timeInStance = new SideDependentList<YoDouble>(lefttimeInStance,righttimeInStance);
   private final SideDependentList<YoBoolean> inLiftState = 
         new SideDependentList<YoBoolean>(new YoBoolean("leftinLiftState","left in LIFT",registry),
                                          new YoBoolean("rightinLiftState","right in LIFT",registry)); // GMN: work-around for incompatible "State" types, UGH!
   private final SideDependentList<YoMatrix> lrSwingCoeffs = new SideDependentList<YoMatrix>(leftSwingCoeffs,rightSwingCoeffs);
   private final SideDependentList<YoMatrix> lrq_d = new SideDependentList<YoMatrix>(leftq_d,rightq_d);
   
   // For logging & debugging:
   private final SideDependentList<YoDouble> desiredxTD = 
         new SideDependentList<YoDouble>(new YoDouble("leftdesiredxTD","left desired xTD",registry),
                                         new YoDouble("rightdesiredxTD","right desired xTD",registry));

   
   public LIPMWalkerControllerGMN(LIPMWalkerRobot robot)
   {
      this.robot = robot;
      initialize();
      
      stateMachines = setupStateMachines();
   }

   @Override
   public void initialize()
   {
      // Pole-placement:   GMN - Can't handle imaginary part yet!
      double p1 = 0;  // set to zero for dead-beat
      double p2 = 0;  // set to zero for dead-beat
      double k = lipTipFreq;
      double Ts = stepTime;
      kxTDLO = (1+Math.exp(2*k*Ts)+2*Math.exp(k*Ts)*(-p1-p2+p1*p2))/(2*Math.pow(-1+Math.exp(k*Ts),2));
      bxTDLO = (1+Math.exp(2*k*Ts)-2*Math.exp(k*Ts)*( p1+p2+p1*p2))/(2*Math.pow( 1+Math.exp(k*Ts),2));
      
      lrLastTime.get(RobotSide.LEFT).set(-1.0);
      lrLastTime.get(RobotSide.RIGHT).set(-1.0);
   }
 
   @Override
   public YoRegistry getYoRegistry()
   {
      return registry;
   }
   
   @Override
   public void doControl()
   {
      // Basic TDLO list:
      // + Need body servo to:
      //   + control height (CoM? Body?)
      //   + control body pitch
      // + Implicit force control in stance leg: tau = J-transpose * F
      // + Need analytical or diff IK to generate joint angles for swing leg
      // + Need task-space traj generation for swing leg towards some desired TD and at desired time
      // + Joint servos with desired values
      // + State machine; step on a timer
      
//      // Simple control for debugging body-servo:
//      controlSupportLeg(RobotSide.LEFT);
//      
//      double q_hip = robot.getHipAngle(RobotSide.RIGHT);
//      double qd_hip = robot.getHipVelocity(RobotSide.RIGHT);
//      double q_knee = robot.getKneeLength(RobotSide.RIGHT);
//      double qd_knee = robot.getKneeVelocity(RobotSide.RIGHT);
//      // Servo joints to the desired joint trajs:
//      double Tauhip  = 10*(0 - q_hip)  + 1*(0 - qd_hip);
//      double Tauknee = 10*(0.6 - q_knee) + 1*(0 - qd_knee);
//      robot.setHipTorque(RobotSide.RIGHT, Tauhip);
//      robot.setKneeForce(RobotSide.RIGHT, Tauknee);
      
      
      for(RobotSide side : RobotSide.values())
      {
         stateMachines.get(side).doAction();
         stateMachines.get(side).doTransitions();
      }

      LogTools.info("StateMachine: Left:{} Right:{}", stateMachines.get(RobotSide.LEFT).getCurrentStateKey(), stateMachines.get(RobotSide.RIGHT).getCurrentStateKey());
   }

   private void controlSupportLeg(RobotSide side)
   {
      // GMN: Would seem we would do body-servo here.
      // Steps?
      // + Do body-servo:  Gives Fx and Fz at foot
      //   + Need measured height (assume body) & veloc
      //   + Need measured pitch & veloc
      // + Do tau = J-transpose * F
      // + Assign leg actuator torques/forces
      
//      Point3DReadOnly centerOfMassPosition = robot.getCenterOfMassPosition();
//      Vector3DReadOnly centerOfMassVelocity = robot.getCenterOfMassVelocity();
      double mass = robot.getMass();
      
      double z_body_w = robot.getBodyZPosition();
      double zd_body_w = robot.getBodyZVelocity();
      double th_body = robot.getBodyPitchAngle();
      double thd_body = robot.getBodyPitchAngularVelocity();
  
      // The body-servo:
      double Fz = 500*(desiredBodyHeight - z_body_w) - 50*zd_body_w + 0.999*mass*9.81; // GMN
      double My = 100*(0 - th_body) - 10*thd_body; // GMN
      
      double xcop = getXcop(side);
      
      double Fx = -1/z_body_w*(Fz*xcop + My);  // Fx based on body-servo & current foot position

      // Do implicit force control:
      DMatrixRMaj Ffoot = new DMatrixRMaj(2,1);
      Ffoot.set(0,0,-Fx); // set desired foot-force (applied by foot to ground)
      Ffoot.set(1,0,-Fz); // set desired foot-force (applied by foot to ground)
      DMatrixRMaj Jleg = robot.getlegJacobian(robot.getBodyPitchAngle(),
                                              robot.getHipAngle(side),
                                              Math.abs(robot.getKneeLength(side))); // get jacobian (includes body pitch)
      CommonOps_DDRM.transpose(Jleg);  // transpose the jacobian
      DMatrixRMaj Tau = new DMatrixRMaj(3,1);
      CommonOps_DDRM.mult(Jleg,Ffoot,Tau); // tau = J-transpose * F
      
//      /* Compute and set orbital energy YoDouble. Based on virtual spring-mass system. */
//      double orbitalEnergyValue = 0.5 * mass * centerOfMassVelocity.lengthSquared() - 0.5 * g / desiredHeight.getDoubleValue() * centerOfMassPosition.distanceFromOriginSquared();
//      orbitalEnergy.set(orbitalEnergyValue);

      // Skip 0 element as that is moment on body
      robot.setHipTorque(side, Tau.get(1, 0));
      robot.setKneeForce(side, Tau.get(2, 0));
   }

   private void controlSwingLeg(RobotSide side, double timeInSwing, double dT, double desiredTD)
   {
      double thd_body = robot.getBodyPitchAngularVelocity();
      
      // Dead-reckoning on foot x pos to desired:
      double dt_denom = (stepTime-0.1) - timeInSwing; // GMN: Added 0.1 sec arrive early...
//      double dt_denom = stepTime - timeInSwing; // GMN: Added 0.1 sec arrive early...
      if (dt_denom < 0.001)
      {
         dt_denom = 0.001; // GMN
      }
      double xd_d_foot = 0.0;
//      if (timeInSwing > 0.05) // GMN: try to prevent dragging while lifting; refactor so this happens in LIFT only; also really needs to be r.t. world
//      {
         xd_d_foot = (desiredTD - lrx_d_foot.get(side).getValue())/dt_denom; // GMN
//      }
      lrx_d_foot.get(side).set(lrx_d_foot.get(side).getValue() + xd_d_foot*dT);
      
      // get zd_d_foot:
      double zd_d_foot = calculateSwingZdot(side,timeInSwing);
      
      // Do diffIK; use measured th_body as FF term?
      DMatrixRMaj q_d = new DMatrixRMaj(2,1);
      lrq_d.get(side).get(q_d);
      DMatrixRMaj Jleg = robot.getlegJacobian(robot.getBodyPitchAngle(),  // includes body-pitch
                                              q_d.get(0,0),
                                              q_d.get(1,0)); 
      DMatrixRMaj Jq = new DMatrixRMaj(2,2); // just leg joints
      Jq.set(0,0,Jleg.get(0,1));
      Jq.set(0,1,Jleg.get(0,2)); 
      Jq.set(1,0,Jleg.get(1,1));
      Jq.set(1,1,Jleg.get(1,2));
      CommonOps_DDRM.invert(Jq);  // Jq now holds its inverse
      DMatrixRMaj RHS = new DMatrixRMaj(2,1);
      RHS.set(0,0,xd_d_foot - 0*Jleg.get(0,0)*thd_body); // GMN: including body-pitch rate bias
      RHS.set(1,0,zd_d_foot - 0*Jleg.get(1,0)*thd_body); // GMN: including body-pitch rate bias
      DMatrixRMaj qd_d = new DMatrixRMaj(2,1);
      CommonOps_DDRM.mult(Jq, RHS, qd_d); // resolved-rate IK
      // Easier way?????
      q_d.set(0,0,q_d.get(0,0)+qd_d.get(0,0)*dT);
      q_d.set(1,0,q_d.get(1,0)+qd_d.get(1,0)*dT);
      lrq_d.get(side).set(q_d);
//      lrq_d.get(side).set(0,0,lrq_d.get(side).get(0,0)+qd_d.get(0,0)*dT); // Easier way?????
//      lrq_d.get(side).set(1,0,lrq_d.get(side).get(1,0)+qd_d.get(1,0)*dT); // Easier way?????
      
      double q_hip = robot.getHipAngle(side);
      double qd_hip = robot.getHipVelocity(side);
      double q_knee = robot.getKneeLength(side);
      double qd_knee = robot.getKneeVelocity(side);

      // Servo joints to the desired joint trajs:
      double Tauhip  = 800*(q_d.get(0,0) - q_hip)  + 80*(qd_d.get(0,0) - qd_hip);
      double Tauknee = 800*(q_d.get(1,0) - q_knee) + 80*(qd_d.get(1,0) - qd_knee);
      
      robot.setHipTorque(side, Tauhip);
      robot.setKneeForce(side, Tauknee);
   }

   // Get current foot x pos r.t. body ewrt world:
   double getXcop(RobotSide side)
   {
      Point3D pos_foot = robot.getFootPosition(side);
      double xcop = pos_foot.getX() - robot.getBodyXPosition(); // xcop r.t. body
//      Point3DReadOnly pos_CoM = robot.getCenterOfMassPosition();
//      double xcop = pos_foot.getX() - pos_CoM.getX(); // xcop r.t. CoM
      
      return xcop;
   }
   
   double simpleTDLOStepLocation(double xLO)
   {
      // Simple TDLO:
      return (kxTDLO*(xTD.getValue()-xLO)-bxTDLO*(xTD.getValue()+xLO));      
//      return (kxTDLO*(xTD.getValue()-xLO)-bxTDLO*(xTD.getValue()+xLO) - 0.08);      
   }

   // predicted-LO TDLO (sort of like "Black Diamond" TDLO)
   double predictedTDLOStepLocation(double t, double xTDk, double xCoP)
   {
      double k = lipTipFreq;
      double Ts = stepTime;
      if (t < 0.1*Ts)
      {
         t = 0.1*Ts;
      }
      
      // See reMarkable notes Feb 23, 2022:
      double cTD = (Math.exp(k*Ts)-Math.exp(2*k*(t-Ts)))/(1-Math.exp(2*k*t));
      double cCoP = Math.exp(k*(t-Ts))*(Math.exp(2*k*Ts)-1)/(Math.exp(2*k*t)-1);
      
      // predicted upcoming LO location:
      double xLOk = cTD*xTDk + cCoP*xCoP;
      
      // predicted final TDLO TD location r.t. body:
      double xTDk1 = kxTDLO*(xTDk-xLOk)-bxTDLO*(xTDk+xLOk);
      
//      // desired TD location: (attempt to get a location ideally fixed in the world)
//      return (xCoP - xLOk + xTDk1);  // GMN: Does world-frame well, but dynamics not good (need to figure out why)
      return xTDk1;
   }
   
   // See reMarkable notes: 2/21/2022
   DMatrixRMaj buildSwingZdot(double zLO)
   {
      double zSH = swingHeight ; // GMN
      double zTD = 0; // desired height of foot at TD
      double zdTD = -1; // desired zd (z-dot) of foot at TD
      
      double tSH = 0.45*stepTime;
      double tTD = stepTime;

      // Cubic-spline fit:
      double data[][] = new double[][] {
         {  Math.pow(tSH,3), Math.pow(tSH,2), tSH },
         {  Math.pow(tTD,3), Math.pow(tTD,2), tTD },
         {3*Math.pow(tTD,2), 2*tTD, 1}
      };
      
      // Solve for spline coeffs:
      DMatrixRMaj m = new DMatrixRMaj(data);
      CommonOps_DDRM.invert(m); // m now holds its inverse
      DMatrixRMaj LHS = new DMatrixRMaj(3,1);
      LHS.set(0,0,zSH - zLO);
      LHS.set(1,0,zTD - zLO);
      LHS.set(2,0,zdTD);
      DMatrixRMaj result = new DMatrixRMaj(3,1);
      CommonOps_DDRM.mult(m, LHS, result);
      
      return result;
   }
   
   double calculateSwingZdot(RobotSide side, double t)
   {
      // This is zd:
      DMatrixRMaj coeffs = new DMatrixRMaj(3,1);
      lrSwingCoeffs.get(side).get(coeffs);
      return (3*coeffs.get(0,0)*Math.pow(t,2) + 2*coeffs.get(1,0)*t + coeffs.get(2,0));
   }

   private SideDependentList<StateMachine<States, State>> setupStateMachines()
   {
      // States and Actions:
      StateMachineFactory<States, State> leftFactory = new StateMachineFactory<>(States.class);
      StateMachineFactory<States, State> rightFactory = new StateMachineFactory<>(States.class);

      leftFactory.setNamePrefix("leftState");
      rightFactory.setNamePrefix("rightState");
      
      leftFactory.setRegistry(registry);
      rightFactory.setRegistry(registry);
      
      leftFactory.buildClock(robot.getRobot().getYoTime());
      rightFactory.buildClock(robot.getRobot().getYoTime());
 
      // Left State Transitions:
      leftFactory.addTransition(States.SUPPORT, States.LIFT, new OtherHeelOnGroundCondition(RobotSide.LEFT));
      leftFactory.addTransition(States.LIFT, States.SWING, new HeelOffGroundCondition(RobotSide.LEFT));
      leftFactory.addTransition(States.SWING, States.SUPPORT, new HeelOnGroundCondition(RobotSide.LEFT));

      // Right State Transitions:
      rightFactory.addTransition(States.SUPPORT, States.LIFT, new OtherHeelOnGroundCondition(RobotSide.RIGHT));
      rightFactory.addTransition(States.LIFT, States.SWING, new HeelOffGroundCondition(RobotSide.RIGHT));
      rightFactory.addTransition(States.SWING, States.SUPPORT, new HeelOnGroundCondition(RobotSide.RIGHT));

      // Assemble the Left State Machine:
      leftFactory.addState(States.SUPPORT, new SupportState(RobotSide.LEFT));
      leftFactory.addState(States.LIFT, new LiftState(RobotSide.LEFT));
      leftFactory.addState(States.SWING, new SwingState(RobotSide.LEFT));

      // Assemble the Right State Machine:
      rightFactory.addState(States.SUPPORT, new SupportState(RobotSide.RIGHT));
      rightFactory.addState(States.LIFT, new LiftState(RobotSide.RIGHT));
      rightFactory.addState(States.SWING, new SwingState(RobotSide.RIGHT));

      // Create the state machines and set their initial states:
      return new SideDependentList<>(leftFactory.build(States.SUPPORT), rightFactory.build(States.LIFT));
   }

   private class SupportState implements State
   {
      private final RobotSide side;

      public SupportState(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public void onEntry()
      {
         // Record xTD for TDLO:
         xTD.set(getXcop(side));
      }

      @Override
      public void doAction(double timeInState)
      {
         // Support Side:
         controlSupportLeg(side);
         timeInStance.get(side).set(timeInState);
      }

      @Override
      public void onExit(double timeInState)
      {

      }
   }

   private class LiftState implements State
   {
      private final RobotSide side;

      public LiftState(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public void onEntry()
      {
         lrLastTime.get(side).set(-1.0);

         // Build the z-spline:
         Point3D pos_foot = robot.getFootPosition(side);
         lrSwingCoeffs.get(side).set(buildSwingZdot(pos_foot.getZ()));
         
         // Initialize the diff IK:
         DMatrixRMaj q_d = new DMatrixRMaj(2,1);
         q_d.set(0,0,robot.getHipAngle(side));
         q_d.set(1,0,robot.getKneeLength(side));
         lrq_d.get(side).set(q_d);
         lrx_d_foot.get(side).set(getXcop(side));
         
         inLiftState.get(side).set(true);
      }

      @Override
      public void doAction(double timeInState)
      {
         if (lrLastTime.get(side).getValue() < 0)
         {
            lrLastTime.get(side).set(timeInState);
         }
         double dT = timeInState - lrLastTime.get(side).getValue();
         lrLastTime.get(side).set(timeInState);
         
         double xCoP = getXcop(side.getOppositeSide());
         double desiredTD = simpleTDLOStepLocation(xCoP); // simple TDLO
//         double desiredTD = predictedTDLOStepLocation(timeInState,xTD.getValue(),xCoP); // predicted TDLO
         desiredxTD.get(side).set(desiredTD); // logging
         controlSwingLeg(side,timeInState,dT,desiredTD);
      }

      @Override
      public void onExit(double timeInState)
      {
         inLiftState.get(side).set(false);
      }
   }

   private class SwingState implements State
   {
      private final RobotSide side;

      public SwingState(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public void onEntry()
      {
         
      }

      @Override
      public void doAction(double timeInState)
      {
         if (lrLastTime.get(side).getValue() < 0)
         {
            lrLastTime.get(side).set(timeInState);
         }
         double dT = timeInState - lrLastTime.get(side).getValue();
         lrLastTime.get(side).set(timeInState);
         
         double xCoP = getXcop(side.getOppositeSide());
         double stanceTime = timeInStance.get(side.getOppositeSide()).getValue();
         double desiredTD = simpleTDLOStepLocation(xCoP); // simple TDLO
//         double desiredTD = predictedTDLOStepLocation(stanceTime,xTD.getValue(),xCoP); // predicted TDLO
         desiredxTD.get(side).set(desiredTD); // logging
         controlSwingLeg(side,stanceTime,dT,desiredTD);
      }

      @Override
      public void onExit(double timeInState)
      {
         
      }
   }

   // LIFT -> SWING: The LIFTING foot has cleared the ground:
   private class HeelOffGroundCondition implements StateTransitionCondition
   {
      private final RobotSide side;

      public HeelOffGroundCondition(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public boolean testCondition(double timeInCurrentState)
      {
         Point3D pos_foot = robot.getFootPosition(side); // GMN: What is behind this function?  Super-sensing?
         boolean result = (pos_foot.getZ() > swingHeight/8); 
//         boolean result = ((robot.getFootZForce(side) <= 0) && 
//                           (inLiftState.get(side).getValue() == true));
         return (result);
      }
   }

   // SUPPORT -> LIFT: The current STANCE foot detects that the OTHER foot has hit the ground:
   private class OtherHeelOnGroundCondition implements StateTransitionCondition
   {
      private final RobotSide side;

      public OtherHeelOnGroundCondition(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public boolean testCondition(double timeInCurrentState)
      {
         // Make the state transition simply be when the OTHER side is on the ground:
         boolean result = ((robot.getFootZForce(this.side.getOppositeSide()) > 10) && 
                           (inLiftState.get(side.getOppositeSide()).getValue() == false));
         return (result);
      }
   }

   // SWING -> SUPPORT: The current SWING foot hits the ground:
   private class HeelOnGroundCondition implements StateTransitionCondition
   {
      private final RobotSide side;

      public HeelOnGroundCondition(RobotSide side)
      {
         this.side = side;
      }

      @Override
      public boolean testCondition(double timeInCurrentState)
      {
         boolean result = ((robot.getFootZForce(this.side) > 10) &&
                           (inLiftState.get(side).getValue() == false));
         return (result);
         
//         LogTools.info("Side: {} getKneeForce(): {}", side, robot.getKneeForce(side));
      }
   }
}