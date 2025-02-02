package us.ihmc.robotics.controllers;

import us.ihmc.commons.MathTools;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoBoolean;
import us.ihmc.yoVariables.variable.YoDouble;
import us.ihmc.yoVariables.variable.YoEnum;

public class BacklashCompensatingPIDController extends PIDController
{
   private final YoDouble maxProportionalGain;
   private final YoDouble maxDerivativeGain;
   
   private final YoDouble lowGainReduction, gainReduction;
   private final YoDouble rampUpTime, rampDownTime, holdLowGainsTime;

   private final YoEnum<GainChangerState> gainChangerState;
   private final YoDouble switchTime;
   private final YoBoolean previousTorqueWasPositive;
   
   private boolean gainReductionUpToDate = false;
   
   public BacklashCompensatingPIDController(String suffix, YoRegistry registry)
   {
      super(suffix, registry);
      
      maxProportionalGain = new YoDouble("max_kp_" + suffix, registry);
      maxDerivativeGain = new YoDouble("max_kd_" + suffix, registry);
      gainReduction = new YoDouble("gainReduction_" + suffix, registry);
      lowGainReduction = new YoDouble("lowGainReduction_" + suffix, registry);

      gainChangerState = new YoEnum<GainChangerState>("gainChangerState_" + suffix, registry, GainChangerState.class);
      
      rampDownTime = new YoDouble("rampDownTime_" + suffix, registry);
      rampUpTime = new YoDouble("rampUpTime_" + suffix, registry);
      holdLowGainsTime = new YoDouble("holdLowGainsTime_" + suffix, registry);
      
      previousTorqueWasPositive = new YoBoolean("previousTorqueWasPositive_" + suffix, registry);
      switchTime = new YoDouble("switchTime_" + suffix, registry);
      
      gainChangerState.set(GainChangerState.LOW_GAINS);
      switchTime.set(0.0);
      previousTorqueWasPositive.set(false);
      gainReduction.set(1.0);
      lowGainReduction.set(0.5);
      
      rampUpTime.set(1.0);
      rampDownTime.set(0.3);
      holdLowGainsTime.set(0.5);
   }
   
   public void setLowGainBacklashReduction(double lowGainReduction)
   {
      this.lowGainReduction.set(lowGainReduction);
   }
   
   public void setRampUpTime(double rampUpTime)
   {
      this.rampUpTime.set(rampUpTime);
   }
   
   public void setRampDownTime(double rampDownTime)
   {
      this.rampDownTime.set(rampDownTime);
   }
   
   public void setHoldLowGainsTime(double holdLowGainsTime)
   {
      this.holdLowGainsTime.set(holdLowGainsTime);
   }
   
   public double getLowGainBacklashReduction()
   {
      return this.lowGainReduction.getDoubleValue();
   }
   
   public double getRampUpTime()
   {
      return this.rampUpTime.getDoubleValue();
   }
   
   public double getRampDownTime()
   {
      return this.rampDownTime.getDoubleValue();
   }
   
   public double getHoldLowGainsTime()
   {
      return this.holdLowGainsTime.getDoubleValue();
   }
   
   public void computeGainReduction(double time, double actuatorTorque)
   {      
      gainReductionUpToDate = true;
      
      switch(gainChangerState.getEnumValue())
      {
      case HIGH_GAINS:
      {
         gainReduction.set(1.0);
         
         if (signChanged(actuatorTorque))
         {
            gainChangerState.set(GainChangerState.RAMPDOWN);
            switchTime.set(time);
         }
         break;
      }
      
      case LOW_GAINS:
      {
         gainReduction.set(lowGainReduction.getDoubleValue());
         
         if (signChanged(actuatorTorque))
         {
            switchTime.set(time);
         }
         
         else if (time > switchTime.getDoubleValue() + holdLowGainsTime.getDoubleValue())
         {
            gainChangerState.set(GainChangerState.RAMPUP);
            switchTime.set(time);
         }

         break;
      }
      
      case RAMPDOWN:
      {
         double deltaTime = time - switchTime.getDoubleValue();
         double percentRampDown = deltaTime / rampDownTime.getDoubleValue();
         
         if (percentRampDown >= 1.0)
         {
            gainChangerState.set(GainChangerState.LOW_GAINS);
            switchTime.set(time);
         }
         
         percentRampDown = MathTools.clamp(percentRampDown, 0.0, 1.0);
         gainReduction.set(1.0 - percentRampDown * (1.0 - lowGainReduction.getDoubleValue()));
         
         break;
      }
      
      case RAMPUP:
      {
         double deltaTime = time - switchTime.getDoubleValue();
         double percentRampUp = deltaTime / rampUpTime.getDoubleValue();
         
         if (percentRampUp >= 1.0)
         {
            gainChangerState.set(GainChangerState.HIGH_GAINS);
            switchTime.set(time);
         }
         
         percentRampUp = MathTools.clamp(percentRampUp, 0.0, 1.0);
         gainReduction.set(lowGainReduction.getDoubleValue() + percentRampUp * (1.0 - lowGainReduction.getDoubleValue()));
         
         break;
      }
      
      default:
      {
         throw new RuntimeException("Shouldn't get here!");
      }
      }
   }
   
   private boolean signChanged(double actuatorTorque)
   {
      if ((previousTorqueWasPositive.getBooleanValue()) && (actuatorTorque <= 0.0))
      {
         previousTorqueWasPositive.set(false);
         return true;
      }
      else if ((!previousTorqueWasPositive.getBooleanValue()) && (actuatorTorque >= 0.0))
      {
         previousTorqueWasPositive.set(true);
         return true;
      }
      else return false;
   }

   @Override
   public double compute(double currentPosition, double desiredPosition, double currentRate, double desiredRate, double deltaTime)
   {
      checkGainReductionUpToDate();
      setGainsReducedIfBacklash();
      return super.compute(currentPosition, desiredPosition, currentRate, desiredRate, deltaTime);
   }

   @Override
   public double computeForAngles(double currentPosition, double desiredPosition, double currentRate, double desiredRate, double deltaTime)
   {
      checkGainReductionUpToDate();
      setGainsReducedIfBacklash();
      return super.computeForAngles(currentPosition, desiredPosition, currentRate, desiredRate, deltaTime);
   }
   
   private void checkGainReductionUpToDate()
   {
      if (!gainReductionUpToDate)
      {
         throw new RuntimeException("gain reduction is not up to date!");
      }
      gainReductionUpToDate = false;
   }
   
   @Override
   public double getProportionalGain()
   {
      return maxProportionalGain.getDoubleValue();
   }

   @Override
   public double getDerivativeGain()
   {
      return maxDerivativeGain.getDoubleValue();
   }

   @Override
   public void setProportionalGain(double proportionalGain)
   {
      maxProportionalGain.set(proportionalGain);
   }

   @Override
   public void setDerivativeGain(double derivativeGain)
   {
      maxDerivativeGain.set(derivativeGain);
   }
  
   
   private void setGainsReducedIfBacklash()
   {
      double proportionalGain = gainReduction.getDoubleValue() * maxProportionalGain.getDoubleValue();
      double derivativeGain = gainReduction.getDoubleValue() * maxDerivativeGain.getDoubleValue();
      
      super.setProportionalGain(proportionalGain);
      super.setDerivativeGain(derivativeGain);
      
   }


   private enum GainChangerState
   {
      LOW_GAINS, HIGH_GAINS, RAMPDOWN, RAMPUP;
   }

}
