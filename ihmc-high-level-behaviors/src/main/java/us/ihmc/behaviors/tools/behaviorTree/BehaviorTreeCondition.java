package us.ihmc.behaviors.tools.behaviorTree;

import java.util.function.BooleanSupplier;

import static us.ihmc.behaviors.tools.behaviorTree.BehaviorTreeNodeStatus.*;

/**
 * A behavior tree action that draws from a boolean supplier.
 */
public class BehaviorTreeCondition extends BehaviorTreeAction
{
   private final BooleanSupplier conditionSupplier;

   public BehaviorTreeCondition(BooleanSupplier conditionSupplier)
   {
      this.conditionSupplier = conditionSupplier;
   }

   protected boolean checkCondition()
   {
      return conditionSupplier.getAsBoolean();
   }

   @Override
   public BehaviorTreeNodeStatus tickInternal()
   {
      boolean success = checkCondition();

      if (success)
      {
         return SUCCESS;
      }
      else
      {
         return FAILURE;
      }
   }
}
