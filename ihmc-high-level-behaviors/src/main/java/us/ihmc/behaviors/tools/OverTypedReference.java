package us.ihmc.behaviors.tools;

import java.util.concurrent.atomic.AtomicReference;

public class OverTypedReference<T>
{
   private AtomicReference atomicReference;

   public OverTypedReference(AtomicReference atomicReference)
   {
      this.atomicReference = atomicReference;
   }

   public T get()
   {
      return (T) atomicReference.get();
   }
}
