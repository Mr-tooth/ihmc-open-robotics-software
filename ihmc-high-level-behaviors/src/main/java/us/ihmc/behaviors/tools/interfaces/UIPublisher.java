package us.ihmc.behaviors.tools.interfaces;

import us.ihmc.messager.MessagerAPIFactory;

public interface UIPublisher
{
   <T> void publishToUI(MessagerAPIFactory.Topic<T> topic, T message);
}
