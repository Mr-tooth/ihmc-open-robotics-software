package us.ihmc.humanoidBehaviors.ui;

import javafx.scene.Group;
import javafx.scene.SubScene;
import us.ihmc.avatar.drcRobot.DRCRobotModel;
import us.ihmc.messager.Messager;
import us.ihmc.ros2.Ros2NodeInterface;

public abstract class BehaviorUIInterface extends Group
{
   public abstract void init(SubScene sceneNode, Group group2D, Ros2NodeInterface ros2Node, Messager behaviorMessager, DRCRobotModel robotModel);

   public abstract void setEnabled(boolean enabled);
}
