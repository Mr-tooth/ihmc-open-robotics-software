package us.ihmc.behaviors.javafx.model.interfaces;

import us.ihmc.euclid.tuple3D.interfaces.Point3DBasics;

public interface PoseEditable extends PositionEditable, OrientationEditable
{
   Point3DBasics getPosition();
}
