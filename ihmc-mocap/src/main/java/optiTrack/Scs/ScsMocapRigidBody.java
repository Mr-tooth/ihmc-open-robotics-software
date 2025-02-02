package optiTrack.Scs;

import java.util.ArrayList;

import optiTrack.MocapMarker;
import optiTrack.MocapRigidBody;
import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.euclid.tuple3D.Vector3D;
import us.ihmc.euclid.tuple4D.Quaternion;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoBoolean;
import us.ihmc.yoVariables.variable.YoDouble;
import us.ihmc.commons.thread.ThreadTools;

public class ScsMocapRigidBody
{
   private int id;

   private ArrayList<MocapMarker> listOfAssociatedMarkers;
   private YoRegistry registry;
   private YoDouble xPos;
   private YoDouble yPos;
   private YoDouble zPos;
   private YoDouble qx;
   private YoDouble qy;
   private YoDouble qz;
   private YoDouble qw;
   private YoBoolean isTracked;
   
   private YoDouble xVel;
   private YoDouble yVel;
   private YoDouble zVel;
   
   private Vector3D lastPosition = new Vector3D();
   private Vector3D currentPosition = new Vector3D();
   long lastTimeUpdated = System.nanoTime();
   
   boolean pause = false;

   public ScsMocapRigidBody(int id, Vector3D position, Quaternion orientation, ArrayList<MocapMarker> listOfAssociatedMarkers, boolean isTracked)
   {
      registry = new YoRegistry("rb_" + id);
      xPos = new YoDouble("xPos", registry);
      yPos = new YoDouble("yPos", registry);
      zPos = new YoDouble("zPos", registry);
      qx = new YoDouble("qx", registry);
      qy = new YoDouble("qy", registry);
      qz = new YoDouble("qz", registry);
      qw = new YoDouble("qw", registry);
      this.isTracked = new YoBoolean("", registry);
      
      
      xVel = new YoDouble("xVel", registry);
      yVel = new YoDouble("yVel", registry);
      zVel = new YoDouble("zVel", registry);

      this.id = id;

      xPos.set(position.getX());
      yPos.set(position.getY());
      zPos.set(position.getZ());
      qx.set(orientation.getX());
      qy.set(orientation.getY());
      qz.set(orientation.getZ());
      qw.set(orientation.getS());

      this.listOfAssociatedMarkers = listOfAssociatedMarkers;
      this.isTracked.set(isTracked);
   }

   public ScsMocapRigidBody(MocapRigidBody mocapRigidBody)
   {
      this(mocapRigidBody.getId(), mocapRigidBody.getPosition(), mocapRigidBody.getOrientation(), mocapRigidBody.getListOfAssociatedMarkers(),
            mocapRigidBody.dataValid);
   }

   public int getId()
   {
      return id;
   }
   
   public YoRegistry getRegistry()
   {
      return registry;
   }
   
   public void pause(boolean pause)
   {
      this.pause = pause;
   }

   public void update(MocapRigidBody rb)
   {
      if(!pause)
      {
         xPos.set(rb.xPosition);
         yPos.set(rb.yPosition);
         zPos.set(rb.zPosition);
         qx.set(rb.qx);
         qy.set(rb.qy);
         qz.set(rb.qz);
         qw.set(rb.qw);
         
         currentPosition.set(rb.xPosition, rb.yPosition, rb.zPosition);
         
         long thisTime = System.currentTimeMillis();
         long timeStep = thisTime - lastTimeUpdated;
         lastTimeUpdated = System.currentTimeMillis();
         
         xVel.set((currentPosition.getX() - lastPosition.getX())/(timeStep/1000.0));
         yVel.set((currentPosition.getY() - lastPosition.getY())/(timeStep/1000.0));
         zVel.set((currentPosition.getZ() - lastPosition.getZ())/(timeStep/1000.0));
         
         lastPosition = new Vector3D(currentPosition.getX(), currentPosition.getY(), currentPosition.getZ());
         ThreadTools.sleep(3);
      }
   }

   public ArrayList<MocapMarker> getListOfAssociatedMarkers()
   {
      return listOfAssociatedMarkers;
   }

   public String toString()
   {
      String message = "\n";
      message = message + "RigidBody ID: " + id;
      message = message + "\nTracked : " + isTracked.getBooleanValue();
      message = message + "\nX: " + this.xPos.getDoubleValue() + " - Y: " + this.yPos.getDoubleValue() + " - Z: " + this.zPos.getDoubleValue();
      message = message + "\nqX: " + this.qx + " - qY: " + this.qy + " - qZ: " + this.qz + " - qW: " + this.qw;
      message = message + "\n# of Markers in rigid body: " + listOfAssociatedMarkers.size();

      for (int i = 0; i < listOfAssociatedMarkers.size(); i++)
      {
         message = message + "\nMarker " + i + " is at: " + listOfAssociatedMarkers.get(i).getPosition() + "  and has size: "
               + listOfAssociatedMarkers.get(i).getMarkerSize() + "m";
      }

      return message;
   }

   private final Quaternion quaternion = new Quaternion();

   public void getPose(RigidBodyTransform pose)
   {
      quaternion.set(qx.getDoubleValue(), qy.getDoubleValue(), qz.getDoubleValue(), qw.getDoubleValue());
      pose.getRotation().set(quaternion);
      pose.getTranslation().set(xPos.getDoubleValue(), yPos.getDoubleValue(), zPos.getDoubleValue());
   }
}

//~ Formatted by Jindent --- http://www.jindent.com
