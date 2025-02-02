package us.ihmc.simulationConstructionSetTools.robotController;

import java.util.List;

import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.simulationconstructionset.ExternalForcePoint;
import us.ihmc.simulationconstructionset.util.RobotController;
import us.ihmc.simulationconstructionset.util.ground.Contactable;
import us.ihmc.simulationconstructionset.util.ground.SimpleStickSlipContactModel;

public class ContactController implements RobotController
{
   private final YoRegistry registry;

   private final SimpleStickSlipContactModel contactModel;

   public ContactController()
   {
      this("");
   }

   public ContactController(String namePrefix)
   {
      registry = new YoRegistry(namePrefix + "ContactController");
      contactModel = new SimpleStickSlipContactModel(namePrefix + "simpleContact", registry);
   }

   public void addContactPoints(List<? extends ExternalForcePoint> contactPoints)
   {
      for (ExternalForcePoint contactPoint : contactPoints)
      {
         addContactPoint(contactPoint);
      }
   }

   public void addContactPoint(ExternalForcePoint contactPoint)
   {
      contactModel.addContactPoint(contactPoint);
   }

   public void addContactables(List<? extends Contactable> contactables)
   {
      for (Contactable contactable : contactables)
      {
         addContactable(contactable);
      }
   }

   public void addContactable(Contactable contactable)
   {
      contactModel.addContactable(contactable);
   }

   public void setContactParameters(double kContact, double bContact, double alphaStick, double alphaSlip)
   {
      contactModel.setKContact(kContact);
      contactModel.setBContact(bContact);
      contactModel.setFrictionCoefficients(alphaStick, alphaSlip);
   }

   @Override
   public void initialize()
   {
   }


   @Override
   public void doControl()
   {
      contactModel.doContact();
   }

   @Override
   public YoRegistry getYoRegistry()
   {
      return registry;
   }

   @Override
   public String getName()
   {
      return registry.getName();
   }

   @Override
   public String getDescription()
   {
      return getName();
   }

}
