package us.ihmc.gdx.simulation.environment.object.objects;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.Model;
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute;
import us.ihmc.euclid.shape.primitives.Box3D;
import us.ihmc.euclid.shape.primitives.Sphere3D;
import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.gdx.lighting.GDXPointLight;
import us.ihmc.gdx.simulation.environment.object.GDXEnvironmentObject;
import us.ihmc.gdx.simulation.environment.object.GDXEnvironmentObjectFactory;
import us.ihmc.gdx.tools.GDXModelPrimitives;
import us.ihmc.gdx.tools.GDXTools;
import us.ihmc.graphicsDescription.appearance.YoAppearance;

public class GDXPointLightObject extends GDXEnvironmentObject
{
   public static final String NAME = "Point Light";
   public static final GDXEnvironmentObjectFactory FACTORY = new GDXEnvironmentObjectFactory(NAME, GDXPointLightObject.class);
   private final GDXPointLight light;

   public GDXPointLightObject()
   {
      super(NAME, FACTORY);
      light = new GDXPointLight();

      Model model = GDXModelPrimitives.buildModel(meshBuilder -> meshBuilder.addSphere(0.1f, Color.YELLOW), "pointModel");

      RigidBodyTransform collisionShapeOffset = new RigidBodyTransform();
      Box3D collisionBox = new Box3D(0.1f, 0.1f, 0.1f);

      Sphere3D boundingSphere = new Sphere3D(collisionBox.getSize().length() / 2.0);

      Model collisionGraphic = GDXModelPrimitives.buildModel(meshBuilder ->
      {
         Color color = GDXTools.toGDX(YoAppearance.LightSkyBlue());
         meshBuilder.addSphere(0.11f, color);
      }, getPascalCasedName() + "CollisionModel" + getObjectIndex());
      collisionGraphic.materials.get(0).set(new BlendingAttribute(true, 0.4f));
      RigidBodyTransform wholeThingOffset = new RigidBodyTransform();
      create(model, collisionShapeOffset, wholeThingOffset, boundingSphere, collisionBox, collisionBox::isPointInside, collisionGraphic);
   }

   @Override
   protected void updateRenderablesPoses()
   {
      super.updateRenderablesPoses();

      light.getPosition().set(getObjectTransform().getTranslation());
      light.update();
   }

   public GDXPointLight getLight()
   {
      return light;
   }
}
