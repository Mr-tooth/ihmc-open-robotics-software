package us.ihmc.gdx.simulation;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.ModelInstance;
import us.ihmc.euclid.geometry.Pose3D;
import us.ihmc.euclid.transform.RigidBodyTransform;
import us.ihmc.gdx.Lwjgl3ApplicationAdapter;
import us.ihmc.gdx.simulation.environment.GDXPhysicsSimulator;
import us.ihmc.gdx.tools.GDXModelPrimitives;
import us.ihmc.gdx.tools.GDXTools;
import us.ihmc.gdx.ui.GDXImGuiBasedUI;
import us.ihmc.scs2.definition.geometry.Box3DDefinition;
import us.ihmc.scs2.definition.geometry.GeometryDefinition;
import us.ihmc.scs2.definition.state.SixDoFJointState;
import us.ihmc.scs2.definition.visual.ColorDefinition;
import us.ihmc.scs2.definition.visual.VisualDefinition;
import us.ihmc.yoVariables.registry.YoNamespace;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoDouble;

public class GDXPhysicsDemo
{
   private final GDXImGuiBasedUI baseUI = new GDXImGuiBasedUI(getClass(),
                                                              "ihmc-open-robotics-software",
                                                              "ihmc-high-level-behaviors/src/test/resources");

   private GDXPhysicsSimulator physicsSimulator = new GDXPhysicsSimulator();
   private ModelInstance boxModelInstance;
   private ModelInstance slopeModelInstance;

   public GDXPhysicsDemo()
   {
      BoxRobotDefinition boxRobot = new BoxRobotDefinition();
      SixDoFJointState initialJointState = new SixDoFJointState();
      initialJointState.setConfiguration(new Pose3D(0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
      boxRobot.getRootJointDefinitions().get(0).setInitialJointState(initialJointState);

      physicsSimulator.getSession().addRobot(boxRobot);
      SlopeGroundDefinition slopeTerrain = new SlopeGroundDefinition();
      physicsSimulator.getSession().addTerrainObject(slopeTerrain);

      baseUI.launchGDXApplication(new Lwjgl3ApplicationAdapter()
      {
         @Override
         public void create()
         {
            baseUI.create();
            baseUI.get3DSceneManager().addModelInstance(new ModelInstance(GDXModelPrimitives.createCoordinateFrame(0.3)));

            VisualDefinition boxVisual = boxRobot.getRigidBodyDefinition("Box").getVisualDefinitions().get(0);
            GeometryDefinition geometryDefinition = boxVisual.getGeometryDefinition();
            if (geometryDefinition instanceof Box3DDefinition)
            {
               Box3DDefinition boxGeometry = (Box3DDefinition) geometryDefinition;
               boxModelInstance = GDXModelPrimitives.buildModelInstance(meshBuilder ->
               {
                  ColorDefinition diffuseColor = boxVisual.getMaterialDefinition().getDiffuseColor();
                  Color color = GDXTools.toGDX(diffuseColor.getRed(), diffuseColor.getGreen(), diffuseColor.getBlue(), diffuseColor.getAlpha());
                  meshBuilder.addBox(boxGeometry.getSizeX(), boxGeometry.getSizeY(), boxGeometry.getSizeZ(), color);
//                  meshBuilder.addBox(1.0f, 1.0f, 1.0f, Color.GREEN);
               }, boxGeometry.getName() + "231");
            }
            baseUI.get3DSceneManager().addModelInstance(boxModelInstance);
            GDXTools.toGDX(boxRobot.getRootBodyDefinition().getInertiaPose(), boxModelInstance.transform);

            VisualDefinition slopeVisual = slopeTerrain.getVisualDefinitions().get(0);
            Box3DDefinition slopeBox = (Box3DDefinition) slopeVisual.getGeometryDefinition();
            slopeModelInstance = GDXModelPrimitives.buildModelInstance(meshBuilder ->
            {
               ColorDefinition diffuseColor = slopeVisual.getMaterialDefinition().getDiffuseColor();
               Color color = GDXTools.toGDX(diffuseColor.getRed(), diffuseColor.getGreen(), diffuseColor.getBlue(), diffuseColor.getAlpha());
               meshBuilder.addBox(slopeBox.getSizeX(), slopeBox.getSizeY(), slopeBox.getSizeZ(), color);
            }, slopeBox.getName());
            baseUI.get3DSceneManager().addModelInstance(slopeModelInstance);
            GDXTools.toGDX(slopeVisual.getOriginPose(), slopeModelInstance.transform);

            physicsSimulator.getSession().startSessionThread();
            baseUI.getImGuiPanelManager().addPanel(physicsSimulator.getControlPanel());
         }

         @Override
         public void render()
         {
            YoRegistry boxRegistry = physicsSimulator.getSession().getPhysicsEngine().getPhysicsEngineRegistry().findRegistry(new YoNamespace("root.Box"));
            YoDouble x = (YoDouble) boxRegistry.getVariable("rootJointX");
            YoDouble y = (YoDouble) boxRegistry.getVariable("rootJointY");
            YoDouble z = (YoDouble) boxRegistry.getVariable("rootJointZ");
            YoDouble qx = (YoDouble) boxRegistry.getVariable("rootJointQx");
            YoDouble qy = (YoDouble) boxRegistry.getVariable("rootJointQy");
            YoDouble qz = (YoDouble) boxRegistry.getVariable("rootJointQz");
            YoDouble qs = (YoDouble) boxRegistry.getVariable("rootJointQs");
            RigidBodyTransform rigidBodyTransform = new RigidBodyTransform();
            rigidBodyTransform.getTranslation().set(x.getValue(), y.getValue(), z.getValue());
            rigidBodyTransform.getRotation().setQuaternion(qx.getValue(), qy.getValue(), qz.getValue(), qs.getValue());

            GDXTools.toGDX(rigidBodyTransform, boxModelInstance.transform);

            baseUI.renderBeforeOnScreenUI();
            baseUI.renderEnd();
         }

         @Override
         public void dispose()
         {
            baseUI.dispose();
         }
      });
   }

   public static void main(String[] args)
   {
      new GDXPhysicsDemo();
   }
}
