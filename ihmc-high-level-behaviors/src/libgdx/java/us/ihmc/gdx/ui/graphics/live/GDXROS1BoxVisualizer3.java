package us.ihmc.gdx.ui.graphics.live;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g3d.*;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.attributes.TextureAttribute;
import com.badlogic.gdx.graphics.g3d.model.MeshPart;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.Pool;
import lidar_obstacle_detection.GDXBoxMessage;
import lidar_obstacle_detection.GDXBoxesMessage;
import org.lwjgl.opengl.GL32;
import us.ihmc.euclid.referenceFrame.FrameBoundingBox3D;
import us.ihmc.euclid.referenceFrame.FrameBox3D;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.referenceFrame.tools.ReferenceFrameTools;
import us.ihmc.euclid.transform.interfaces.RigidBodyTransformReadOnly;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;
import us.ihmc.euclid.tuple4D.Quaternion;
import us.ihmc.gdx.mesh.GDXMultiColorMeshBuilder;
import us.ihmc.tools.thread.MissingThreadTools;
import us.ihmc.tools.thread.ResettableExceptionHandlingExecutorService;
import us.ihmc.utilities.ros.RosMainNode;
import us.ihmc.utilities.ros.subscriber.AbstractRosTopicSubscriber;

import java.util.ArrayList;

public class GDXROS1BoxVisualizer3 implements RenderableProvider
{
   private final ResettableExceptionHandlingExecutorService executorService = MissingThreadTools.newSingleThreadExecutor(getClass().getSimpleName(), true, 1);
   private final ModelBuilder modelBuilder = new ModelBuilder();
   private ModelInstance modelInstance;
   private Model lastModel;
   private volatile Runnable toRender = null;

   private final ReferenceFrame sensorFrame;
   private final FrameBoundingBox3D boundingBox = new FrameBoundingBox3D();
   private final FrameBox3D box = new FrameBox3D();
   private final Point3D center = new Point3D();
   private final Quaternion zeroOrientation = new Quaternion();
   private final Point3D[] vertices = new Point3D[8];
   private final ArrayList<Point3DReadOnly> orderedVerticesForDrawing;

   public GDXROS1BoxVisualizer3(RosMainNode ros1Node, String ros1BoxTopic, ReferenceFrame sensorBaseFrame, RigidBodyTransformReadOnly baseToSensorTransform)
   {
      sensorFrame = ReferenceFrameTools.constructFrameWithUnchangingTransformFromParent("baseFrame",
                                                                                        ReferenceFrame.getWorldFrame(),
                                                                                        baseToSensorTransform);
      for (int i = 0; i < vertices.length; i++)
      {
         vertices[i] = new Point3D();
      }
      orderedVerticesForDrawing = new ArrayList<>();
      orderedVerticesForDrawing.add(vertices[0]); // x+y+z+  draw top
      orderedVerticesForDrawing.add(vertices[1]); // x-y-z+
      orderedVerticesForDrawing.add(vertices[3]); // x-y+z+
      orderedVerticesForDrawing.add(vertices[2]); // x+y-z+
      orderedVerticesForDrawing.add(vertices[0]); // x+y+z+
      orderedVerticesForDrawing.add(vertices[4]); // x+y+z-  go down
      orderedVerticesForDrawing.add(vertices[5]); // x-y-z-  leg 1
      orderedVerticesForDrawing.add(vertices[1]); // x-y-z+
      orderedVerticesForDrawing.add(vertices[5]); // x-y-z-
      orderedVerticesForDrawing.add(vertices[7]); // x-y+z-  leg 2
      orderedVerticesForDrawing.add(vertices[3]); // x-y+z+
      orderedVerticesForDrawing.add(vertices[7]); // x-y+z-
      orderedVerticesForDrawing.add(vertices[6]); // x+y-z-  leg 3
      orderedVerticesForDrawing.add(vertices[2]); // x+y-z+
      orderedVerticesForDrawing.add(vertices[6]); // x+y-z-
      orderedVerticesForDrawing.add(vertices[4]); // x+y+z-  leg 4

      ros1Node.attachSubscriber(ros1BoxTopic, new AbstractRosTopicSubscriber<GDXBoxesMessage>(GDXBoxesMessage._TYPE)
      {
         @Override
         public void onNewMessage(GDXBoxesMessage boxes)
         {
            queueRenderBoxesAsync(boxes);
         }
      });
   }

   public void render()
   {
      if (toRender != null)
      {
         toRender.run();
         toRender = null;
      }
   }

   private void queueRenderBoxesAsync(GDXBoxesMessage boxes)
   {
      executorService.clearQueueAndExecute(() -> generateMeshes(boxes));
   }

   public synchronized void generateMeshes(GDXBoxesMessage boxes)
   {
      double lineWidth = 0.03;
      GDXMultiColorMeshBuilder meshBuilder = new GDXMultiColorMeshBuilder();
      for (GDXBoxMessage message : boxes.getBoxes())
      {
         boundingBox.setToZero(sensorFrame);
         // Be robust to incorrect incoming data
         double xMin = Math.min(message.getXMin(), message.getXMax());
         double yMin = Math.min(message.getYMin(), message.getYMax());
         double zMin = Math.min(message.getZMin(), message.getZMax());
         double xMax = Math.max(message.getXMin(), message.getXMax());
         double yMax = Math.max(message.getYMin(), message.getYMax());
         double zMax = Math.max(message.getZMin(), message.getZMax());
         boundingBox.set(xMin, yMin, zMin, xMax, yMax, zMax);
         boundingBox.getCenterPoint(center);
         box.setIncludingFrame(sensorFrame,
                               center,
                               zeroOrientation,
                               boundingBox.getMaxX() - boundingBox.getMinX(),
                               boundingBox.getMaxY() - boundingBox.getMinY(),
                               boundingBox.getMaxZ() - boundingBox.getMinZ());
         box.changeFrame(ReferenceFrame.getWorldFrame());
         box.getVertices(vertices);

         meshBuilder.addMultiLine(orderedVerticesForDrawing, lineWidth, Color.RED, false);
      }

      toRender = () ->
      {
         modelBuilder.begin();
         Mesh mesh = meshBuilder.generateMesh();
         MeshPart meshPart = new MeshPart("xyz", mesh, 0, mesh.getNumIndices(), GL32.GL_TRIANGLES);
         Material material = new Material();
         Texture paletteTexture = new Texture(Gdx.files.classpath("palette.png"));
         material.set(TextureAttribute.createDiffuse(paletteTexture));
         material.set(ColorAttribute.createDiffuse(new Color(0.7f, 0.7f, 0.7f, 1.0f)));
         modelBuilder.part(meshPart, material);

         if (lastModel != null)
            lastModel.dispose();

         lastModel = modelBuilder.end();
         modelInstance = new ModelInstance(lastModel); // TODO: Clean up garbage and look into reusing the Model
      };
   }

   public void dispose()
   {
      executorService.destroy();
   }

   @Override
   public void getRenderables(Array<Renderable> renderables, Pool<Renderable> pool)
   {
      // sync over current and add
      if (modelInstance != null)
      {
         modelInstance.getRenderables(renderables, pool);
      }
   }
}
