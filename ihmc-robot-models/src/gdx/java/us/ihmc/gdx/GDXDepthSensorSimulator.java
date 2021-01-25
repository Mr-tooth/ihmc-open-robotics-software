package us.ihmc.gdx;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g3d.ModelBatch;
import com.badlogic.gdx.graphics.glutils.*;
import com.badlogic.gdx.math.Matrix4;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.utils.BufferUtils;
import com.badlogic.gdx.utils.viewport.ScreenViewport;
import imgui.internal.ImGui;
import org.lwjgl.opengl.GL32;
import us.ihmc.commons.MathTools;
import us.ihmc.commons.lists.RecyclingArrayList;
import us.ihmc.euclid.tuple3D.Point3D32;
import us.ihmc.gdx.imgui.ImGuiTools;
import us.ihmc.gdx.sceneManager.GDX3DSceneManager;
import us.ihmc.gdx.sceneManager.GDXSceneLevel;
import us.ihmc.gdx.tools.GDXTools;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Random;

/**
 * Lidar could be simulated with a viewportHeight of 1 and rotating the camera
 */
public class GDXDepthSensorSimulator
{
   private final String depthWindowName = "Depth";
   private final String colorWindowName = "Color";

   private final Random random = new Random();

   private final float fieldOfViewY;
   private final int imageWidth;
   private final int imageHeight;
   private final float minRange;
   private final float maxRange;
   private final Vector3 depthPoint = new Vector3();

   private PerspectiveCamera camera;
   private ModelBatch modelBatch;
   private ScreenViewport viewport;
   private SensorFrameBuffer frameBuffer;
   private RecyclingArrayList<Point3D32> points;

   private Pixmap depthWindowPixmap;
   private Texture depthWindowTexture;

   private ByteBuffer depthByteBuffer;
   private FloatBuffer depthFloatBuffer;

   public GDXDepthSensorSimulator(double fieldOfViewY, int imageWidth, int imageHeight, double minRange, double maxRange)
   {
      this.fieldOfViewY = (float) fieldOfViewY;
      this.imageWidth = imageWidth;
      this.imageHeight = imageHeight;
      this.minRange = (float) minRange;
      this.maxRange = (float) maxRange;
   }

   public void create()
   {
      camera = new PerspectiveCamera(fieldOfViewY, imageWidth, imageHeight);
      camera.near = minRange;
      camera.far = maxRange;
      viewport = new ScreenViewport(camera);

      modelBatch = new ModelBatch();

      SensorFrameBufferBuilder frameBufferBuilder = new SensorFrameBufferBuilder(imageWidth, imageHeight);
      frameBufferBuilder.addBasicColorTextureAttachment(Pixmap.Format.RGBA8888);
      frameBufferBuilder.addDepthTextureAttachment(GL30.GL_DEPTH_COMPONENT32F, GL30.GL_FLOAT);
      frameBuffer = frameBufferBuilder.build();

      depthByteBuffer = BufferUtils.newByteBuffer(imageWidth * imageHeight * 4);
      depthFloatBuffer = depthByteBuffer.asFloatBuffer();

      depthWindowPixmap = new Pixmap(imageWidth, imageHeight, Pixmap.Format.RGBA8888);
      depthWindowTexture = new Texture(new PixmapTextureData(depthWindowPixmap, null, false, false));

      points = new RecyclingArrayList<>(imageWidth * imageHeight, Point3D32::new);
   }

   public void render(GDX3DSceneManager sceneManager)
   {
      frameBuffer.begin();

      float clear = 0.0f;
      Gdx.gl.glClearColor(clear, clear, clear, 1.0f);
      Gdx.gl.glClear(GL32.GL_COLOR_BUFFER_BIT | GL32.GL_DEPTH_BUFFER_BIT);

      viewport.update(imageWidth, imageHeight);
      modelBatch.begin(camera);
      Gdx.gl.glViewport(0, 0, imageWidth, imageHeight);

      sceneManager.renderRegisteredObjectsWithEnvironment(modelBatch, GDXSceneLevel.REAL_ENVIRONMENT);

      modelBatch.end();

      Gdx.gl.glPixelStorei(GL20.GL_PACK_ALIGNMENT, 4);
      depthByteBuffer.rewind();
      Gdx.gl.glReadPixels(0, 0, imageWidth, imageHeight, GL30.GL_DEPTH_COMPONENT, GL30.GL_FLOAT, depthByteBuffer);

      frameBuffer.end();

      points.clear();

//      depthByteBuffer.rewind();
      depthFloatBuffer.rewind();
      for (int y = 0; y < imageHeight; y++)
      {
         for (int x = 0; x < imageWidth; x++)
         {
            float depthReading = depthFloatBuffer.get();

            float clippedDepth = (float) MathTools.clamp(depthReading, camera.near, camera.far);
            float pastNear = clippedDepth - camera.near;
            float worldRange = camera.far - camera.near;
            float colorRange = 1.0f;
            float grayscale = pastNear * colorRange / worldRange;
            grayscale = depthReading * colorRange / camera.far;
            int flippedY = imageHeight - y;

            depthWindowPixmap.drawPixel(x, flippedY, Color.rgba8888(grayscale, grayscale, grayscale, 1.0f));

            if (depthReading > camera.near && depthReading < camera.far)
            {
               depthPoint.x = (2.0f * x) / imageWidth - 1.0f;
               depthPoint.y = (2.0f * y) / imageHeight - 1.0f;
               depthPoint.z = 2.0f * depthReading - 1.0f;
               depthPoint.prj(camera.invProjectionView);

               Point3D32 point = points.add();
               GDXTools.toEuclid(depthPoint, point);
               point.addZ(random.nextDouble() * 0.007); // add some noise
            }
         }
      }

      depthWindowTexture.draw(depthWindowPixmap, 0, 0);
   }

   public void renderImGuiDepthWindow(GDX3DSceneManager sceneManager)
   {
      ImGui.begin(depthWindowName);

      float posX = ImGui.getWindowPosX();
      float posY = ImGui.getWindowPosY();
      float sizeX = ImGui.getWindowSizeX();
      float sizeY = ImGui.getWindowSizeY();

      int textureId = depthWindowTexture.getTextureObjectHandle();

      ImGui.getWindowDrawList().addImage(textureId, posX, posY, posX + sizeX, posY + sizeY);

      sceneManager.getCamera3D().addInputExclusionBox(ImGuiTools.windowBoundingBox());
      ImGui.end();
   }

   public void renderImGuiColorWindow(GDX3DSceneManager sceneManager)
   {
      ImGui.begin(colorWindowName);

      float posX = ImGui.getWindowPosX();
      float posY = ImGui.getWindowPosY();
      float sizeX = ImGui.getWindowSizeX();
      float sizeY = ImGui.getWindowSizeY();

      int textureId = frameBuffer.getColorTexture().getTextureObjectHandle();

      ImGui.getWindowDrawList().addImage(textureId, posX, posY + sizeY, posX + sizeX, posY);

      sceneManager.getCamera3D().addInputExclusionBox(ImGuiTools.windowBoundingBox());
      ImGui.end();
   }

   public void dispose()
   {
      frameBuffer.dispose();
      depthWindowTexture.dispose();
      modelBatch.dispose();
   }

   public void setCameraWorldTransform(Matrix4 worldTransform)
   {
      camera.position.setZero();
      camera.up.set(0.0f, 0.0f, 1.0f);
      camera.direction.set(1.0f, 0.0f, 0.0f);
      camera.transform(worldTransform);
   }

   public PerspectiveCamera getCamera()
   {
      return camera;
   }

   public RecyclingArrayList<Point3D32> getPoints()
   {
      return points;
   }

   public String getDepthWindowName()
   {
      return depthWindowName;
   }

   public String getColorWindowName()
   {
      return colorWindowName;
   }
}
