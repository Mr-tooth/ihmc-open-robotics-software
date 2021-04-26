package us.ihmc.gdx.ui.graphics;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Mesh;
import com.badlogic.gdx.graphics.g3d.Material;
import com.badlogic.gdx.graphics.g3d.ModelInstance;
import com.badlogic.gdx.graphics.g3d.Renderable;
import com.badlogic.gdx.graphics.g3d.RenderableProvider;
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.Pool;
import us.ihmc.euclid.Axis3D;
import us.ihmc.euclid.geometry.Line3D;
import us.ihmc.euclid.matrix.RotationMatrix;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple4D.Quaternion;
import us.ihmc.euclid.yawPitchRoll.YawPitchRoll;
import us.ihmc.gdx.mesh.GDXMeshBuilder;
import us.ihmc.gdx.mesh.GDXMeshDataInterpreter;
import us.ihmc.gdx.tools.GDXModelPrimitives;
import us.ihmc.gdx.tools.GDXTools;
import us.ihmc.graphicsDescription.MeshDataGenerator;
import us.ihmc.graphicsDescription.MeshDataHolder;

public class GDXPose3DWidget implements RenderableProvider
{
   private static final Color xAxisDefaultColor = new Color(0.9f, 0.4f, 0.4f, 0.4f);
   private static final Color yAxisDefaultColor = new Color(0.4f, 0.9f, 0.4f, 0.4f);
   private static final Color zAxisDefaultColor = new Color(0.4f, 0.4f, 0.9f, 0.4f);
   private static final Color centerDefaultColor = new Color(0.7f, 0.7f, 0.7f, 0.4f);

   private static final Color xAxisSelectedDefaultColor = new Color(0.9f, 0.3f, 0.3f, 0.9f);
   private static final Color yAxisSelectedDefaultColor = new Color(0.3f, 0.9f, 0.3f, 0.9f);
   private static final Color zAxisSelectedDefaultColor = new Color(0.3f, 0.3f, 0.9f, 0.9f);
   private static final Color centerSelectedDefaultColor = new Color(0.5f, 0.5f, 0.5f, 0.9f);

   private final Color[] axisColors = {xAxisDefaultColor, yAxisDefaultColor, zAxisDefaultColor};
   private final Color[] axisSelectedColors = {xAxisSelectedDefaultColor, yAxisSelectedDefaultColor, zAxisSelectedDefaultColor};

   private final RotationMatrix[] axisRotations = new RotationMatrix[3];
   private final ModelInstance[] angularControlModelInstances = new ModelInstance[3];

   public void create()
   {
//      Mesh angularControlHighlightMesh = angularHighlightMesh(radius, thickness);

      axisRotations[0] = new RotationMatrix(0.0, Math.PI / 2.0, 0.0);
      axisRotations[1] = new RotationMatrix(0.0, 0.0, -Math.PI / 2.0);
      axisRotations[2] = new RotationMatrix();

      for (Axis3D axis : Axis3D.values)
      {
         String axisName = axis.name().toLowerCase();

         double radius = 0.075f;
         double thickness = 0.01f;
         int resolution = 25;
         ModelInstance ring = GDXModelPrimitives.buildModelInstance(meshBuilder ->
            meshBuilder.addArcTorus(0.0, Math.PI * 2.0f, radius, thickness, resolution, axisColors[axis.ordinal()]), axisName);
         ring.materials.get(0).set(new BlendingAttribute(true, 0.5f));
         GDXTools.toGDX(axisRotations[axis.ordinal()], ring.transform);
         angularControlModelInstances[axis.ordinal()] = ring;
      }

   }

   public SixDoFSelection intersect(Line3D pickRay)
   {
      return SixDoFSelection.LINEAR_X;
   }


   @Override
   public void getRenderables(Array<Renderable> renderables, Pool<Renderable> pool)
   {
      for (Axis3D axis : Axis3D.values)
      {
         angularControlModelInstances[axis.ordinal()].getRenderables(renderables, pool);
      }
   }

   public static Mesh angularHighlightMesh(double majorRadius, double minorRadius)
   {
      return tetrahedronRingMesh(1.75 * minorRadius, 1.25 * minorRadius, 5);
   }

   public static Mesh linearControlMesh(double bodyRadius, double bodyLength, double headRadius, double headLength, double spacing)
   {
      GDXMeshBuilder meshBuilder = new GDXMeshBuilder();
      meshBuilder.addCylinder(bodyLength, bodyRadius, new Point3D(0.0, 0.0, 0.5 * spacing));
      meshBuilder.addCone(headLength, headRadius, new Point3D(0.0, 0.0, 0.5 * spacing + bodyLength));
      meshBuilder.addCylinder(bodyLength, bodyRadius, new Point3D(0.0, 0.0, -0.5 * spacing), new YawPitchRoll(0.0, Math.PI, 0.0));
      meshBuilder.addCone(headLength, headRadius, new Point3D(0.0, 0.0, -0.5 * spacing - bodyLength), new YawPitchRoll(0.0, Math.PI, 0.0));
      return meshBuilder.generateMesh();
   }

   public static Mesh linearControlHighlightMesh(double bodyRadius, double bodyLength, double spacing)
   {
      GDXMeshBuilder meshBuilder = new GDXMeshBuilder();

      int numberOfHighlights = 5;

      Point3D center = new Point3D(0, 0, 0.5 * spacing + 0.33 * bodyLength);
      MeshDataHolder ringMesh = tetrahedronRingMeshDataHolder(1.75 * bodyRadius, 1.25 * bodyRadius, numberOfHighlights);
      meshBuilder.addMesh(ringMesh, center);
      center.negate();
      meshBuilder.addMesh(ringMesh, center);

      return meshBuilder.generateMesh();
   }

   public static Mesh tetrahedronRingMesh(double ringRadius, double tetrahedronSize, int numberOfTetrahedrons)
   {
      return GDXMeshDataInterpreter.interpretMeshData(tetrahedronRingMeshDataHolder(ringRadius, tetrahedronSize, numberOfTetrahedrons));
   }

   public static MeshDataHolder tetrahedronRingMeshDataHolder(double ringRadius, double tetrahedronSize, int numberOfTetrahedrons)
   {
      GDXMeshBuilder meshBuilder = new GDXMeshBuilder();

      Point3D position = new Point3D();
      Point3D offset = new Point3D();
      Quaternion orientation = new Quaternion();

      for (int i = 0; i < numberOfTetrahedrons; i++)
      {
         MeshDataHolder tetrahedron = MeshDataGenerator.Tetrahedron(tetrahedronSize);
         orientation.setToYawOrientation(i * 2.0 * Math.PI / numberOfTetrahedrons);
         orientation.appendPitchRotation(0.5 * Math.PI);

         offset.set(0.0, 0.0, ringRadius);
         orientation.transform(offset);
         position.set(offset);
         meshBuilder.addMesh(tetrahedron, position, orientation);
      }

      return meshBuilder.generateMeshDataHolder();
   }
}
