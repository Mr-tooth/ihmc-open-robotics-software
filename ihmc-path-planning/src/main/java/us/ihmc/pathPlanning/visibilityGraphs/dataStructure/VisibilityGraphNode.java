package us.ihmc.pathPlanning.visibilityGraphs.dataStructure;

import java.util.ArrayList;
import java.util.List;

import us.ihmc.euclid.interfaces.EpsilonComparable;
import us.ihmc.euclid.interfaces.Transformable;
import us.ihmc.euclid.transform.interfaces.Transform;
import us.ihmc.euclid.tuple2D.Point2D;
import us.ihmc.euclid.tuple2D.interfaces.Point2DReadOnly;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;

public class VisibilityGraphNode implements Transformable, EpsilonComparable<VisibilityGraphNode>
{
   private final VisibilityGraphNavigableRegion visibilityGraphNavigableRegion;
   private final ConnectionPoint3D pointInWorld;
   private final Point2D point2DInLocal;

   private boolean edgesHaveBeenDetermined = false;

   private double costFromStart = Double.NaN;
   private double estimatedCostToGoal = Double.NaN;

   private boolean hasBeenExpanded = false;
   private VisibilityGraphNode bestParentNode = null;

   private final ArrayList<VisibilityGraphEdge> edges = new ArrayList<>();

   public int getRegionId()
   {
      return pointInWorld.getRegionId();
   }

   public VisibilityGraphNode(Point3DReadOnly pointInWorld, Point2DReadOnly pointInLocal, VisibilityGraphNavigableRegion visibilityGraphNavigableRegion)
   {
      this.visibilityGraphNavigableRegion = visibilityGraphNavigableRegion;
      this.pointInWorld = new ConnectionPoint3D(pointInWorld, visibilityGraphNavigableRegion.getMapId());
      this.point2DInLocal = new Point2D(pointInLocal);
   }

   public VisibilityGraphNavigableRegion getVisibilityGraphNavigableRegion()
   {
      return visibilityGraphNavigableRegion;
   }

   public Point2DReadOnly getPoint2DInLocal()
   {
      return point2DInLocal;
   }

   public void addEdge(VisibilityGraphEdge edge)
   {
      edges.add(edge);
   }

   public List<VisibilityGraphEdge> getEdges()
   {
      return edges;
   }

   public ConnectionPoint3D getPointInWorld()
   {
      return pointInWorld;
   }

   public double distance(VisibilityGraphNode target)
   {
      return pointInWorld.distance(target.pointInWorld);
   }

   public double distanceSquared(VisibilityGraphNode target)
   {
      return pointInWorld.distanceSquared(target.pointInWorld);
   }

   @Override
   public boolean epsilonEquals(VisibilityGraphNode other, double epsilon)
   {
      return pointInWorld.epsilonEquals(other.pointInWorld, epsilon);
   }

   @Override
   public void applyTransform(Transform transform)
   {
      pointInWorld.applyTransform(transform);
   }

   @Override
   public void applyInverseTransform(Transform transform)
   {
      pointInWorld.applyInverseTransform(transform);
   }

   @Override
   public String toString()
   {
      return pointInWorld.toString();
   }

   public double getCostFromStart()
   {
      return costFromStart;
   }

   public void setCostFromStart(double costFromStart, VisibilityGraphNode bestParentNode)
   {
      this.costFromStart = costFromStart;
      this.bestParentNode = bestParentNode;
   }

   public VisibilityGraphNode getBestParentNode()
   {
      return bestParentNode;
   }

   public double getEstimatedCostToGoal()
   {
      return estimatedCostToGoal;
   }

   public void setEstimatedCostToGoal(double estimatedCostToGoal)
   {
      this.estimatedCostToGoal = estimatedCostToGoal;
   }

   public boolean getHasBeenExpanded()
   {
      return hasBeenExpanded;
   }

   public void setHasBeenExpanded(boolean hasBeenExpanded)
   {
      this.hasBeenExpanded = hasBeenExpanded;
   }

   public boolean getEdgesHaveBeenDetermined()
   {
      return edgesHaveBeenDetermined;
   }

   public void setEdgesHaveBeenDetermined(boolean edgesHaveBeenDetermined)
   {
      this.edgesHaveBeenDetermined = edgesHaveBeenDetermined;
   }

}
