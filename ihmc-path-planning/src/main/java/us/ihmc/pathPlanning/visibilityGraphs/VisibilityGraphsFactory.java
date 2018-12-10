package us.ihmc.pathPlanning.visibilityGraphs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import us.ihmc.euclid.geometry.BoundingBox3D;
import us.ihmc.euclid.tuple2D.Point2D;
import us.ihmc.euclid.tuple2D.interfaces.Point2DReadOnly;
import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;
import us.ihmc.pathPlanning.visibilityGraphs.clusterManagement.Cluster;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.Connection;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.ConnectionPoint3D;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.InterRegionVisibilityMap;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.NavigableRegion;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.SingleSourceVisibilityMap;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityMap;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityMapWithNavigableRegion;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.InterRegionConnectionFilter;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.NavigableExtrusionDistanceCalculator;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.ObstacleExtrusionDistanceCalculator;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.ObstacleRegionFilter;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.PlanarRegionFilter;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.VisibilityGraphsParameters;
import us.ihmc.pathPlanning.visibilityGraphs.tools.PlanarRegionTools;
import us.ihmc.pathPlanning.visibilityGraphs.tools.VisibilityTools;
import us.ihmc.robotics.geometry.PlanarRegion;

public class VisibilityGraphsFactory
{
   /**
    * I believe these filters are now not useful anymore, but I haven't had the time to make sure
    * they're obsolete. When disabled, everything still looks good.
    * 
    * +++JEP: Pretty sure they are obsolete now. Turning them off.
    */
   private static final boolean ENABLE_GREEDY_FILTERS = false;

   // Whether to create the inter regions using the cluster points or , if false, to create them
   // after the inner regions are created, using the inner region maps,
   // The former (true) I believe is preferable since I think we should be creating inter regions first, then inner regions.
   // Also, could then do the two in parallel instead of sequentially.
   private static final boolean CREATE_INTER_REGIONS_USING_CLUSTER_POINTS = true;

   public static List<VisibilityMapWithNavigableRegion> createVisibilityMapsWithNavigableRegions(List<PlanarRegion> allRegions,
                                                                                                 VisibilityGraphsParameters parameters)
   {
      if (allRegions.isEmpty())
         return null;

      List<VisibilityMapWithNavigableRegion> navigableRegions = createNavigableRegionAndListOfVisibilityMaps(allRegions, parameters);
      createStaticVisibilityMapsForNavigableRegions(navigableRegions);

      return navigableRegions;
   }

   public static List<VisibilityMapWithNavigableRegion> createNavigableRegionAndListOfVisibilityMaps(List<PlanarRegion> allRegions,
                                                                                                     VisibilityGraphsParameters parameters)
   {
      if (allRegions.isEmpty())
         return null;

      List<NavigableRegion> navigableRegions = NavigableRegionsFactory.createNavigableRegionButNotVisibilityMaps(allRegions, parameters);

      List<VisibilityMapWithNavigableRegion> visibilityMapsWithNavigableRegions = new ArrayList<>(allRegions.size());

      for (int i = 0; i < navigableRegions.size(); i++)
      {
         NavigableRegion navigableRegion = navigableRegions.get(i);
         visibilityMapsWithNavigableRegions.add(new VisibilityMapWithNavigableRegion(navigableRegion));
      }

      return visibilityMapsWithNavigableRegions;
   }

   

   public static VisibilityMapWithNavigableRegion createVisibilityMapsWithNavigableRegions(PlanarRegion region, List<PlanarRegion> otherRegions,
                                                                                           double orthogonalAngle, double clusterResolution,
                                                                                           ObstacleRegionFilter obstacleRegionFilter, PlanarRegionFilter filter,
                                                                                           NavigableExtrusionDistanceCalculator navigableCalculator,
                                                                                           ObstacleExtrusionDistanceCalculator obstacleCalculator)
   {
      NavigableRegion navigableRegion = NavigableRegionsFactory.createNavigableRegionButNotVisibilityMaps(region, otherRegions, orthogonalAngle, clusterResolution,
                                                                                  obstacleRegionFilter, filter, navigableCalculator, obstacleCalculator);

      VisibilityMapWithNavigableRegion visibilityMapWithNavigableRegion = new VisibilityMapWithNavigableRegion(navigableRegion);
      createStaticVisibilityMapsForNavigableRegion(visibilityMapWithNavigableRegion);

      return visibilityMapWithNavigableRegion;
   }

   
   public static void createStaticVisibilityMapsForNavigableRegions(List<VisibilityMapWithNavigableRegion> navigableRegions)
   {
      if (navigableRegions == null)
         return;

      for (int navigableRegionIndex = 0; navigableRegionIndex < navigableRegions.size(); navigableRegionIndex++)
      {
         VisibilityMapWithNavigableRegion navigableRegion = navigableRegions.get(navigableRegionIndex);
         createStaticVisibilityMapsForNavigableRegion(navigableRegion);
      }
   }

   private static void createStaticVisibilityMapsForNavigableRegion(VisibilityMapWithNavigableRegion navigableRegion)
   {
      Collection<Connection> connectionsForMap = VisibilityTools.createStaticVisibilityMap(navigableRegion);

      if (ENABLE_GREEDY_FILTERS)
      {
         PlanarRegion homeRegion = navigableRegion.getHomePlanarRegion();
         connectionsForMap = VisibilityTools.removeConnectionsFromExtrusionsOutsideRegions(connectionsForMap, homeRegion);
         connectionsForMap = VisibilityTools.removeConnectionsFromExtrusionsInsideNoGoZones(connectionsForMap, navigableRegion.getAllClusters());
      }

      VisibilityMap visibilityMap = new VisibilityMap();
      visibilityMap.setConnections(connectionsForMap);
      navigableRegion.setVisibilityMapInLocal(visibilityMap);
   }

   /**
    * Creates a visibility map using the given {@code source} and connect it to all the visibility
    * connection points of the host region's map.
    * <p>
    * The host region is defined as the region that contains the given {@code source}.
    * </p>
    * <p>
    * When the source is located inside non accessible zone on the host region, it then either
    * connected to the closest connection point of the host region's map or the closest connection
    * from the given {@code potentialFallbackMap}, whichever is the closest.
    * </p>
    * 
    * @param source the single source used to build the visibility map.
    * @param navigableRegions the list of navigable regions among which the host is to be found. Not
    *           modified.
    * @param searchHostEpsilon espilon used during the search. When positive, it is equivalent to
    *           growing all the regions before testing if the {@code source} is inside.
    * @param potentialFallbackMap in case the source is located in a non accessible zone, the
    *           fallback map might be used to connect the source. Additional connections may be
    *           added to the map.
    * @return the new map or {@code null} if a host region could not be found.
    */
   public static SingleSourceVisibilityMap createSingleSourceVisibilityMap(Point3DReadOnly source, NavigableRegions navigableRegions,
                                                                           double searchHostEpsilon, VisibilityMap potentialFallbackMap)
   {
      NavigableRegion hostNavigableRegion = PlanarRegionTools.getNavigableRegionContainingThisPoint(source, navigableRegions, searchHostEpsilon);

      if (hostNavigableRegion == null)
         return null;

      Point3D sourceInLocal = new Point3D(source);
      hostNavigableRegion.transformFromWorldToLocal(sourceInLocal);
      int mapId = hostNavigableRegion.getMapId();
      
      VisibilityMapWithNavigableRegion hostRegion = new VisibilityMapWithNavigableRegion(hostNavigableRegion);

      Set<Connection> connections = VisibilityTools.createStaticVisibilityMap(sourceInLocal, mapId, hostRegion.getAllClusters(), mapId);

      if (!connections.isEmpty())
         return new SingleSourceVisibilityMap(source, connections, hostRegion);

      return connectSourceToHostOrFallbackMap(source, potentialFallbackMap, hostRegion);
   }

   public static SingleSourceVisibilityMap connectToFallbackMap(Point3DReadOnly source, int sourceId, double maxConnectionLength, VisibilityMap fallbackMap)
   {

      double minDistance = Double.POSITIVE_INFINITY;
      Connection closestConnection = null;

      for (Connection connection : fallbackMap)
      {
         double distance = connection.distanceSquared(source);

         if (distance < minDistance)
         {
            minDistance = distance;
            closestConnection = connection;
         }
      }

      if (minDistance > maxConnectionLength)
         return null;

      Set<Connection> connections = new HashSet<>();
      ConnectionPoint3D sourceConnectionPoint = new ConnectionPoint3D(source, sourceId);
      double percentage = closestConnection.percentageAlongConnection(source);
      double epsilon = 1.0e-3;
      if (percentage <= epsilon)
      {
         connections.add(new Connection(sourceConnectionPoint, closestConnection.getSourcePoint()));
      }
      else if (percentage >= 1.0 - epsilon)
      {
         connections.add(new Connection(sourceConnectionPoint, closestConnection.getTargetPoint()));
      }
      else
      { // Let's create an connection point on the connection.
         ConnectionPoint3D newConnectionPoint = closestConnection.getPointGivenPercentage(percentage, sourceId);

         fallbackMap.addConnection(new Connection(closestConnection.getSourcePoint(), newConnectionPoint));
         fallbackMap.addConnection(new Connection(newConnectionPoint, closestConnection.getTargetPoint()));

         connections.add(new Connection(sourceConnectionPoint, newConnectionPoint));
      }

      return new SingleSourceVisibilityMap(source, sourceId, connections);
   }

   private static SingleSourceVisibilityMap connectSourceToHostOrFallbackMap(Point3DReadOnly sourceInworld, VisibilityMap fallbackMap,
                                                                             VisibilityMapWithNavigableRegion hostRegion)
   {
      Point3D sourceInLocal = new Point3D(sourceInworld);
      hostRegion.transformFromWorldToLocal(sourceInLocal);
      int mapId = hostRegion.getMapId();

      Set<Connection> connections = new HashSet<>();
      double minDistance = Double.POSITIVE_INFINITY;
      ConnectionPoint3D closestHostPoint = null;

      VisibilityMap hostMapInLocal = hostRegion.getVisibilityMapInLocal();
      hostMapInLocal.computeVertices();

      for (ConnectionPoint3D connectionPoint : hostMapInLocal.getVertices())
      {
         double distance = connectionPoint.distanceSquared(sourceInLocal);

         if (distance < minDistance)
         {
            minDistance = distance;
            closestHostPoint = connectionPoint;
         }
      }

      Connection closestFallbackConnection = null;

      for (Connection connection : fallbackMap)
      {
         double distance = connection.distanceSquared(sourceInworld);

         if (distance < minDistance)
         {
            minDistance = distance;
            closestFallbackConnection = connection;
            closestHostPoint = null;
         }
      }

      if (closestHostPoint != null)
      { // Make the connection to the host
         ConnectionPoint3D sourceConnectionPoint = new ConnectionPoint3D(sourceInLocal, mapId);
         connections.add(new Connection(sourceConnectionPoint, closestHostPoint));
         return new SingleSourceVisibilityMap(sourceInworld, connections, hostRegion);
      }
      else
      { // Make the connection to the fallback map
         ConnectionPoint3D sourceConnectionPoint = new ConnectionPoint3D(sourceInworld, mapId);
         double percentage = closestFallbackConnection.percentageAlongConnection(sourceInworld);
         double epsilon = 1.0e-3;
         if (percentage <= epsilon)
         {
            connections.add(new Connection(sourceConnectionPoint, closestFallbackConnection.getSourcePoint()));
         }
         else if (percentage >= 1.0 - epsilon)
         {
            connections.add(new Connection(sourceConnectionPoint, closestFallbackConnection.getTargetPoint()));
         }
         else
         { // Let's create an connection point on the connection.
            ConnectionPoint3D newConnectionPoint = closestFallbackConnection.getPointGivenPercentage(percentage, mapId);

            fallbackMap.addConnection(new Connection(closestFallbackConnection.getSourcePoint(), newConnectionPoint));
            fallbackMap.addConnection(new Connection(newConnectionPoint, closestFallbackConnection.getTargetPoint()));

            connections.add(new Connection(sourceConnectionPoint, newConnectionPoint));
         }

         return new SingleSourceVisibilityMap(sourceInworld, mapId, connections);
      }
   }

   public static SingleSourceVisibilityMap connectToClosestPoints(ConnectionPoint3D source, int maximumNumberOfConnections,
                                                                  List<VisibilityMapWithNavigableRegion> navigableRegions, int mapId)
   {
      List<Connection> allConnections = new ArrayList<>();

      for (int i = 0; i < navigableRegions.size(); i++)
      {
         VisibilityMap targetMap = navigableRegions.get(i).getVisibilityMapInWorld();
         Set<ConnectionPoint3D> targetPoints = targetMap.getVertices();

         for (ConnectionPoint3D targetPoint : targetPoints)
         {
            allConnections.add(new Connection(source, targetPoint));
         }
      }

      Collections.sort(allConnections, (c1, c2) -> {
         double c1LengthSquared = c1.lengthSquared();
         double c2LengthSquared = c2.lengthSquared();

         return c1LengthSquared < c2LengthSquared ? -1 : 1;
      });

      HashSet<Connection> connections = new HashSet<>();
      connections.addAll(allConnections.subList(0, maximumNumberOfConnections));

      return new SingleSourceVisibilityMap(source, mapId, connections);
   }

   public static InterRegionVisibilityMap createInterRegionVisibilityMap(List<VisibilityMapWithNavigableRegion> navigableRegions,
                                                                         InterRegionConnectionFilter filter)
   {
      if (CREATE_INTER_REGIONS_USING_CLUSTER_POINTS)
      {
         return createInterRegionVisibilityMapUsingClusterPoints(navigableRegions, filter);
      }
      else
      {
         return createInterRegionVisibilityMapUsingInnerVisibilityMaps(navigableRegions, filter);
      }
   }

   public static InterRegionVisibilityMap createInterRegionVisibilityMapUsingClusterPoints(List<VisibilityMapWithNavigableRegion> navigableRegions,
                                                                                           InterRegionConnectionFilter filter)
   {
      InterRegionVisibilityMap map = new InterRegionVisibilityMap();

      for (int sourceMapIndex = 0; sourceMapIndex < navigableRegions.size(); sourceMapIndex++)
      {
         VisibilityMapWithNavigableRegion sourceRegion = navigableRegions.get(sourceMapIndex);
         for (int targetMapIndex = sourceMapIndex + 1; targetMapIndex < navigableRegions.size(); targetMapIndex++)
         {
            VisibilityMapWithNavigableRegion targetRegion = navigableRegions.get(targetMapIndex);

            ArrayList<Connection> connectionsBetweenTwoNavigableRegions = createInterRegionVisibilityConnectionsUsingClusterPoints(sourceRegion, targetRegion,
                                                                                                                                   filter);
            map.addConnections(connectionsBetweenTwoNavigableRegions);
         }
      }

      return map;
   }

   //TODO: +++JEP: Get rid of these stats after optimized.
   private static int numberPlanarRegionBoundingBoxesTooFar = 0, totalSourceTargetChecks = 0, numberPassValidFilter = 0, numberNotInsideSourceRegion = 0,
         numberNotInsideTargetRegion = 0, numberSourcesInNoGoZones = 0, numberTargetsInNoGoZones = 0, numberValidConnections = 0;

   public static ArrayList<Connection> createInterRegionVisibilityConnectionsUsingClusterPoints(VisibilityMapWithNavigableRegion sourceNavigableRegion,
                                                                                                VisibilityMapWithNavigableRegion targetNavigableRegion,
                                                                                                InterRegionConnectionFilter filter)
   {
      ArrayList<Connection> connections = new ArrayList<Connection>();

      int sourceId = sourceNavigableRegion.getMapId();
      int targetId = targetNavigableRegion.getMapId();

      if (sourceId == targetId)
         return connections;

      // If the source and target regions are simply too far apart, then do not check their individual points.
      PlanarRegion sourceHomePlanarRegion = sourceNavigableRegion.getHomePlanarRegion();
      PlanarRegion targetHomePlanarRegion = targetNavigableRegion.getHomePlanarRegion();

      BoundingBox3D sourceHomeRegionBoundingBox = sourceHomePlanarRegion.getBoundingBox3dInWorld();
      BoundingBox3D targetHomeRegionBoundingBox = targetHomePlanarRegion.getBoundingBox3dInWorld();

      if (!sourceHomeRegionBoundingBox.intersectsEpsilon(targetHomeRegionBoundingBox, filter.getMaximumInterRegionConnetionDistance()))
      {
         numberPlanarRegionBoundingBoxesTooFar++;
         return connections;
      }

      List<Cluster> sourceClusters = sourceNavigableRegion.getAllClusters();
      List<Cluster> sourceObstacleClusters = sourceNavigableRegion.getObstacleClusters();
      for (Cluster sourceCluster : sourceClusters)
      {
         List<Point3DReadOnly> sourcePointsInWorld = sourceCluster.getNavigablePointsInsideHomeRegionInWorld(sourceHomePlanarRegion);

         List<Cluster> targetClusters = targetNavigableRegion.getAllClusters();
         List<Cluster> targetObstacleClusters = targetNavigableRegion.getObstacleClusters();
         for (Cluster targetCluster : targetClusters)
         {
            List<Point3DReadOnly> targetPointsInWorld = targetCluster.getNavigablePointsInsideHomeRegionInWorld(targetHomePlanarRegion);

            for (int sourceIndex = 0; sourceIndex < sourcePointsInWorld.size(); sourceIndex++)
            {
               Point3DReadOnly sourcePoint3DInWorld = sourcePointsInWorld.get(sourceIndex);

               for (Point3DReadOnly targetPoint3D : targetPointsInWorld)
               {
                  totalSourceTargetChecks++;
                  if ((totalSourceTargetChecks % 100000002) == 0)
                  {
                     printStats();
                  }

                  ConnectionPoint3D source = new ConnectionPoint3D(sourcePoint3DInWorld, sourceId);
                  ConnectionPoint3D target = new ConnectionPoint3D(targetPoint3D, targetId);

                  if (filter.isConnectionValid(source, target))
                  {
                     numberPassValidFilter++;

//                     Point2D sourcePoint2DInLocal = getPoint2DInLocal(sourceNavigableRegion, sourcePoint3DInWorld);

                     //TODO: Get rid of these check since now down beforehand.
                     //                     double epsilonForInsidePlanarRegion = 1e-4; //Add a little just to make sure we do not miss anything.

                     //                     //TODO: +++JEP: Taking up lots of time, but necessary. 4.7sec. 1,998,512 calls
                     //                     if (!PlanarRegionTools.isPointInLocalInsidePlanarRegion(sourceHomePlanarRegion, sourcePoint2DInLocal, epsilonForInsidePlanarRegion))
                     //                     {
                     //                        numberNotInsideSourceRegion++;
                     //                        continue;
                     //                     }

//                     Point2D targetPoint2DInLocal = getPoint2DInLocal(targetNavigableRegion, targetPoint3D);

                     //                     //TODO: +++JEP: Taking up lots of time, but necessary. 1.1 sec. 912,250 calls
                     //                     if (!PlanarRegionTools.isPointInLocalInsidePlanarRegion(targetHomePlanarRegion, targetPoint2DInLocal, epsilonForInsidePlanarRegion))
                     //                     {
                     //                        numberNotInsideTargetRegion++;
                     //                        continue;
                     //                     }
//
                     Connection connection = new Connection(source, target);
//
//                     boolean sourceIsInsideNoGoZone = isInsideANonNavigableZone(sourcePoint2DInLocal, sourceObstacleClusters);
//                     if (sourceIsInsideNoGoZone)
//                     {
//                        numberSourcesInNoGoZones++;
//                        continue;
//                     }
//
//                     boolean targetIsInsideNoGoZone = isInsideANonNavigableZone(targetPoint2DInLocal, targetObstacleClusters);
//                     if (targetIsInsideNoGoZone)
//                     {
//                        numberTargetsInNoGoZones++;
//                        continue;
//                     }

                     numberValidConnections++;
                     connections.add(connection);
                  }
               }
            }
         }
      }

      //      printStats();
      return connections;
   }

   private static void printStats()
   {
      System.out.println("numberPlanarRegionBoundingBoxesTooFar = " + numberPlanarRegionBoundingBoxesTooFar);
      System.out.println("totalSourceTargetChecks = " + totalSourceTargetChecks);
      System.out.println("numberPassValidFilter = " + numberPassValidFilter);
      System.out.println("numberNotInsideSourceRegion = " + numberNotInsideSourceRegion);
      System.out.println("numberNotInsideTargetRegion = " + numberNotInsideTargetRegion);
      System.out.println("numberSourcesInNoGoZones = " + numberSourcesInNoGoZones);
      System.out.println("numberTargetsInNoGoZones = " + numberTargetsInNoGoZones);
      System.out.println("numberValidConnections = " + numberValidConnections);

      System.out.println();
   }

   public static boolean isInsideANonNavigableZone(Point2DReadOnly pointInLocal, List<Cluster> clusters)
   {
      for (Cluster cluster : clusters)
      {
         if (cluster.isInsideNonNavigableZone(pointInLocal))
            return true;
      }
      return false;
   }

   public static Point2D getPoint2DInLocal(VisibilityMapWithNavigableRegion region, Point3DReadOnly point3DInWorld)
   {
      Point3D pointInLocal = new Point3D();
      pointInLocal.set(point3DInWorld);
      region.transformFromWorldToLocal(pointInLocal);
      Point2D pointInLocal2D = new Point2D(pointInLocal.getX(), pointInLocal.getY());
      return pointInLocal2D;
   }
   
   public static Point2D getPoint2DInLocal(NavigableRegion region, Point3DReadOnly point3DInWorld)
   {
      Point3D pointInLocal = new Point3D();
      pointInLocal.set(point3DInWorld);
      region.transformFromWorldToLocal(pointInLocal);
      Point2D pointInLocal2D = new Point2D(pointInLocal.getX(), pointInLocal.getY());
      return pointInLocal2D;
   }

   public static InterRegionVisibilityMap createInterRegionVisibilityMapUsingInnerVisibilityMaps(List<VisibilityMapWithNavigableRegion> navigableRegions,
                                                                                                 InterRegionConnectionFilter filter)
   {
      InterRegionVisibilityMap map = new InterRegionVisibilityMap();

      for (int sourceMapIndex = 0; sourceMapIndex < navigableRegions.size(); sourceMapIndex++)
      {
         VisibilityMap sourceMap = navigableRegions.get(sourceMapIndex).getVisibilityMapInWorld();
         Set<ConnectionPoint3D> sourcePoints = sourceMap.getVertices();

         for (ConnectionPoint3D source : sourcePoints)
         {
            for (int targetMapIndex = sourceMapIndex + 1; targetMapIndex < navigableRegions.size(); targetMapIndex++)
            {
               VisibilityMapWithNavigableRegion targetRegion = navigableRegions.get(targetMapIndex);

               VisibilityMap targetMap = targetRegion.getVisibilityMapInWorld();
               Set<ConnectionPoint3D> targetPoints = targetMap.getVertices();

               for (ConnectionPoint3D target : targetPoints)
               {
                  if (source.getRegionId() == target.getRegionId())
                  {
                     continue;
                  }

                  if (filter.isConnectionValid(source, target))
                  {
                     map.addConnection(source, target);
                  }
               }
            }
         }
      }

      return map;
   }
}
