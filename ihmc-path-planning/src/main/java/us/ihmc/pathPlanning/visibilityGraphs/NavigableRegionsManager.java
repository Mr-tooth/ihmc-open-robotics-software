package us.ihmc.pathPlanning.visibilityGraphs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import us.ihmc.euclid.tuple3D.Point3D;
import us.ihmc.euclid.tuple3D.interfaces.Point3DReadOnly;
import us.ihmc.log.LogTools;
import us.ihmc.pathPlanning.visibilityGraphs.clusterManagement.Cluster;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.ConnectionPoint3D;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.InterRegionVisibilityMap;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.NavigableRegion;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.SingleSourceVisibilityMap;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityGraphEdge;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityGraphNode;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityMapSolution;
import us.ihmc.pathPlanning.visibilityGraphs.dataStructure.VisibilityMapWithNavigableRegion;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.VisibilityGraphsParameters;
import us.ihmc.pathPlanning.visibilityGraphs.interfaces.VisibilityMapHolder;
import us.ihmc.pathPlanning.visibilityGraphs.tools.ClusterTools;
import us.ihmc.pathPlanning.visibilityGraphs.tools.OcclusionTools;
import us.ihmc.pathPlanning.visibilityGraphs.tools.PlanarRegionTools;
import us.ihmc.robotics.geometry.PlanarRegion;

public class NavigableRegionsManager
{
   private final static boolean debug = false;
   private final VisibilityGraphsParameters parameters;

   private final VisibilityMapSolution visibilityMapSolution = new VisibilityMapSolution();

   public NavigableRegionsManager()
   {
      this(null, null);
   }

   public NavigableRegionsManager(VisibilityGraphsParameters parameters)
   {
      this(parameters, null);
   }

   public NavigableRegionsManager(List<PlanarRegion> regions)
   {
      this(null, regions);
   }

   public NavigableRegionsManager(VisibilityGraphsParameters parameters, List<PlanarRegion> regions)
   {
      visibilityMapSolution.setNavigableRegions(new NavigableRegions(parameters, regions));
      this.parameters = parameters == null ? new DefaultVisibilityGraphParameters() : parameters;
   }

   private static ArrayList<VisibilityMapWithNavigableRegion> createListOfVisibilityMapsWithNavigableRegions(NavigableRegions navigableRegions)
   {
      ArrayList<VisibilityMapWithNavigableRegion> list = new ArrayList<>();

      List<NavigableRegion> naviableRegionsList = navigableRegions.getNaviableRegionsList();

      for (NavigableRegion navigableRegion : naviableRegionsList)
      {
         VisibilityMapWithNavigableRegion visibilityMapWithNavigableRegion = new VisibilityMapWithNavigableRegion(navigableRegion);
         list.add(visibilityMapWithNavigableRegion);
      }

      return list;
   }

   public List<VisibilityMapWithNavigableRegion> getNavigableRegionsList()
   {
      return visibilityMapSolution.getVisibilityMapsWithNavigableRegions();
   }

   public void setPlanarRegions(List<PlanarRegion> planarRegions)
   {
      visibilityMapSolution.getNavigableRegions().setPlanarRegions(planarRegions);
   }

   public List<Point3DReadOnly> calculateBodyPath(final Point3DReadOnly start, final Point3DReadOnly goal)
   {
      boolean calculateVisibilityMapAllAtOnce = false;

      if (calculateVisibilityMapAllAtOnce)
      {
         return calculateVisibilityMapAllAtOnceThenFindPath(start, goal);
      }
      else
      {
         return calculateVisibilityMapLazilyWhileFindingPath(start, goal);
      }
   }

   private List<Point3DReadOnly> calculateVisibilityMapLazilyWhileFindingPath(Point3DReadOnly startInWorld, Point3DReadOnly goalInWorld)
   {
      boolean areStartAndGoalValid = checkIfStartAndGoalAreValid(startInWorld, goalInWorld);
      if (!areStartAndGoalValid)
         return null;

      NavigableRegions navigableRegions = visibilityMapSolution.getNavigableRegions();
      navigableRegions.filterPlanarRegionsWithBoundingCapsule(startInWorld, goalInWorld, parameters.getExplorationDistanceFromStartGoal());
      long startBodyPathComputation = System.currentTimeMillis();

      navigableRegions.createNavigableRegions();

      VisibilityGraph visibilityGraph = new VisibilityGraph(navigableRegions, parameters.getInterRegionConnectionFilter());

      //      visibilityGraph.fullyExpandVisibilityGraph();

      double searchHostEpsilon = parameters.getSearchHostRegionEpsilon();

      visibilityGraph.setStart(startInWorld, searchHostEpsilon);
      visibilityGraph.setGoal(goalInWorld, searchHostEpsilon);

      VisibilityGraphNode startNode = visibilityGraph.getStartNode();
      VisibilityGraphNode goalNode = visibilityGraph.getGoalNode();

      if (startNode == null)
         return null;

      //TODO: Pull this out to somewhere else since it is A-Star solver.

      Comparator<VisibilityGraphNode> comparator = new Comparator<VisibilityGraphNode>()
      {
         @Override
         public int compare(VisibilityGraphNode nodeOne, VisibilityGraphNode nodeTwo)
         {
            //TODO: Check the statement below. It might be false, since just doing compare not equals?
            //Note: Can only return 0 if the two nodes are ==. 
            // This is because queue.remove(node) will remove the first one with .equals()
            if (nodeOne == nodeTwo)
               return 0;

            if (nodeOne.getCostFromStart() + nodeOne.getEstimatedCostToGoal() < nodeTwo.getCostFromStart() + nodeTwo.getEstimatedCostToGoal())
               return -1;
            return 1;
         }
      };

      PriorityQueue<VisibilityGraphNode> queue = new PriorityQueue<>(comparator);

      startNode.setEdgesHaveBeenDetermined(true);
      startNode.setCostFromStart(0.0, null);
      startNode.setEstimatedCostToGoal(startInWorld.distance(goalInWorld));
      expandNode(visibilityGraph, startNode, goalInWorld, queue);

      while (!queue.isEmpty())
      {
         VisibilityGraphNode nodeToExpand = queue.poll();

         if (nodeToExpand == goalNode)
            break;
         expandNode(visibilityGraph, nodeToExpand, goalInWorld, queue);
      }

      //

      VisibilityMapSolution visibilityMapSolutionFromNewVisibilityGraph = visibilityGraph.createVisibilityMapSolution();

      visibilityMapSolution.setVisibilityMapsWithNavigableRegions(visibilityMapSolutionFromNewVisibilityGraph.getVisibilityMapsWithNavigableRegions());
      visibilityMapSolution.setInterRegionVisibilityMap(visibilityMapSolutionFromNewVisibilityGraph.getInterRegionVisibilityMap());

      visibilityMapSolution.setStartMap(visibilityMapSolutionFromNewVisibilityGraph.getStartMap());
      visibilityMapSolution.setGoalMap(visibilityMapSolutionFromNewVisibilityGraph.getGoalMap());

      List<Point3DReadOnly> path = new ArrayList<>();
      VisibilityGraphNode nodeWalkingBack = goalNode;
      
      while(nodeWalkingBack != null)
      {
         path.add(nodeWalkingBack.getPointInWorld());
         nodeWalkingBack = nodeWalkingBack.getBestParentNode();
      }
      Collections.reverse(path);
      
      printResults(startBodyPathComputation, path);
      return path;
   }

   private void expandNode(VisibilityGraph visibilityGraph, VisibilityGraphNode nodeToExpand, Point3DReadOnly goalInWorld,
                           PriorityQueue<VisibilityGraphNode> queue)
   {
      if (nodeToExpand.getHasBeenExpanded())
      {
         throw new RuntimeException("Node has already been expanded!!");
      }

      boolean edgesHaveBeenDetermined = nodeToExpand.getEdgesHaveBeenDetermined();

      if (!edgesHaveBeenDetermined)
      {
         visibilityGraph.computeInnerAndInterEdges(nodeToExpand);
         nodeToExpand.setEdgesHaveBeenDetermined(true);
      }

      List<VisibilityGraphEdge> edges = nodeToExpand.getEdges();

      for (VisibilityGraphEdge edge : edges)
      {
         VisibilityGraphNode nextNode = null;

         VisibilityGraphNode sourceNode = edge.getSourceNode();
         VisibilityGraphNode targetNode = edge.getTargetNode();

         if (nodeToExpand == sourceNode)
         {
            nextNode = targetNode;
         }
         else if (nodeToExpand == targetNode)
         {
            nextNode = sourceNode;
         }

         ConnectionPoint3D nodeToExpandInWorld = nodeToExpand.getPointInWorld();
         ConnectionPoint3D nextNodeInWorld = nextNode.getPointInWorld();

         //TODO: Something besides distance later...
         double distance = nodeToExpandInWorld.distance(nextNodeInWorld);
         double newCostFromStart = nodeToExpand.getCostFromStart() + distance;

         double currentCostFromStart = nextNode.getCostFromStart();

         if (Double.isNaN(currentCostFromStart) || (newCostFromStart < currentCostFromStart))
         {
            nextNode.setCostFromStart(newCostFromStart, nodeToExpand);

            //TODO: Heuristic for AStar.
            nextNode.setEstimatedCostToGoal(nextNodeInWorld.distance(goalInWorld));

            queue.remove(nextNode);
            queue.add(nextNode);
         }

      }

      nodeToExpand.setHasBeenExpanded(true);
   }

   public List<Point3DReadOnly> calculateVisibilityMapAllAtOnceThenFindPath(final Point3DReadOnly startInWorld, final Point3DReadOnly goalInWorld)
   {
      boolean areStartAndGoalValid = checkIfStartAndGoalAreValid(startInWorld, goalInWorld);
      if (!areStartAndGoalValid)
         return null;

      NavigableRegions navigableRegions = visibilityMapSolution.getNavigableRegions();
      navigableRegions.filterPlanarRegionsWithBoundingCapsule(startInWorld, goalInWorld, parameters.getExplorationDistanceFromStartGoal());
      long startBodyPathComputation = System.currentTimeMillis();

      navigableRegions.createNavigableRegions();

      VisibilityGraph visibilityGraph = new VisibilityGraph(navigableRegions, parameters.getInterRegionConnectionFilter());
      visibilityGraph.fullyExpandVisibilityGraph();

      double searchHostEpsilon = parameters.getSearchHostRegionEpsilon();

      visibilityGraph.setStart(startInWorld, searchHostEpsilon);
      visibilityGraph.setGoal(goalInWorld, searchHostEpsilon);

      VisibilityGraphNode startNode = visibilityGraph.getStartNode();
      VisibilityGraphNode goalNode = visibilityGraph.getGoalNode();

      if (startNode == null)
         return null;

      VisibilityMapSolution visibilityMapSolutionFromNewVisibilityGraph = visibilityGraph.createVisibilityMapSolution();

      visibilityMapSolution.setVisibilityMapsWithNavigableRegions(visibilityMapSolutionFromNewVisibilityGraph.getVisibilityMapsWithNavigableRegions());
      visibilityMapSolution.setInterRegionVisibilityMap(visibilityMapSolutionFromNewVisibilityGraph.getInterRegionVisibilityMap());

      visibilityMapSolution.setStartMap(visibilityMapSolutionFromNewVisibilityGraph.getStartMap());
      visibilityMapSolution.setGoalMap(visibilityMapSolutionFromNewVisibilityGraph.getGoalMap());

      ConnectionPoint3D projectedStartInWorld = startNode.getPointInWorld();
      ConnectionPoint3D projectedGoalInWorld = goalNode.getPointInWorld();

      int START_GOAL_ID = 0;

      ConnectionPoint3D startConnection = new ConnectionPoint3D(projectedStartInWorld, START_GOAL_ID);
      ConnectionPoint3D goalConnection = new ConnectionPoint3D(projectedGoalInWorld, START_GOAL_ID);

      List<VisibilityMapHolder> visibilityMapHolders = createVisibilityMapHoldersFromVisibilityMapSolution(visibilityMapSolution);

      List<Point3DReadOnly> path = parameters.getPathPlanner().calculatePath(startConnection, goalConnection, visibilityMapHolders);
      printResults(startBodyPathComputation, path);
      return path;
   }

   private static List<VisibilityMapHolder> createVisibilityMapHoldersFromVisibilityMapSolution(VisibilityMapSolution visibilityMapSolution)
   {
      SingleSourceVisibilityMap startMap = visibilityMapSolution.getStartMap();
      SingleSourceVisibilityMap goalMap = visibilityMapSolution.getGoalMap();

      if (startMap == null)
         return null;

      List<VisibilityMapHolder> visibilityMapHolders = new ArrayList<>();
      visibilityMapHolders.addAll(visibilityMapSolution.getVisibilityMapsWithNavigableRegions());
      visibilityMapHolders.add(startMap);
      visibilityMapHolders.add(goalMap);
      visibilityMapHolders.add(visibilityMapSolution.getInterRegionVisibilityMap());

      return visibilityMapHolders;
   }

   private boolean checkIfStartAndGoalAreValid(Point3DReadOnly start, Point3DReadOnly goal)
   {
      boolean areStartAndGoalValid = true;
      if (start == null)
      {
         LogTools.error("Start is null!");
         areStartAndGoalValid = false;
      }

      if (goal == null)
      {
         LogTools.error("Goal is null!");
         areStartAndGoalValid = false;
      }

      if (debug)
         LogTools.info("Starting to calculate body path");

      return areStartAndGoalValid;
   }

   private void printResults(long startBodyPathComputation, List<Point3DReadOnly> path)
   {
      if (debug)
      {
         if (path != null)
         {
            LogTools.info("Total time to find solution was: " + (System.currentTimeMillis() - startBodyPathComputation) + "ms");
         }
         else
         {
            LogTools.info("NO BODY PATH SOLUTION WAS FOUND!" + (System.currentTimeMillis() - startBodyPathComputation) + "ms");
         }
      }
   }

   public List<Point3DReadOnly> calculateBodyPathWithOcclusions(Point3DReadOnly start, Point3DReadOnly goal)
   {
      List<Point3DReadOnly> path = calculateBodyPath(start, goal);

      if (path == null)
      {
         NavigableRegions navigableRegions = visibilityMapSolution.getNavigableRegions();
         ArrayList<VisibilityMapWithNavigableRegion> visibilityMapsWithNavigableRegions = createListOfVisibilityMapsWithNavigableRegions(navigableRegions);

         if (!OcclusionTools.isTheGoalIntersectingAnyObstacles(visibilityMapsWithNavigableRegions.get(0), start, goal))
         {
            if (debug)
            {
               LogTools.info("StraightLine available");
            }

            path = new ArrayList<>();
            path.add(start);
            path.add(goal);

            return path;
         }

         NavigableRegion regionContainingPoint = PlanarRegionTools.getNavigableRegionContainingThisPoint(start, navigableRegions);
         List<Cluster> intersectingClusters = OcclusionTools.getListOfIntersectingObstacles(regionContainingPoint.getObstacleClusters(), start, goal);
         Cluster closestCluster = ClusterTools.getTheClosestCluster(start, intersectingClusters);
         Point3D closestExtrusion = ClusterTools.getTheClosestVisibleExtrusionPoint(1.0, start, goal, closestCluster.getNavigableExtrusionsInWorld(),
                                                                                    regionContainingPoint.getHomePlanarRegion());

         path = calculateBodyPath(start, closestExtrusion);
         path.add(goal);

         return path;
      }
      else
      {
         return path;
      }
   }

   public VisibilityMapHolder getStartMap()
   {
      return visibilityMapSolution.getStartMap();
   }

   public VisibilityMapHolder getGoalMap()
   {
      return visibilityMapSolution.getGoalMap();
   }

   public InterRegionVisibilityMap getInterRegionConnections()
   {
      return visibilityMapSolution.getInterRegionVisibilityMap();
   }

}
