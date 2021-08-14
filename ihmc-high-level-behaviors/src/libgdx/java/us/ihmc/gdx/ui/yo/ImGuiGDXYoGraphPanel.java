package us.ihmc.gdx.ui.yo;

import imgui.ImVec2;
import imgui.extension.implot.ImPlot;
import imgui.extension.implot.ImPlotContext;
import imgui.extension.implot.ImPlotStyle;
import imgui.internal.ImGui;
import imgui.type.ImBoolean;
import imgui.type.ImInt;
import imgui.type.ImString;
import us.ihmc.commons.exception.DefaultExceptionHandler;
import us.ihmc.commons.thread.ThreadTools;
import us.ihmc.communication.configuration.NetworkParameterKeys;
import us.ihmc.communication.configuration.NetworkParameters;
import us.ihmc.gdx.imgui.ImGuiTools;
import us.ihmc.log.LogTools;
import us.ihmc.robotDataLogger.YoVariableClient;
import us.ihmc.robotDataLogger.logger.DataServerSettings;
import us.ihmc.robotDataVisualizer.BasicYoVariablesUpdatedListener;
import us.ihmc.tools.thread.MissingThreadTools;
import us.ihmc.yoVariables.registry.YoRegistry;
import us.ihmc.yoVariables.variable.YoVariable;

import java.util.*;

public class ImGuiGDXYoGraphPanel
{
   private final String title;
   private final String controllerHost = NetworkParameters.getHost(NetworkParameterKeys.robotController);
   private final int bufferSize;
   private final ArrayList<GDXYoGraphRunnable> graphs = new ArrayList<>();
   private volatile boolean handshakeComplete = false;
   private volatile boolean disconnecting = false;
   private volatile boolean connecting = false;
   private YoRegistry registry;
   private YoVariableClient yoVariableClient;

   private final ImInt serverSelectedIndex = new ImInt(0);
   private String[] serverGraphGroupNames = new String[0];
   private final HashMap<String, GDXYoGraphGroup> serverGraphGroups = new HashMap<>();

   private ImPlotContext context = null;

   private final ImBoolean showAllVariables = new ImBoolean(false);
   private final ImString searchBar = new ImString();

   private GDXYoGraphRunnable graphRequesting = null;

   public ImGuiGDXYoGraphPanel(String title, int bufferSize)
   {
      this.title = title;
      this.bufferSize = bufferSize;
   }

   public void startYoVariableClient()
   {
      //      Set<String> graphGroupNameKeySet = graphGroups.keySet();
      //      String robotControllerHost = NetworkParameters.getHost(NetworkParameterKeys.robotController);
      //      String graphGroupName;
      //      if (robotControllerHost.trim().toLowerCase().contains("cpu4") && graphGroupNameKeySet.contains("Real robot"))
      //      {
      //         Arrays.asList(graphGroupNames).indexOf("Real robot");
      //         graphGroupSelectedIndex.set(graphGroupNames.);
      //      }
   }

   public void startYoVariableClient(String graphGroupName)
   {
      for (int i = 0; i < serverGraphGroupNames.length; i++)
      {
         if (serverGraphGroupNames[i].equals(graphGroupName))
         {
            serverSelectedIndex.set(i);
         }
      }

      connecting = true;
      handshakeComplete = false;
      this.registry = new YoRegistry(getClass().getSimpleName());
      BasicYoVariablesUpdatedListener clientUpdateListener = new BasicYoVariablesUpdatedListener(registry);
      yoVariableClient = new YoVariableClient(clientUpdateListener);
      MissingThreadTools.startAsDaemon("YoVariableClient", DefaultExceptionHandler.MESSAGE_AND_STACKTRACE, () ->
      {
         LogTools.info("Connecting to {}:{}", controllerHost, DataServerSettings.DEFAULT_PORT);
         boolean connectionRefused = true;
         while (connectionRefused)
         {
            try
            {
               yoVariableClient.start(controllerHost, DataServerSettings.DEFAULT_PORT);
               connectionRefused = false;
            }
            catch (RuntimeException e)
            {
               if (e.getMessage().contains("Connection refused"))
               {
                  ThreadTools.sleep(100);
               }
               else
               {
                  throw e;
               }
            }
         }
         if (!clientUpdateListener.isHandshakeComplete())
            LogTools.info("Waiting for handshake...");
         while (!clientUpdateListener.isHandshakeComplete())
         {
            ThreadTools.sleep(100);
         }
         LogTools.info("Handshake complete.");
         handshakeComplete = true;

         //         listVariables(registry, 4);
      });

      context = ImPlot.createContext();
      ImPlotStyle style = ImPlot.getStyle();
      style.setPlotPadding(new ImVec2(0, 0));

      GDXYoGraphGroup graphGroup = serverGraphGroups.get(graphGroupName);
      for (String variableName : graphGroup.getVariableNames())
      {
         MissingThreadTools.startAsDaemon("AddGraph", DefaultExceptionHandler.MESSAGE_AND_STACKTRACE, () ->
         {
            while (!handshakeComplete)
            {
               ThreadTools.sleep(150);
            }
            final YoVariable variable = registry.findVariable(registry.getName() + "." + variableName);
            if (variable != null)
            {
               LogTools.info("Setting up graph for variable: {}", variable.getFullName());
               Double[] values = new Double[bufferSize];
               synchronized (graphs)
               {
                  graphs.add(new GDXYoGraphRunnable(context, variable, values, registry, bufferSize));
               }
            }
            else
            {
               LogTools.error("Variable not found: {}", variableName);
            }
            connecting = false;
         });
      }
   }

   private void getAllVariablesHelper(YoRegistry registry, ArrayList<YoVariable> output)
   {
      output.addAll(registry.getVariables());

      for (YoRegistry child : registry.getChildren())
      {
         getAllVariablesHelper(child, output);
      }
   }

   private List<YoVariable> getAllVariables(YoRegistry registry)
   {
      ArrayList<YoVariable> output = new ArrayList<>();
      getAllVariablesHelper(registry, output);
      return output;
   }

   public void renderImGuiWidgetsVariablePanel()
   {
      if (graphRequesting == null)
      {
         ImGui.text("Select a graph by right clicking it to add a variable.");
         return;
      }

      List<YoVariable> variables = getAllVariables(registry);
      variables.sort(Comparator.comparing(YoVariable::getName));

      ImGui.inputText("Variable Search", searchBar);

      ImGui.sameLine();
      if (ImGui.button("Cancel"))
      {
         showAllVariables.set(false);
         graphRequesting.cancelWantVariable();
         graphRequesting = null;
      }

      if (ImGui.beginListBox("##YoVariables", ImGui.getColumnWidth(), ImGui.getWindowSizeY() - 100))
      {
         for (YoVariable variable : variables)
         {
            if (!variable.getName().toLowerCase().contains(searchBar.get().toLowerCase()))
               continue;

            ImGui.selectable(variable.getName());
            if (ImGui.isItemClicked())
            {
               graphRequesting.addVariable(variable);
               showAllVariables.set(false);
               graphRequesting = null;
            }
         }
         ImGui.endListBox();
      }
   }

   public void renderImGuiWidgetsGraphPanel()
   {
      ImGui.text("Controller host: " + controllerHost);

      ImGui.combo(ImGuiTools.uniqueIDOnly(this, "Profile"), serverSelectedIndex, serverGraphGroupNames, serverGraphGroupNames.length);
      ImGui.sameLine();
      if (connecting)
      {
         ImGui.text("Connecting...");
      }
      else if (disconnecting)
      {
         ImGui.text("Disconnecting...");
      }
      else if (yoVariableClient == null)
      {
         if (ImGui.button(ImGuiTools.uniqueLabel(this, "Connect")))
         {
            startYoVariableClient(serverGraphGroupNames[serverSelectedIndex.get()]);
         }
      }
      else
      {
         if (ImGui.button(ImGuiTools.uniqueLabel(this, "Disconnect")))
         {
            destroy();
         }
      }

      synchronized (graphs)
      {
         Iterator<GDXYoGraphRunnable> graphsIterator = graphs.iterator();
         while (graphsIterator.hasNext())
         {
            GDXYoGraphRunnable graph = graphsIterator.next();
            graph.run();
            if (!graph.shouldGraphExist())
               graphsIterator.remove();
            else if (graph.graphWantsVariable())
            {
               showAllVariables.set(true);
               graphRequesting = graph;
            }
         }
      }

      if (ImGui.button("Add new graph"))
      {
         graphs.add(new GDXYoGraphRunnable(context, registry, bufferSize));
      }
   }

   public void graphVariable(String serverName, String yoVariableName)
   {
      GDXYoGraphGroup graphGroup = serverGraphGroups.get(serverName);
      if (graphGroup == null)
      {
         graphGroup = new GDXYoGraphGroup();
         serverGraphGroups.put(serverName, graphGroup);
      }

      graphGroup.addVariable(yoVariableName);
      serverGraphGroupNames = serverGraphGroups.keySet().toArray(new String[0]);
   }

   public void destroy()
   {
      disconnecting = true;
      graphs.clear();
      yoVariableClient.stop();
      yoVariableClient = null;
      handshakeComplete = false;
      disconnecting = false;
      ImPlot.destroyContext(context);
   }

   public String getWindowName()
   {
      return title;
   }

   public String getVarWindowName()
   {
      return title + " List";
   }
}
