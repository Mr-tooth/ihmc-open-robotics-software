package us.ihmc.gdx.ui.yo;

import imgui.internal.ImGui;
import us.ihmc.gdx.imgui.ImGuiPanel;
import us.ihmc.gdx.imgui.ImGuiUniqueLabelMap;
import us.ihmc.gdx.simulation.scs2.GDXYoManager;

import java.util.ArrayList;
import java.util.function.Consumer;

public class ImPlotModifiableYoPlotPanel extends ImGuiPanel
{
   private final ArrayList<ImPlotModifiableYoPlot> yoPlots = new ArrayList<>();
   private final ImPlotPlotPanelLayout layout = new ImPlotPlotPanelLayout();
   private final ImGuiUniqueLabelMap labels = new ImGuiUniqueLabelMap(getClass());
   private final ImGuiYoVariableSearchPanel yoVariableSearchPanel;
   private final GDXYoManager yoManager;
   private final Consumer<ImPlotModifiableYoPlotPanel> removeSelf;

   public ImPlotModifiableYoPlotPanel(String panelName,
                                      ImGuiYoVariableSearchPanel yoVariableSearchPanel,
                                      GDXYoManager yoManager,
                                      Consumer<ImPlotModifiableYoPlotPanel> removeSelf)
   {
      super(panelName, null, false, true);
      this.yoVariableSearchPanel = yoVariableSearchPanel;
      this.yoManager = yoManager;
      this.removeSelf = removeSelf;
      setRenderMethod(this::render);
      layout.setPlotRenderer(this::renderPlot, yoPlots::size);
   }

   private void renderPlot(int i)
   {
      yoPlots.get(i).render(layout.getPlotWidth(), layout.getPlotHeight());
   }

   public void render()
   {
      ImGui.beginMenuBar();
      layout.renderLayoutMenu();
      if (ImGui.beginMenu(labels.get("Plots")))
      {
         if (ImGui.menuItem("Add Plot"))
         {
            addPlot();
         }
         if (ImGui.menuItem(labels.get("Remove this panel")))
         {
            removeSelf.accept(this);
         }
         ImGui.endMenu();
      }

      ImGui.endMenuBar();

      layout.renderPlots();
   }

   public ImPlotModifiableYoPlot addPlot()
   {
      ImPlotModifiableYoPlot imPlotModifiableYoPlot = new ImPlotModifiableYoPlot(yoVariableSearchPanel, this, yoManager, this::removePlot);
      yoPlots.add(imPlotModifiableYoPlot);
      return imPlotModifiableYoPlot;
   }

   private void removePlot(ImPlotModifiableYoPlot plot)
   {
      yoPlots.remove(plot);
   }

   public ArrayList<ImPlotModifiableYoPlot> getYoPlots()
   {
      return yoPlots;
   }
}
