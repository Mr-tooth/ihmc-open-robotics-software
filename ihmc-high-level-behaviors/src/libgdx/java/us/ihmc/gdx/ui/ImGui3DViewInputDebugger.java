package us.ihmc.gdx.ui;

import imgui.flag.ImGuiMouseButton;
import us.ihmc.gdx.imgui.ImGuiPlot;

public class ImGui3DViewInputDebugger
{
   private final int bufferSize = 1000;
   private ImGuiPlot screenXPlot;
   private ImGuiPlot screenYPlot;
   private ImGuiPlot deltaXPlot;
   private ImGuiPlot deltaYPlot;
   private ImGuiPlot scrolledYPlot;

   public void create(GDXImGuiBasedUI baseUI)
   {
      baseUI.addImGui3DViewInputProcessor(input ->
      {
         if (input.isWindowHovered())
         {
            scrolledYPlot.setValue(input.getMouseWheelDelta());
            screenXPlot.setValue(input.getMousePosX());
            screenYPlot.setValue(input.getMousePosY());
         }
         if (input.isDragging(ImGuiMouseButton.Left))
         {
            deltaXPlot.setValue(input.getMouseDraggedX(ImGuiMouseButton.Left));
            deltaYPlot.setValue(input.getMouseDraggedY(ImGuiMouseButton.Left));
         }
      });

      screenXPlot = new ImGuiPlot("screenX", bufferSize);
      screenYPlot = new ImGuiPlot("screenY", bufferSize);
      deltaXPlot = new ImGuiPlot("deltaX", bufferSize);
      deltaYPlot = new ImGuiPlot("deltaY", bufferSize);
      scrolledYPlot = new ImGuiPlot("scrolledY", bufferSize);
   }

   public void render()
   {
      screenXPlot.render();
      screenYPlot.render();
      deltaXPlot.render();
      deltaYPlot.render();
      scrolledYPlot.render();
   }

   public String getWindowName()
   {
      return "Input Debugger";
   }
}
