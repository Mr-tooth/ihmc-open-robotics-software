package us.ihmc.gdx.ui;

import imgui.ImGui;
import imgui.ImVec2;
import imgui.extension.implot.ImPlot;
import us.ihmc.gdx.Lwjgl3ApplicationAdapter;
import us.ihmc.gdx.ui.tools.ImPlotTools;
import us.ihmc.log.LogTools;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Seizure warning - this stress test generates rapidly flashing, random colors
 */
public class ImGuiGDXImPlotStressTestDemo
{
   private final String WINDOW_NAME = "ImPlot Stress Test";
   private GDXImGuiBasedUI baseUI;
   private AtomicInteger numPlotsToShow = new AtomicInteger(50);
   private final Timer timer = new Timer();
   private final Double[] xs = new Double[500];
   private final Double[] ys = new Double[500];
   private final Random random = new Random();
   private boolean recalculate = true;

   public ImGuiGDXImPlotStressTestDemo()
   {
      LogTools.info("Starting UI");
      baseUI = new GDXImGuiBasedUI(getClass(), "ihmc-open-robotics-software", "ihmc-high-level-behaviors/src/libgdx/resources", WINDOW_NAME);
   }

   public void launch()
   {
      baseUI.launchGDXApplication(new Lwjgl3ApplicationAdapter()
      {
         @Override
         public void create()
         {
            baseUI.create();
            ImPlotTools.ensureImPlotInitialized();

            timer.scheduleAtFixedRate(new TimerTask()
            {
               @Override
               public void run()
               {
                  numPlotsToShow.incrementAndGet();
               }
            }, 1000, 1500);

            timer.scheduleAtFixedRate(new TimerTask()
            {
               @Override
               public void run()
               {
                  recalculate = true;
               }
            }, 0, 50);
         }

         @Override
         public void render()
         {
            baseUI.renderBeforeOnScreenUI();
            ImGui.begin(WINDOW_NAME);

            if (recalculate)
            {
               for (int i = 0; i < 500; i++)
               {
                  xs[i] = random.nextInt(500) + random.nextDouble();
                  ys[i] = random.nextInt(500) + random.nextDouble();
               }
               recalculate = false;
            }

            int max = numPlotsToShow.get();
            for (int i = 0; i < max; i++)
            {
               if (i % 8 != 0)
                  ImGui.sameLine();

               ImPlot.pushColormap(random.nextInt(16));
               if (ImPlot.beginPlot("Plot " + i, "X", "Y", new ImVec2(225, 150)))
               {
                  if (random.nextBoolean())
                     ImPlot.plotLine("line" + i, xs, ys);
                  if (random.nextBoolean())
                     ImPlot.plotBars("bars" + i, xs, ys);
                  if (random.nextBoolean())
                     ImPlot.plotScatter("bars" + i, xs, ys);
                  if (random.nextBoolean())
                     ImPlot.plotBarsH("bars" + i, xs, ys);

                  ImPlot.endPlot();
               }
               ImPlot.popColormap();
            }

            ImGui.end();
            baseUI.renderEnd();
         }

         @Override
         public void dispose()
         {
            baseUI.dispose();
         }
      });
   }

   public static void main(String[] args) //Seizure warning - this stress test generates rapidly flashing, random colors
   {
      ImGuiGDXImPlotStressTestDemo ui = new ImGuiGDXImPlotStressTestDemo();
      ui.launch();
   }
}
