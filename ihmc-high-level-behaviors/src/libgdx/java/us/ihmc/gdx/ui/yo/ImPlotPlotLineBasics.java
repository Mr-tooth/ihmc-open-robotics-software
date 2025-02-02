package us.ihmc.gdx.ui.yo;

import imgui.extension.implot.ImPlot;
import us.ihmc.commons.time.Stopwatch;
import us.ihmc.gdx.ui.tools.ImPlotTools;

public abstract class ImPlotPlotLineBasics implements ImPlotPlotLine
{
   private Runnable legendPopupImGuiRenderer;
   private final String variableNameBase;
   private final String variableNamePostfix;
   private final String variableName;
   private String labelID;
   private ImPlotPlotLineSwapBuffer swapBuffer;
   private double history = 3.0;
   private final Stopwatch stopwatch = new Stopwatch();
   private int bufferSize = 100;
   private double timeForOneBufferEntry = history / bufferSize;
   private long lastTickIndex = 0;
   private long tickIndex = 0;
   private final Integer[] xValues = ImPlotTools.createIndex(bufferSize);
   private boolean isA = true;
   private int filledIndex = 0;

   public ImPlotPlotLineBasics(String variableName, String initialValueString)
   {
      this.variableName = variableName;
      variableNameBase = variableName + " ";
      variableNamePostfix = "###" + variableName;
      labelID = variableNameBase + initialValueString + variableNamePostfix;
   }

   public void setSwapBuffer(ImPlotPlotLineSwapBuffer swapBuffer)
   {
      this.swapBuffer = swapBuffer;
      swapBuffer.initialize(bufferSize);
   }

   public void addValue(String valueString)
   {
      labelID = variableNameBase + valueString + variableNamePostfix;

      int numberOfTicksToAdvance = 0;

      double totalElapsed = stopwatch.totalElapsed();
      if (Double.isNaN(totalElapsed))
      {
         stopwatch.start();
      }
      else
      {
         tickIndex = (long) (totalElapsed / timeForOneBufferEntry);
         numberOfTicksToAdvance = (int) (tickIndex - lastTickIndex);
         lastTickIndex = tickIndex;
      }

      int numberOfTicksToAdvanceBeforeFilling = Math.min(bufferSize, filledIndex + numberOfTicksToAdvance) - filledIndex;
      int numberOfTicksToAdvanceAfterFilling = Math.min(bufferSize, numberOfTicksToAdvance - numberOfTicksToAdvanceBeforeFilling);
      if (filledIndex < bufferSize)
      {
         if (numberOfTicksToAdvanceBeforeFilling == 0)
         {
            swapBuffer.setAValue(filledIndex);
         }
         else
         {
            for (int i = 0; i < numberOfTicksToAdvanceBeforeFilling && filledIndex < bufferSize; i++)
            {
               ++filledIndex;
               if (filledIndex < bufferSize)
                  swapBuffer.setAValue(filledIndex);
            }

            if (filledIndex == bufferSize)
            {
               swapBuffer.copyAToB(); // copy A to B when we fill up the first time
            }
         }
      }

      if (filledIndex == bufferSize)
      {
         if (numberOfTicksToAdvanceAfterFilling > 0)
         {
            swapBuffer.copyPreviousToUpdated(numberOfTicksToAdvanceAfterFilling, 0, bufferSize - numberOfTicksToAdvanceAfterFilling);
            isA = !isA;

            for (int i = 0; i < numberOfTicksToAdvanceAfterFilling; i++)
            {
               swapBuffer.setUpdatedValue(bufferSize - 1 - i);
            }
         }
         else
         {
            swapBuffer.setPreviousValue(bufferSize - 1);
         }
      }
   }

   @Override
   public boolean render()
   {
      int offset = 0; // This is believed to be the index in the array we are passing in which implot will start reading
      swapBuffer.plot(labelID, xValues, offset);

      boolean showingLegendPopup = false;
      if (legendPopupImGuiRenderer != null && ImPlot.beginLegendPopup(labelID))
      {
         showingLegendPopup = true;
         legendPopupImGuiRenderer.run();
         ImPlot.endLegendPopup();
      }
      return showingLegendPopup;
   }

   public void setLegendPopupImGuiRenderer(Runnable legendPopupImGuiRenderer)
   {
      this.legendPopupImGuiRenderer = legendPopupImGuiRenderer;
   }

   @Override
   public String getVariableName()
   {
      return variableName;
   }
}
