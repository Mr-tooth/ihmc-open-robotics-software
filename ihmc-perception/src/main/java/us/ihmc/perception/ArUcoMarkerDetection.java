package us.ihmc.perception;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_aruco;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_aruco.Dictionary;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;

import java.util.ArrayList;

public class ArUcoMarkerDetection
{
   public static final int DEFAULT_DICTIONARY = opencv_aruco.DICT_4X4_100;

   private Dictionary dictionary;
   private BytedecoImage abgr8888ColorImage;
   private MatVector corners;
   private Mat ids;
   private ArrayList<Integer> idsAsList = new ArrayList<>();
   private IntPointer fromABGRToRGBA = new IntPointer(0, 3, 1, 2, 2, 1, 3, 0);
   private BytedecoImage rgbaImage;
   private BytedecoImage alphaRemovedImage;

   public void create(BytedecoImage abgr8888ColorImage)
   {
      this.abgr8888ColorImage = abgr8888ColorImage;
      dictionary = opencv_aruco.getPredefinedDictionary(DEFAULT_DICTIONARY);

      rgbaImage = new BytedecoImage(abgr8888ColorImage.getImageWidth(), abgr8888ColorImage.getImageHeight(), opencv_core.CV_8UC4);
      alphaRemovedImage = new BytedecoImage(abgr8888ColorImage.getImageWidth(), abgr8888ColorImage.getImageHeight(), opencv_core.CV_8UC3);
      corners = new MatVector();
      ids = new Mat();
   }
   
   public void update()
   {
      opencv_core.mixChannels(abgr8888ColorImage.getBytedecoOpenCVMat(), 1, rgbaImage.getBytedecoOpenCVMat(), 1, fromABGRToRGBA, 4);

      // ArUco library doesn't support alpha channel being in there
      opencv_imgproc.cvtColor(rgbaImage.getBytedecoOpenCVMat(), alphaRemovedImage.getBytedecoOpenCVMat(), opencv_imgproc.COLOR_RGBA2RGB);

//      opencv_aruco.drawMarker(dictionary, 23, 6, alphaRemovedImage.getBytedecoOpenCVMat(), 1);

      opencv_aruco.detectMarkers(alphaRemovedImage.getBytedecoOpenCVMat(), dictionary, corners, ids);
      idsAsList.clear();
      for (int i = 0; i < ids.cols(); i++)
      {
         idsAsList.add(ids.ptr(0, i).getInt());
      }
   }

   public MatVector getCorners()
   {
      return corners;
   }

   public ArrayList<Integer> getIds()
   {
      return idsAsList;
   }

   public BytedecoImage getAlphaRemovedImage()
   {
      return alphaRemovedImage;
   }

   public BytedecoImage getAbgr8888ColorImage()
   {
      return abgr8888ColorImage;
   }

   /**
    * Save a ArUco marker image of id to file.
    */
   public static void main(String[] args)
   {
      Mat markerToSave = new Mat();
      Dictionary dictionary = opencv_aruco.getPredefinedDictionary(DEFAULT_DICTIONARY);
      int markerID = 0;
      int totalImageSizePixels = 200;
      for (; markerID < 100; markerID++)
      {
         opencv_aruco.drawMarker(dictionary, markerID, totalImageSizePixels, markerToSave, 2);
         opencv_imgcodecs.imwrite("marker" + markerID + ".png", markerToSave);
      }
   }
}
