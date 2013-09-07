/* \author Radu Bogdan Rusu
 * adaptation Raphael Favier*/
#include "stdafx.h"
/* \author Geoffrey Biggs */

#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>

#include <iostream>
#include <vector>
#include <ctime>

int
main11 (int argc, char** argv)
{
  srand ((unsigned int) time (NULL));

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // Generate pointcloud data
  cloud->width = 1000;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud->points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud->points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud->points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
  }

  float resolution = 128.0f;

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);

  octree.setInputCloud (cloud);
  octree.addPointsFromInputCloud ();

  pcl::PointXYZ searchPoint;

  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);

  // Neighbors within voxel search

  std::vector<int> pointIdxVec;

  if (octree.voxelSearch (searchPoint, pointIdxVec))
  {
    std::cout << "Neighbors within voxel search at (" << searchPoint.x 
     << " " << searchPoint.y 
     << " " << searchPoint.z << ")" 
     << std::endl;
              
    for (size_t i = 0; i < pointIdxVec.size (); ++i)
   std::cout << "    " << cloud->points[pointIdxVec[i]].x 
       << " " << cloud->points[pointIdxVec[i]].y 
       << " " << cloud->points[pointIdxVec[i]].z << std::endl;
  }

  // K nearest neighbor search

  int K = 10;

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;

  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;

  if (octree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
  {
    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
                << " " << cloud->points[ pointIdxNKNSearch[i] ].y 
                << " " << cloud->points[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
  }

  // Neighbors within radius search

  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);

  std::cout << "Neighbors within radius search at (" << searchPoint.x 
      << " " << searchPoint.y 
      << " " << searchPoint.z
      << ") with radius=" << radius << std::endl;


  if (octree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
  {
    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      std::cout << "    "  <<   cloud->points[ pointIdxRadiusSearch[i] ].x 
                << " " << cloud->points[ pointIdxRadiusSearch[i] ].y 
                << " " << cloud->points[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  }
  return 0;
}










//#include <iostream>
//
//#include <boost/thread/thread.hpp>
//#include <pcl/common/common_headers.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/console/parse.h>
//
//// --------------
//// -----Help-----
//// --------------
//void
//printUsage (const char* progName)
//{
//  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
//            << "Options:\n"
//            << "-------------------------------------------\n"
//            << "-h           this help\n"
//            << "-s           Simple visualisation example\n"
//            << "-r           RGB colour visualisation example\n"
//            << "-c           Custom colour visualisation example\n"
//            << "-n           Normals visualisation example\n"
//            << "-a           Shapes visualisation example\n"
//            << "-v           Viewports example\n"
//            << "-i           Interaction Customization example\n"
//            << "\n\n";
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
//{
//  // --------------------------------------------
//  // -----Open 3D viewer and add point cloud-----
//  // --------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//  return (viewer);
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
//{
//  // --------------------------------------------
//  // -----Open 3D viewer and add point cloud-----
//  // --------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//  return (viewer);
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
//{
//  // --------------------------------------------
//  // -----Open 3D viewer and add point cloud-----
//  // --------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
//  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//  return (viewer);
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
//    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
//{
//  // --------------------------------------------------------
//  // -----Open 3D viewer and add point cloud and normals-----
//  // --------------------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//  return (viewer);
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> shapesVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
//{
//  // --------------------------------------------
//  // -----Open 3D viewer and add point cloud-----
//  // --------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//
//  //------------------------------------
//  //-----Add shapes at cloud points-----
//  //------------------------------------
//  
//  pcl::PointXYZRGB a = cloud->points[5];
//  
//  pcl::PointXYZRGB b = cloud->points[cloud->size() - 1];
//  viewer->addLine<pcl::PointXYZRGB> (a,
//                                     b, 1, 0.5, 0.2, "line");
//  
//  viewer->addLine<pcl::PointXYZRGB> (cloud->points[0],
//                                     cloud->points[cloud->size() - 1], "line2");
//  viewer->addSphere (cloud->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");
//
//  //---------------------------------------
//  //-----Add shapes at other locations-----
//  //---------------------------------------
//  pcl::ModelCoefficients coeffs;
//  coeffs.values.push_back (0.0);
//  coeffs.values.push_back (0.0);
//  coeffs.values.push_back (1.0);
//  coeffs.values.push_back (0.0);
//  //viewer->addPlane (coeffs, "plane");
//  coeffs.values.clear ();
//  coeffs.values.push_back (0.3);
//  coeffs.values.push_back (0.3);
//  coeffs.values.push_back (0.0);
//  coeffs.values.push_back (0.0);
//  coeffs.values.push_back (1.0);
//  coeffs.values.push_back (0.0);
//  coeffs.values.push_back (5.0);
//  viewer->addCone (coeffs, "cone");
//
//  return (viewer);
//}
//
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis (
//    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2)
//{
//  // --------------------------------------------------------
//  // -----Open 3D viewer and add point cloud and normals-----
//  // --------------------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->initCameraParameters ();
//
//  int v1(0);
//  viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
//  viewer->setBackgroundColor (0, 0, 0, v1);
//  viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud1", v1);
//
//  int v2(0);
//  viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
//  viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
//  viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
//  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", v2);
//
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
//  viewer->addCoordinateSystem (1.0);
//
//  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals1, 10, 0.05, "normals1", v1);
//  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals2, 10, 0.05, "normals2", v2);
//
//  return (viewer);
//}
//
//
//unsigned int text_id = 0;
//void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
//                            void* viewer_void)
//{
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
//  if (event.getKeySym () == "r" && event.keyDown ())
//  {
//    std::cout << "r was pressed => removing all text" << std::endl;
//
//    char str[512];
//    for (unsigned int i = 0; i < text_id; ++i)
//    {
//      sprintf (str, "text#%03d", i);
//      viewer->removeShape (str);
//    }
//    text_id = 0;
//  }
//}
//
//void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
//                         void* viewer_void)
//{
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
//  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
//      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
//  {
//    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;
//
//    char str[512];
//    sprintf (str, "text#%03d", text_id ++);
//    viewer->addText ("clicked here", event.getX (), event.getY (), str);
//  }
//}
//
//boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
//{
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  viewer->addCoordinateSystem (1.0);
//
//  viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
//  viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);
//
//  return (viewer);
//}
//
//
//// --------------
//// -----Main-----
//// --------------
//int
//main11 (int argc, char** argv)
//{
//  // --------------------------------------
//  // -----Parse Command Line Arguments-----
//  // --------------------------------------
//  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
//  {
//    printUsage (argv[0]);
//    return 0;
//  }
//  bool simple(false), rgb(false), custom_c(false), normals(false),
//    shapes(false), viewports(false), interaction_customization(false);
//  if (pcl::console::find_argument (argc, argv, "-s") >= 0)
//  {
//    simple = true;
//    std::cout << "Simple visualisation example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-c") >= 0)
//  {
//    custom_c = true;
//    std::cout << "Custom colour visualisation example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-r") >= 0)
//  {
//    rgb = true;
//    std::cout << "RGB colour visualisation example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-n") >= 0)
//  {
//    normals = true;
//    std::cout << "Normals visualisation example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-a") >= 0)
//  {
//    shapes = true;
//    std::cout << "Shapes visualisation example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-v") >= 0)
//  {
//    viewports = true;
//    std::cout << "Viewports example\n";
//  }
//  else if (pcl::console::find_argument (argc, argv, "-i") >= 0)
//  {
//    interaction_customization = true;
//    std::cout << "Interaction Customization example\n";
//  }
//  else
//  {
//    printUsage (argv[0]);
//    return 0;
//  }
//
//  // ------------------------------------
//  // -----Create example point cloud-----
//  // ------------------------------------
//  pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
//  std::cout << "Genarating example point clouds.\n\n";
//  // We're going to make an ellipse extruded along the z-axis. The colour for
//  // the XYZRGB cloud will gradually go from red to green to blue.
//  uint8_t r(255), g(15), b(15);
//  for (float z(-1.0); z <= 1.0; z += 0.05)
//  {
//    for (float angle(0.0); angle <= 360.0; angle += 5.0)
//    {
//      pcl::PointXYZ basic_point;
//      basic_point.x = 0.5 * cosf (pcl::deg2rad(angle));
//      basic_point.y = sinf (pcl::deg2rad(angle));
//      basic_point.z = z;
//      basic_cloud_ptr->points.push_back(basic_point);
//
//      pcl::PointXYZRGB point;
//      point.x = basic_point.x;
//      point.y = basic_point.y;
//      point.z = basic_point.z;
//      uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
//              static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
//      point.rgb = *reinterpret_cast<float*>(&rgb);
//      point_cloud_ptr->points.push_back (point);
//    }
//    if (z < 0.0)
//    {
//      r -= 12;
//      g += 12;
//    }
//    else
//    {
//      g -= 12;
//      b += 12;
//    }
//  }
//  basic_cloud_ptr->width = (int) basic_cloud_ptr->points.size ();
//  basic_cloud_ptr->height = 1;
//  point_cloud_ptr->width = (int) point_cloud_ptr->points.size ();
//  point_cloud_ptr->height = 1;
//
//  // ----------------------------------------------------------------
//  // -----Calculate surface normals with a search radius of 0.05-----
//  // ----------------------------------------------------------------
//  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//  ne.setInputCloud (point_cloud_ptr);
//  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
//  ne.setSearchMethod (tree);
//  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
//  ne.setRadiusSearch (0.05);
//  ne.compute (*cloud_normals1);
//
//  // ---------------------------------------------------------------
//  // -----Calculate surface normals with a search radius of 0.1-----
//  // ---------------------------------------------------------------
//  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
//  ne.setRadiusSearch (0.1);
//  ne.compute (*cloud_normals2);
//
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//  if (simple)
//  {
//    viewer = simpleVis(basic_cloud_ptr);
//  }
//  else if (rgb)
//  {
//    viewer = rgbVis(point_cloud_ptr);
//  }
//  else if (custom_c)
//  {
//    viewer = customColourVis(basic_cloud_ptr);
//  }
//  else if (normals)
//  {
//    viewer = normalsVis(point_cloud_ptr, cloud_normals2);
//  }
//  else if (shapes)
//  {
//    viewer = shapesVis(point_cloud_ptr);
//  }
//  else if (viewports)
//  {
//    viewer = viewportsVis(point_cloud_ptr, cloud_normals1, cloud_normals2);
//  }
//  else if (interaction_customization)
//  {
//    viewer = interactionCustomizationVis();
//  }
//
//  //--------------------
//  // -----Main loop-----
//  //--------------------
//  while (!viewer->wasStopped ())
//  {
//    viewer->spinOnce (100);
//    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//  }
//}