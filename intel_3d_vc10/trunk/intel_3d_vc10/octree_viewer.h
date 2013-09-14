/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*
*  \author Raphael Favier
* */
# include <queue>

#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/common/common.h>

//#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>

#include <pcl/filters/filter.h>
#include "pcl_tools_boost.h"

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCubeSource.h>
#include <vtkGlyph3D.h>
//=============================
// Displaying cubes is very long!
// so we limit their numbers.
const int MAX_DISPLAYED_CUBES(15000);
//=============================

class OctreeViewer
{
public:
	OctreeViewer (double resolution) :
	  viz ("Octree visualizator"),
		  displayCloud (new pcl::PointCloud<pcl::PointXYZRGBA>()), octree (resolution), octree_search(resolution), displayCubes(false),
		  showPointsWithCubes (false), wireframe (true), m_resolution(resolution), pointSize(1), octree1(resolution)
	  {

		  //try to load the cloud
		  //if (!loadCloud(filename))
		  // return;



		  //viz.initCameraParameters();
		  //register keyboard callbacks
		  viz.registerKeyboardCallback(&OctreeViewer::keyboardEventOccurred, *this, 0);

		  //key legends
		  viz.addText("Keys:", 0, 170, 0.0, 1.0, 0.0, "keys_t");
		  viz.addText("a -> Increment displayed depth", 10, 155, 0.0, 1.0, 0.0, "key_a_t");
		  viz.addText("z -> Decrement displayed depth", 10, 140, 0.0, 1.0, 0.0, "key_z_t");
		  viz.addText("d -> Toggle Point/Cube representation", 10, 125, 0.0, 1.0, 0.0, "key_d_t");
		  viz.addText("x -> Show/Hide original cloud", 10, 110, 0.0, 1.0, 0.0, "key_x_t");
		  viz.addText("s/w -> Surface/Wireframe representation", 10, 95, 0.0, 1.0, 0.0, "key_sw_t");

		  //set current level to half the maximum one
		  displayedDepth = 1;

		  //show octree at default depth


		  //reset camera
		  viz.resetCameraViewpoint("cloud");

		  //viz.addCoordinateSystem(1.0);
		  //run main loop
		  //run();

	  };
	  OctreeViewer (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_colud, double resolution) :
	  viz ("Octree visualizator"), cloud (point_colud),
		  displayCloud (new pcl::PointCloud<pcl::PointXYZRGBA>()), octree (resolution), octree_search (resolution), displayCubes(false),
		  showPointsWithCubes (false), wireframe (true),m_resolution(resolution), pointSize(1), octree1(resolution)
	  {

		  //try to load the cloud
		  //if (!loadCloud(filename))
		  // return;

		  octree.setInputCloud(cloud);
		  //update bounding box automatically
		  octree.defineBoundingBox();
		  //add points in the tree
		  octree.addPointsFromInputCloud();


		  //viz.initCameraParameters();
		  //register keyboard callbacks
		  viz.registerKeyboardCallback(&OctreeViewer::keyboardEventOccurred, *this, 0);

		  //key legends
		  viz.addText("Keys:", 0, 170, 0.0, 1.0, 0.0, "keys_t");
		  viz.addText("a -> Increment displayed depth", 10, 155, 0.0, 1.0, 0.0, "key_a_t");
		  viz.addText("z -> Decrement displayed depth", 10, 140, 0.0, 1.0, 0.0, "key_z_t");
		  viz.addText("d -> Toggle Point/Cube representation", 10, 125, 0.0, 1.0, 0.0, "key_d_t");
		  viz.addText("x -> Show/Hide original cloud", 10, 110, 0.0, 1.0, 0.0, "key_x_t");
		  viz.addText("s/w -> Surface/Wireframe representation", 10, 95, 0.0, 1.0, 0.0, "key_sw_t");

		  //set current level to half the maximum one
		  displayedDepth = static_cast<int> (floor (octree.getTreeDepth() / 2.0));
		  if (displayedDepth == 0)
			  displayedDepth = 1;

		  //show octree at default depth
		  extractPointsAtLevel(displayedDepth);

		  //reset camera
		  viz.resetCameraViewpoint("cloud");

		  //viz.addCoordinateSystem(1.0);
		  //run main loop
		  run();

	  };

	  void update_data(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud)
	  {
		  cloud = point_cloud;
		  cloud->points.size();
		  octree = pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndices >(m_resolution);
		  octree_search = pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>(m_resolution);

		  octree.setInputCloud(point_cloud);
		  octree.addPointsFromInputCloud();
		  octree_search.setInputCloud(point_cloud);
		  octree_search.addPointsFromInputCloud();  
	  };

	  void drawAndSpin(){
		  extractPointsAtLevel(displayedDepth);
		  viz.spinOnce();
	  }

	  pcl::visualization::PCLVisualizer viz;
	  //octree
	  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndices > octree;
	  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA> octree1;
	  pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA> octree_search;
	  double m_resolution;
private:
	//========================================================
	// PRIVATE ATTRIBUTES
	//========================================================
	//visualizer
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr xyz;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_rgb;


	//original cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
	//displayed_cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr displayCloud;

	//level
	int displayedDepth;
	//bool to decide if we display points or cubes
	bool displayCubes, showPointsWithCubes, wireframe;
	int pointSize;
	//========================================================

	/* \brief Callback to interact with the keyboard
	*
	*/
	void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *)
	{

		if (event.getKeySym() == "a" && event.keyDown())
		{
			IncrementLevel();
		}
		else if (event.getKeySym() == "z" && event.keyDown())
		{
			DecrementLevel();
		}
		else if (event.getKeySym() == "d" && event.keyDown())
		{
			displayCubes = !displayCubes;
			update();
		}
		else if (event.getKeySym() == "x" && event.keyDown())
		{
			showPointsWithCubes = !showPointsWithCubes;
			update();
		}
		else if (event.getKeySym() == "w" && event.keyDown())
		{
			if(!wireframe)
				wireframe=true;
			update();
		}
		else if (event.getKeySym() == "s" && event.keyDown())
		{
			if(wireframe)
				wireframe=false;
			update();
		}
		else if (event.getKeySym() == "r" && event.keyDown())
		{
			IncrementPointSIze();
			update();
		}
		else if (event.getKeySym() == "t" && event.keyDown())
		{
			DecrementPointSIze();
			update();
		}


		
	}

	/* \brief Graphic loop for the viewer
	*
	*/
	void run()
	{
		while (!viz.wasStopped())
		{
			//main loop of the visualizer
			viz.spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}

	/* \brief Helper function that draw info for the user on the viewer
	*
	*/
	void showLegend(bool showCubes)
	{
		/*char pointSizeDisplay[256];
		sprintf(pointSizeDisplay, "Point Size %s", pointSize);
		viz.removeShape("disp_ps");
		viz.addText(pointSizeDisplay, 0, 75, 1.0, 0.0, 0.0, "disp_ps");*/

		char dataDisplay[256];
		if(showCubes)
			sprintf(dataDisplay, "Displaying data as %s", ("CUBES"));
		else
			sprintf(dataDisplay, "Displaying data as %s with size %d (r->increase, t->decrease)", ("POINTS"), pointSize);
		viz.removeShape("disp_t");
		viz.addText(dataDisplay, 0, 60, 1.0, 0.0, 0.0, "disp_t");

		char level[256];
		sprintf(level, "Displayed depth is %d on %d", displayedDepth, octree.getTreeDepth());
		viz.removeShape("level_t1");
		viz.addText(level, 0, 45, 1.0, 0.0, 0.0, "level_t1");

		viz.removeShape("level_t2");
		sprintf(level, "Voxel size: %.4fm [%zu voxels]", sqrt(octree.getVoxelSquaredSideLen(displayedDepth)),
			displayCloud->points.size());
		viz.addText(level, 0, 30, 1.0, 0.0, 0.0, "level_t2");

		viz.removeShape("org_t");
		if (showPointsWithCubes)
			viz.addText("Displaying original cloud", 0, 15, 1.0, 0.0, 0.0, "org_t");
	}

	/* \brief Visual update. Create visualizations and add them to the viewer
	*
	*/
	void update()
	{
		//remove existing shapes from visualizer
		clearView();

		//prevent the display of too many cubes
		bool displayCubeLegend = displayCubes && static_cast<int> (displayCloud->points.size ()) <= MAX_DISPLAYED_CUBES;

		showLegend(displayCubeLegend);

		if (displayCubeLegend)
		{
			//show octree as cubes
			showCubes(sqrt(octree.getVoxelSquaredSideLen(displayedDepth)));
			if (showPointsWithCubes)
			{
				//add original cloud in visualizer
				pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGBA> color_handler(cloud, "z");
				viz.addPointCloud(cloud, color_handler, "cloud");
				viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "cloud");
			}
		}
		else
		{
			//printf("maybe here....point size : %d\n", pointSize);
			//add current cloud in visualizer
			//pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGBA> color_handler(displayCloud,"z");
			viz.addPointCloud(displayCloud, "cloud");
			viz.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "cloud");
		}
	}

	/* \brief remove dynamic objects from the viewer
	*
	*/
	void clearView()
	{
		//remove cubes if any
		vtkRenderer *renderer = viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();
		while (renderer->GetActors()->GetNumberOfItems() > 0)
			renderer->RemoveActor(renderer->GetActors()->GetLastActor());
		//remove point clouds if any
		viz.removePointCloud("cloud");
	}

	/* \brief Create a vtkSmartPointer object containing a cube
	*
	*/
	vtkSmartPointer<vtkPolyData> GetCuboid(double minX, double maxX, double minY, double maxY, double minZ, double maxZ)
	{
		vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
		cube->SetBounds(minX, maxX, minY, maxY, minZ, maxZ);
		return cube->GetOutput();
	}

	/* \brief display octree cubes via vtk-functions
	*
	*/
	void showCubes(double voxelSideLen)
	{
		//get the renderer of the visualizer object
		vtkRenderer *renderer = viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();

		vtkSmartPointer<vtkAppendPolyData> treeWireframe = vtkSmartPointer<vtkAppendPolyData>::New();
		size_t i;
		double s = voxelSideLen / 2.0;
		for (i = 0; i < displayCloud->points.size(); i++)
		{

			double x = displayCloud->points[i].x;
			double y = displayCloud->points[i].y;
			double z = displayCloud->points[i].z;

			vtkSmartPointer<vtkPolyData> cubeData = GetCuboid(x - s, x + s, y - s, y + s, z - s, z + s);
			treeWireframe->AddInput(cubeData);
		}

		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
		colors->SetName("colors");
		colors->SetNumberOfComponents(3);
		for (i = 0; i < displayCloud->points.size(); i++)
		{

			double x = displayCloud->points[i].x;
			double y = displayCloud->points[i].y;
			double z = displayCloud->points[i].z;
			points->InsertNextPoint(x, y, z);

			int r = displayCloud->points[i].r;
			int g = displayCloud->points[i].g;;
			int b = displayCloud->points[i].b;;
			unsigned char color[3] = {r, g, b};
			colors->InsertNextTupleValue(color);
		}

		vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
		polydata->SetPoints(points);
		polydata->GetPointData()->SetScalars(colors);

		vtkSmartPointer<vtkCubeSource> cubeSource = 
			vtkSmartPointer<vtkCubeSource>::New();

		vtkSmartPointer<vtkGlyph3D> glyph3D = 
			vtkSmartPointer<vtkGlyph3D>::New();
		glyph3D->SetColorModeToColorByScalar();
		glyph3D->SetSourceConnection(treeWireframe->GetOutputPort());
#if VTK_MAJOR_VERSION <= 5
		glyph3D->SetInput(polydata);
#else
		glyph3D->SetInputData(polydata);
#endif
		glyph3D->ScalingOff();
		glyph3D->Update();

		vtkSmartPointer<vtkActor> treeActor = vtkSmartPointer<vtkActor>::New();

		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
		//mapper->SetInput(treeWireframe->GetOutput());
		mapper->SetInput(glyph3D->GetOutput());
		//mapper->SetInputConnection(glyph3D->GetOutputPort());

		treeActor->SetMapper(mapper);

		//treeActor->GetProperty()->SetColor(1.0, .5, .5);
		treeActor->GetProperty()->SetLineWidth(2);
		if(wireframe)
		{
			treeActor->GetProperty()->SetRepresentationToWireframe();
			treeActor->GetProperty()->SetOpacity(0.35);
		}
		else
			treeActor->GetProperty()->SetRepresentationToSurface();

		renderer->AddActor(treeActor);
	}

	/* \brief Extracts all the points of depth = level from the octree
	*
	*/
	void extractPointsAtLevel(int depth)
	{
		displayCloud->points.clear();

		pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndices >::BreadthFirstIterator tree_it;
		pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndices >::BreadthFirstIterator tree_it_end = octree.breadth_end(); 


		//for (tree_it = octree.breadth_begin(depth); tree_it!=tree_it_end; ++tree_it)
		//{
		//	//Diaply only speficed depth
		////	if(tree_it.getCurrentOctreeDepth() != depth) continue;
		//pcl::octree::OctreeNode* node = tree_it.getCurrentOctreeNode();
		//pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndices>* node_l = NULL;
		//bool isLeaf = false;
		//while(node->getNodeType() == pcl::octree::BRANCH_NODE && !isLeaf){
		//	pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>* node_b = (pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>*)node;

		//	for(int i = 0 ; i < 8 ; i++){
		//		if(node_b->hasChild(i)){
		//			node = node_b->getChildPtr(i);
		//			if(node->getNodeType() == pcl::octree::LEAF_NODE){
		//				isLeaf = true;
		//				break;
		//			}
		//		}
		//	}
		//}
		//pcl::octree::OctreeContainerPointIndices* leaf_container = ((pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndices>*)node)->getContainerPtr();
		//printf("here!\n");
		//}

		//return;


		pcl::PointXYZRGBA pt;
		//std::cout << "===== Extracting data at depth " << depth << "... " << std::flush;
		double start = pcl::getTime ();

		int branch_cnt, leaf_cnt;
		branch_cnt = 0;
		leaf_cnt = 0;
		for (tree_it = octree.breadth_begin(depth); tree_it!=tree_it_end; ++tree_it)
		{
			//Diaply only speficed depth
			if(tree_it.getCurrentOctreeDepth() != depth) continue;

			Eigen::Vector3f voxel_min, voxel_max;
			octree.getVoxelBounds(tree_it, voxel_min, voxel_max);

			pt.x = (voxel_min.x() + voxel_max.x()) / 2.0f;
			pt.y = (voxel_min.y() + voxel_max.y()) / 2.0f;
			pt.z = (voxel_min.z() + voxel_max.z()) / 2.0f; 

			
			//pcl::PointXYZRGBA a;
			//octree.getVoxelCentroidAtPoint(point_idx, a);
			
			pt.r = 255;//255;
			pt.g = 255;
			pt.b = 255;//255;

			

			
			if(tree_it.isBranchNode()){
				branch_cnt++;
				pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndices >::BranchContainer bc = tree_it.getBranchContainer();
				pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>* branchNode = (pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>*)tree_it.getCurrentOctreeNode();
				vector<pcl::PointXYZRGBA> subPoints = getSubtreePoints<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices>(branchNode);

				float r,g,b;
				r=0;
				g=0;
				b=0;
				for(int i = 0 ; i < subPoints.size() ; i++){
					r += subPoints[i].r;
					g += subPoints[i].g;
					b += subPoints[i].b;
				}

				pt.r = r/subPoints.size();//255;
				pt.g = g/subPoints.size();//255;
				pt.b = b/subPoints.size();//0;//255;

				//displayCloud->points.push_back(pt);
			}else{
				leaf_cnt++;
				//printf("this is leaf node!\n");
				
				
				//pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndex >::LeafContainer lc = tree_it.getLeafContainer();
				
				//lc.addPointIndex(20);
				
				//lc.getPointIndex(point_idx);
				//lc.addPoint(pt);
				int point_idx = tree_it.getLeafContainer().getPointIndex();
				pcl::PointXYZRGBA leaf_p = octree.getInputCloud()->points[point_idx];
				//printf("point_idx : %d\n", point_idx);
				pt.r = leaf_p.r;
				pt.g = leaf_p.g;
				pt.b = leaf_p.b;

				//printf("this is leaf node! and index : %d, %d, <%d. %d. %d>\n", point_idx, lc.getSize(), pt.r, pt.g, pt.b);

			}
			//pcl::octree::OctreeIteratorBase<pcl::PointXYZRGBA>::BranchContainer a = tree_it.getBranchContainer();

			displayCloud->points.push_back(pt);

		}
		printf("display points # : %d\n", displayCloud->points.size());
		//printf("branch_cnt : %d\n", branch_cnt);
		//printf("leaf_cnt : %d\n", leaf_cnt);

		//int node_cnt=0;
		//for (tree_it = octree.begin(depth); tree_it!=tree_it_end; ++tree_it)
		//{
		//	node_cnt++;

		//}
		//printf("node_cnt : %d\n", node_cnt);

		//pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndex >::LeafNodeIterator it1;
		//pcl::octree::OctreePointCloud<pcl::PointXYZRGBA, pcl::octree::OctreeContainerPointIndices, pcl::octree::OctreeContainerPointIndex >::LeafNodeIterator it1_end = octree.leaf_end();
		//int leaf_node_cnt=0;
		//for (it1 = octree.leaf_begin(); it1 != it1_end; ++it1)
		//{
		//	//it1.getLeafContainer().getPointIndices(indexVector);
		//	//int p_idx;
		//	//it1.getLeafContainer().getPointIndex(p_idx);
		//	
		//	Eigen::Vector3f voxel_min, voxel_max;
		//	octree.getVoxelBounds(it1, voxel_min, voxel_max);
		//	

		//	pt.x = (voxel_min.x() + voxel_max.x()) / 2.0f;
		//	pt.y = (voxel_min.y() + voxel_max.y()) / 2.0f;
		//	pt.z = (voxel_min.z() + voxel_max.z()) / 2.0f; 

		//	//octree.getVoxelCentroidAtPoint(p_idx, pt);
		//	//printf("p_idx : %d\n", p_idx);

		//	int point_idx = it1.getLeafContainer().getPointIndex();
		//	pcl::PointXYZRGBA leaf_p = octree.getInputCloud()->points[point_idx];
		//	//printf("p_idx : %d, //// %d, %d, %d\n", point_idx, leaf_p.r, leaf_p.r, leaf_p.r);
		//	pt.r = leaf_p.r;
		//	pt.g = leaf_p.g;
		//	pt.b = leaf_p.b;

		//	displayCloud->points.push_back(pt);


		//	leaf_node_cnt++;
		//}
		//printf("leaf_node_cnt : %d\n", leaf_node_cnt);

		double end = pcl::getTime ();
		//printf("%zu pts, %.4gs. %.4gs./pt. =====\n", displayCloud->points.size (), end - start,			(end - start) / static_cast<double> (displayCloud->points.size ()));

		update();
	}

	template<typename PointT, typename ContainerT>
	vector<PointT> getSubtreePoints(pcl::octree::OctreeBranchNode<ContainerT>* node_){
		vector<PointT> pointVec;
		std::queue<pcl::octree::OctreeNode*> nodeVec;
		

		pcl::octree::OctreeNode* node = node_;//tree_it.getCurrentOctreeNode();
		nodeVec.push(node);

		while(nodeVec.size() > 0){
			pcl::octree::OctreeNode* cur_n = (pcl::octree::OctreeBranchNode<ContainerT>*)nodeVec.front();
			nodeVec.pop();
			
			if(cur_n->getNodeType() == pcl::octree::BRANCH_NODE){
				pcl::octree::OctreeBranchNode<ContainerT>* cur_bn = (pcl::octree::OctreeBranchNode<ContainerT>*)cur_n;
				for(int i = 0 ; i < 8 ; i++){
					if(cur_bn->hasChild(i)){
						nodeVec.push(cur_bn->getChildPtr(i));
					}
				}
			}
			else if(cur_n->getNodeType() == pcl::octree::LEAF_NODE){
				pcl::octree::OctreeContainerPointIndices* leaf_container = ((pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndices>*)cur_n)->getContainerPtr();
				//int a = 1;
				int point_idx = leaf_container->getPointIndex();
				//printf("point_idx : %d\n", point_idx);
				PointT leaf_p = octree.getInputCloud()->points[point_idx];
				//printf("%d, %d, %d\n", leaf_p.r, leaf_p.g, leaf_p.b);
				pointVec.push_back(leaf_p);
			}
			//break;
		}
		/*node = nodeVec.front();

		pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndices>* node_l = NULL;
		bool isLeaf = false;
		while(node->getNodeType() == pcl::octree::BRANCH_NODE && !isLeaf){
			pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>* node_b = (pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices>*)node;

			for(int i = 0 ; i < 8 ; i++){
				if(node_b->hasChild(i)){
					node = node_b->getChildPtr(i);
					if(node->getNodeType() == pcl::octree::LEAF_NODE){
						isLeaf = true;
						break;
					}
				}
			}
		}
		pcl::octree::OctreeContainerPointIndices* leaf_container = ((pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndices>*)node)->getContainerPtr();
		printf("here!\n");*/

		/*vector<PointT> pointVec;
		std::queue<pcl::octree::OctreeNode*> nodeVec;
		nodeVec.push(node);
		
		while(nodeVec.size() > 0){
			pcl::octree::OctreeNode* cur_n = (pcl::octree::OctreeBranchNode<ContainerT>*)nodeVec.front();
			nodeVec.pop();

			if(cur_n->getNodeType() == pcl::octree::BRANCH_NODE){
				pcl::octree::OctreeBranchNode<ContainerT>* cur_bn = (pcl::octree::OctreeBranchNode<ContainerT>*)cur_n;
				for(int i = 0 ; i < 8 ; i++){
					if(cur_bn->hasChild(i)){
						nodeVec.push(cur_bn->getChildPtr(i));
					}
				}
			}
			else if(cur_n->getNodeType() == pcl::octree::LEAF_NODE){
				pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndex>* cur_ln = (pcl::octree::OctreeLeafNode<pcl::octree::OctreeContainerPointIndex>*)cur_n;
				pcl::octree::OctreeContainerPointIndex leaf_container= cur_ln->getContainer();
				int point_idx = leaf_container.getPointIndex();
				printf("point_idx : %d\n", point_idx);
				//PointT leaf_p = octree.getInputCloud()->points[point_idx];
				//printf("%d, %d, %d\n", leaf_p.r, leaf_p.g, leaf_p.b);
				//pointVec.push_back(leaf_p);
			}
		}*/
		return pointVec;
	}

	/* \brief Helper function to increase the octree display level by one
	*
	*/
	bool IncrementLevel()
	{
		if (displayedDepth < static_cast<int> (octree.getTreeDepth ()))
		{
			displayedDepth++;
			extractPointsAtLevel(displayedDepth);
			return true;
		}
		else
			return false;
	}

	/* \brief Helper function to decrease the octree display level by one
	*
	*/
	bool DecrementLevel()
	{
		if (displayedDepth > 0)
		{
			displayedDepth--;
			extractPointsAtLevel(displayedDepth);
			return true;
		}
		return false;
	}

	bool IncrementPointSIze()
	{
		pointSize++;
		return true;
	}

	bool DecrementPointSIze()
	{
		if(pointSize > 1){ 
			pointSize--;
			return true;
		}
		return false;
	}

};