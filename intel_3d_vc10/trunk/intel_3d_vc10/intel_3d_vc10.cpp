#include "stdafx.h"
#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxcimage.h"
#include "pxcaccelerator.h"
#include "util_capture_file.h"
#include "util_render.h"
#include "util_pipeline.h"
#include "util_cmdline.h"
#include "Settings.h"

#include <gl/glut.h>
#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\gpu\gpu.hpp>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>

#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>



using namespace std;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

#define SCALE_MM_TO_M 1000

#define CALIBRATION_UX 0
#define CALIBRATION_UY -28
#define CALIBRATION_PX 4200

const int VOXEL_DIM = 120;
const int VOXEL_SIZE = VOXEL_DIM*VOXEL_DIM*VOXEL_DIM;
const int VOXEL_SLICE = VOXEL_DIM*VOXEL_DIM;
const int distance_threshold = 7000;



struct startParams {
    float startX;
    float startY;
    float startZ;
    float voxelWidth;
    float voxelHeight;
    float voxelDepth;
};

void renderModel(float fArray[], startParams params);

void setInitialFrameLocation(vector<UtilRender*> renders);
static double computeReprojectionErrors( const vector<vector<cv::Point3f> >& objectPoints,
                                         const vector<vector<cv::Point2f> >& imagePoints,
                                         const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
                                         const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                         vector<float>& perViewErrors);
static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, vector<cv::Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/);
bool runCalibrationAndSave(Settings& s, cv::Size imageSize, cv::Mat&  cameraMatrix, cv::Mat& distCoeffs,vector<vector<cv::Point2f> > imagePoints );
static bool runCalibration( Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                            vector<vector<cv::Point2f> > imagePoints, vector<cv::Mat>& rvecs, vector<cv::Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr);
static void saveCameraParams( Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                              const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<cv::Point2f> >& imagePoints,
                              double totalAvgErr );
//#pragma comment (lib, "glut64.lib")     /* link with OpenGL Utility lib */

typedef struct sVertexType{
	float fX;
	float fY;
	float fZ;
	float fNX;
	float fNY;
	float fNZ;
	float fU;
	float fV;
}SVERTEXTYPE;

bool isFunctionWork = false;
PXCPointU32 prevLoc;



int wmain(int argc, WCHAR* argv[]) { 
	
	Settings s;
    const string inputSettingsFile = "in_VID5.xml";
	const string inputCameraInfo = "our_camera_data.xml";
    cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
	cv::FileStorage fs_our(inputCameraInfo, cv::FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
	if (!fs_our.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputCameraInfo << "\"" << endl;
        return -1;
    }
	s.read(fs["Settings"]);
    fs.release();                                         // close Settings file

	cv::Mat cameraMatrix, distCoeffs;
	fs_our["Camera_Matrix"] >> cameraMatrix;
	fs_our["Distortion_Coefficients"] >> distCoeffs;
	fs_our.release();

	cv::Mat cameraMatrix_inv = cameraMatrix.inv();

	if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<cv::Point2f> > imagePoints;
    

    cv::Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	int mode_sheet_capture = true;
    clock_t prevTimestamp = 0;
    const cv::Scalar RED(0,0,255), GREEN(0,255,0);
    const char ESC_KEY = 27;
	
	//SVERTEXTYPE* pVertexMem = new SVERTEXTYPE[320*240];

	pxcStatus sts; 

	// Create session 
	PXCSmartPtr<PXCSession> session; 
	sts = PXCSession_Create(&session); 
	if (sts<PXC_STATUS_NO_ERROR) { 
		wprintf(L"Failed to create a session\n"); 
		return 3; 
	} 

	UtilCaptureFile capture(session, 0, false); 

	// Set source device search critieria 
	capture.SetFilter(L"DepthSense Device 325"); 
	PXCSizeU32 size_VGA = {640, 480}; 
	capture.SetFilter(PXCImage::IMAGE_TYPE_COLOR, size_VGA); 
	PXCSizeU32 size_QVGA = {320, 240}; 
	capture.SetFilter(PXCImage::IMAGE_TYPE_DEPTH, size_QVGA); 

	PXCCapture::VideoStream::DataDesc request; 
	memset(&request, 0, sizeof(request)); 

	request.streams[0].format=PXCImage::COLOR_FORMAT_RGB32; 
	request.streams[1].format=PXCImage::COLOR_FORMAT_DEPTH; 
	request.streams[2].format=PXCImage::COLOR_FORMAT_VERTICES; 

	sts = capture.LocateStreams (&request); 
	if (sts<PXC_STATUS_NO_ERROR) { 
		wprintf(L"Failed to locate color and depth streams\n"); 
		return 1; 
	} 

	PXCCapture::Device* device = capture.QueryDevice(); 

	PXCCapture::VideoStream::ProfileInfo pinfo1; 
	capture.QueryVideoStream(0)->QueryProfile(&pinfo1); 
	PXCCapture::VideoStream::ProfileInfo pinfo2; 
	capture.QueryVideoStream(1)->QueryProfile(&pinfo2); 
	PXCCapture::VideoStream::ProfileInfo pinfo3; 
	capture.QueryVideoStream(2)->QueryProfile(&pinfo3); 


	std::vector<UtilRender*> renders; 
	renders.push_back(new UtilRender(L"Color with UVMap")); 
	renders.push_back(new UtilRender(L"Depth")); 
	renders.push_back(new UtilRender(L"Color + Calibration ")); 
	renders.push_back(new UtilRender(L"Undistorted Color")); 
	renders.push_back(new UtilRender(L"Depth Denoised")); 
	renders.push_back(new UtilRender(L"Vertices")); 

	PXCImage::ImageData data0; 
	PXCImage::ImageData data1; 
	PXCImage::ImageData data2; 
	PXCImage::ImageData data3;
	PXCImage::ImageData data4; 
	PXCImage::ImageData data5; 

	//CvPoint2D32f* pointBuf = new CvPoint2D32f[ 100 ];
	vector<cv::Point2f> pointBuf2;
	int corner_count;
	vector<vector<cv::Point2f> > points_vec;

	setInitialFrameLocation(renders);
	
	boost::shared_ptr<pcl::visualization::PCLVisualizer> scene_viewer (new pcl::visualization::PCLVisualizer ("Coordination Viewer"));
	scene_viewer->initCameraParameters ();
	int vp_1(1), vp_2(1);
	scene_viewer->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
	scene_viewer->setBackgroundColor(0, 0, 0, vp_1);
	scene_viewer->addText ("Camera Coordination", 10, 10, "v1 text", vp_1);

	scene_viewer->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
	scene_viewer->setBackgroundColor(0.2, 0.2, 0.2, vp_2);
	

	//scene_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
	//scene_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
	scene_viewer->addCoordinateSystem (1.0);

	float scene_scale_factor = SCALE_MM_TO_M;

	//pcl::visualization::PCLVisualizer camera_position_viewer ("Camera position");
	
	vector<cv::Point3f> objectPoints;
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints, s.calibrationPattern);

	vector<cv::Point3f> obj_pnts;
	int x_min = -2000;
	int x_max = 2000;
	float x_interval = (x_max - x_min) / VOXEL_DIM;
	int y_min = -2000;
	int y_max = 200;
	float y_interval = (y_max - y_min) / VOXEL_DIM;
	int z_min = 0;
	int z_max = 300;
	float z_interval = (z_max - z_min) / VOXEL_DIM;

	for(float i = x_min ; i < x_max ; i+=x_interval)
		for(float j = y_min; j < y_max ; j+=y_interval){
			for(float k = z_min; k < z_max ; k+=z_interval){
			//printf("%f, %f, %f\n", i, j, 0);
				obj_pnts.push_back(cv::Point3f(i, j, k));
			}
		}

	
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr worldCoord_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	for (int f=0;;f++, Sleep(5)) { 
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cameraCoord_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		
		//pcl::octree::OctreePointCloud<pcl::PointXYZRGBA> octreeA (1.0f);
		
		//octreeA.setInputCloud (cameraCoord_cloud);
		//octreeA.addPointsFromInputCloud ();
		
		//pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>::LeafNodeIterator itL ((octreeA);
		/*
		unsigned int leafCount = 0;
		while ( *++itL )
		{
			++leafCount;
		}
		printf("%d\n", leafCount);*/

		bool blinkOutput = false;
		//printf("%d\n", f);
		PXCSmartArray<PXCImage> images(3); 
		PXCSmartArray<PXCImage> images_modified(3); 
		PXCSmartSP sp; 

		capture.ReadStreamAsync(images, &sp); 
		sp->Synchronize();

		//images[2] = images[0];
		

		//RGB Source
		images[0]->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::COLOR_FORMAT_RGB32,&data0); 
		//Depth Source
		images[1]->AcquireAccess(PXCImage::ACCESS_READ,&data1); 
		//Vertidex Source
		images[2]->AcquireAccess(PXCImage::ACCESS_READ,&data5); 
		

		PXCAccelerator *accelerator;
		session->CreateAccelerator(PXCAccelerator::ACCEL_TYPE_CPU, &accelerator);

		PXCImage::ImageInfo info;
		memset(&info, 0, sizeof(info));
		info.width=640; 
		info.height=480; 
		info.format=PXCImage::COLOR_FORMAT_RGB32;

		PXCImage *test = 0;
		memset(&data2, 0, sizeof(data2));
		data2.format = PXCImage::COLOR_FORMAT_RGB24;
		data2.planes[0] = data0.planes[0];
		data2.pitches[0] = data0.pitches[0];

		memset(&test, 0, sizeof(test));
		sts = accelerator->CreateImage(&info, 0, 0, &images_modified[0]);
		sts = accelerator->CreateImage(&info, 0, 0, &images_modified[1]);
		sts = accelerator->CreateImage(&info, 0, &data0, &test);
		images_modified[0]->CopyData(test);
		images_modified[1]->CopyData(test);
		test->Release();

		memset(&test, 0, sizeof(test));
		PXCImage::ImageInfo info_depth;
		memset(&info_depth, 0, sizeof(info_depth));
		info_depth.width=320; 
		info_depth.height=240; 
		info_depth.format=PXCImage::COLOR_FORMAT_DEPTH;
		sts = accelerator->CreateImage(&info_depth, 0, 0, &images_modified[2]);
		sts = accelerator->CreateImage(&info_depth, 0, &data1, &test);
		images_modified[2]->CopyData(test);
		test->Release();


		images_modified[0]->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::COLOR_FORMAT_RGB32,&data2); 
		IplImage* colorimg = cvCreateImageHeader(cvSize(640, 480), 8, 4);
		cvSetData(colorimg, (uchar*)data2.planes[0], 640*4*sizeof(uchar));		

		images_modified[1]->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::COLOR_FORMAT_RGB32,&data3); 
		IplImage* colorimg_undistorted = cvCreateImageHeader(cvSize(640, 480), 8, 4);
		cvSetData(colorimg_undistorted, (uchar*)data3.planes[0], 640*4*sizeof(uchar));		

		images_modified[2]->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::COLOR_FORMAT_DEPTH,&data4); 
		IplImage* depth_denoise = cvCreateImageHeader(cvSize(320, 240), 8, 1);
		cvSetData(depth_denoise, (uchar*)data4.planes[0], 320*1*sizeof(uchar));		
		//cvCircle(depth_denoise, cvPoint(10, 10), 10, cvScalar(0, 0, 0, 0), 2);

		float *uvmap=(float*)data1.planes[2]; 
		float *depthmap = (float*)data1.planes[0]; 
		short *vertidex = (short*)data5.planes[0];

		

		//f(1==1) return 0;
		
		
		cv::Mat depthImage(240/*DepthSize.height*/, 320/*DepthSize.width*/, CV_16UC1, data4.planes[0], data4.pitches[0]);

		cv::Mat depthImage_up;
		cv::resize(depthImage, depthImage_up, cv::Size(640, 480));
		double m, M;
		cv::minMaxIdx(depthImage, &m, &M, 0, 0, depthImage < 32000);
		cv::Mat dstImage(depthImage.size(), CV_8UC1);
		depthImage.convertTo(dstImage, CV_8UC1, 255/(M-m), 1.0*(-m)/(M-m));
		//dstImage = 255 - dstImage;


		cv::Mat image(colorimg);
		cv::Mat image_gray;
		cv::cvtColor(image, image_gray, CV_BGRA2GRAY);

		cv::Mat image_undistorted(colorimg_undistorted);
		cv::Mat temp = image_undistorted.clone();
		cv::undistort(temp, image_undistorted, cameraMatrix, distCoeffs);
		//vector<Point2f> pointBuf;

		cv::Mat UVImage(240/*DepthSize.height*/, 320/*DepthSize.width*/, CV_16UC2);

		for (int y=0;y<(int)pinfo2.imageInfo.height;y++) { 
			for (int x=0;x<(int)pinfo2.imageInfo.width;x++) { 
				int xx=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+0] *pinfo1.imageInfo.width+0.5); 
				int yy=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+1] *pinfo1.imageInfo.height+0.5); 
				int ux = CALIBRATION_UX;
				int uy = CALIBRATION_UY;
				float px = CALIBRATION_PX;
				short depth_value = depthImage.at<short>(y, x);
				//printf("%f, %d, %f\n", px, depth_value, px / depth_value);
				xx = xx + ux + px / depth_value;
				yy = yy + uy;
				UVImage.at<cv::Vec2s>(y, x) = cv::Vec2s(xx, yy);
			}
		}

		//cv::Mat UVImage_undistorted;// = UVImage.clone();
		//cv::Mat temp2;
		//cv::resize(UVImage, temp2, cv::Size(640, 320), 0.0, 0.0, CV_INTER_NN);
		//
		//cv::undistort(temp2, UVImage_undistorted, cameraMatrix, distCoeffs);

		

		

		bool found;
        switch( s.calibrationPattern ) // Find feature points on the input format
        {
        case Settings::CHESSBOARD:
            //found = cvFindChessboardCorners( colorimg, s.boardSize, pointBuf, &corner_count,
            //   CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
			found = cv::findChessboardCorners(image_gray, s.boardSize, pointBuf2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
            break;
        case Settings::CIRCLES_GRID:
            found = cv::findCirclesGrid( image_gray, s.boardSize, pointBuf2 );
			
            break;
        case Settings::ASYMMETRIC_CIRCLES_GRID:
			
            found = cv::findCirclesGrid( image_gray, s.boardSize, pointBuf2, cv::CALIB_CB_ASYMMETRIC_GRID );

            break;
        default:
            found = false;
            break;
        }
			
		
		//If chessboard is found,
		if ( found)              
        {
			// improve the found corners' coordinate accuracy for chessboard
            if( s.calibrationPattern == Settings::CHESSBOARD)
            {
                cv::Mat viewGray;
                //cv::cvtColor(image, viewGray, CV_BGR2GRAY);
                cornerSubPix( image_gray, pointBuf2, cv::Size(11,11),
                    cv::Size(-1,-1), cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
            }

            if( mode_sheet_capture && mode == CAPTURING &&  // For camera only take new samples after delay time
                (clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
            { 
				points_vec.push_back(pointBuf2);
				//runCalibrationAndSave(s, s.boardSize, cameraMatrix, distCoeffs, points_vec);
				printf("points set num : %d, delay : %d\n", points_vec.size(), s.delay);
                prevTimestamp = clock(); 
                blinkOutput = s.inputCapture.isOpened();
				mode_sheet_capture = false;
            }
				
            // Draw the corners.
			drawChessboardCorners( image, s.boardSize, pointBuf2,  found );

			CvFont font;
			cvInitFont(&font, CV_FONT_VECTOR0, 0.45, 0.45, 0, 1);
			char text[300];
			sprintf(text, "Points set num : %d", points_vec.size());
			cvPutText(colorimg, text, cvPoint(10, 10), &font, CV_RGB(255, 255, 0));

			cv::Mat img_pnt_Mat(pointBuf2);
			CvMat img_pnt_cvMat = img_pnt_Mat;

			//Set chessboard edge positions. (0,0 0,20 20,20 ... )
			vector<cv::Point3f> objectPoints;
			calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints, s.calibrationPattern);
								

			////Calculate Extrinsic Parameters
			//Just conversion..
			cv::Mat obj_pnt_Mat(objectPoints);
			CvMat obj_pnt_cvMat = obj_pnt_Mat;
			CvMat* rvecs = cvCreateMat( 3, 1, CV_64F);
			CvMat* tvecs  = cvCreateMat( 3, 1, CV_64F);

			cv::Mat rvecs_mat;
			cv::Mat tvecs_mat;

			CvMat cameraMatrix_cvmat = cameraMatrix;
			CvMat distCoeffs_cvmat = distCoeffs;
			//Compute Extrinsic Parameter
			cvFindExtrinsicCameraParams2(&obj_pnt_cvMat, &img_pnt_cvMat, &cameraMatrix_cvmat, NULL, rvecs, tvecs);
			
			//cv::calibrateCamera(objectPoints, pointBuf2, cv::Size(640, 320), cameraMatrix, distCoeffs, rvecs_mat, tvecs_mat);
			//Get camera position
			/*CvScalar t1 = cvGet2D(tvecs, 0, 0);
			CvScalar t2 = cvGet2D(tvecs, 1, 0);
			CvScalar t3 = cvGet2D(tvecs, 2, 0);*/

			cv::Mat objPoints = cv::Mat(cv::Mat(obj_pnts));
			CvMat objPoints2 = objPoints;

			//CvMat rvecs2 = rvecs[i];
			//CvMat tvecs2 = tvecs[i];

			cv::Mat rvecs2 = rvecs;
			cv::Mat tvecs2 = tvecs;

			rvecs2.convertTo(rvecs2, CV_32F);
			tvecs2.convertTo(tvecs2, CV_32F);

			tvecs2 = tvecs2.t();
			rvecs2 = rvecs2.t();
			//cv::Mat rvces_32f;
			//cv::Mat tvces_32f;
				
				
			cv::Size size1 = rvecs2.size();
			cv::Size size2 = tvecs2.size();

			cv::Mat cameraMatrix_32f;
			cv::Mat distCoeffs_32f;

			//cameraMatrix.convertTo(cameraMatrix_32f, CV_32FC1);
			//distCoeffs.convertTo(distCoeffs_32f, CV_32FC1);
				
			CvMat camera_matrix2 = cameraMatrix;
			CvMat distCoeffs2 = distCoeffs;
			//cv::Mat pointMat = cv::Mat(imagePoints2);
			CvMat* imagePoints22 = cvCreateMat(obj_pnts.size(),1,CV_32FC2);;
					
			cv::Mat rot;
			Rodrigues(rvecs2, rot);
			cv::Mat extMat_3;
			cv::hconcat(rot, tvecs2.t(), extMat_3);
			cv::Mat b_row(1, 4, CV_32F);
			b_row.at<float>(cv::Point(0, 0)) = 0;
			b_row.at<float>(cv::Point(1, 0)) = 0;
			b_row.at<float>(cv::Point(2, 0)) = 0;
			b_row.at<float>(cv::Point(3, 0)) = 1;

			cv::Mat extMat;
			cv::vconcat(extMat_3, b_row, extMat);
			cv::Mat extMatInv = extMat.inv();
			
			/*printf("Intrinsic Matrix\n");
			for(int i = 0 ; i < 3 ; i++){
				for(int j = 0 ; j < 3 ; j++){
					printf("%f\t", cameraMatrix.at<double>(i, j));
				}
				printf("\n");
			}
			printf("\n");
			printf("extrinsic Matrix\n");
			printf("%d, %d\n", extMat.size().height, extMat.size().width);
			for(int i = 0 ; i < 4 ; i++){
				for(int j = 0 ; j < 4 ; j++){
					printf("%f\t", extMat.at<float>(i, j));
				}
				printf("\n");
			}
			printf("\n");
			printf("extrinsic Matrix inverse\n");
			printf("%d, %d\n", extMatInv.size().height, extMatInv.size().width);
			for(int i = 0 ; i < 4 ; i++){
				for(int j = 0 ; j < 4 ; j++){
					printf("%f\t", extMatInv.at<float>(i, j));
				}
				printf("\n");
			}
			printf("\n");*/

			//printf("%f, %f, %f\n", t1.val[0], t2.val[0], t3.val[0]);

			
			pcl::PointXYZRGBA p_prev;

			const int line_max = 7;
			static const CvScalar line_colors[line_max] =
			{
				{{0,0,255}},
				{{0,128,255}},
				{{0,200,200}},
				{{0,255,0}},
				{{200,200,0}},
				{{255,0,0}},
				{{255,0,255}}
			};

			//Draw object poinsts = chessboard features.
			for(int i = 0 ; i < objectPoints.size() ; i++){
				cv::Point3f objPoint = objectPoints[i];

				

				pcl::PointXYZRGBA p;
				//Scaling ( mm to m) , 
				p.x = objPoint.x/scene_scale_factor;
				p.y = objPoint.y/scene_scale_factor;
				p.z = objPoint.z/scene_scale_factor;

				int y = floor(i/9.0);
				CvScalar color = line_colors[y % line_max];
				p.r = color.val[0];
				p.g = color.val[1];
				p.b = color.val[2];
				worldCoord_cloud->push_back(p);

				if(i > 0){
					char line_id_char[200];
					sprintf(line_id_char, "line_id_%d_%d_%d_to_%d_%d_%d", p.x, p.y, p.z, p_prev.x, p_prev.y, p_prev.z);
					std::string line_id = line_id_char;
						
					//camera_position_viewer.addLine<pcl::PointXYZRGBA, pcl::PointXYZRGBA>(p, p_prev, line_id);
					//camera_position_viewer.spinOnce();
					//camera_position_viewer.removePointCloud(line_id);
				}

				p_prev = p;

				cv::Mat pos(4, 1, CV_32FC1);
				pos.at<float>(0, 0) = objPoint.x;//
				pos.at<float>(1, 0) = objPoint.y;
				pos.at<float>(2, 0) = objPoint.z;
				pos.at<float>(3, 0) = 1;

				cv::Mat conv_pos = extMat*pos;
					

				pcl::PointXYZRGBA p1;				
				p1.x = conv_pos.at<float>(0, 0);///conv_pos.at<float>(2, 0);
				p1.y = conv_pos.at<float>(1, 0);///conv_pos.at<float>(2, 0);
				p1.z = conv_pos.at<float>(2, 0);///conv_pos.at<float>(2, 0);

				//printf("%f, %f, %f\n", conv_pos.at<float>(0, 0), conv_pos.at<float>(1, 0), conv_pos.at<float>(2, 0));

				p1.x /= scene_scale_factor;
				p1.y /= scene_scale_factor;
				p1.z /= scene_scale_factor;

				p1.r = 230;//color.val[0];
				p1.g = 230;//color.val[1];
				p1.b = 100;//color.val[2];
				cameraCoord_cloud->push_back(p1);


				cv::Mat conv_pos2(3, 1, CV_64FC1);
				conv_pos2.at<double>(0, 0) = conv_pos.at<float>(0, 0);
				conv_pos2.at<double>(1, 0) = conv_pos.at<float>(1, 0);
				conv_pos2.at<double>(2, 0) = conv_pos.at<float>(2, 0);
				
				cv::Mat conv_pos_img = cameraMatrix*conv_pos2;
				double reproject_z = conv_pos_img.at<double>(2, 0);
				double reproject_x = conv_pos_img.at<double>(0, 0);
				double reproject_y = conv_pos_img.at<double>(1, 0);

				//Add distortion matrix computation.

				//printf("%f, %f, %f\n", reproject_x, reproject_y, reproject_z);

				reproject_x /= (reproject_z);
				reproject_y /= (reproject_z);
				
				cv::circle(image, cv::Point(reproject_x, reproject_y), 2, cv::Scalar(color.val[2], color.val[1], color.val[0]), 2);


				cv::Mat pos_img(3, 1, CV_64FC1);
				pos_img.at<double>(0, 0) = reproject_x;
				pos_img.at<double>(1, 0) = reproject_y;
				pos_img.at<double>(2, 0) = 1;
				cv::Mat conv_pos_camera = cameraMatrix_inv*pos_img;
				pcl::PointXYZRGBA p2;				

				
				p2.z = conv_pos.at<float>(2, 0);//conv_pos_camera.at<double>(2, 0);///scene_scale_factor;
				//1120
				//1650
				//printf("z : %f\n", p2.z);
				p2.x = conv_pos_camera.at<double>(0, 0) * p2.z;///scene_scale_factor;
				p2.y = conv_pos_camera.at<double>(1, 0) * p2.z;///scene_scale_factor;

				p2.x /= scene_scale_factor;
				p2.y /= scene_scale_factor;
				p2.z /= scene_scale_factor;


				p2.r = 230;//color.val[0];
				p2.g = 100;//color.val[1];
				p2.b = 230;//color.val[2];
				cameraCoord_cloud->push_back(p2);

			}

			/*pcl::PointXYZRGBA p;
			p.x = t1.val[0]/scene_scale_factor;
			p.y = t2.val[0]/scene_scale_factor;
			p.z = t3.val[0]/scene_scale_factor;
			p.r = 255;
			p.b = 0;
			p.g = 0;
			worldCoord_cloud->push_back(p);*/

				
				

			//camera_position_viewer.addPointCloud (worldCoord_cloud, "worldCoord_cloud");
			//camera_position_viewer.spinOnce();
			//camera_position_viewer.removePointCloud("worldCoord_cloud");

			/*for (int y=0;y<(int)pinfo2.imageInfo.height;y++) { 
				for (int x=0;x<(int)pinfo2.imageInfo.width;x++) { 
					cv::Mat locMat(3, 1, CV_64FC1);
						
					locMat.at<double>(0,0) = x;
					locMat.at<double>(1,0) = y;
					locMat.at<double>(2,0) = 1;

					cv::Mat conv_locMat = cameraMatrix_inv * locMat;

					pcl::PointXYZRGBA p;
					p.x = conv_locMat.at<double>(0,0);///scene_scale_factor;
					p.y = conv_locMat.at<double>(1,0);///scene_scale_factor;
					p.z = conv_locMat.at<double>(2,0);///scene_scale_factor;
					printf("%f, %f, %f\n", p.x, p.y, p.z);

					cv::Vec4b colorValue = image.at<cv::Vec4b>(y, x);
					p.r = colorValue.val[2];
					p.g = colorValue.val[1];
					p.b = colorValue.val[0];
					worldCoord_cloud->push_back(p);
				}
			}*/

				

			
			

			char text_tmp[100];
			sprintf(text_tmp, "World Coordination, Current Camera Position: x:%.2f, y:%.2f, z:%.2f", extMat.at<float>(0, 3), extMat.at<float>(1, 3), extMat.at<float>(2, 3));
			std::string str_cam_pos = text_tmp;
			scene_viewer->addText (str_cam_pos, 10, 10, "v2 text", vp_2);

			//float *uvmap=(float*)data1.planes[2]; 
			
			//Mark Camera Position.
			//Origin
			cv::Mat pos_origin(4, 1, CV_32F);
			pos_origin.at<float>(0, 0) = 0;
			pos_origin.at<float>(1, 0) = 0;
			pos_origin.at<float>(2, 0) = 0;
			pos_origin.at<float>(3, 0) = 1;
			//camera_posiion_in_world_coordination = (M_ext)^-1 * origin
			cv::Mat conv_pos_origin = extMatInv*pos_origin;
			pcl::PointXYZRGBA p_origin;		
			p_origin.x = conv_pos_origin.at<float>(0, 0)/scene_scale_factor;
			p_origin.y = conv_pos_origin.at<float>(1, 0)/scene_scale_factor;
			p_origin.z = conv_pos_origin.at<float>(2, 0)/scene_scale_factor;
			p_origin.r = 255;
			p_origin.g = 120;
			p_origin.b = 120;
			worldCoord_cloud->push_back(p_origin);


			

			for (int y=0;y<(int)pinfo2.imageInfo.height;y++) { 
				for (int x=0;x<(int)pinfo2.imageInfo.width;x++) { 

					int vert_idx = 3*(y*pinfo2.imageInfo.width+x);
					int xx=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+0] *pinfo1.imageInfo.width+0.5); 
					int yy=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+1] *pinfo1.imageInfo.height+0.5); 
					int ux = CALIBRATION_UX;
					int uy = CALIBRATION_UY;
					float px = CALIBRATION_PX;
					
					short depth_value = depthImage.at<short>(y, x);
					//Set depth threshold 1000 = 1 meter
					if(depth_value > distance_threshold) continue;
					//printf("%f, %d, %f\n", px, depth_value, px / depth_value);
					xx = xx + ux + px / depth_value;
					yy = yy + uy;

					//If matched point over the image boundary, then throw away.
					if (xx<0 || xx>=(int)pinfo1.imageInfo.width-1 || yy<0 || yy>= (int)pinfo1.imageInfo.height-1) continue;

					//Get camera coordination with depth = 1
					cv::Mat pos_img(3, 1, CV_64FC1);
					pos_img.at<double>(0, 0) = xx;
					pos_img.at<double>(1, 0) = yy;
					pos_img.at<double>(2, 0) = 1;
					cv::Mat conv_pos_camera = cameraMatrix_inv*pos_img;

					double c_coord_x = conv_pos_camera.at<double>(0, 0);///scene_scale_factor;
					double c_coord_y = conv_pos_camera.at<double>(1, 0);///scene_scale_factor;
					double c_coord_z = conv_pos_camera.at<double>(2, 0);///scene_scale_factor;

					c_coord_z *= depth_value*1.5;//vertidex[vert_idx+2];
					c_coord_x *= c_coord_z;
					c_coord_y *= c_coord_z;

					/*double distance_o_to_xy = sqrt((double)1*1+xx*xx+yy*yy);
					double scale = distance_o_to_xy / depth_value;
												
					double c_coord_x_c = xx * scale;
					double c_coord_y_c = yy * scale;
					double c_coord_z_c = 1 * scale;*/

					pcl::PointXYZRGBA p2;				
					p2.x = c_coord_x/scene_scale_factor;
					p2.y = c_coord_y/scene_scale_factor;
					p2.z = c_coord_z/scene_scale_factor;

					cv::Vec4b colorValue = image.at<cv::Vec4b>(yy, xx);
					p2.r = colorValue.val[2];
					p2.g = colorValue.val[1];
					p2.b = colorValue.val[0];
					cameraCoord_cloud->push_back(p2);

					/*int xx_max = xx+1;
					int yy_max = yy+1;

					for(int xx_t = xx ; xx_t < xx_max ; xx_t++){
						for(int yy_t = yy ; yy_t < yy_max ; yy_t++){

						}
					}*/

					//cv::Mat pos_img(3, 1, CV_64FC1);
					//pos_img.at<double>(0, 0) = x;
					//pos_img.at<double>(1, 0) = y;
					//pos_img.at<double>(2, 0) = 1;

					//cv::Mat conv_pos_img = cameraMatrix_inv*pos_img;
					//pcl::PointXYZRGBA p_img_c_coord;		
					//p_img_c_coord.x = pos_img.at<double>(0, 0);///scene_scale_factor;
					//p_img_c_coord.y = pos_img.at<double>(1, 0);///scene_scale_factor;
					//p_img_c_coord.z = pos_img.at<double>(2, 0);///scene_scale_factor;
					//cv::Vec4b colorValue = image.at<cv::Vec4b>(y, x);

					//p_img_c_coord.r = colorValue.val[2];
					//p_img_c_coord.g = colorValue.val[1];
					//p_img_c_coord.b = colorValue.val[0];

					///*p_img_c_coord.r = 120;
					//p_img_c_coord.g = 255;
					//p_img_c_coord.b = 120;*/
					//cameraCoord_cloud->push_back(p_img_c_coord);




					cv::Mat pos(4, 1, CV_32F);
					pos.at<float>(0, 0) = c_coord_x;//*scene_scale_factor;//vertidex[vert_idx];
					pos.at<float>(1, 0) = c_coord_y;//*scene_scale_factor;//vertidex[vert_idx+1];
					pos.at<float>(2, 0) = c_coord_z;//vertidex[vert_idx+2];///scene_scale_factor;//vertidex[vert_idx+2]; 
					pos.at<float>(3, 0) = 1;

					cv::Mat conv_pos = extMatInv*pos;
					

					pcl::PointXYZRGBA p1;				
					p1.x = conv_pos.at<float>(0, 0)/scene_scale_factor;
					p1.y = conv_pos.at<float>(1, 0)/scene_scale_factor;
					p1.z = conv_pos.at<float>(2, 0)/scene_scale_factor;
					//printf("%f, %f, %f\n", p1.x, p1.y, p1.z);
					
					int color = ((pxcU32*)data2.planes[0])[yy*pinfo1.imageInfo.width+xx];
						
					//cv::Vec4b colorValue = image.at<cv::Vec4b>(yy, xx);

					p1.r = colorValue.val[2];
					p1.g = colorValue.val[1];
					p1.b = colorValue.val[0];
					worldCoord_cloud->push_back(p1);
					
					

				}
			}


				
			
        }
		
		
		
		//int step = 640*4*sizeof(uchar);
		//colorimg = &IplImage(image);
		//cvGetRawData(colorimg, &data2.planes[0], &step);		
		//cvGetRawData(colorimg_features, &data3.planes[0], &step);		

		//cv::imwrite("aa.bmp", depthImage);
		
		unsigned char *depth_val = (unsigned char*)(depthImage.data);
		

		if (renders[0]) { 
			
			
			//images[2]->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::COLOR_FORMAT_RGB32,&data2); 
			//float *uvmap=(float*)data1.planes[2]; 
			//float *depthmap = (float*)data1.planes[0]; 
			//short *vertidex = (short*)data5.planes[0];
			
			int n = 0;
			for (int y=0;y<(int)pinfo2.imageInfo.height;y++) { 
				for (int x=0;x<(int)pinfo2.imageInfo.width;x++) { 
					
					////Set UV Map in original distorted image.

					//Get original UV map. It requires calibration.
					//int xx=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+0] *pinfo1.imageInfo.width+0.5); 
					//int yy=(int)(uvmap[(y*pinfo2.imageInfo.width+x)*2+1] *pinfo1.imageInfo.height+0.5); 
					//int idx = 3*(y*pinfo2.imageInfo.width+x);

					////Calibration result from calibration tool set
					//int ux = CALIBRATION_UX;
					//int uy = CALIBRATION_UY;
					//float px = CALIBRATION_PX;
					//short depth_value = vertidex[idx+2];//depthImage.at<short>(y, x);
					////printf("%f, %d, %f\n", px, depth_value, px / depth_value);
					//xx = xx + ux + px / depth_value;
					//yy = yy + uy;

					cv::Vec2s loc1 = UVImage.at<cv::Vec2s>(y, x);
					int xx = (int)loc1.val[0];
					int yy = (int)loc1.val[1];
					
					int depth = dstImage.at<unsigned char>(y, x);
					if (xx>=0 && xx<(int)pinfo1.imageInfo.width && yy>=0 && yy<(int)pinfo1.imageInfo.height && depth < 255) 
						((pxcU32 *)data0.planes[0])[yy*pinfo1.imageInfo.width+xx] = 0x80FF0000; 


					////Set UV Map in undistorted image.
					//cv::Vec2s loc = UVImage_undistorted.at<cv::Vec2s>(y, x);
					//int xx_u = (int)loc.val[0];
					//int yy_u = (int)loc.val[1];

					///*cv::Vec2s loc1 = UVImage.at<cv::Vec2s>(y, x);
					//int xx_x = (int)loc1.val[0];
					//int yy_x = (int)loc1.val[1];

					//printf("%d, %d, /// %d, %d\n", xx_u, yy_u, xx_x, yy_x);*/

					//if (xx_u>=0 && xx_u<(int)pinfo1.imageInfo.width && yy_u>=0 && yy_u<(int)pinfo1.imageInfo.height && depth < 255) {
					//	cv::circle(image_undistorted, cv::Point(xx_u, yy_u), 1, cv::Scalar(0, 0, 255));
					//	cv::circle(image_undistorted, cv::Point(xx, yy), 1, cv::Scalar(255, 0, 0));
					//}

					//float depth = vertidex[(y*pinfo2.imageInfo.width+x)];
					//Visualizae 3D
					//printf("%f\t", depth); 
					//int detph_val = dstImage.at<(x, y);
					//pcl::PointXYZRGBA p1;
					////p1.x = x/scene_scale_factor;
					////p1.y = y/scene_scale_factor;
					////p1.z = depth/scene_scale_factor;//dstImage.at<unsigned char>(y, x);

					//
					//p1.x = vertidex[idx]/scene_scale_factor;
					//p1.y = -vertidex[idx+1]/scene_scale_factor;
					//p1.z = vertidex[idx+2]/scene_scale_factor;
					//

					//if(abs(p1.x) > 5 || abs(p1.y) > 5 || p1.z > 5 ) continue;
					//if(idx == 0){
					//	printf("%d, %d, %d\n", vertidex[idx], vertidex[idx+1], vertidex[idx+2]);
					//}
					//
					//if (xx>=0 && xx<(int)pinfo1.imageInfo.width && yy>=0 && yy<(int)pinfo1.imageInfo.height) {
					//	int color = ((pxcU32*)data2.planes[0])[yy*pinfo1.imageInfo.width+xx];
					//	
					//	cv::Vec4b colorValue = image.at<cv::Vec4b>(yy, xx);

					//	p1.r = colorValue.val[2];
					//	p1.g = colorValue.val[1];
					//	p1.b = colorValue.val[0];
					//	cameraCoord_cloud->push_back(p1);	
					//}else{
					//	//p1.r = 255;//((pxcU8*)data2.planes[0])[y*pinfo1.imageInfo.width+x];
					//	//p1.g = 255;//((pxcU8*)data2.planes[0])[y*pinfo1.imageInfo.width+x+1];
					//	//p1.b = 255;//((pxcU8*)data2.planes[0])[y*pinfo1.imageInfo.width+x+2];
					//}

					

					
				
				} 
			} 

			
			

			

			images[0]->ReleaseAccess(&data0); 
			images[1]->ReleaseAccess(&data1); 
			images_modified[0]->ReleaseAccess(&data2);
			images_modified[1]->ReleaseAccess(&data3);
			

			if (!renders[0]->RenderFrame(images[0])) { 
				delete renders[0]; 
				renders[0]=0; 
				break; 
			} 
		} 
		
		if (!renders[1]->RenderFrame(images[1])) { 
			delete renders[1]; 
			renders[1]=0; 
			break; 
		}

		if (!renders[5]->RenderFrame(images[2])) { 
			delete renders[5]; 
			renders[5]=0; 
			break; 
		}

		if (!renders[2]->RenderFrame(images_modified[0])) { 
			delete renders[2]; 
			renders[2]=0; 
			break; 
		}
		
		if (!renders[3]->RenderFrame(images_modified[1])) { 
			delete renders[3]; 
			renders[3]=0; 
			break; 
		}

		if (!renders[4]->RenderFrame(images_modified[2])) { 
			delete renders[4]; 
			renders[4]=0; 
			break; 
		}
		

		
		/*PXCPointU32 curVertexMouseLoc = renders[5]->m_mouse;
		short *vertidex = (short*)data5.planes[0];
		int idx = 3*(curVertexMouseLoc.y*pinfo2.imageInfo.width+curVertexMouseLoc.x);
		printf("<%d, %d> -> <%d, %d, %d>\n", curVertexMouseLoc.x, curVertexMouseLoc.y, vertidex[idx], vertidex[idx+1], vertidex[idx+2]);*/

		
		PXCPointU32 curDepthMouseLoc = renders[1]->m_mouse;
		//long *depthidex = (long*)data4.planes[0];
		//int idx = (curDepthMouseLoc.y*pinfo2.imageInfo.width+curDepthMouseLoc.x);
		//short *vertidex = (short*)data5.planes[0];
		/*
		printf("<%d, %d>\n", curDepthMouseLoc.x, curDepthMouseLoc.y);
		int mx = curDepthMouseLoc.x;
		int my = curDepthMouseLoc.y;
		if(mx > 0 && my > 0){
			int vidx = 3*(curDepthMouseLoc.y*pinfo2.imageInfo.width+curDepthMouseLoc.x);
			printf("<%d, %d>\n", curDepthMouseLoc.x, curDepthMouseLoc.y);
			printf("\t-> <%d>\n", depthImage.at<short>(curDepthMouseLoc.y, curDepthMouseLoc.x));
			printf("\t-> <%d, %d, %d>, D = %f\n", vertidex[vidx], vertidex[vidx+1], vertidex[vidx+2], sqrt((double)vertidex[vidx]*vertidex[vidx]+vertidex[vidx+1]*vertidex[vidx+1]+vertidex[vidx+2]*vertidex[vidx+2]));
		}
		*/
		
		//Add function by clicking the corner of img
		PXCPointU32 curMouseLoc = renders[0]->m_mouse;
		if(prevLoc.x != 0 && prevLoc.y != 0 && (prevLoc.x != curMouseLoc.x || prevLoc.y != curMouseLoc.y)){
			//printf("%d, %d /// %d, %d\n", prevLoc.x, prevLoc.y, curMouseLoc.x, curMouseLoc.y);
			isFunctionWork = true;
		}

		prevLoc = curMouseLoc;

		if(isFunctionWork){
			//printf("%d, %d, [%d, %d, %d]\n", curMouseLoc.x, curMouseLoc.y);
			if(curMouseLoc.x < size_VGA.width * 0.3 && curMouseLoc.y < size_VGA.height *0.3){
				printf("Function work #1 save img!\n");
				images_modified[0]->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::COLOR_FORMAT_RGB32,&data0); 
				IplImage* colorimg = cvCreateImageHeader(cvSize(640, 480), 8, 4);
				cvSetData(colorimg, (uchar*)data0.planes[0], 640*4*sizeof(uchar));
				
				cv::Mat image(colorimg);
				cv::Mat image_rgb;
				cv::cvtColor(image, image_rgb, CV_RGBA2RGB);
				imshow("Current Frame", image_rgb);
				cv::imwrite("test.png", image_rgb);
				cv::waitKey(0);
			}else if(curMouseLoc.x > size_VGA.width * 0.7 && curMouseLoc.y < size_VGA.height *0.3){
				printf("Function work #2! Calibration\n");
				runCalibrationAndSave(s, s.boardSize, cameraMatrix, distCoeffs, points_vec);
			}else if(curMouseLoc.x < size_VGA.width * 0.3 && curMouseLoc.y > size_VGA.height *0.7){
				printf("Function work #3!\n");
				mode_sheet_capture = true;
			}else if(curMouseLoc.x > size_VGA.width * 0.7 && curMouseLoc.y > size_VGA.height *0.7){
				printf("Function work #4!\n");
			}
			isFunctionWork = false;
		}
				

		//printf("%d, %d\n", renders[0]->m_frame, renders[0]->m_pause);

		images.ReleaseRef(0); 
		images.ReleaseRef(1); 


		
		
		
		scene_viewer->addPointCloud (cameraCoord_cloud, "cameraCoord_cloud", vp_1);
		scene_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cameraCoord_cloud");
		scene_viewer->addPointCloud ( worldCoord_cloud, "worldCoord_cloud", vp_2);
		scene_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "worldCoord_cloud");

		scene_viewer->spinOnce();
		scene_viewer->removeText3D("v2 text", vp_2);
		scene_viewer->removePointCloud("cameraCoord_cloud", vp_1);
		scene_viewer->removePointCloud ( "worldCoord_cloud" , vp_2);
		

		
		
		

		

	} 
	// destroy resources 
	if (renders[0]) 
		delete renders[0]; 
	if (renders[1]) 
		delete renders[1]; 
	return 0; 
}

void setInitialFrameLocation(vector<UtilRender*> renders){
	//Set Window frame position
	LPRECT rct_1 = new RECT();
	GetWindowRect(renders[0]->m_hWnd, rct_1);
	//printf("%d, %d, %d, %d\n", rct_1->top, rct_1->bottom, rct_1->left, rct_1->right);
	SetWindowPos(renders[0]->m_hWnd, 0, 0, 0, rct_1->right-rct_1->left, rct_1->bottom-rct_1->top, SWP_NOZORDER);
	GetWindowRect(renders[0]->m_hWnd, rct_1);

	LPRECT rct_2 = new RECT();
	GetWindowRect(renders[1]->m_hWnd, rct_2);
	//printf("%d, %d, %d, %d\n", rct_2->top, rct_2->bottom, rct_2->left, rct_2->right);
	SetWindowPos(renders[1]->m_hWnd, 0, 0, 530, rct_2->right-rct_2->left, rct_2->bottom-rct_2->top, SWP_NOZORDER);
	GetWindowRect(renders[1]->m_hWnd, rct_2);

	LPRECT rct_3 = new RECT();
	GetWindowRect(renders[2]->m_hWnd, rct_3);
	//printf("%d, %d, %d, %d\n", rct_3->top, rct_3->bottom, rct_3->left, rct_3->right);
	SetWindowPos(renders[2]->m_hWnd, 0, 670, 0, rct_3->right-rct_3->left, rct_3->bottom-rct_3->top, SWP_NOZORDER);
	GetWindowRect(renders[2]->m_hWnd, rct_3);

	LPRECT rct_4 = new RECT();
	GetWindowRect(renders[3]->m_hWnd, rct_4);
	//printf("%d, %d, %d, %d\n", rct_4->top, rct_4->bottom, rct_4->left, rct_4->right);
	SetWindowPos(renders[3]->m_hWnd, 0, 670, 530, rct_4->right-rct_4->left, rct_4->bottom-rct_4->top, SWP_NOZORDER);
	GetWindowRect(renders[3]->m_hWnd, rct_4);
}

bool runCalibrationAndSave(Settings& s, cv::Size imageSize, cv::Mat&  cameraMatrix, cv::Mat& distCoeffs,vector<vector<cv::Point2f> > imagePoints )
{
    vector<cv::Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(s,imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                             reprojErrs, totalAvgErr);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
        << ". avg re projection error = "  << totalAvgErr ;

    if( ok ){
		printf("Calibration ok!!\n");
		printf("%f, %f, %f\n", tvecs[0].at<double>(0, 0), tvecs[0].at<double>(1, 0), tvecs[0].at<double>(2, 0));
        saveCameraParams( s, imageSize, cameraMatrix, distCoeffs, rvecs ,tvecs, reprojErrs,
                            imagePoints, totalAvgErr);
	}
    return ok;
}

static bool runCalibration( Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                            vector<vector<cv::Point2f> > imagePoints, vector<cv::Mat>& rvecs, vector<cv::Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr)
{

    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = 1.0;

    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    vector<vector<cv::Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
	
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);
	//printf("totalAveErr: %f\n", totalAvgErr);
    return ok;
}

// Print camera parameters to the output file
static void saveCameraParams( Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                              const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<cv::Point2f> >& imagePoints,
                              double totalAvgErr )
{
    cv::FileStorage fs( s.outputFileName, cv::FileStorage::WRITE );

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_Time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_Width" << imageSize.width;
    fs << "image_Height" << imageSize.height;
    fs << "board_Width" << s.boardSize.width;
    fs << "board_Height" << s.boardSize.height;
    fs << "square_Size" << s.squareSize;

    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "FixAspectRatio" << s.aspectRatio;

    if( s.flag )
    {
        sprintf( buf, "flags: %s%s%s%s",
            s.flag & CV_CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
            s.flag & CV_CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
            s.flag & CV_CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
            s.flag & CV_CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );

    }

    fs << "flagValue" << s.flag;

    fs << "Camera_Matrix" << cameraMatrix;
    fs << "Distortion_Coefficients" << distCoeffs;

    fs << "Avg_Reprojection_Error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "Per_View_Reprojection_Errors" << cv::Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            cv::Mat r = bigmat(cv::Range(i, i+1), cv::Range(0,3));
            cv::Mat t = bigmat(cv::Range(i, i+1), cv::Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "Extrinsic_Parameters" << bigmat;

		cv::Mat roMatrix;
		cv::Rodrigues(cv::Mat(rvecs[0]), roMatrix);
		fs << "Rotation_Matrix" << roMatrix;
    }

	

    if( !imagePoints.empty() )
    {
        cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            cv::Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            cv::Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "Image_points" << imagePtMat;
    }
}

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, vector<cv::Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear();

    switch(patternType)
    {
    case Settings::CHESSBOARD:
    case Settings::CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; ++i )
            for( int j = 0; j < boardSize.width; ++j )
                corners.push_back(cv::Point3f(float( j*squareSize ), float( i*squareSize ), 0));
        break;

    case Settings::ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
        break;
    default:
        break;
    }
}

static double computeReprojectionErrors( const vector<vector<cv::Point3f> >& objectPoints,
                                         const vector<vector<cv::Point2f> >& imagePoints,
                                         const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
                                         const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
		cv::Mat objPoints = cv::Mat(cv::Mat(objectPoints[i]));
		CvMat objPoints2 = objPoints;

		CvMat rvecs2 = rvecs[i];
		CvMat tvecs2 = tvecs[i];
		CvMat camera_matrix2 = cameraMatrix;
		CvMat distCoeffs2 = distCoeffs;
		cv::Mat pointMat = cv::Mat(imagePoints2);
		CvMat* imagePoints22 = cvCreateMat(objectPoints[i].size(),1,CV_32FC2);;
		cvProjectPoints2(&objPoints2, &rvecs2, &tvecs2, &camera_matrix2, &distCoeffs2, imagePoints22);

		cv::Mat a = cv::Mat(imagePoints[i]);
		cv::Mat b = cv::Mat(imagePoints22);
		//printf("size : %d/%d, %d/%d, type: %d/%d....%d\n", a.rows, b.rows, a.cols, b.cols, a.type(), b.type(), objectPoints[i].size());
        //projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
        //               distCoeffs, imagePoints2);
        err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints22), CV_L2);

        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}
