//
// Created by ghm on 2021/10/19.
//

#ifndef LIDAR_ODO_FEATURE_EXTRACTION_H
#define LIDAR_ODO_FEATURE_EXTRACTION_H
#include "param_server.h"
#include "lidar_odo/cloud_info.h"

const int queueLength = 2000;

struct smoothness_t {
	float value;
	size_t ind;
};

struct by_value {
	bool operator()(smoothness_t const &left, smoothness_t const &right) {
		return left.value < right.value;
	}
};

class FeatureExtraction : public ParamServer {
public:
	FeatureExtraction();

	~FeatureExtraction() = default;

private:
	std::mutex imuLock;
	std::mutex odoLock;

	ros::Subscriber subLaserCloud;
	ros::Publisher pubLaserCloud;

	ros::Publisher pubExtractedCloud;
	ros::Publisher pubLaserCloudInfo;

	ros::Subscriber subImu;
	std::deque<sensor_msgs::Imu> imuQueue;

	ros::Subscriber subOdom;
	std::deque<nav_msgs::Odometry> odomQueue;

	std::deque<sensor_msgs::PointCloud2> cloudQueue;
	sensor_msgs::PointCloud2 currentCloudMsg;

	double *imuTime = new double[queueLength];
	double *imuRotX = new double[queueLength];
	double *imuRotY = new double[queueLength];
	double *imuRotZ = new double[queueLength];

	int imuPointerCur;
	bool firstPointFlag;
	Eigen::Affine3f transStartInverse;

	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
	pcl::PointCloud<PointType>::Ptr fullCloud;
	pcl::PointCloud<PointType>::Ptr extractedImgCloud;

	int deskewFlag;
	cv::Mat rangeMat;

	bool odomDeskewFlag;
	float odomIncreX;
	float odomIncreY;
	float odomIncreZ;

	lidar_odo::cloud_info cloudInfo;
	double timeScanCur;
	double timeScanEnd;
	std_msgs::Header cloudHeader;

	// feeature extraction
	ros::Subscriber subLaserCloudInfo;
	ros::Publisher pubCornerPoints;
	ros::Publisher pubSurfacePoints;

	pcl::PointCloud<PointType>::Ptr extractedCloud;
	pcl::PointCloud<PointType>::Ptr cornerCloud;
	pcl::PointCloud<PointType>::Ptr surfaceCloud;

	pcl::VoxelGrid<PointType> downSizeFilter;

	std::vector<smoothness_t> cloudSmoothness;
	float *cloudCurvature;
	int *cloudNeighborPicked;
	int *cloudLabel;

private:
	inline void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
	{
		sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
		std::lock_guard<std::mutex> lock1(imuLock);
		imuQueue.emplace_back(thisImu);
	}

	inline void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
	{
		std::lock_guard<std::mutex> lock2(odoLock);
		odomQueue.emplace_back(*odometryMsg);
	}

	inline void freeCloudInfoMemory()
	{
		cloudInfo.startRingIndex.clear();
		cloudInfo.endRingIndex.clear();
		cloudInfo.pointColInd.clear();
		cloudInfo.pointRange.clear();
	}

	inline void publishFeatureCloud()
	{
		// free cloud info memory
		freeCloudInfoMemory();
		// save newly extracted features
		cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
		cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
		// publish to mapOptimization
		pubLaserCloudInfo.publish(cloudInfo);
	}

	inline void publishClouds()
	{
		cloudInfo.header = cloudHeader;
		cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedImgCloud, cloudHeader.stamp, lidarFrame);
	}

	void allocateMemory();

	void resetParameters();

	void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

	bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

	bool deskewInfo();

	void imuDeskewInfo();

	void odomDeskewInfo();

	void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);

	void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);

	PointType deskewPoint(PointType *point, double pointTime);

	void projectPointCloud();

	void cloudExtraction();

	void initializationValue();

	void laserCloudFeature();

	void calculateSmoothness();

	void markOccludedPoints();

	void extractFeatures();
};


#endif //LIDAR_ODO_FEATURE_EXTRACTION_H
