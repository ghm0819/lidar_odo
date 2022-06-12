//
// Created by ghm on 2021/10/19.
//

#ifndef LIDAR_ODO_SCAN_MATCH_H
#define LIDAR_ODO_SCAN_MATCH_H

#include "lidar_odo/cloud_info.h"
#include "param_server.h"

struct PoseInfo {
	double x;
	double y;
	double z;
	double roll;
	double pitch;
	double yaw;
	double time;

	PoseInfo() : x(0.0), y(0.0), z(0.0), roll(0.0), pitch(0.0), yaw(0.0), time(0.0) {};

	PoseInfo(const double xIn, const double yIn, const double zIn, const double rollIn, const double pitchIn,
		const double yawIn, const double timeIn) :
		x(xIn), y(yIn), z(zIn), roll(rollIn), pitch(pitchIn), yaw(yawIn), time(timeIn) {};
};

class ScanMatch : public ParamServer {
public:
	ScanMatch();

	~ScanMatch() = default;

	inline void visualizeGlobalMapThread()
	{
		ros::Rate rate(0.2);
		while (ros::ok()) {
			rate.sleep();
			publishGlobalMap();
		}
	}
private:
	ros::Publisher pubLaserCloudSurround;
	ros::Publisher pubLaserOdometryGlobal;
	ros::Publisher pubLaserOdometryIncremental;
	ros::Publisher pubPath;

	ros::Subscriber subCloud;
	lidar_odo::cloud_info cloudInfo;

	pcl::PointCloud<PointType>::Ptr cloudKeyPose_;
	std::deque<std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer_;
	std::deque<PoseInfo> keyPoseInfo_;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
	pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;   // surf feature set from odoOptimization
	pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
	pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

	pcl::PointCloud<PointType>::Ptr laserCloudOri;
	pcl::PointCloud<PointType>::Ptr coeffSel;

	std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
	std::vector<PointType> coeffSelCornerVec;
	std::vector<bool> laserCloudOriCornerFlag;
	std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
	std::vector<PointType> coeffSelSurfVec;
	std::vector<bool> laserCloudOriSurfFlag;

	std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

	pcl::VoxelGrid<PointType> downSizeFilterCorner;
	pcl::VoxelGrid<PointType> downSizeFilterSurf;
	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map

	ros::Time timeLaserInfoStamp;
	double timeLaserInfoCur;

	float transformTobeMapped[6];

	std::mutex mtx;

	bool isDegenerate = false;
	Eigen::Matrix<float, 6, 6> matP;
	int laserCloudCornerLastDSNum = 0;
	int laserCloudSurfLastDSNum = 0;

	nav_msgs::Path globalPath;

	Eigen::Affine3f transPointAssociateToMap;
	Eigen::Affine3f incrementalOdometryAffineFront;
	Eigen::Affine3f incrementalOdometryAffineBack;

	Eigen::Isometry3d imu2LidarMatrix = Eigen::Isometry3d::Identity();

private:
	void allocateMemory();

	void laserCloudInfoHandler(const lidar_odo::cloud_infoConstPtr &cloud_info_msg);

	void pointAssociateToMap(PointType const *const pi, PointType *const po);

	pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
		const PoseInfo& currPose);

	inline Eigen::Affine3f pclPointToAffine3f(const PoseInfo& currPose)
	{
		return pcl::getTransformation(currPose.x, currPose.y, currPose.z, currPose.roll, currPose.pitch,
			currPose.yaw);
	}

	inline Eigen::Affine3f trans2Affine3f(float transformIn[])
	{
		return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1],
			transformIn[2]);
	}

	void publishGlobalMap();

	void updateInitialGuess();

	void extractNeighbourhoodKeyFrames();

	void downsampleCurrentScan();

	void updatePointAssociateToMap();

	void cornerOptimization();

	void surfOptimization();

	void combineOptimizationCoeffs();

	bool LMOptimization(int iterCount);

	void scan2MapOptimization();

	void transformUpdate();

	float constraintTransformation(float value, float limit);

	bool saveFrame();

	void saveKeyFramesAndFactor();

	void updatePath(const PoseInfo& poseIn);

	void publishOdometry();

	void publishFrames();
};
#endif //LIDAR_ODO_SCAN_MATCH_H
