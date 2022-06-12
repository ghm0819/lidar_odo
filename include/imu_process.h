//
// Created by ghm on 2021/10/19.
//

#ifndef LIDAR_ODO_IMU_PROCESS_H
#define LIDAR_ODO_IMU_PROCESS_H
#include "param_server.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

// for viewer
class TransformFusion : public ParamServer {
public:
	TransformFusion()
	{
		subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lidar_odo/mapping/odometry", 5,
			&TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
		subImuOdometry = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",
			2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

		pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
		pubImuPath = nh.advertise<nav_msgs::Path>("lidar_odo/imu/path", 1);
	}

private:
	std::mutex mtx;

	ros::Subscriber subImuOdometry;
	ros::Subscriber subLaserOdometry;

	ros::Publisher pubImuOdometry;
	ros::Publisher pubImuPath;

	Eigen::Affine3f lidarOdomAffine;

	double lidarOdomTime = -1;
	std::deque<nav_msgs::Odometry> imuOdomQueue;

	inline Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
	{
		double x, y, z, roll, pitch, yaw;
		x = odom.pose.pose.position.x;
		y = odom.pose.pose.position.y;
		z = odom.pose.pose.position.z;
		tf::Quaternion orientation;
		tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
		return pcl::getTransformation(x, y, z, roll, pitch, yaw);
	}

	inline void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
	{
		std::lock_guard<std::mutex> lock(mtx);

		lidarOdomAffine = odom2affine(*odomMsg);

		lidarOdomTime = odomMsg->header.stamp.toSec();
	}

	void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg);
};

class ImuProcess : public ParamServer {
public:
	ImuProcess();

	~ImuProcess() = default;

private:
	std::mutex mtx;

	ros::Subscriber subImu;
	ros::Subscriber subOdometry;
	ros::Publisher pubImuOdometry;

	bool systemInitialized = false;

	gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
	gtsam::Vector noiseModelBetweenBias;


	gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
	gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

	std::deque<sensor_msgs::Imu> imuQueOpt;
	std::deque<sensor_msgs::Imu> imuQueImu;

	gtsam::Pose3 prevPose_;
	gtsam::Vector3 prevVel_;
	gtsam::NavState prevState_;
	gtsam::imuBias::ConstantBias prevBias_;

	gtsam::NavState prevStateOdom;
	gtsam::imuBias::ConstantBias prevBiasOdom;

	bool doneFirstOpt = false;
	double lastImuT_imu = -1;
	double lastImuT_opt = -1;

	gtsam::ISAM2 optimizer;
	gtsam::NonlinearFactorGraph graphFactors;
	gtsam::Values graphValues;

	const double delta_t = 0;

	int key = 1;

	gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTransLidar2Veh.x(),
		extTransLidar2Veh.y(), extTransLidar2Veh.z()));
	gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTransLidar2Veh.x(),
		extTransLidar2Veh.y(), extTransLidar2Veh.z()));
private:
	inline void resetOptimization()
	{
		gtsam::ISAM2Params optParameters;
		optParameters.relinearizeThreshold = 0.1;
		optParameters.relinearizeSkip = 1;
		optimizer = gtsam::ISAM2(optParameters);

		gtsam::NonlinearFactorGraph newGraphFactors;
		graphFactors = newGraphFactors;

		gtsam::Values NewGraphValues;
		graphValues = NewGraphValues;
	}

	inline void resetParams()
	{
		lastImuT_imu = -1;
		doneFirstOpt = false;
		systemInitialized = false;
	}

	void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg);

	bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur);

	void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw);
};


#endif //LIDAR_ODO_IMU_PROCESS_H
