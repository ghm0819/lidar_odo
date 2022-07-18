//
// Created by ghm on 2021/10/18.
//

#ifndef LIDAR_ODO_PARAM_SERVER_H
#define LIDAR_ODO_PARAM_SERVER_H
#include "utility.h"

class ParamServer {
public:
	ros::NodeHandle nh;
	// Topics
	std::string pointCloudTopic;
	std::string imuTopic;
	std::string odomTopic;

	// Frames
	std::string lidarFrame;
	std::string baselinkFrame;
	std::string odometryFrame;
	std::string mapFrame;

	// Sensor Configuration
	int N_SCAN;
	int Horizon_SCAN;
	std::string timeField;
	int downsampleRate;
	float lidarMinRange;
	float lidarMaxRange;
	float x_range_min;
	float x_range_max;
	float y_range_min;
	float y_range_max;

	// IMU
	float imuAccNoise;
	float imuGyrNoise;
	float imuAccBiasN;
	float imuGyrBiasN;
	float imuGravity;
	float imuRPYWeight;

	std::vector<double> extTransV;
	std::vector<double> extRotV;
	Eigen::Matrix3d extRotLidar2Veh;
	Eigen::Vector3d extTransLidar2Veh;
	Eigen::Matrix3d extRotVeh2Lidar;
	Eigen::Vector3d extTransVeh2Lidar;
	Eigen::Quaterniond extQRPY;

	// LOAM
	float edgeThreshold;
	float surfThreshold;
	int edgeFeatureMinValidNum;
	int surfFeatureMinValidNum;

	// voxel filter paprams
	float odometrySurfLeafSize;
	float mappingCornerLeafSize;
	float mappingSurfLeafSize;

	float z_tollerance;
	float rotation_tollerance;

	// CPU Params
	int numberOfCores;
	double mappingProcessInterval;

	// Surrounding map
	float surroundingkeyframeAddingDistThreshold;
	float surroundingkeyframeAddingAngleThreshold;
	float surroundingKeyframeDensity;
	float surroundingKeyframeSearchRadius;

	// global map visualization radius
	float globalMapVisualizationSearchRadius;
	float globalMapVisualizationPoseDensity;
	float globalMapVisualizationLeafSize;

	ParamServer() {
		nh.param<std::string>("lidar_odo/pointCloudTopic", pointCloudTopic, "points_raw");
		nh.param<std::string>("lidar_odo/imuTopic", imuTopic, "imu_correct");
		nh.param<std::string>("lidar_odo/odomTopic", odomTopic, "odometry/imu");

		nh.param<std::string>("lidar_odo/lidarFrame", lidarFrame, "lidar_link");
		nh.param<std::string>("lidar_odo/baselinkFrame", baselinkFrame, "base_link");
		nh.param<std::string>("lidar_odo/odometryFrame", odometryFrame, "odom");
		nh.param<std::string>("lidar_odo/mapFrame", mapFrame, "map");

		nh.param<int>("lidar_odo/N_SCAN", N_SCAN, 40);
		nh.param<int>("lidar_odo/Horizon_SCAN", Horizon_SCAN, 1800);
		nh.param<std::string>("lidar_odo/timeField", timeField, "time");
		nh.param<int>("lidar_odo/downsampleRate", downsampleRate, 1);
		nh.param<float>("lidar_odo/lidarMinRange", lidarMinRange, 1.0);
		nh.param<float>("lidar_odo/lidarMaxRange", lidarMaxRange, 100.0);
        nh.param<float>("lidar_odo/x_range_min", x_range_min, -4.0);
        nh.param<float>("lidar_odo/x_range_max", x_range_max, 0.5);
        nh.param<float>("lidar_odo/y_range_min", y_range_min, -1.2);
        nh.param<float>("lidar_odo/y_range_max", y_range_max, 1.2);

		nh.param<float>("lidar_odo/imuAccNoise", imuAccNoise, 0.01);
		nh.param<float>("lidar_odo/imuGyrNoise", imuGyrNoise, 0.001);
		nh.param<float>("lidar_odo/imuAccBiasN", imuAccBiasN, 0.0002);
		nh.param<float>("lidar_odo/imuGyrBiasN", imuGyrBiasN, 0.00003);
		nh.param<float>("lidar_odo/imuGravity", imuGravity, 9.80511);
		nh.param<float>("lidar_odo/imuRPYWeight", imuRPYWeight, 0.01);
		nh.param <std::vector <double>> ("lidar_odo/extrinsicTrans", extTransV, std::vector<double>());
		nh.param <std::vector <double>> ("lidar_odo/extrinsicRot", extRotV, std::vector<double>());
		extTransVeh2Lidar = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
		extRotVeh2Lidar = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
		extRotLidar2Veh = extRotVeh2Lidar.inverse();
		extTransLidar2Veh = -extRotVeh2Lidar.inverse() * extTransVeh2Lidar;
		extQRPY = Eigen::Quaterniond(extRotVeh2Lidar);

		nh.param<float>("lidar_odo/edgeThreshold", edgeThreshold, 0.1);
		nh.param<float>("lidar_odo/surfThreshold", surfThreshold, 0.1);
		nh.param<int>("lidar_odo/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
		nh.param<int>("lidar_odo/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

		nh.param<float>("lidar_odo/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
		nh.param<float>("lidar_odo/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
		nh.param<float>("lidar_odo/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

		nh.param<float>("lidar_odo/z_tollerance", z_tollerance, FLT_MAX);
		nh.param<float>("lidar_odo/rotation_tollerance", rotation_tollerance, FLT_MAX);

		nh.param<int>("lidar_odo/numberOfCores", numberOfCores, 2);
		nh.param<double>("lidar_odo/mappingProcessInterval", mappingProcessInterval, 0.15);

		nh.param<float>("lidar_odo/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
		nh.param<float>("lidar_odo/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
		nh.param<float>("lidar_odo/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
		nh.param<float>("lidar_odo/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

		nh.param<float>("lidar_odo/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
		nh.param<float>("lidar_odo/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
		nh.param<float>("lidar_odo/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
		usleep(100);
	}

	sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
	{
		sensor_msgs::Imu imu_out = imu_in;
		// rotate acceleration
		Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
		acc = extRotLidar2Veh * acc;
		imu_out.linear_acceleration.x = acc.x();
		imu_out.linear_acceleration.y = acc.y();
		imu_out.linear_acceleration.z = acc.z();

		// rotate gyroscope
		Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
		gyr = extRotLidar2Veh * gyr;
		imu_out.angular_velocity.x = gyr.x();
		imu_out.angular_velocity.y = gyr.y();
		imu_out.angular_velocity.z = gyr.z();

		return imu_out;
	}
};
#endif //LIDAR_ODO_PARAM_SERVER_H
