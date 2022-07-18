//
// Created by ghm on 2021/10/19.
//

#include "feature_extraction.h"

FeatureExtraction::FeatureExtraction() : deskewFlag(0)
{
	subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &FeatureExtraction::imuHandler,
		this, ros::TransportHints().tcpNoDelay());
	subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &FeatureExtraction::odometryHandler,
		this, ros::TransportHints().tcpNoDelay());
	subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &FeatureExtraction::cloudHandler,
		this, ros::TransportHints().tcpNoDelay());

	pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lidar_odo/deskew/cloud_deskewed", 1);
	pubLaserCloudInfo = nh.advertise<lidar_odo::cloud_info> ("lidar_odo/feature/cloud_info", 1);
	pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lidar_odo/feature/cloud_corner", 1);
	pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lidar_odo/feature/cloud_surface", 1);

	allocateMemory();
	resetParameters();

	pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

void FeatureExtraction::allocateMemory()
{
	laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
	fullCloud.reset(new pcl::PointCloud<PointType>());
	extractedImgCloud.reset(new pcl::PointCloud<PointType>());

	fullCloud->points.resize(N_SCAN * Horizon_SCAN);

	cloudInfo.startRingIndex.assign(N_SCAN, 0);
	cloudInfo.endRingIndex.assign(N_SCAN, 0);

	cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
	cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

	resetParameters();
	initializationValue();
}

void FeatureExtraction::resetParameters()
{
	laserCloudIn->clear();
	extractedImgCloud->clear();
	// reset range matrix for range image projection
	rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

	imuPointerCur = 0;
	firstPointFlag = true;
	odomDeskewFlag = false;

	for (int i = 0; i < queueLength; ++i)
	{
		imuTime[i] = 0;
		imuRotX[i] = 0;
		imuRotY[i] = 0;
		imuRotZ[i] = 0;
	}
}

void FeatureExtraction::cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
	if (!cachePointCloud(laserCloudMsg)) {
		return;
	}

	if (!deskewInfo()) {
		return;
	}

	projectPointCloud();

	cloudExtraction();

	publishClouds();

	laserCloudFeature();

	resetParameters();
}

bool FeatureExtraction::cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	// cache point cloud
	cloudQueue.emplace_back(*laserCloudMsg);
	if (cloudQueue.size() <= 2) {
		return false;
	}

	// convert cloud
	currentCloudMsg = cloudQueue.front();
	cloudQueue.pop_front();
	/////////// filter the self cloud
	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudInTemp;
    laserCloudInTemp.reset(new pcl::PointCloud<PointXYZIRT>());
	pcl::fromROSMsg(currentCloudMsg, *laserCloudInTemp);
    std::vector<int> mapping_indices;
    pcl::removeNaNFromPointCloud(*laserCloudInTemp, *laserCloudInTemp, mapping_indices);

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudInTempX;
    laserCloudInTempX.reset(new pcl::PointCloud<PointXYZIRT>());
	pcl::PassThrough<PointXYZIRT> pass;
	pass.setInputCloud(laserCloudInTemp);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(x_range_min, x_range_max);
    pass.setFilterLimitsNegative(true);
    pass.filter(*laserCloudInTempX);

    pass.setInputCloud(laserCloudInTempX);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_range_min, y_range_max);
    pass.setFilterLimitsNegative(true);
    pass.filter(*laserCloudIn);
    /////////////////////////

	// get timestamp
	cloudHeader = currentCloudMsg.header;
	timeScanCur = cloudHeader.stamp.toSec(); // the start time of this point cloud
	timeScanEnd = laserCloudIn->points.back().timestamp; // the end time of this point cloud
	if (std::abs(timeScanEnd - timeScanCur) < 1e-3) {
        timeScanEnd = timeScanCur + 1e-3;
	}
	// check dense flag
	if (!laserCloudIn->is_dense) {
		ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
		ros::shutdown();
	}

	// check ring channel
	static int ringFlag = 0;
	if (ringFlag == 0) {
		ringFlag = -1;
		for (auto & field : currentCloudMsg.fields) {
			if (field.name == "ring") {
				ringFlag = 1;
				break;
			}
		}
		if (ringFlag == -1) {
			ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
			ros::shutdown();
		}
	}

	// check point time
	if (deskewFlag == 0) {
		deskewFlag = -1;
		for (auto & field : currentCloudMsg.fields) {
			if (field.name == timeField) {
				deskewFlag = 1;
				break;
			}
		}
		if (deskewFlag == -1) {
			ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
		}
	}

	return true;
}

bool FeatureExtraction::deskewInfo()
{
	std::lock_guard<std::mutex> lock1(imuLock);
	std::lock_guard<std::mutex> lock2(odoLock);

	// make sure IMU data available for the scan
	if (imuQueue.empty() || (imuQueue.front().header.stamp.toSec() > timeScanCur) ||
	    (imuQueue.back().header.stamp.toSec() < timeScanEnd)) {
		ROS_DEBUG("Waiting for IMU data ...");
		return false;
	}

	imuDeskewInfo();

	odomDeskewInfo();

	return true;
}

void FeatureExtraction::imuDeskewInfo()
{
	cloudInfo.imuAvailable = false;

	while (!imuQueue.empty()) {
		if (imuQueue.front().header.stamp.toSec() < (timeScanCur - 0.01)) {
			imuQueue.pop_front();
		} else {
			break;
		}
	}

	if (imuQueue.empty()) {
		return;
	}

	imuPointerCur = 0;
	for (auto thisImuMsg : imuQueue) {
		double currentImuTime = thisImuMsg.header.stamp.toSec();
		if (currentImuTime > (timeScanEnd + 0.01)) {
			break;
		}

		if (imuPointerCur == 0) {
			imuRotX[0] = 0;
			imuRotY[0] = 0;
			imuRotZ[0] = 0;
			imuTime[0] = currentImuTime;
			++imuPointerCur;
			continue;
		}

		// get angular velocity
		double angular_x, angular_y, angular_z;
		imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

		// integrate rotation
		double timeDiff = currentImuTime - imuTime[imuPointerCur - 1U];
		imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
		imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
		imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
		imuTime[imuPointerCur] = currentImuTime;
		++imuPointerCur;
	}

	--imuPointerCur;
	if (imuPointerCur <= 0) {
		return;
	}
	cloudInfo.imuAvailable = true;
}

void FeatureExtraction::odomDeskewInfo()
{
	cloudInfo.odomAvailable = false;

	while (!odomQueue.empty()) {
		if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01) {
			odomQueue.pop_front();
		} else {
			break;
		}
	}

	if (odomQueue.empty()) {
		return;
	}

	if (odomQueue.front().header.stamp.toSec() > timeScanCur) {
		return;
	}

	// get start odometry at the beinning of the scan
	nav_msgs::Odometry startOdomMsg;

	for (auto& odom : odomQueue) {
		startOdomMsg = odom;
		if (ROS_TIME(&startOdomMsg) < timeScanCur) {
			continue;
		} else {
			break;
		}
	}

	tf::Quaternion orientation;
	tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

	double roll, pitch, yaw;
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

	// Initial guess used in mapOptimization
	cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
	cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
	cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
	cloudInfo.initialGuessRoll = roll;
	cloudInfo.initialGuessPitch = pitch;
	cloudInfo.initialGuessYaw = yaw;

	cloudInfo.odomAvailable = true;

	// get end odometry at the end of the scan
	odomDeskewFlag = false;

	if (odomQueue.back().header.stamp.toSec() < timeScanEnd) {
		return;
	}

	nav_msgs::Odometry endOdomMsg;

	for (auto& odom : odomQueue)
	{
		endOdomMsg = odom;
		if (ROS_TIME(&endOdomMsg) < timeScanEnd) {
			continue;
		} else {
			break;
		}
	}

	if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0]))) {
		return;
	}

	Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x,
		startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

	tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
	Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x,
		endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

	Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
	float rollIncre, pitchIncre, yawIncre;
	pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);
	odomDeskewFlag = true;
}

void FeatureExtraction::findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
{
	*rotXCur = 0;
	*rotYCur = 0;
	*rotZCur = 0;

	int imuPointerFront = 0;
	while (imuPointerFront < imuPointerCur) {
		if (pointTime < imuTime[imuPointerFront]) {
			break;
		}
		++imuPointerFront;
	}

	if ((pointTime > imuTime[imuPointerFront]) || (imuPointerFront == 0)) {
		*rotXCur = imuRotX[imuPointerFront];
		*rotYCur = imuRotY[imuPointerFront];
		*rotZCur = imuRotZ[imuPointerFront];
	} else {
		int imuPointerBack = imuPointerFront - 1;
		double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		*rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
		*rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
		*rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
	}
}

void FeatureExtraction::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
{
	*posXCur = 0;
	*posYCur = 0;
	*posZCur = 0;

	// If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

	if ((!cloudInfo.odomAvailable) || (!odomDeskewFlag)) {
		return;
	}

	float ratio = relTime / (timeScanEnd - timeScanCur);

	*posXCur = ratio * odomIncreX;
	*posYCur = ratio * odomIncreY;
	*posZCur = ratio * odomIncreZ;
}

PointType FeatureExtraction::deskewPoint(PointType *point, double pointTime)
{
	if ((deskewFlag == -1) || (cloudInfo.imuAvailable == false)) {
		return *point;
	}
	float rotXCur, rotYCur, rotZCur;
	findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

	float posXCur, posYCur, posZCur;
	findPosition(pointTime, &posXCur, &posYCur, &posZCur);

	if (firstPointFlag) {
		transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
		firstPointFlag = false;
	}

	// transform points to start
	Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
	Eigen::Affine3f transBt = transStartInverse * transFinal;

	PointType newPoint;
	newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
	newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
	newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
	newPoint.intensity = point->intensity;

	return newPoint;
}

void FeatureExtraction::projectPointCloud()
{
	auto cloudSize = laserCloudIn->points.size();
	// range image projection
	for (size_t i = 0U; i < cloudSize; ++i) {
		PointType thisPoint;
		thisPoint.x = laserCloudIn->points[i].x;
		thisPoint.y = laserCloudIn->points[i].y;
		thisPoint.z = laserCloudIn->points[i].z;
		thisPoint.intensity = laserCloudIn->points[i].intensity;

		float range = pointDistance(thisPoint);
		if ((range < lidarMinRange) || (range > lidarMaxRange)) {
			continue;
		}

		int rowIdn = laserCloudIn->points[i].ring;
		if ((rowIdn < 0) || (rowIdn >= N_SCAN)) {
			continue;
		}

		if ((rowIdn % downsampleRate) != 0) {
			continue;
		}

		auto horizonAngle = std::atan2(thisPoint.y, thisPoint.z) * 180 / M_PI;

		static float ang_res_x = 360.0 / float(Horizon_SCAN);
		int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
		if (columnIdn >= Horizon_SCAN) {
			columnIdn -= Horizon_SCAN;
		}

		if (columnIdn < 0 || columnIdn >= Horizon_SCAN) {
			continue;
		}
		if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) {
			continue;
		}

		thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].timestamp); // rslidar
		// thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // Velodyne
		// thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0); // Ouster

		rangeMat.at<float>(rowIdn, columnIdn) = range;

		int index = columnIdn + rowIdn * Horizon_SCAN;
		fullCloud->points[index] = thisPoint;
	}
}

void FeatureExtraction::cloudExtraction()
{
	int count = 0;
	// extract segmented cloud for lidar odometry
	for (int i = 0; i < N_SCAN; ++i) {
		cloudInfo.startRingIndex[i] = count - 1 + 5;
		for (int j = 0; j < Horizon_SCAN; ++j) {
			if (rangeMat.at<float>(i, j) != FLT_MAX) {
				// mark the points' column index for marking occlusion later
				cloudInfo.pointColInd[count] = j;
				// save range info
				cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
				// save extracted cloud
				extractedImgCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
				// size of extracted cloud
				++count;
			}
		}
		cloudInfo.endRingIndex[i] = count - 1 - 5;
	}
}

// feature extraction
void FeatureExtraction::initializationValue()
{
	cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

	downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

	extractedCloud.reset(new pcl::PointCloud<PointType>());
	cornerCloud.reset(new pcl::PointCloud<PointType>());
	surfaceCloud.reset(new pcl::PointCloud<PointType>());

	cloudCurvature = new float[N_SCAN*Horizon_SCAN];
	cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
	cloudLabel = new int[N_SCAN*Horizon_SCAN];
}

void FeatureExtraction::laserCloudFeature()
{
	cloudInfo = cloudInfo; // new cloud info
	cloudHeader = cloudInfo.header; // new cloud header
	pcl::fromROSMsg(cloudInfo.cloud_deskewed, *extractedCloud); // new cloud for extraction

	calculateSmoothness();

	markOccludedPoints();

	extractFeatures();

	publishFeatureCloud();
}

void FeatureExtraction::calculateSmoothness()
{
	int cloudSize = extractedImgCloud->points.size();
	for (size_t i = 5; i < cloudSize - 5; i++) {
		float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
		                  + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
		                  + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
		                  + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
		                  + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
		                  + cloudInfo.pointRange[i+5];

		cloudCurvature[i] = diffRange*diffRange; //diffX * diffX + diffY * diffY + diffZ * diffZ;

		cloudNeighborPicked[i] = 0;
		cloudLabel[i] = 0;
		// cloudSmoothness for sorting
		cloudSmoothness[i].value = cloudCurvature[i];
		cloudSmoothness[i].ind = i;
	}
}

void FeatureExtraction::markOccludedPoints()
{
	int cloudSize = extractedImgCloud->points.size();
	// mark occluded points and parallel beam points
	for (int i = 5; i < cloudSize - 6; ++i) {
		// occluded points
		float depth1 = cloudInfo.pointRange[i];
		float depth2 = cloudInfo.pointRange[i+1];
		int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

		if (columnDiff < 10) {
			// 10 pixel diff in range image
			if ((depth1 - depth2) > 0.3) {
				cloudNeighborPicked[i - 5] = 1;
				cloudNeighborPicked[i - 4] = 1;
				cloudNeighborPicked[i - 3] = 1;
				cloudNeighborPicked[i - 2] = 1;
				cloudNeighborPicked[i - 1] = 1;
				cloudNeighborPicked[i] = 1;
			} else if ((depth2 - depth1) > 0.3) {
				cloudNeighborPicked[i + 1] = 1;
				cloudNeighborPicked[i + 2] = 1;
				cloudNeighborPicked[i + 3] = 1;
				cloudNeighborPicked[i + 4] = 1;
				cloudNeighborPicked[i + 5] = 1;
				cloudNeighborPicked[i + 6] = 1;
			} else {
				// do nothing
			}
		}
		// parallel beam
		float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
		float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

		if (diff1 > (0.02 * cloudInfo.pointRange[i]) && diff2 > (0.02 * cloudInfo.pointRange[i])) {
			cloudNeighborPicked[i] = 1;
		}
	}
}

void FeatureExtraction::extractFeatures()
{
	cornerCloud->clear();
	surfaceCloud->clear();

	pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

	for (int i = 0; i < N_SCAN; ++i) {
		surfaceCloudScan->clear();

		for (int j = 0; j < 6; ++j) {

			int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
			int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

			if (sp >= ep) {
				continue;
			}

			std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

			int largestPickedNum = 0;
			for (int k = ep; k >= sp; --k) {
				auto ind = cloudSmoothness[k].ind;
				if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) {
					largestPickedNum++;
					if (largestPickedNum <= 20){
						cloudLabel[ind] = 1;
						cornerCloud->push_back(extractedImgCloud->points[ind]);
					} else {
						break;
					}

					cloudNeighborPicked[ind] = 1;
					for (int l = 1; l <= 5; ++l) {
						int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
						if (columnDiff > 10) {
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
					for (int l = -1; l >= -5; --l) {
						int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
						if (columnDiff > 10) {
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			for (int k = sp; k <= ep; ++k) {
				auto ind = cloudSmoothness[k].ind;
				if ((cloudNeighborPicked[ind] == 0) && (cloudCurvature[ind] < surfThreshold)) {

					cloudLabel[ind] = -1;
					cloudNeighborPicked[ind] = 1;

					for (int l = 1; l <= 5; ++l) {
						int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
						if (columnDiff > 10) {
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
					for (int l = -1; l >= -5; --l) {
						int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
						if (columnDiff > 10) {
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			for (int k = sp; k <= ep; ++k) {
				if (cloudLabel[k] <= 0) {
					surfaceCloudScan->push_back(extractedImgCloud->points[k]);
				}
			}
		}

		surfaceCloudScanDS->clear();
		downSizeFilter.setInputCloud(surfaceCloudScan);
		downSizeFilter.filter(*surfaceCloudScanDS);

		*surfaceCloud += *surfaceCloudScanDS;
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "lidar_odo");

	FeatureExtraction FE;

	ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

	ros::MultiThreadedSpinner spinner(3);
	spinner.spin();

	return 0;
}