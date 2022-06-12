//
// Created by ghm on 2021/10/19.
//

#include "scan_match.h"

ScanMatch::ScanMatch() {
	imu2LidarMatrix.rotate(extQRPY);
	imu2LidarMatrix.pretranslate(extTransVeh2Lidar);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("lidar_odo/mapping/map_global", 1);
	pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lidar_odo/mapping/odometry", 1);
	pubPath = nh.advertise<nav_msgs::Path>("lidar_odo/mapping/path", 1);

	subCloud = nh.subscribe<lidar_odo::cloud_info>("lidar_odo/feature/cloud_info", 1,
		&ScanMatch::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

	downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize,
		mappingCornerLeafSize);
	downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
	downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity,
		surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

	allocateMemory();
}

void ScanMatch::allocateMemory()
{
	cloudKeyPose_.reset(new pcl::PointCloud<PointType>());
	laserCloudCornerLast.reset(
			new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
	laserCloudSurfLast.reset(
			new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
	laserCloudCornerLastDS.reset(
			new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
	laserCloudSurfLastDS.reset(
			new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

	laserCloudOri.reset(new pcl::PointCloud<PointType>());
	coeffSel.reset(new pcl::PointCloud<PointType>());

	laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
	coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
	laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
	laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
	coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
	laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

	std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
	std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

	laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
	laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
	laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
	laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

	kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
	kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

	for (float& i : transformTobeMapped) {
		i = 0;
	}
	matP.setZero();
}

void ScanMatch::laserCloudInfoHandler(const lidar_odo::cloud_infoConstPtr& cloud_info_msg)
{
	// extract time stamp
	timeLaserInfoStamp = cloud_info_msg->header.stamp;
	timeLaserInfoCur = cloud_info_msg->header.stamp.toSec();

	// extract info and feature cloud
	cloudInfo = *cloud_info_msg;
	pcl::fromROSMsg(cloud_info_msg->cloud_corner, *laserCloudCornerLast);
	pcl::fromROSMsg(cloud_info_msg->cloud_surface, *laserCloudSurfLast);

	if (keyPoseInfo_.empty()) {
		Eigen::Isometry3d initialVelMatrix = Eigen::Isometry3d::Identity();
		auto initialLidarMatrix = initialVelMatrix * imu2LidarMatrix;
		transformTobeMapped[3] = initialLidarMatrix.translation().x();
		transformTobeMapped[4] = initialLidarMatrix.translation().y();
		transformTobeMapped[5] = initialLidarMatrix.translation().z();
		auto eulerAngle = initialLidarMatrix.rotation().eulerAngles(2, 1, 0);
		transformTobeMapped[0] = eulerAngle[2];
		transformTobeMapped[1] = eulerAngle[1];
		transformTobeMapped[2] = eulerAngle[0];
	}

	std::lock_guard<std::mutex> lock(mtx);

	static double timeLastProcessing = -1;
	if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {
		timeLastProcessing = timeLaserInfoCur;

		updateInitialGuess();

		extractNeighbourhoodKeyFrames();

		downsampleCurrentScan();

		scan2MapOptimization();

		saveKeyFramesAndFactor();

		publishOdometry();

		publishFrames();
	}
}

void ScanMatch::pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y +
		transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
	po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y +
		transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
	po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y +
		transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
	po->intensity = pi->intensity;
}

pcl::PointCloud<PointType>::Ptr ScanMatch::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
	const PoseInfo& currPose)
{
	pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

	PointType *pointFrom;

	int cloudSize = cloudIn->size();
	cloudOut->resize(cloudSize);

	Eigen::Affine3f transCur = pcl::getTransformation(currPose.x, currPose.y, currPose.z,
		currPose.roll, currPose.pitch, currPose.yaw);

#pragma omp parallel for num_threads(numberOfCores)
	for (int i = 0; i < cloudSize; ++i) {
		pointFrom = &cloudIn->points[i];
		cloudOut->points[i].x = transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y +
			transCur(0, 2) * pointFrom->z + transCur(0, 3);
		cloudOut->points[i].y = transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y +
			transCur(1, 2) * pointFrom->z + transCur(1, 3);
		cloudOut->points[i].z = transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y +
			transCur(2, 2) * pointFrom->z + transCur(2, 3);
		cloudOut->points[i].intensity = pointFrom->intensity;
	}
	return cloudOut;
}

void ScanMatch::publishGlobalMap()
{
	if (keyPoseInfo_.empty()) {
		return;
	}
	pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

	// extract visualized and downsampled key frames
	for (const auto& keyFrame : laserCloudMapContainer_) {
		*globalMapKeyFrames += keyFrame.first;
		*globalMapKeyFrames += keyFrame.second;
	}
	// downsample visualized points
	pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
	downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
		globalMapVisualizationLeafSize); // for global map visualization
	downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
	downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
	publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}

void ScanMatch::updateInitialGuess()
{
	incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

	static Eigen::Affine3f lastImuTransformation;
	// initialization
	if (keyPoseInfo_.empty()) {
		return;
	}

	// use imu pre-integration estimation for pose guess
	static bool lastImuPreTransAvailable = false;
	static Eigen::Affine3f lastImuPreTransformation;
	if (cloudInfo.odomAvailable) {
		Eigen::Affine3f transBack = pcl::getTransformation(
				cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
				cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
		if (!lastImuPreTransAvailable) {
			lastImuPreTransformation = transBack;
			lastImuPreTransAvailable = true;
		} else {
			Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
			Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
			Eigen::Affine3f transFinal = transTobe * transIncre;
			pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4],
				transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

			lastImuPreTransformation = transBack;
			return;
		}
	}
}

void ScanMatch::extractNeighbourhoodKeyFrames()
{
	if (keyPoseInfo_.empty()) {
		return;
	}
	laserCloudCornerFromMap->clear();
	laserCloudSurfFromMap->clear();
	const auto& lastKeyPose = keyPoseInfo_.back();
	size_t curIndex = 0U;
	for (const auto& keyPose : keyPoseInfo_) {
		if ((std::sqrt((keyPose.x - lastKeyPose.x) * (keyPose.x - lastKeyPose.x) + (keyPose.y - lastKeyPose.y) *
			(keyPose.y - lastKeyPose.y)) > 30.0)) {
			continue;
		}
		*laserCloudCornerFromMap += laserCloudMapContainer_[curIndex].first;
		*laserCloudSurfFromMap += laserCloudMapContainer_[curIndex].second;
		++curIndex;
	}
	// Downsample the surrounding corner key frames (or map)
	downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
	downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
	// Downsample the surrounding surf key frames (or map)
	downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
	downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
}

void ScanMatch::downsampleCurrentScan()
{
	// Downsample cloud from current scan
	laserCloudCornerLastDS->clear();
	downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
	downSizeFilterCorner.filter(*laserCloudCornerLastDS);
	laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

	laserCloudSurfLastDS->clear();
	downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
	downSizeFilterSurf.filter(*laserCloudSurfLastDS);
	laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
}

void ScanMatch::updatePointAssociateToMap()
{
	transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

void ScanMatch::cornerOptimization()
{
	updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
	for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
		PointType pointOri, pointSel, coeff;
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqDis;

		pointOri = laserCloudCornerLastDS->points[i];
		pointAssociateToMap(&pointOri, &pointSel);
		kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

		cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
		cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
		cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

		if (pointSearchSqDis[4] < 1.0) {
			float cx = 0, cy = 0, cz = 0;
			for (int j = 0; j < 5; j++) {
				cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
				cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
				cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
			}
			cx /= 5;
			cy /= 5;
			cz /= 5;

			float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
			for (int j = 0; j < 5; ++j) {
				float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
				float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
				float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

				a11 += ax * ax;
				a12 += ax * ay;
				a13 += ax * az;
				a22 += ay * ay;
				a23 += ay * az;
				a33 += az * az;
			}
			a11 /= 5;
			a12 /= 5;
			a13 /= 5;
			a22 /= 5;
			a23 /= 5;
			a33 /= 5;

			matA1.at<float>(0, 0) = a11;
			matA1.at<float>(0, 1) = a12;
			matA1.at<float>(0, 2) = a13;
			matA1.at<float>(1, 0) = a12;
			matA1.at<float>(1, 1) = a22;
			matA1.at<float>(1, 2) = a23;
			matA1.at<float>(2, 0) = a13;
			matA1.at<float>(2, 0) = a13;
			matA1.at<float>(2, 1) = a23;
			matA1.at<float>(2, 2) = a33;

			cv::eigen(matA1, matD1, matV1);

			if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
				float x0 = pointSel.x;
				float y0 = pointSel.y;
				float z0 = pointSel.z;
				float x1 = cx + 0.1 * matV1.at<float>(0, 0);
				float y1 = cy + 0.1 * matV1.at<float>(0, 1);
				float z1 = cz + 0.1 * matV1.at<float>(0, 2);
				float x2 = cx - 0.1 * matV1.at<float>(0, 0);
				float y2 = cy - 0.1 * matV1.at<float>(0, 1);
				float z2 = cz - 0.1 * matV1.at<float>(0, 2);

				float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
				                  ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
				                  ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
				                  ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
				                  ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
				                  ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

				float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

				float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
				            (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
				           a012 / l12;

				float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
				             (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
				           a012 / l12;

				float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
				             (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
				           a012 / l12;

				float ld2 = a012 / l12;

				float s = 1 - 0.9 * fabs(ld2);

				coeff.x = s * la;
				coeff.y = s * lb;
				coeff.z = s * lc;
				coeff.intensity = s * ld2;

				if (s > 0.1) {
					laserCloudOriCornerVec[i] = pointOri;
					coeffSelCornerVec[i] = coeff;
					laserCloudOriCornerFlag[i] = true;
				}
			}
		}
	}
}

void ScanMatch::surfOptimization()
{
	updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
	for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
		PointType pointOri, pointSel, coeff;
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqDis;

		pointOri = laserCloudSurfLastDS->points[i];
		pointAssociateToMap(&pointOri, &pointSel);
		kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

		Eigen::Matrix<float, 5, 3> matA0;
		Eigen::Matrix<float, 5, 1> matB0;
		Eigen::Vector3f matX0;

		matA0.setZero();
		matB0.fill(-1);
		matX0.setZero();

		if (pointSearchSqDis[4] < 1.0) {
			for (int j = 0; j < 5; ++j) {
				matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
				matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
				matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
			}

			matX0 = matA0.colPivHouseholderQr().solve(matB0);

			float pa = matX0(0, 0);
			float pb = matX0(1, 0);
			float pc = matX0(2, 0);
			float pd = 1;

			float ps = sqrt(pa * pa + pb * pb + pc * pc);
			pa /= ps;
			pb /= ps;
			pc /= ps;
			pd /= ps;

			bool planeValid = true;
			for (int j = 0; j < 5; ++j) {
				if (std::abs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
				             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
				             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
					planeValid = false;
					break;
				}
			}

			if (planeValid) {
				float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
				float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y +
					pointSel.z * pointSel.z));

				coeff.x = s * pa;
				coeff.y = s * pb;
				coeff.z = s * pc;
				coeff.intensity = s * pd2;

				if (s > 0.1) {
					laserCloudOriSurfVec[i] = pointOri;
					coeffSelSurfVec[i] = coeff;
					laserCloudOriSurfFlag[i] = true;
				}
			}
		}
	}
}

void ScanMatch::combineOptimizationCoeffs() {
	// combine corner coeffs
	for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
		if (laserCloudOriCornerFlag[i]) {
			laserCloudOri->push_back(laserCloudOriCornerVec[i]);
			coeffSel->push_back(coeffSelCornerVec[i]);
		}
	}
	// combine surf coeffs
	for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
		if (laserCloudOriSurfFlag[i]) {
			laserCloudOri->push_back(laserCloudOriSurfVec[i]);
			coeffSel->push_back(coeffSelSurfVec[i]);
		}
	}
	// reset flag for next iteration
	std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
	std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

bool ScanMatch::LMOptimization(int iterCount)
{
	float srx = sin(transformTobeMapped[1]);
	float crx = cos(transformTobeMapped[1]);
	float sry = sin(transformTobeMapped[2]);
	float cry = cos(transformTobeMapped[2]);
	float srz = sin(transformTobeMapped[0]);
	float crz = cos(transformTobeMapped[0]);

	int laserCloudSelNum = laserCloudOri->size();
	if (laserCloudSelNum < 50) {
		return false;
	}

	cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
	cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
	cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
	cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
	cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
	cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
	cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

	PointType pointOri, coeff;

	for (int i = 0; i < laserCloudSelNum; i++) {
		// lidar -> camera
		pointOri.x = laserCloudOri->points[i].y;
		pointOri.y = laserCloudOri->points[i].z;
		pointOri.z = laserCloudOri->points[i].x;
		// lidar -> camera
		coeff.x = coeffSel->points[i].y;
		coeff.y = coeffSel->points[i].z;
		coeff.z = coeffSel->points[i].x;
		coeff.intensity = coeffSel->points[i].intensity;
		// in camera
		float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) *
			coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y +
			(crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

		float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y +
			crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x +
			(cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

		float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) *
			coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) *
			pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
		// lidar -> camera
		matA.at<float>(i, 0) = arz;
		matA.at<float>(i, 1) = arx;
		matA.at<float>(i, 2) = ary;
		matA.at<float>(i, 3) = coeff.z;
		matA.at<float>(i, 4) = coeff.x;
		matA.at<float>(i, 5) = coeff.y;
		matB.at<float>(i, 0) = -coeff.intensity;
	}

	cv::transpose(matA, matAt);
	matAtA = matAt * matA;
	matAtB = matAt * matB;
	cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

	if (iterCount == 0) {
		cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

		cv::eigen(matAtA, matE, matV);
		matV.copyTo(matV2);

		isDegenerate = false;
		float eignThre[6] = {100, 100, 100, 100, 100, 100};
		for (int i = 5; i >= 0; i--) {
			if (matE.at<float>(0, i) < eignThre[i]) {
				for (int j = 0; j < 6; j++) {
					matV2.at<float>(i, j) = 0;
				}
				isDegenerate = true;
			} else {
				break;
			}
		}
		matP = matV.inv() * matV2;
	}

	if (isDegenerate) {
		cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
		matX.copyTo(matX2);
		matX = matP * matX2;
	}

	transformTobeMapped[0] += matX.at<float>(0, 0);
	transformTobeMapped[1] += matX.at<float>(1, 0);
	transformTobeMapped[2] += matX.at<float>(2, 0);
	transformTobeMapped[3] += matX.at<float>(3, 0);
	transformTobeMapped[4] += matX.at<float>(4, 0);
	transformTobeMapped[5] += matX.at<float>(5, 0);

	float deltaR = std::sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
		pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
	float deltaT = std::sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) +
		pow(matX.at<float>(5, 0) * 100, 2));

	if (deltaR < 0.05 && deltaT < 0.05) {
		return true; // converged
	}
	return false; // keep optimizing
}

void ScanMatch::scan2MapOptimization()
{
	if (keyPoseInfo_.empty()) {
		return;
	}
	if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
	    laserCloudSurfLastDSNum > surfFeatureMinValidNum) {
		kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
		kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
		for (int iterCount = 0; iterCount < 30; ++iterCount) {
			laserCloudOri->clear();
			coeffSel->clear();

			cornerOptimization();
			surfOptimization();

			combineOptimizationCoeffs();

			if (LMOptimization(iterCount)) {
				break;
			}
		}
	} else {
		ROS_WARN("Not enough features! Only %d edge and %d planar features available.",
		         laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
	}
}

float ScanMatch::constraintTransformation(float value, float limit)
{
	if (value < -limit) {
		value = -limit;
	}
	if (value > limit) {
		value = limit;
	}
	return value;
}

bool ScanMatch::saveFrame()
{
	if (keyPoseInfo_.empty()) {
		return true;
	}

	Eigen::Affine3f transStart = pclPointToAffine3f(keyPoseInfo_.back());
	Eigen::Affine3f transFinal = pcl::getTransformation(
			transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
			transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
	Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
	float x, y, z, roll, pitch, yaw;
	pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

	if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
	    abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
	    abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
	    std::sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold) {
		return false;
	}
	return true;
}

void ScanMatch::saveKeyFramesAndFactor()
{
	if (!saveFrame()) {
		return;
	}

	// save key poses
	PointType thisPose3D;
	thisPose3D.x = transformTobeMapped[3];
	thisPose3D.y = transformTobeMapped[4];
	thisPose3D.z = transformTobeMapped[5];
	thisPose3D.intensity = cloudKeyPose_->size(); // this can be used as index
	cloudKeyPose_->push_back(thisPose3D);

	keyPoseInfo_.emplace_back(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
		transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2], timeLaserInfoCur);

	// save all the received edge and surf points
	pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
	pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
	pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

	// transformed cloud not available
	pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(thisCornerKeyFrame, keyPoseInfo_.back());
	pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(thisSurfKeyFrame, keyPoseInfo_.back());
	laserCloudMapContainer_.emplace_back(std::make_pair(laserCloudCornerTemp, laserCloudSurfTemp));

	if (keyPoseInfo_.size() > 50U) {
		keyPoseInfo_.pop_front();
		laserCloudMapContainer_.pop_front();
	}

	// save path for visualization
	updatePath(keyPoseInfo_.back());
}

void ScanMatch::updatePath(const PoseInfo& poseIn) {
	geometry_msgs::PoseStamped pose_stamped;
	pose_stamped.header.stamp = ros::Time().fromSec(poseIn.time);
	pose_stamped.header.frame_id = odometryFrame;
	pose_stamped.pose.position.x = poseIn.x;
	pose_stamped.pose.position.y = poseIn.y;
	pose_stamped.pose.position.z = poseIn.z;
	tf::Quaternion q = tf::createQuaternionFromRPY(poseIn.roll, poseIn.pitch, poseIn.yaw);
	pose_stamped.pose.orientation.x = q.x();
	pose_stamped.pose.orientation.y = q.y();
	pose_stamped.pose.orientation.z = q.z();
	pose_stamped.pose.orientation.w = q.w();
	globalPath.poses.emplace_back(pose_stamped);
}

void ScanMatch::publishOdometry() {
	// Publish odometry for ROS (global)
	nav_msgs::Odometry laserOdometryROS;
	laserOdometryROS.header.stamp = timeLaserInfoStamp;
	laserOdometryROS.header.frame_id = odometryFrame;
	laserOdometryROS.child_frame_id = "odom";
	laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
	laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
	laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
	laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0],
		transformTobeMapped[1], transformTobeMapped[2]);
	pubLaserOdometryGlobal.publish(laserOdometryROS);
}

void ScanMatch::publishFrames() {
	if (keyPoseInfo_.empty()) {
		return;
	}
	// publish path
	if (pubPath.getNumSubscribers() != 0) {
		globalPath.header.stamp = timeLaserInfoStamp;
		globalPath.header.frame_id = odometryFrame;
		pubPath.publish(globalPath);
	}
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "lidar_odo");
	ScanMatch SM;
	ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
	std::thread visualizeMapThread(&ScanMatch::visualizeGlobalMapThread, &SM);
	ros::spin();
	visualizeMapThread.join();
	return 0;
}

