#include "pcl_helper_functions.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_helper_functions::convertPCLFromROS(
    const sensor_msgs::PointCloud2ConstPtr& inputCloud
){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*inputCloud,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2,*outputCloud);
    return outputCloud;
}

sensor_msgs::PointCloud2 pcl_helper_functions::convertROSFromPCL(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
){
    sensor_msgs::PointCloud2 outputCloud;
    pcl::toROSMsg(*inputCloud,outputCloud);
    return outputCloud;
}
sensor_msgs::PointCloud2 pcl_helper_functions::convertROSFromPCL(
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud
){
    sensor_msgs::PointCloud2 outputCloud;
    pcl::toROSMsg(*inputCloud,outputCloud);
    return outputCloud;
}

std::string pcl_helper_functions::getCurrTimeAsString(){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,80,"cloud_%d-%m-%Y_%I:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

Eigen::Matrix4f pcl_helper_functions::getTransformMatrix(
    double roll, double pitch, double yaw, double x, double y, double z
){

    Eigen::AngleAxisd rollAngle(roll / 180.0 * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw / 180.0 * M_PI, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch / 180.0 * M_PI, Eigen::Vector3d::UnitX());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

    Eigen::Matrix3d rotationMatrix = q.matrix();

    Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
    transformMatrix(0,0) = rotationMatrix(0,0);
    transformMatrix(0,1) = rotationMatrix(0,1);
    transformMatrix(0,2) = rotationMatrix(0,2);
    transformMatrix(0,3) = x;
    transformMatrix(1,0) = rotationMatrix(1,0);
    transformMatrix(1,1) = rotationMatrix(1,1);
    transformMatrix(1,2) = rotationMatrix(1,2);
    transformMatrix(1,3) = y;
    transformMatrix(2,0) = rotationMatrix(2,0);
    transformMatrix(2,1) = rotationMatrix(2,1);
    transformMatrix(2,2) = rotationMatrix(2,2);
    transformMatrix(2,3) = z;
    transformMatrix(3,0) = 0;
    transformMatrix(3,1) = 0;
    transformMatrix(3,2) = 0;
    transformMatrix(3,3) = 1;
    return transformMatrix;
}

Eigen::Matrix4f pcl_helper_functions::getExternalCalibration(
    std::string ID
){
    Eigen::Matrix4f transformMatrix;
    std::string path = ros::package::getPath("data_collection") + "/calibration_data/" + ID + "/external_calibration.yaml";
    YAML::Node matrix = YAML::LoadFile(path);
    std::vector<std::vector<double> > worldMatrix =
        matrix["WorldMatrix"].as<std::vector<std::vector<double> > >();
    transformMatrix <<  worldMatrix[0][0], worldMatrix[0][1], worldMatrix[0][2], worldMatrix[0][3],
                        worldMatrix[1][0], worldMatrix[1][1], worldMatrix[1][2], worldMatrix[1][3],
                        worldMatrix[2][0], worldMatrix[2][1], worldMatrix[2][2], worldMatrix[2][3],
                        worldMatrix[3][0], worldMatrix[3][1], worldMatrix[3][2], worldMatrix[3][3];
    transformMatrix = transformMatrix.inverse().eval();
    return transformMatrix;
}

void pcl_helper_functions::performRangeThresholding(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
    std::string axis, double startRange, double endRange
){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rangedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> passthroughFilter;
    passthroughFilter.setInputCloud(inputCloud);
    passthroughFilter.setFilterFieldName(axis);
    passthroughFilter.setFilterLimits(startRange, endRange);
    passthroughFilter.filter(*rangedCloud);
    inputCloud = rangedCloud;
}

void pcl_helper_functions::performSmoothing(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
    double searchRadius
){
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr smoothedCloudNormal(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> smoothingFilter;
    smoothingFilter.setInputCloud(inputCloud);
    smoothingFilter.setSearchRadius(searchRadius);
    smoothingFilter.setPolynomialFit(true);
    smoothingFilter.setComputeNormals(true);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr smoothingkdtree;
    smoothingFilter.setSearchMethod(smoothingkdtree);
    smoothingFilter.process(*smoothedCloudNormal);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*smoothedCloudNormal, *smoothedCloud);
    inputCloud = smoothedCloud;
}

pcl::ModelCoefficients::Ptr pcl_helper_functions::performPlaneFitting(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& inputCloud,
    double maxProjectError
){
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudProjected(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (maxProjectError);
    seg.setInputCloud (inputCloud);
    seg.segment (*inliers, *coefficients);

    pcl::ProjectInliers<pcl::PointXYZRGBNormal> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setIndices (inliers);
    proj.setInputCloud (inputCloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloudProjected);
    inputCloud = cloudProjected;

    return coefficients;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcl_helper_functions::extractEuclidianClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
    double tolerance, int minClusterSize, int maxClusterSize
){
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr segkdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    segkdtree->setInputCloud(inputCloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> clustering;
    clustering.setClusterTolerance(tolerance);
    clustering.setMinClusterSize(minClusterSize);
    clustering.setMaxClusterSize(maxClusterSize);
    clustering.setSearchMethod(segkdtree);
    clustering.setInputCloud(inputCloud);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    int currentClusterNum = 1;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clustersAllocated;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(inputCloud->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        clustersAllocated.push_back(cluster);

        if (cluster->points.size() <= 0)
            break;
        currentClusterNum++;
    }

    return clustersAllocated;
}

pcl::PointXYZ pcl_helper_functions::extractCentroid(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
){
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*inputCloud, centroid);
    pcl::PointXYZ pointCentroid(centroid[0],centroid[1],centroid[2]);
    return pointCentroid;
}

pcl_helper_functions::OBB pcl_helper_functions::extractOBB(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
){
    pcl_helper_functions::OBB obbData;

    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*inputCloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*inputCloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*inputCloud, *cloudPointsProjected, projectionTransform);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjectedConv(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*cloudPointsProjected, *cloudPointsProjectedConv);
    pcl::getMinMax3D(*cloudPointsProjectedConv, obbData.minPoint, obbData.maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(obbData.maxPoint.getVector3fMap() + obbData.minPoint.getVector3fMap());

    obbData.xDist = obbData.maxPoint.x - obbData.minPoint.x;
    obbData.yDist = obbData.maxPoint.y - obbData.minPoint.y;
    obbData.zDist = obbData.maxPoint.z - obbData.minPoint.z;
    const Eigen::Quaternionf bboxQuaternion2(eigenVectorsPCA);
    obbData.bboxQuaternion = bboxQuaternion2;
    obbData.bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
    return obbData;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_helper_functions::extractConcaveHull(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
    double alpha
){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudHull (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ConcaveHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud (inputCloud);
    chull.setAlpha (alpha);
    chull.reconstruct (*cloudHull);
    pcl::PointCloud<pcl::PointXYZ>::Ptr hullConv(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*cloudHull,*hullConv);
    return hullConv;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcl_helper_functions::extractNormals(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
    double searchRadius, pcl_helper_functions::Mode searchMode
){
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud (*inputCloud, *normals);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> normalEstimation;
    normalEstimation.setInputCloud(inputCloud);
    switch(searchMode)
    {
        case EXTRACT_NORMALS_RADIUSSEARCH: normalEstimation.setRadiusSearch(searchRadius);
        case EXTRACT_NORMALS_KSEARCH: normalEstimation.setKSearch((int)searchRadius);
        default: {}
    }
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
    return normals;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> pcl_helper_functions::extractRegionGrowingNormalsClusters(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr inputCloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &coloredCloud,
    int neighbours, int minClusterSize, int maxClusterSize, double smoothnessThreshold, double curvatureThreshold
){
    pcl::PointCloud<pcl::Normal>::Ptr convNormal(new pcl::PointCloud<pcl::Normal>);
    copyPointCloud(*inputCloud,*convNormal);
    pcl::search::Search<pcl::PointXYZRGBNormal>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBNormal> > (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    pcl::RegionGrowing<pcl::PointXYZRGBNormal, pcl::Normal> reg;
    reg.setMinClusterSize (minClusterSize);
    reg.setMaxClusterSize (maxClusterSize);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (neighbours);
    reg.setInputCloud (inputCloud);
    reg.setInputNormals (convNormal);
    reg.setSmoothnessThreshold (smoothnessThreshold / 180.0 * M_PI);
    reg.setCurvatureThreshold (curvatureThreshold);
    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);
    coloredCloud = reg.getColoredCloud ();

    int currentClusterNum = 1;
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clustersAllocated;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(inputCloud->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        clustersAllocated.push_back(cluster);

        if (cluster->points.size() <= 0)
            break;
        currentClusterNum++;
    }

    return clustersAllocated;
}

bool pcl_helper_functions::clusterSizeSort(pcl::PointCloud<pcl::PointXYZRGB>::Ptr i,pcl::PointCloud<pcl::PointXYZRGB>::Ptr j)
{ 
  return (i->points.size()<j->points.size()); 
}

void pcl_helper_functions::sortClusterBySize(
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputCluster
){
    std::sort(inputCluster.begin(), inputCluster.end(), pcl_helper_functions::clusterSizeSort);
}



