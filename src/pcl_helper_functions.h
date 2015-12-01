#ifndef PCL_HELPER_FUNCTIONS_H
#define PCL_HELPER_FUNCTIONS_H

// C++
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

// PCL Conversions
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>

// Filters/Features/Algorithms
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/registration/icp.h>

// Viz
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>

// Eigen
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/StdVector>

// YAML
#include <yaml-cpp/yaml.h>

class pcl_helper_functions
{
public:

    struct OBB{
        Eigen::Vector3f bboxTransform;
        Eigen::Quaternionf bboxQuaternion;
        pcl::PointXYZ minPoint;
        pcl::PointXYZ maxPoint;
        double xDist;
        double yDist;
        double zDist;
    };

    enum Mode{
      EXTRACT_NORMALS_RADIUSSEARCH,
      EXTRACT_NORMALS_KSEARCH
    };
    
    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertPCLFromROS(
        const sensor_msgs::PointCloud2ConstPtr& inputCloud
    );

    static sensor_msgs::PointCloud2 convertROSFromPCL(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
    );
    static sensor_msgs::PointCloud2 convertROSFromPCL(
        pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud
    );

    static Eigen::Matrix4f getExternalCalibration(
        std::string ID
    );

    static std::string getCurrTimeAsString(
    );

    static Eigen::Matrix4f getTransformMatrix(
        double roll, double pitch, double yaw, double x, double y, double z
    );

    static void performRangeThresholding(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
        std::string axis, double startRange, double endRange
    );

    static void performSmoothing(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
        double searchRadius
    );

    static pcl::ModelCoefficients::Ptr performPlaneFitting(
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& inputCloud,
        double maxProjectError
    );

    static std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extractEuclidianClusters(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
        double tolerance, int minClusterSize, int maxClusterSize
    );

    static pcl::PointXYZ extractCentroid(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
    );

    static pcl_helper_functions::OBB extractOBB(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud
    );

    static pcl::PointCloud<pcl::PointXYZ>::Ptr extractConcaveHull(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
        double alpha
    );

    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extractNormals(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
        double searchRadius, pcl_helper_functions::Mode searchMode
    );

    static std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> extractRegionGrowingNormalsClusters(
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr inputCloud,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &coloredCloud,
        int neighbours, int minClusterSize, int maxClusterSize, double smoothnessThreshold, double curvatureThreshold
    );

    static void sortClusterBySize(
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputCluster
    );

    static bool clusterSizeSort(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr j
    );


};

#endif // PCL_HELPER_FUNCTIONS_H
