#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <algorithm>

#include "pcl_helper_functions.h"
#include "geometry_msgs/Pose.h"
#include <pcl/registration/icp.h>
#include <omp.h>

#include "particle_filter.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

using namespace std;
using namespace Eigen;

int
main (int argc, char** argv)
{
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Palletization Detector"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);

    particle_filter pf;
    pf.initialize(0.5,0.4,0.40,0.70,0.47,0.6,-20);

    int i=0;
    while(1){
        pf.update_state();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud = pf.generate_box(0.8,0.8,0.8,0.70,0.27,0.6,40,0.05,0.01);
        pf.update_weights(inputCloud);
        pf.resample_particles();
        MeanVector meanVector = pf.compute_mean();
        cout << meanVector.transpose() << endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr meanCloud = pf.generate_box(meanVector(0),meanVector(1),meanVector(2),meanVector(3),meanVector(4),meanVector(5),meanVector(6),0.05,0);

        for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = inputCloud->begin(); it!= inputCloud->end(); it++){
            it->r = 0; it->g = 255; it->b = 255;
        }
        for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = meanCloud->begin(); it!= meanCloud->end(); it++){
            it->r = 255; it->g = 0; it->b = 0;
        }

        StateMatrix stateMatrix = pf.get_state();
        pcl::PointCloud<pcl::PointXYZ> pointCloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudPtr(new pcl::PointCloud<pcl::PointXYZRGB>);
        //cout << stateMatrix << endl;
        for(int i=0;i<NPOP_PARTICLES;i++){
            pcl::PointXYZ p1;
            p1.x=stateMatrix(0,i); p1.y=stateMatrix(1,i); p1.z=stateMatrix(2,i);
            pointCloud.push_back(p1);
        }
        pcl::copyPointCloud(pointCloud,*pointCloudPtr);
        for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = pointCloudPtr->begin(); it!= pointCloudPtr->end(); it++){
            it->r = 255; it->g = 255; it->b = 255;
        }
        viewer->addPointCloud(pointCloudPtr,"distCloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "distCloud");
        viewer->addPointCloud(inputCloud,"inputCloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inputCloud");
        viewer->addPointCloud(meanCloud,"meanCloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "meanCloud");
        viewer->addPointCloud(dataCloud,"dataCloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dataCloud");
        viewer->spinOnce(10);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        viewer->removeAllPointClouds();

    }

}


