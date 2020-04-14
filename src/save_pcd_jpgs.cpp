#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <time.h>
#include <fstream>

#define MAP_POW 7
#define RESOLUTION 0.1
#define Z_MIN 0.4
#define Z_MAX 1.8

using namespace std;
using namespace message_filters;

string path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/sudoku/";

void mapCallback(const sensor_msgs::PointCloud2ConstPtr& cloud, const geometry_msgs::PointStampedConstPtr& center)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud, *cloud_in);

    static int size = 1 << (MAP_POW-1);
//    static int size = 1 << MAP_POW;
    static int center_x_y_img = size / 2;

    cv::Mat img = cv::Mat::ones(size, size, CV_8UC1) * 127;

    static int add_value = 128 / ((Z_MAX -Z_MIN) / RESOLUTION);

    // Add "add_value" to the pixel corresponds to a occupied voxel in map. Then dark pixel in the image suggests the place is occupied.
    // 3D to 2D projection
    for(int nIndex = 0; nIndex < cloud_in->points.size (); nIndex++)
    {
        if(cloud_in->points[nIndex].z > Z_MIN && cloud_in->points[nIndex].z < Z_MAX){
            int x = (int)((cloud_in->points[nIndex].x - center->point.x) / RESOLUTION)/2 + center_x_y_img;
            int y = (int)((cloud_in->points[nIndex].y - center->point.y) / RESOLUTION)/2 + center_x_y_img;
//            int x = (int)((cloud_in->points[nIndex].x - center->point.x) / RESOLUTION) + center_x_y_img;
//            int y = (int)((cloud_in->points[nIndex].y - center->point.y) / RESOLUTION) + center_x_y_img;

            x = min(x,size-1);
            y = min(y,size-1);
            x = max(0,x);
            y = max(0,y);

            // int add_value_this_point = (int)(add_value * cloud_in->points[nIndex].intensity);

            if(cloud_in->points[nIndex].intensity < 0.8){ //Freespace
                if(img.at<uchar>(y, x) + add_value <= 255){
                    img.at<uchar>(y, x) += add_value;
                }else{
                    img.at<uchar>(y, x) = 255;
                }
            } else{
                if(img.at<uchar>(y, x) - add_value*3 >= 0){
                    img.at<uchar>(y, x) -= add_value*3;
                }else{
                    img.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    cv::imshow("projected img", img);
    char key = cv::waitKey(50);

    // save at a rate of 1 Hz at most, otherwise the new file would cover the old one.
    if(key=='s'){
        time_t tt = time(NULL);
        tm* t=localtime(&tt);
        string file_name = path + "pow"+to_string(MAP_POW)+"resolution"+ to_string(RESOLUTION*10)+"T"+to_string(t->tm_year)+to_string(t->tm_mon)+to_string(t->tm_mday)+to_string(t->tm_hour)
                           +to_string(t->tm_hour)+to_string(t->tm_min)+to_string(t->tm_sec);

        cv::imwrite(file_name+".png", img);
        pcl::io::savePCDFileASCII(file_name+".pcd", *cloud_in);

        ofstream outFile;
        outFile.open(file_name+".csv", ios::out);
        outFile << center->point.x <<','<<center->point.y<<','<<center->point.z;
        outFile.close();
        cout << "Image saved!" << endl;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_pcd_jpgs");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::PointCloud2> map_sub(nh, "/ring_buffer/cloud_all", 1);
    message_filters::Subscriber<geometry_msgs::PointStamped> center_sub(nh, "/map_center", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PointStamped> sync_policy_classification;

    message_filters::Synchronizer<sync_policy_classification> sync(sync_policy_classification(10), map_sub, center_sub);
    sync.registerCallback(boost::bind(&mapCallback, _1, _2));

    ros::spin();

    return 0;

}
