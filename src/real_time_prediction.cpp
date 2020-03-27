//
// Created by cc on 2020/3/27.
//

#include "tiny_dnn/tiny_dnn.h"
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
#include <algorithm>
#include "voronoi_skeleton_points.h"
#include "preprocess.h"

#define MAP_POW 7
#define RESOLUTION 0.1
#define Z_MIN 0.4
#define Z_MAX 1.8

#define RECT_SIZE_X 30 //26
#define RECT_SIZE_Y 30 //26

using namespace std;
using namespace message_filters;
using namespace tiny_dnn;


tiny_dnn::network<tiny_dnn::sequential> model, angle_model;

void convert_image(const cv::Mat &img, vec_t &d)
{
    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(img.cols, img.rows));
    double minv = -1.0;
    double maxv = 1.0;

    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void convert_images(std::vector<cv::Mat> &rects, std::vector<vec_t>& data)
{
    for(auto &rect : rects){
        vec_t d;
        convert_image(rect, d);
        data.push_back(d);
    }
}

float pointSquareDistance(cv::Point p1, cv::Point p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

void rankFromLargeToSmallConfidence(std::vector<cv::Point> &p_in, std::vector<float> &confidence)
{
    for(int i=0;  i<p_in.size()-1; i++){
        for(int j=i+1; j<p_in.size(); j++){
            if(confidence[i] < confidence[j]){
                float temp_confidence = confidence[i];
                cv::Point temp_point = p_in[i];
                confidence[i] = confidence[j];
                p_in[i] = p_in[j];
                confidence[j] = temp_confidence;
                p_in[j] = temp_point;
            }
        }
    }
}

void nonMaximumSuppression(std::vector<cv::Point> &points_in, std::vector<float> &confidences, std::vector<cv::Point> &p_out, float dist_threshold = 4)
{
    if(points_in.size() == 0)
        return;
    else if(points_in.size() == 1){
        p_out = points_in;
    }

    float square_threshold = dist_threshold * dist_threshold;
    rankFromLargeToSmallConfidence(points_in, confidences);
    std::vector<int> merged_flag(points_in.size(), 0);
    for(int i=0; i<points_in.size()-1; i++){
        if(!merged_flag[i]){
            for(int j=i+1; j<points_in.size(); j++){
                if(!merged_flag[j]){
                    float square_dist = pointSquareDistance(points_in[i], points_in[j]);
                    if(square_dist <= square_threshold){
                        if(confidences[i]==confidences[j]){
                            points_in[i] = (points_in[i] + points_in[j]) / 2;
                        }
                        merged_flag[j] = 1;
                    }
                }
            }
            merged_flag[i] = 1;
            p_out.push_back(points_in[i]);
        }
    }
}

void mapCallback(const sensor_msgs::PointCloud2ConstPtr& cloud, const geometry_msgs::PointStampedConstPtr& center) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud, *cloud_in);

    static int size = 1 << (MAP_POW - 1);
//    static int size = 1 << MAP_POW;
    static int center_x_y_img = size / 2;

    cv::Mat img = cv::Mat::ones(size, size, CV_8UC1) * 127;

    static int add_value = 128 / ((Z_MAX - Z_MIN) / RESOLUTION);

    // Add "add_value" to the pixel corresponds to a occupied voxel in map. Then dark pixel in the image suggests the place is occupied.
    // 3D to 2D projection
    for (int nIndex = 0; nIndex < cloud_in->points.size(); nIndex++) {
        if (cloud_in->points[nIndex].z > Z_MIN && cloud_in->points[nIndex].z < Z_MAX) {
            int x = (int) ((cloud_in->points[nIndex].x - center->point.x) / RESOLUTION) / 2 + center_x_y_img;
            int y = (int) ((cloud_in->points[nIndex].y - center->point.y) / RESOLUTION) / 2 + center_x_y_img;

            x = min(x, size - 1);
            y = min(y, size - 1);
            x = max(0, x);
            y = max(0, y);

            // int add_value_this_point = (int)(add_value * cloud_in->points[nIndex].intensity);

            if (cloud_in->points[nIndex].intensity < 0.8) { //Freespace
                if (img.at<uchar>(y, x) + add_value <= 255) {
                    img.at<uchar>(y, x) += add_value;
                } else {
                    img.at<uchar>(y, x) = 255;
                }
            } else {
                if (img.at<uchar>(y, x) - add_value * 3 >= 0) {
                    img.at<uchar>(y, x) -= add_value * 3;
                } else {
                    img.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    cv::imshow("projected img", img);

    /// Load model and testing
    clock_t start_time, end_time;

    cv::Mat image_this = img;
    turnBlacktoGray(image_this);

    cv::Mat image_this_copy = image_this.clone();
    start_time = clock();

    /// Extract skeleton points
    std::vector<cv::Point> skeleton_points;
    findVoronoiSkeletonPoints(image_this, skeleton_points, false);  /// CHG
    if(skeleton_points.size() < 1){
        std::cout << "Found no skeleton_points in this image, skip!" << std::endl;
        return;
    }
    /// Predict on skeleton points
    std::vector<float> confidences_vec;
    std::vector<cv::Point> valid_points;

    for(auto &sk_point : skeleton_points){
        if(sk_point.x < RECT_SIZE_X/2 || sk_point.y <RECT_SIZE_Y/2 || sk_point.x > image_this.cols-RECT_SIZE_X/2-1 || sk_point.y > image_this.rows-RECT_SIZE_Y/2-1){
            continue;
        }else{
            cv::Mat rect_this = image_this(cv::Rect(sk_point.x-RECT_SIZE_X/2, sk_point.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
            vec_t data_this;
            convert_image(rect_this, data_this);

            vec_t label_this = model.predict(data_this);

            if(label_this[0] < 0.5){
                float confidence = 1.f - label_this[0];
                confidences_vec.push_back(confidence);
                valid_points.push_back(sk_point);
            }
        }
    }

    /// nonMaximumSuppression
    std::vector<cv::Point> result_points;
    std::vector<float> result_angles;

    nonMaximumSuppression(valid_points, confidences_vec, result_points, 5); //5 for 30, 4 for 26 image

    /// Angle predicton
    for(auto &p : result_points){
        if(p.x < RECT_SIZE_X/2 || p.y <RECT_SIZE_Y/2 || p.x > image_this.cols-RECT_SIZE_X/2-1 || p.y > image_this.rows-RECT_SIZE_Y/2-1){
            continue;
        }else{
            cv::Mat rect_this = image_this(cv::Rect(p.x-RECT_SIZE_X/2, p.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
            vec_t data_this;
            convert_image(rect_this, data_this);

            vec_t angle_label_this = angle_model.predict(data_this);
            float angle = (angle_label_this[0] - 0.5)*2*CV_PI;
            result_angles.push_back(angle);
        }
    }

    /// Correct directions for angle output in range[0, Pi]
    for(int i=0; i<result_points.size(); i++){
        auto p = result_points[i];
        float dy1 = p.x - image_this_copy.cols/2;
        float dx1 = p.y - image_this_copy.rows/2;
        float temp_direction = atan2(dy1, dx1);
        std::cout << "P "<< p.x<< " " << p.y << " temp_direction="<<temp_direction<<std::endl;
        temp_direction = -temp_direction + CV_PI / 2.f;
        if(temp_direction > CV_PI) temp_direction -= CV_2PI;
        else if(temp_direction < -CV_PI) temp_direction += CV_2PI;

        if( fabs(result_angles[i] - temp_direction) < CV_PI / 2.f || fabs(result_angles[i] - temp_direction) > CV_PI / 2.f * 3.f){
            continue;
        }else{
            std::cout << "angle corrected " << std::endl;
            if(result_angles[i] > 0){
                result_angles[i] -= CV_PI;
            }else{
                result_angles[i] += CV_PI;
            }
        }
    }

    end_time = clock();
    std::cout <<"gateway_num = " << result_points.size() << std::endl;
    std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    /// Display
    cv::circle(image_this_copy, cv::Point(image_this_copy.cols/2, image_this_copy.rows/2), 1, cv::Scalar(150), 2);
    for(int i=0; i<result_points.size(); i++){
        auto p = result_points[i];
        cv::circle(image_this_copy, p, 2, cv::Scalar(0), 1);
        cv::Point direction_end_point;

        float direction_angle = result_angles[i];
        direction_end_point.x = 10 * cos(direction_angle) + p.x;
        direction_end_point.y = 10 * sin(direction_angle) + p.y;
        cv::line(image_this_copy, p, direction_end_point, cv::Scalar(160), 1);
    }

    cv::imshow("image_this_copy", image_this_copy);
    cv::waitKey(5);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_pcd_jpgs");
    ros::NodeHandle nh;

    model.load("LeNet-model-rects-regression");
    angle_model.load("LeNet-model-angles-regression");

    message_filters::Subscriber<sensor_msgs::PointCloud2> map_sub(nh, "/ring_buffer/cloud_all", 1);
    message_filters::Subscriber<geometry_msgs::PointStamped> center_sub(nh, "/map_center", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PointStamped> sync_policy_classification;

    message_filters::Synchronizer<sync_policy_classification> sync(sync_policy_classification(10), map_sub, center_sub);
    sync.registerCallback(boost::bind(&mapCallback, _1, _2));

    ros::spin();

    return 0;

}