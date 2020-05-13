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
#include "local_feature_map.h"
#include "local_ssh/Feature.h"
#include "local_ssh/Features.h"

#define MAP_POW 7
#define RESOLUTION 0.1
#define Z_MIN 0.4
#define Z_MAX 1.8
#define IMG_SCALE 2

#define RECT_SIZE_X 30 //26
#define RECT_SIZE_Y 30 //26
#define RECT_SHRINK_SIZE 10

using namespace std;
using namespace message_filters;
using namespace tiny_dnn;


tiny_dnn::network<tiny_dnn::sequential> model, angle_model;
std::vector<LocalFeatureMap> local_maps_on_nodes;
int node_counter = 0;

ros::Publisher node_features_pub;

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

void findImgBoundary(int px, int py, int &x_min, int &x_max, int &y_min, int &y_max)
{
    x_min = std::min(x_min, px);
    x_max = std::max(x_max, px);
    y_min = std::min(y_min, py);
    y_max = std::max(y_max, py);
}

void showAllStoredFeatures(std::vector<Feature> &feature_map, float center_x, float center_y, cv::Mat &current_local_map, string display_name="map")
{
    if(feature_map.empty())
        return;

    std::vector<std::vector<cv::Point>> gateway_boundary_points;
    std::vector<cv::Point> gateway_direction_point;
    std::vector<cv::Scalar> colors;

    int x_min = 10000, x_max = -10000;
    int y_min = 10000, y_max = -10000;

    /* Find Edge of the map and store gateway positions and color */
    for(const auto &feature_stored : feature_map){
        if(feature_stored.label == "gateway" ){
            float boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y;
            local_maps_on_nodes[node_counter].getGatewayBoundaryPoints(feature_stored, boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y);
            std::vector<cv::Point> gateway_boundary_point_pair;
            cv::Point boundary_point1_img, boundary_point2_img;
            boundary_point1_img.x = (boundary_point1_x - center_x) / RESOLUTION /IMG_SCALE;
            boundary_point1_img.y = (boundary_point1_y - center_y) / RESOLUTION /IMG_SCALE;
            findImgBoundary(boundary_point1_img.x, boundary_point1_img.y, x_min, x_max, y_min, y_max);

            boundary_point2_img.x = (boundary_point2_x - center_x) / RESOLUTION /IMG_SCALE;
            boundary_point2_img.y = (boundary_point2_y - center_y) / RESOLUTION /IMG_SCALE;
            findImgBoundary(boundary_point2_img.x, boundary_point2_img.y, x_min, x_max, y_min, y_max);

            gateway_boundary_point_pair.push_back(boundary_point1_img);
            gateway_boundary_point_pair.push_back(boundary_point2_img);
            gateway_boundary_points.push_back(gateway_boundary_point_pair);

            int gateway_img_center_x = (boundary_point1_img.x + boundary_point2_img.x) / 2;
            int gateway_img_center_y = (boundary_point1_img.y + boundary_point2_img.y) / 2;
            int direction_point_x = gateway_img_center_x + 6* cos(feature_stored.pose.yaw);
            int direction_point_y = gateway_img_center_y + 6* sin(feature_stored.pose.yaw);
            gateway_direction_point.push_back(cv::Point(direction_point_x, direction_point_y));

            if(feature_stored.exsitence_confidence < local_maps_on_nodes[node_counter].Confidence_High){
                colors.push_back(cv::Scalar(50,50,50)); //gray: low confidence
            }else if(!feature_stored.in_fov_flag){
                colors.push_back(cv::Scalar(255,0,0));  //Blue: high confidence, outside of view
            }else{
                colors.push_back(cv::Scalar(0, 255, 0));  //Green: high confidence, in view field (local map range)
            }
        }
    }

    /* Create a color image with local map in the center */
    int image_cols = x_max - x_min + 60;
    image_cols = std::min(1200, image_cols);
    image_cols = std::max(240, image_cols);
    int image_rows = y_max - y_min + 60;
    image_rows = std::min(1200, image_cols);
    image_rows = std::max(240, image_cols);

    cv::Mat map_img = cv::Mat::zeros(cv::Size(image_cols, image_rows), CV_8UC3);
    std::vector<cv::Mat> local_map_three_channels;
    local_map_three_channels.push_back(current_local_map);
    local_map_three_channels.push_back(current_local_map);
    local_map_three_channels.push_back(current_local_map);
    cv::Mat current_local_map_color;
    cv::merge(local_map_three_channels, current_local_map_color);

    int channel = current_local_map_color.channels();
    for(int i=0; i<current_local_map_color.rows; i++)  //insert local map to the black background
    {
        uchar* inRgb = current_local_map_color.ptr<uchar>(i);
        uchar* outRgb = map_img.ptr<uchar>(i + (map_img.rows-current_local_map_color.rows)/2);

        for(int j=0; j<current_local_map_color.cols; j++)
        {
            int col_in_map_img = j + (map_img.cols-current_local_map_color.cols)/2;
            outRgb[channel*col_in_map_img] = inRgb[channel*j];
            outRgb[channel*col_in_map_img+1] = inRgb[channel*j + 1];
            outRgb[channel*col_in_map_img+2] = inRgb[channel*j + 2];
        }
    }

    for(int i=0; i<gateway_boundary_points.size(); i++){
        const auto &gateway_boundary = gateway_boundary_points[i];
        cv::Point boundary_point1_img, boundary_point2_img;
        boundary_point1_img.x = gateway_boundary[0].x + map_img.cols/2;
        boundary_point1_img.y = gateway_boundary[0].y + map_img.rows/2;
        boundary_point2_img.x = gateway_boundary[1].x + map_img.cols/2;
        boundary_point2_img.y = gateway_boundary[1].y + map_img.rows/2;
        cv::line(map_img, boundary_point1_img, boundary_point2_img, colors[i], 2);
        cv::line(map_img, cv::Point(gateway_direction_point[i].x + map_img.cols/2, gateway_direction_point[i].y + map_img.rows/2),
                cv::Point((boundary_point1_img.x+boundary_point2_img.x)/2, (boundary_point1_img.y+boundary_point2_img.y)/2), colors[i], 2);
    }
    cv::imshow(display_name, map_img);
    cv::waitKey(1);
}


bool checkIfNearFloodRegion(cv::Point &p, cv::Mat &flooded_img, int filled_flag, int check_size = 2)
{
    if(p.x < check_size || p.x > flooded_img.cols-check_size-1 || p.y < check_size || p.y > flooded_img.rows-check_size-1){
        return false;
    }else{
        for(int i=p.x-check_size; i<p.x+check_size; i++){
            for(int j=p.y-check_size; j<p.y+check_size; j++){
                if(flooded_img.at<uchar>(j,i) == filled_flag){
                    return true;
                }
            }
        }
        return false;
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
            int x = (int) ((cloud_in->points[nIndex].x - center->point.x) / RESOLUTION) / IMG_SCALE + center_x_y_img;
            int y = (int) ((cloud_in->points[nIndex].y - center->point.y) / RESOLUTION) / IMG_SCALE + center_x_y_img;

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
    findVoronoiSkeletonPoints(image_this, skeleton_points, true);  /// CHG

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
            cv::Mat rect_this = image_this(cv::Rect(sk_point.x-(RECT_SIZE_X-RECT_SHRINK_SIZE)/2, sk_point.y-(RECT_SIZE_Y-RECT_SHRINK_SIZE)/2, RECT_SIZE_X-RECT_SHRINK_SIZE, RECT_SIZE_Y-RECT_SHRINK_SIZE));
            cv::resize(rect_this, rect_this, cv::Size(RECT_SIZE_X, RECT_SIZE_Y), 0,0, cv::INTER_NEAREST);
            vec_t data_this;
            convert_image(rect_this, data_this);

            vec_t label_this = model.predict(data_this);

            if(label_this[0] < 0.7){
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

    /// Correct directions for angle output
    for(int i=0; i<result_points.size(); i++){
        auto p = result_points[i];
        float dy1 = p.x - image_this_copy.cols/2;
        float dx1 = p.y - image_this_copy.rows/2;
        float temp_direction = atan2(dy1, dx1);
//        std::cout << "P "<< p.x<< " " << p.y << " temp_direction="<<temp_direction<<std::endl;
        temp_direction = -temp_direction + CV_PI / 2.f;
        if(temp_direction > CV_PI) temp_direction -= CV_2PI;
        else if(temp_direction < -CV_PI) temp_direction += CV_2PI;

        if( fabs(result_angles[i] - temp_direction) < CV_PI / 2.f || fabs(result_angles[i] - temp_direction) > CV_PI / 2.f * 3.f){
            continue;
        }else{
//            std::cout << "angle corrected " << std::endl;
            if(result_angles[i] > 0){
                result_angles[i] -= CV_PI;
            }else{
                result_angles[i] += CV_PI;
            }
        }
    }

    cv::Mat img_to_flood_fill, img_to_show_all;
    image_this_copy.copyTo(img_to_flood_fill);
    image_this_copy.copyTo(img_to_show_all);

    /// Display
    cv::circle(image_this_copy, cv::Point(image_this_copy.cols/2, image_this_copy.rows/2), 1, cv::Scalar(150), 2);
    for(int i=0; i<result_points.size(); i++){
        auto p = result_points[i];
        cv::circle(image_this_copy, p, 2, cv::Scalar(0), 1);
        cv::Point direction_end_point;

        float direction_angle_rad = result_angles[i];
        direction_end_point.x = 10 * cos(direction_angle_rad) + p.x;
        direction_end_point.y = 10 * sin(direction_angle_rad) + p.y;
        cv::line(image_this_copy, p, direction_end_point, cv::Scalar(160), 1);
    }

    cv::imshow("image_this_copy", image_this_copy);

    /// Retrieve the gateways in local area with high confidence to do the flood fill
    for(const auto &feature_stored : local_maps_on_nodes[node_counter].feature_map){
        if(feature_stored.label == "gateway" && feature_stored.in_fov_flag && feature_stored.exsitence_confidence >= local_maps_on_nodes[node_counter].Confidence_High){
            float boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y;
            local_maps_on_nodes[node_counter].getGatewayBoundaryPoints(feature_stored, boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y);
            cv::Point boundary_point1_img, boundary_point2_img;
            boundary_point1_img.x = (boundary_point1_x - center->point.x) / RESOLUTION /IMG_SCALE + img_to_flood_fill.cols/2;
            boundary_point1_img.y = (boundary_point1_y - center->point.y) / RESOLUTION /IMG_SCALE + img_to_flood_fill.rows/2;
            boundary_point2_img.x = (boundary_point2_x - center->point.x) / RESOLUTION /IMG_SCALE + img_to_flood_fill.cols/2;
            boundary_point2_img.y = (boundary_point2_y - center->point.y) / RESOLUTION /IMG_SCALE + img_to_flood_fill.rows/2;
            cv::line(img_to_flood_fill, boundary_point1_img, boundary_point2_img, cv::Scalar(0), 2);
        }
    }

    for(int i=0; i<result_points.size(); i++)
    {
        auto gateway_point = result_points[i];
        float boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y;
        local_maps_on_nodes[node_counter].getGatewayBoundaryPixelPoints(gateway_point.x, gateway_point.y, result_angles[i],
                                                                        boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y);
        cv::line(img_to_flood_fill, cv::Point(boundary_point1_x, boundary_point1_y), cv::Point(boundary_point2_x, boundary_point2_y), cv::Scalar(50), 2);
    }

    cv::floodFill(img_to_flood_fill, cv::Point(img_to_flood_fill.cols/2, img_to_flood_fill.rows/2), cv::Scalar(188)); //188 is a flag
    cv::imshow("img_to_flood_fill", img_to_flood_fill);

    std::vector<FeatureIN> features_in;
    for(int i=0; i<result_points.size(); i++){  // Convert to map frame
        if(checkIfNearFloodRegion(result_points[i], img_to_flood_fill, 188, 2)){
            FeatureIN temp_feature;
            temp_feature.label = "gateway";
            temp_feature.pose.x = (result_points[i].x - image_this_copy.cols/2) * RESOLUTION * IMG_SCALE + center->point.x;
            temp_feature.pose.y = (result_points[i].y - image_this_copy.rows/2) * RESOLUTION * IMG_SCALE + center->point.y;
            temp_feature.pose.yaw = result_angles[i];
            features_in.push_back(temp_feature);
        }
    }

    /// Update position and check if passed a gateway
    Pose2D vehicle_position;
    vehicle_position.x = center->point.x;
    vehicle_position.y = center->point.y;
    vehicle_position.yaw = 0.f;
    bool if_passed_a_gateway = local_maps_on_nodes[node_counter].updateVehiclePosition(vehicle_position);
    if(if_passed_a_gateway){
        //remove the nodes that should belong to the new node
        local_maps_on_nodes[node_counter].deleteFromFeatureMap(features_in);
        //publish
        local_ssh::Features feature_map_to_publish;
        feature_map_to_publish.header.stamp = ros::Time::now();
        for(const auto& valid_feature : local_maps_on_nodes[node_counter].feature_map){
            if(valid_feature.exsitence_confidence >= local_maps_on_nodes[node_counter].Confidence_High){
                local_ssh::Feature temp_feature;
                temp_feature.label = valid_feature.label;
                temp_feature.x = valid_feature.pose.x;
                temp_feature.y = valid_feature.pose.y;
                temp_feature.yaw = valid_feature.pose.yaw;
                temp_feature.confidence = valid_feature.exsitence_confidence;
                feature_map_to_publish.features.push_back(temp_feature);
            }
        }
        feature_map_to_publish.passed_gateway.label = local_maps_on_nodes[node_counter].gateway_just_passed.label;
        feature_map_to_publish.passed_gateway.x = local_maps_on_nodes[node_counter].gateway_just_passed.pose.x;
        feature_map_to_publish.passed_gateway.y = local_maps_on_nodes[node_counter].gateway_just_passed.pose.y;
        feature_map_to_publish.passed_gateway.yaw = local_maps_on_nodes[node_counter].gateway_just_passed.pose.yaw;
        feature_map_to_publish.passed_gateway.confidence = local_maps_on_nodes[node_counter].gateway_just_passed.exsitence_confidence;
        node_features_pub.publish(feature_map_to_publish);

        // Start another node
        LocalFeatureMap *node = new LocalFeatureMap;
        local_maps_on_nodes.push_back(*node);
        node_counter ++;
    }

    /// Add currently detected features to local feature map
    local_maps_on_nodes[node_counter].addToFeatureMap(features_in);

    end_time = clock();
    std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    /// Display map
    showAllStoredFeatures(local_maps_on_nodes[node_counter].feature_map, center->point.x, center->point.y, img_to_show_all, "current map");
    if(local_maps_on_nodes.size()>1){
        showAllStoredFeatures(local_maps_on_nodes[node_counter-1].feature_map, center->point.x, center->point.y, img_to_show_all, "last map");
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_pcd_jpgs");
    ros::NodeHandle nh;

    model.load("LeNet-model-rects-regression-30combined-vgg");
    angle_model.load("LeNet-model-angles-regression-30combined-vgg");

    LocalFeatureMap *node = new LocalFeatureMap;
    local_maps_on_nodes.push_back(*node);

    message_filters::Subscriber<sensor_msgs::PointCloud2> map_sub(nh, "/ring_buffer/cloud_all", 1);
    message_filters::Subscriber<geometry_msgs::PointStamped> center_sub(nh, "/map_center", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PointStamped> sync_policy_classification;

    message_filters::Synchronizer<sync_policy_classification> sync(sync_policy_classification(10), map_sub, center_sub);
    sync.registerCallback(boost::bind(&mapCallback, _1, _2));

    node_features_pub = nh.advertise<local_ssh::Features>("/features_in_last_map", 1);

    ros::spin();

    return 0;

}