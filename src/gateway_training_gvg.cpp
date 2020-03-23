//
// Created by cc on 2020/3/20.
//

#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <opencv2/ml.hpp>
#include "labelimg_xml_reader.h"
#include "voronoi_skeleton_points.h"

#define KERNEL_X 13 // 17 when use rotated kernel
#define KERNEL_Y 21
#define EDGE_X 6
#define EDGE_Y 10

const float scale_factor = 0.85;
const int scale_times = 3;  //1, when use rotated kernel
const float rotate_angle = 30;  // 30x12
const int rotate_times = 12; //1, when use rotated kernel

bool use_noised_data = false;
bool use_rotated_data = false;

bool save_rect_images = true;
const int rect_to_save_size_x = 26;
const int rect_to_save_size_y = 26;
int rect_save_counter = 0;
int negative_data_save_dist_threshold = 3;

const float kernel_scale_factor = 0.85;
const int kernel_scale_times = 4;
const float kernel_rotate_angle = 30;
const int kernel_rotate_times = 12;


bool ifCloseToAnyPointInVector(Eigen::Vector2i p, const std::vector<Eigen::Vector2i>& v, const float threshold){
    for(const auto &point : v){
        float dist = sqrt( (p(0)-point(0))^2 + (p(1)-point(1))^2);
        if(dist <= threshold){
            return true;
        }
    }
    return false;
}


float pointSquareDistance(cv::Point p1, cv::Point p2){
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}


float findNearestPoint(cv::Point &p, std::vector<cv::Point> &reference_points, cv::Point &nearest_point){
    const int reference_points_size = reference_points.size();
    float min_distance = 100000000.f;
    int min_seq = 0;
    for(int i=0; i<reference_points_size; i++){
        float distance_this = pointSquareDistance(p, reference_points[i]);
        if(distance_this < min_distance){
            min_distance = distance_this;
            min_seq = i;
        }
    }

    nearest_point = reference_points[min_seq];

    return sqrt(min_distance);
}

void defineKernels(std::vector<Eigen::MatrixXf> &kernels)
{
    /// Kernel size should all be 13x21, if not, an edge adjustment is to be done when generating cost maps
    /*Kernel 1*/
    Eigen::MatrixXf kernel1(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel1.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernel1.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel1.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6, 1.f);
    kernel1.block(7, 0, 6, 21) = Eigen::MatrixXf::Constant(6, 21, -0.5);
//    std::cout << std::endl << kernel1 << std::endl;
    kernels.push_back(kernel1);

    /*Kernel 2*/
    Eigen::MatrixXf kernel2(KERNEL_X,KERNEL_Y); //13 rows, 21 cols
    kernel2.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernel2.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel2.block(0, 15, 13, 6) = Eigen::MatrixXf::Constant(13, 6, 1.f);
    kernel2.block(7, 0, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
//    std::cout << std::endl << kernel2 << std::endl;
    kernels.push_back(kernel2);

    /*Kernel 3*/
    Eigen::MatrixXf kernel3(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel3.block(0, 0, 13, 6) = Eigen::MatrixXf::Constant(13, 6,1.f);
    kernel3.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel3.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6, 1.f);
    kernel3.block(7, 6, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
//    std::cout << std::endl << kernel3 << std::endl;
    kernels.push_back(kernel3);

    /*Kernel 4*/

}

void setMatBlockAsConstant(cv::Mat &img, int start_row, int start_col, int block_rows, int block_cols, int value)
{
    for (int i = start_row; i < start_row + block_rows; i++) {
        uchar *row_in_img = img.ptr<uchar>(i);
        for (int j = start_col; j < start_col + block_cols; j++) {
            row_in_img[j] = value;
        }
    }
}

void defineRotatedKernels(std::vector<Eigen::MatrixXf> &kernels)
{
    std::vector<cv::Mat> kernels_mat_form;
    /* Kernel 1 */
    cv::Mat kernel1 = cv::Mat::zeros(cv::Size(KERNEL_Y, KERNEL_X), CV_8UC1);
    setMatBlockAsConstant(kernel1, 0, 0, 9, 6, 200);
    setMatBlockAsConstant(kernel1, 0, 6, 9, 9, 100);
    setMatBlockAsConstant(kernel1, 0, 15, 9, 6, 200);
    setMatBlockAsConstant(kernel1, 9, 0, 8, 21, 100);
    kernels_mat_form.push_back(kernel1);

    for(int i=1; i<kernel_rotate_times; i++){
        float rotate_angle_total = i * kernel_rotate_angle;
        cv::Mat rotated_kernel;
        imgRotateCutEdge(kernel1, rotated_kernel, rotate_angle_total, 0);
        kernels_mat_form.push_back(rotated_kernel);
    }

    /*Kernel 2*/
    cv::Mat kernel2 = cv::Mat::zeros(cv::Size(KERNEL_Y, KERNEL_X), CV_8UC1);
    setMatBlockAsConstant(kernel2, 0, 0, 9, 6, 200);
    setMatBlockAsConstant(kernel2, 0, 6, 9, 9, 100);
    setMatBlockAsConstant(kernel2, 0, 15, 17, 6, 200);
    setMatBlockAsConstant(kernel2, 9, 0, 8, 15, 100);
    kernels_mat_form.push_back(kernel2);

    for(int i=1; i<kernel_rotate_times; i++){
        float rotate_angle_total = i * kernel_rotate_angle;
        cv::Mat rotated_kernel;
        imgRotateCutEdge(kernel2, rotated_kernel, rotate_angle_total, 0);
        kernels_mat_form.push_back(rotated_kernel);
    }

    /*Kernel 3*/
    cv::Mat kernel3 = cv::Mat::zeros(cv::Size(KERNEL_Y, KERNEL_X), CV_8UC1);
    setMatBlockAsConstant(kernel3, 0, 0, 17, 6, 200);
    setMatBlockAsConstant(kernel3, 0, 6, 9, 9, 100);
    setMatBlockAsConstant(kernel3, 0, 15, 9, 6, 200);
    setMatBlockAsConstant(kernel3, 9, 6, 8, 15, 100);
    kernels_mat_form.push_back(kernel3);

    for(int i=1; i<kernel_rotate_times; i++){
        float rotate_angle_total = i * kernel_rotate_angle;
        cv::Mat rotated_kernel;
        imgRotateCutEdge(kernel3, rotated_kernel, rotate_angle_total, 0);
        kernels_mat_form.push_back(rotated_kernel);
    }

    for(auto &kernel_mat : kernels_mat_form){
        Eigen::MatrixXf kernel_matrix = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y);
        for (int i = 0; i < kernel_mat.rows; i++) {
            uchar *row_in_img = kernel_mat.ptr<uchar>(i);
            for (int j = 0; j < kernel_mat.cols; j++) {
                if(row_in_img[j] > 150){
                    kernel_matrix(i, j) = 1.f;
                }else if(row_in_img[j] > 0){
                    kernel_matrix(i, j) = -0.5f;
                }
            }
        }
        kernels.push_back(kernel_matrix);
//        std::cout <<std::endl << kernel_matrix <<std::endl;

//        cv::imshow("kernel_mat", kernel_mat);
//        cv::waitKey();
    }
}

void getRotatedPointPosition(cv::Point &point, cv::Point &rotated_point, cv::Point &rotation_center, float rotation_angle)
{
    float angle_rad = rotation_angle / 180.f * CV_PI;
//    std::cout << "angle_rad="<<angle_rad<<std::endl;
    rotated_point.x = (point.x - rotation_center.x) * cos(angle_rad) +
                         (point.y - rotation_center.y) * sin(angle_rad) + rotation_center.x;
    rotated_point.y = -(point.x - rotation_center.x) * sin(angle_rad) +
                         (point.y - rotation_center.y) * cos(angle_rad) + rotation_center.y;
}

bool ifRectAvaliable(cv::Point p, int img_x_size, int img_y_size, int edge_x, int edge_y){
    edge_x = edge_x / 2;
    edge_y = edge_y / 2;
    if(p.x < edge_x || p.x > img_x_size - edge_x || p.y < edge_y || p.y > img_y_size - edge_y){
        return false;
    }else{
        return true;
    }
}

void generateTrainingData(std::string data_dir, int extra_negative_sample_num = 100)
{
    clock_t start_time, end_time;

    /// Read an XML file names for training
    std::vector<std::string> filenames;
    getFileNames(data_dir, filenames, ".xml");
    std::cout << "Found " << filenames.size() << " xml files in " << data_dir <<std::endl;

    /// Generate training dataset for gateways
    std::ofstream positive_data_file, negative_data_file;
    positive_data_file.open(data_dir +"positive_data.csv", std::ios::out);
    negative_data_file.open(data_dir +"negative_data.csv", std::ios::out);

    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);  // define kernels for cost maps
//    defineRotatedKernels(kernels);

    unsigned seed;  // Random generator seed for collecting extra negative samples
    seed = time(0);
    srand(seed);

    for(const auto& filename : filenames){
        /// Read xml
        std::string xml_path = data_dir + filename;
        std::string img_path;
        int img_width, img_height, img_depth;
        std::vector<Object> objects;
        readLabelIMGObjectDetectionXML(xml_path, img_path, img_width, img_height, img_depth, objects);

        /// Abort if there is no gateway
        int labeled_gateway_counter = 0;
        for(const auto &object : objects){
            if(object.label == "gateway"){
                labeled_gateway_counter ++;
            }
        }
        if(labeled_gateway_counter < 1) continue;

        /// Read image and find skeleton points
        cv::Mat img_in = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(img_in); // CHG

        std::vector<cv::Point> skeleton_points;
        findVoronoiSkeletonPoints(img_in, skeleton_points);  /// CHG
        if(skeleton_points.size() < 2){
            std::cout << "Found no skeleton_points in file " <<  img_path << ", skip!" << std::endl;
            continue;
        }

        /*** Consider only skeleton points later when generating extra data ***/
//        std::cout << "processing image "<< img_path << std::endl;

        /// Generate noised dat
        std::vector<MAP_XD> noised_cost_maps;
        std::vector<MAP_XD> noised_corresponding_angles;

        if(use_noised_data){
            cv::Mat noised_img;
            pepperAndSaltNoise(img_in, noised_img);

            std::vector<std::vector<cv::Mat>> noised_result_imgs;
            getScaledAndRotatedImgs(noised_img, noised_result_imgs, scale_factor, scale_times, rotate_angle, rotate_times); //4, 24
            getCostMaps(noised_result_imgs, scale_factor, rotate_angle, kernels, noised_cost_maps, noised_corresponding_angles);
        }


        /// Scale and rotate
        std::vector<std::vector<cv::Mat>> result_imgs;
        getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, scale_times, rotate_angle, rotate_times); //4, 24

//        std::cout<<"00"<<std::endl;

        /// Get cost maps
        std::vector<MAP_XD> cost_maps;
        std::vector<MAP_XD> corresponding_angles;
        getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

//        std::cout<<"11"<<std::endl;

        /// Add samples corresponding to the hand-labeled gateway position in csv files
        std::vector<Eigen::Vector2i> positive_sample_positions;
        for(const auto &object : objects){
            if(object.label == "gateway"){

                /// Use nearest skeleton point to correct hand label error
                cv::Point gateway_img_pos((object.x_min + object.x_max)/2, (object.y_min + object.y_max)/2);
                cv::Point nearest_skeleton;
                float nearest_dist = findNearestPoint(gateway_img_pos, skeleton_points, nearest_skeleton);
                if(nearest_dist > 3){
                    std::cout << "GVG point miss match hand labeled point, skip..." << std::endl;
                    break;
                }

                // Save rect for other training
                if(save_rect_images){
                    if(ifRectAvaliable(nearest_skeleton, img_in.cols, img_in.rows, rect_to_save_size_x, rect_to_save_size_y)){
                        cv::Mat rect_to_save = img_in(cv::Rect(nearest_skeleton.x-rect_to_save_size_x/2, nearest_skeleton.y-rect_to_save_size_y/2,
                                                               rect_to_save_size_x, rect_to_save_size_y));
                        cv::imwrite(data_dir + "positive_"+std::to_string(rect_save_counter) + ".png", rect_to_save);
                        rect_save_counter ++;
                    }
                }

                const int gateway_x = nearest_skeleton.y; ///Note the x in a image is y in a matrix
                const int gateway_y = nearest_skeleton.x;
                Eigen::Vector2i gateway_pos;
                gateway_pos << gateway_x, gateway_y;
                positive_sample_positions.push_back(gateway_pos);

//                cv::circle(img_in, cv::Point(gateway_pos(1), gateway_pos(0)), 2, cv::Scalar(0), 1);
//                cv::imshow("gateway", img_in);
//                cv::waitKey();

                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq < cost_maps[kernel_seq][gateway_x][gateway_y].size(); feature_seq++){
                        positive_data_file << cost_maps[kernel_seq][gateway_x][gateway_y][feature_seq] << ",";
                    }
                }
                positive_data_file << "\n";

                if(use_noised_data){  // Add samples from noised image
                    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                        for(int feature_seq=0; feature_seq < cost_maps[kernel_seq][gateway_x][gateway_y].size(); feature_seq++){
                            positive_data_file << noised_cost_maps[kernel_seq][gateway_x][gateway_y][feature_seq] << ",";
                        }
                    }
                    positive_data_file << "\n";
                }

            }
        }

//        std::cout<<"22"<<std::endl;
        /// Add some more negative samples from skeleton points, "extra_negative_sample_num" samples in one image
        int max_negative_sample_num = skeleton_points.size() - positive_sample_positions.size();
        extra_negative_sample_num = std::min(max_negative_sample_num, extra_negative_sample_num);

        int time_out_seq = 0;
        for(int neg_extra_sample_seq=0; neg_extra_sample_seq<extra_negative_sample_num; neg_extra_sample_seq++)
        {
            int random_seq = rand() % skeleton_points.size();

            Eigen::Vector2i point;
            point << skeleton_points[random_seq].y, skeleton_points[random_seq].x;

            if(!ifCloseToAnyPointInVector(point, positive_sample_positions, negative_data_save_dist_threshold)){

                // Save rect for other training
                if(save_rect_images){
                    if(ifRectAvaliable(skeleton_points[random_seq], img_in.cols, img_in.rows, rect_to_save_size_x, rect_to_save_size_y)){
                        cv::Mat rect_to_save = img_in(cv::Rect(skeleton_points[random_seq].x-rect_to_save_size_x/2, skeleton_points[random_seq].y-rect_to_save_size_y/2,
                                                               rect_to_save_size_x, rect_to_save_size_y));
                        cv::imwrite(data_dir + "negative_"+std::to_string(rect_save_counter) + ".png", rect_to_save);
                        rect_save_counter ++;
                    }
                }

                // Add to feature csv file
                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][point(0)][point(1)].size(); feature_seq++){
                        negative_data_file << cost_maps[kernel_seq][point(0)][point(1)][feature_seq] << ",";
                    }
                }
                negative_data_file << "\n";

                if(use_noised_data){  // Add samples from noised image
                    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                        for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][point(0)][point(1)].size(); feature_seq++){
                            negative_data_file << noised_cost_maps[kernel_seq][point(0)][point(1)][feature_seq] << ",";
                        }
                    }
                    negative_data_file << "\n";
                }

            }else{
                neg_extra_sample_seq --;
                time_out_seq ++;
            }

            if(time_out_seq > 100) {
                std::cout << "Time out error. Can not generate more negative samples." <<std::endl;
                break;
            }

        }

//        std::cout<<"33"<<std::endl;

        /// Add samples from rotated images ***********************
        if(use_rotated_data){
            std::vector<std::vector<cv::Mat>> rotate_result_imgs;
            const float Rotate_Angle = 52;
            const int Rotate_Times = 6;
            getScaledAndRotatedImgs(img_in, rotate_result_imgs, 1, 1, Rotate_Angle, Rotate_Times); //4, 24

            const int img_center_x = img_in.cols / 2;
            const int img_center_y = img_in.rows / 2;
            cv::Point img_center(img_center_x, img_center_y);

            for (int rotated_seq = 0; rotated_seq < rotate_result_imgs[0].size(); rotated_seq++) {

                /// Calculate costmaps
                std::vector<std::vector<cv::Mat>> rotate_result_imgs_rotate_scale_results;
                getScaledAndRotatedImgs(rotate_result_imgs[0][rotated_seq], rotate_result_imgs_rotate_scale_results, scale_factor, scale_times, rotate_angle, rotate_times);

                std::vector<MAP_XD> cost_maps_this_rotate;
                std::vector<MAP_XD> corresponding_angles_this_rotate;
                getCostMaps(rotate_result_imgs_rotate_scale_results, scale_factor, rotate_angle, kernels, cost_maps_this_rotate,
                            corresponding_angles_this_rotate);

                /// Calculate rotated skeletons
                std::vector<cv::Point> rotated_skeleton_points;
                for(auto & skeleton_point : skeleton_points){
                    cv::Point rotated_skeleton_point;
                    getRotatedPointPosition(skeleton_point, rotated_skeleton_point, img_center, Rotate_Angle * rotated_seq);

                    if(rotated_skeleton_point.x < EDGE_X || rotated_skeleton_point.x > rotate_result_imgs[0][rotated_seq].cols - EDGE_X ||
                       rotated_skeleton_point.y < EDGE_Y || rotated_skeleton_point.y > rotate_result_imgs[0][rotated_seq].rows - EDGE_Y){
                        continue;
                    }else{
                        rotated_skeleton_points.push_back(rotated_skeleton_point);
                    }
                }

                if(rotated_skeleton_points.size() < 5){
                    continue;
                }

//                for(auto &p : rotated_skeleton_points){
//                    cv::circle(rotate_result_imgs[0][rotated_seq], p, 1, cv::Scalar(0), 1);
//                }
//                cv::imshow("rotate_result_imgs[0][rotated_seq]", rotate_result_imgs[0][rotated_seq]);
//                cv::waitKey();

                std::vector<Eigen::Vector2i> positive_sample_positions_this_rotate;
                for (const auto &object : objects) {

                    if (object.label == "gateway") {
                        int gateway_x = (object.x_min + object.x_max) / 2;
                        int gateway_y = (object.y_min + object.y_max) / 2;

                        cv::Point gateway_image_position(gateway_x, gateway_y);
                        cv::Point gateway_rotated_image_position;
                        getRotatedPointPosition(gateway_image_position, gateway_rotated_image_position, img_center, Rotate_Angle * rotated_seq);

                        /// Use nearest skeleton point to correct hand label error
                        cv::Point gateway_img_pos(gateway_rotated_image_position.x, gateway_rotated_image_position.y);
                        cv::Point nearest_skeleton;
                        float nearest_dist = findNearestPoint(gateway_img_pos, skeleton_points, nearest_skeleton);
                        if(nearest_dist > 3){
                            std::cout << "GVG point miss match hand labeled point, skip..." << std::endl;
                            break;
                        }

                        // Save rect for other training
                        if(save_rect_images){
                            if(ifRectAvaliable(nearest_skeleton, rotate_result_imgs[0][rotated_seq].cols, rotate_result_imgs[0][rotated_seq].rows, rect_to_save_size_x, rect_to_save_size_y)){
                                cv::Mat rect_to_save = rotate_result_imgs[0][rotated_seq](cv::Rect(nearest_skeleton.x-rect_to_save_size_x/2, nearest_skeleton.y-rect_to_save_size_y/2,
                                                                       rect_to_save_size_x, rect_to_save_size_y));
                                cv::imwrite(data_dir + "positive_rotated_"+std::to_string(rect_save_counter) + ".png", rect_to_save);
                                rect_save_counter ++;
                            }
                        }

                        int gateway_x_this = nearest_skeleton.y; ///Note the x in a image is y in a matrix
                        int gateway_y_this = nearest_skeleton.x;

                        if(gateway_x_this < EDGE_X || gateway_x_this > cost_maps_this_rotate[0].size()-EDGE_X ||
                           gateway_y_this < EDGE_Y || gateway_y_this > cost_maps_this_rotate[0][0].size()-EDGE_Y){
                            continue;
                        }

//                        cv::circle(rotate_result_imgs[0][rotated_seq], cv::Point(gateway_y_this, gateway_x_this), 2, cv::Scalar(0), 1);
//                        cv::imshow("rotate_result_imgs[0][rotated_seq]", rotate_result_imgs[0][rotated_seq]);
//                        cv::waitKey();

                        Eigen::Vector2i gateway_pos;
                        gateway_pos << gateway_x_this, gateway_y_this;
                        positive_sample_positions_this_rotate.push_back(gateway_pos);

                        for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                            for (int feature_seq = 0; feature_seq <
                                                      cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this].size(); feature_seq++) {
                                positive_data_file
                                        << cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this][feature_seq]
                                        << ",";
                            }
                        }
                        positive_data_file << "\n";
                    }

                }

                /// Add some more negative samples, "extra_negative_sample_num" samples per rotated image
                int max_negative_sample_num = skeleton_points.size() - positive_sample_positions.size();
                extra_negative_sample_num = std::min(max_negative_sample_num, extra_negative_sample_num);
                int time_out_seq = 0;
                for (int neg_extra_sample_seq = 0;
                     neg_extra_sample_seq < extra_negative_sample_num; neg_extra_sample_seq++) {

                    int random_seq = rand() % rotated_skeleton_points.size();
                    Eigen::Vector2i point;
                    point << rotated_skeleton_points[random_seq].y, rotated_skeleton_points[random_seq].x;

                    if (!ifCloseToAnyPointInVector(point, positive_sample_positions_this_rotate, negative_data_save_dist_threshold)) {

                        // Save rect for other training
                        if(save_rect_images){
                            if(ifRectAvaliable(skeleton_points[random_seq], rotate_result_imgs[0][rotated_seq].cols, rotate_result_imgs[0][rotated_seq].rows, rect_to_save_size_x, rect_to_save_size_y)){
                                cv::Mat rect_to_save = rotate_result_imgs[0][rotated_seq](cv::Rect(skeleton_points[random_seq].x-rect_to_save_size_x/2, skeleton_points[random_seq].y-rect_to_save_size_y/2,
                                                                       rect_to_save_size_x, rect_to_save_size_y));
                                cv::imwrite(data_dir + "negative_rotated_"+std::to_string(rect_save_counter) + ".png", rect_to_save);
                                rect_save_counter ++;
                            }
                        }

                        for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                            for (int feature_seq = 0;
                                 feature_seq < cost_maps_this_rotate[kernel_seq][point(0)][point(1)].size(); feature_seq++) {
                                negative_data_file << cost_maps_this_rotate[kernel_seq][point(0)][point(1)][feature_seq] << ",";
                            }
                        }
                        negative_data_file << "\n";
                    } else {
                        neg_extra_sample_seq--;
                        time_out_seq ++;
                    }

                    if(time_out_seq > 100) {
                        std::cout << "Time out error. Can not generate more negative samples." <<std::endl;
                        break;
                    }
                }

                /// ***********************************
            }

        }
    }

    positive_data_file.close();
    negative_data_file.close();
}

void countCSVSize(const std::string &csv_file, int &line_num, int &elements_in_one_line)
{
    line_num = 0;
    elements_in_one_line = 0;

    std::ifstream file(csv_file);
    std::string line;
    while(getline(file, line)){
        std::istringstream data(line);
        std::string data_string;
        if(line_num == 0){
            while(getline(data, data_string, ',')){
                elements_in_one_line ++;
            }
        }
        line_num ++;
    }
}

void readCSVstoMat(const std::string positive_samples_csv_path, const std::string negative_samples_csv_path, cv::Mat &mat_data, cv::Mat &label)
{
    int line_num_positive, elements_in_one_line_positive;
    countCSVSize(positive_samples_csv_path, line_num_positive, elements_in_one_line_positive);
    std::cout << "positive samples line_num=" << line_num_positive << " elements_in_one_line="<<elements_in_one_line_positive<<std::endl;

    int line_num_negative, elements_in_one_line_negative;
    countCSVSize(negative_samples_csv_path, line_num_negative, elements_in_one_line_negative);
    std::cout << "negative samples line_num=" << line_num_negative << " elements_in_one_line="<<elements_in_one_line_negative<<std::endl;

    if(elements_in_one_line_negative != elements_in_one_line_positive){
        std::cout << "Error: the feature vector in both csv files should has the same size" << std::endl;
        return;
    }

    /// Define container
    const int data_size_x = line_num_positive + line_num_negative;
    const int data_size_y = elements_in_one_line_positive;
    mat_data = cv::Mat::zeros(data_size_x, data_size_y, CV_32FC1);
    label = cv::Mat::zeros(data_size_x, 1, CV_32SC1);
    int positive_tag = 1, negative_tag = -1;
    int line_seq = 0;

    /// Read positive data
    std::ifstream positive_file(positive_samples_csv_path);
    std::string positive_line;
    while(getline(positive_file, positive_line)){
        std::istringstream data(positive_line);
        std::string data_string;
        int element_seq = 0;
        while(getline(data, data_string, ',')){
            std::stringstream ss;
            ss << data_string;
            float dd;
            ss >> dd;
            mat_data.at<float>(line_seq, element_seq) = dd;
            element_seq ++;
        }
        label.at<int>(line_seq, 0) = positive_tag;
        line_seq ++;
    }

    /// Read negative data
    std::ifstream negative_file(negative_samples_csv_path);
    std::string negative_line;
    while(getline(negative_file, negative_line)){
        std::istringstream data(negative_line);
        std::string data_string;
        int element_seq = 0;
        while(getline(data, data_string, ',')){
            std::stringstream ss;
            ss << data_string;
            float dd;
            ss >> dd;
            mat_data.at<float>(line_seq, element_seq) = dd;
            element_seq ++;
        }
        label.at<int>(line_seq, 0) = negative_tag;
        line_seq ++;
    }

//    std::cout << "-------------------------------------" <<std::endl;
//    std::cout << label <<std::endl;
//    std::cout << "-------------------------------------" <<std::endl;
}


void trainingSVM(cv::Ptr<cv::ml::SVM> &model, cv::Mat &data, cv::Mat &labels)
{
//    std::cout << data << std::endl;
//    std::cout << "label 187=" << labels[186]<<" ,label 188=" <<labels[188] << std::endl;
    model->setType(cv::ml::SVM::C_SVC);
    model->setKernel(cv::ml::SVM::RBF);
    model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, FLT_EPSILON));

    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);

    std::cout << "SVM: start train ..." << std::endl;
    model->trainAuto(tData);
    std::cout << "SVM: train success ..." << std::endl;
}

void predictSVM(cv::Ptr<cv::ml::SVM> &model, cv::Mat &test, cv::Mat &result)
{
    float rst = model->predict(test, result);
    for (int i = 0; i < result.rows; i++){
        std::cout << result.at<float>(i, 0)<<" ";
    }
    std::cout<<"\n rst is " << rst << std::endl;
}

float validateSVM(cv::Ptr<cv::ml::SVM> &model, cv::Mat &test, cv::Mat &ground_truth)
{
    cv::Mat predicted_result;
    predictSVM(model, test, predicted_result);
    int right_num = 0, wrong_num = 0;
    for(int i=0; i<predicted_result.rows; i++)
    {
        std::cout << ground_truth.at<int>(i,0) << " ";
        if(predicted_result.at<int>(i,0) * ground_truth.at<int>(i,0) > 0){
            right_num ++;
        }else{
            wrong_num ++;
        }
    }
    float correct_ratio = (float)right_num / (right_num+wrong_num);
    std::cout << std::endl<<"right_num="<<right_num<<", wrong_num="<<wrong_num<<std::endl;
    std::cout << "validation correct_ratio is " << correct_ratio << std::endl;
    return correct_ratio;
}

void imgTest(cv::Ptr<cv::ml::SVM> &model, cv::Mat &img_in)
{
    /// Extract skeleton points
    std::vector<cv::Point> skeleton_points;
    findVoronoiSkeletonPoints(img_in, skeleton_points);  /// CHG
    if(skeleton_points.size() < 1){
        std::cout << "Found no skeleton_points in this image, skip!" << std::endl;
        return;
    }

    /// define kernels for cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);
//    defineRotatedKernels(kernels);

    /// Scale and rotate
    std::vector<std::vector<cv::Mat>> result_imgs;
    getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, scale_times, rotate_angle, rotate_times); //4, 24

    /// Get cost maps
    std::vector<MAP_XD> cost_maps;
    std::vector<MAP_XD> corresponding_angles;

    getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

    /// Transform to SVM input data
    int pixels_num_for_test = cost_maps[0].size()*cost_maps[0][0].size();
    int feature_length = cost_maps.size()*cost_maps[0][0][0].size();
    cv::Mat test_mat = cv::Mat::zeros(pixels_num_for_test, feature_length, CV_32FC1);

    for(int row_seq=0; row_seq<skeleton_points.size(); row_seq++){
        int cost_seq_x = skeleton_points[row_seq].y;
        int cost_seq_y = skeleton_points[row_seq].x;
        for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
            for(int feature_seq=0; feature_seq<cost_maps[0][0][0].size(); feature_seq++){
                int col_seq = kernel_seq*cost_maps[0][0][0].size()+feature_seq;
                test_mat.at<float>(row_seq, col_seq) = cost_maps[kernel_seq][cost_seq_x][cost_seq_y][feature_seq];
            }
        }
    }

    /// Predict
    clock_t start_time, end_time;
    cv::Mat result;
    start_time = clock();
    float rst = model->predict(test_mat, result);
    end_time = clock();
    std::cout << "Predict Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    /// Draw the result
    int gateway_num = 0;
    for(int row_seq=0; row_seq<skeleton_points.size(); row_seq++){
        if(result.at<int>(row_seq,0) > 0){
            cv::circle(img_in, skeleton_points[row_seq], 2, cv::Scalar(0), 1);
            gateway_num ++;
        }
    }
    std::cout << "Found " << gateway_num <<" gateways" << std::endl;
}

int main(int argc, char** argv)
{
    unsigned seed;  // Random generator seed for collecting extra negative samples
    seed = time(0);
    srand(seed);

    for(int training_times=0; training_times < 20; training_times++){
        std::cout<<"************************************************" <<std::endl;
        std::cout << "start " << training_times << " times training" << std::endl;
        int maximum_extra_sample = rand() % 10 + 3;
        std::cout << "maximum_extra_sample = " << maximum_extra_sample << std::endl;
        rect_save_counter = 0; //reset counter to avoid save the same images

        /// Generating training data
        std::string training_data_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";


        generateTrainingData(training_data_path, maximum_extra_sample);

        /// Read training data
        cv::Mat data;
        cv::Mat labels;
        readCSVstoMat(training_data_path+"positive_data.csv",
                      training_data_path+"negative_data.csv",
                      data, labels);

        /// Training
        cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
        trainingSVM(model, data, labels);

        /// Save model
        model->save("svm_model.xml");

        ///Validating
        std::string validation_data_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";
        generateTrainingData(validation_data_path, 3);
        cv::Mat validation_data;
        cv::Mat validation_labels;
        readCSVstoMat(validation_data_path+"positive_data.csv",
                      validation_data_path+"negative_data.csv",
                      validation_data, validation_labels);

        float success_ratio = validateSVM(model, validation_data, validation_labels);
        if(success_ratio > 0.9){
            break;
        }
    }


//    validateSVM(model, data, labels);

    /// Load model
    cv::Ptr<cv::ml::SVM> model_loaded = cv::ml::StatModel::load<cv::ml::SVM>("svm_model.xml");

    /// Testing with images
    std::string test_images_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";  // Floor2_test
    std::vector<std::string> test_image_names;
    getFileNames(test_images_path, test_image_names, ".png");
    clock_t start_time, end_time;

    for(const auto &image_to_test : test_image_names) {
        cv::Mat img_in = cv::imread(test_images_path + image_to_test, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(img_in);

        start_time = clock();
        imgTest(model_loaded, img_in);
        end_time = clock();
        std::cout << "Time = " << (double) (end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("img_in", img_in);
        cv::waitKey();
    }

//    for(const auto &image_to_test : test_image_names) {
//        cv::Mat img_in = cv::imread(test_images_path + image_to_test, cv::IMREAD_GRAYSCALE);
//        turnBlacktoGray(img_in);
//        std::vector<std::vector<cv::Mat>> transformed_imgs;
//        getScaledAndRotatedImgs(img_in, transformed_imgs, scale_factor, scale_times, rotate_angle, rotate_times);
//
//        for(int i=0; i<transformed_imgs.size();i++){
//            for(int j=0; j<transformed_imgs[0].size(); j++){
//                start_time = clock();
//                imgTest(model_loaded, transformed_imgs[i][j]);
//                end_time = clock();
//                std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//
//                cv::imshow("img_in", transformed_imgs[i][j]);
//                cv::waitKey();
//            }
//        }
//    }

    std::cout << "Bye" << std::endl;


    return 0;
}
