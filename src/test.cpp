//
// Created by cc on 2020/3/12.
//

#include <ros/ros.h>
#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>

#define KERNEL_X 11
#define KERNEL_Y 19

void defineKernels(std::vector<Eigen::MatrixXf> &kernels)
{
    /// Kernel size should all be 11x19, if not, an edge adjustment is to be done when generating cost maps
    /*Kernel 1*/
    Eigen::MatrixXf kernel1(KERNEL_X, KERNEL_Y); //11 rows, 19 cols
    kernel1.block(0, 0, 6, 5) = Eigen::MatrixXf::Constant(6, 5,1.f);
    kernel1.block(0, 5, 6, 9) = Eigen::MatrixXf::Constant(6, 9,-0.5);
    kernel1.block(0, 14, 6, 5) = Eigen::MatrixXf::Constant(6, 5, 1.f);
    kernel1.block(6, 0, 5, 19) = Eigen::MatrixXf::Constant(5, 19, -0.5);
    std::cout << std::endl << kernel1 << std::endl;
    kernels.push_back(kernel1);

    /*Kernel 2*/
    Eigen::MatrixXf kernel2(KERNEL_X,KERNEL_Y); //11 rows, 19 cols
    kernel2.block(0, 0, 6, 5) = Eigen::MatrixXf::Constant(6, 5,1.f);
    kernel2.block(0, 5, 6, 9) = Eigen::MatrixXf::Constant(6, 9,-0.5);
    kernel2.block(0, 14, 11, 5) = Eigen::MatrixXf::Constant(11, 5, 1.f);
    kernel2.block(6, 0, 5, 14) = Eigen::MatrixXf::Constant(5, 14, -0.5);
    std::cout << std::endl << kernel2 << std::endl;
    kernels.push_back(kernel2);

    /*Kernel 3*/
    Eigen::MatrixXf kernel3(KERNEL_X, KERNEL_Y); //11 rows, 19 cols
    kernel3.block(0, 0, 11, 5) = Eigen::MatrixXf::Constant(11, 5,1.f);
    kernel3.block(0, 5, 6, 9) = Eigen::MatrixXf::Constant(6, 9,-0.5);
    kernel3.block(0, 14, 6, 5) = Eigen::MatrixXf::Constant(6, 5, 1.f);
    kernel3.block(6, 5, 5, 14) = Eigen::MatrixXf::Constant(5, 14, -0.5);
    std::cout << std::endl << kernel3 << std::endl;
    kernels.push_back(kernel3);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detect_test");
    clock_t start_time, end_time;

    /// Read image
//    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811112649.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811114344.png", cv::IMREAD_GRAYSCALE);

    start_time = clock();
    /// Scale and rotate
    std::vector<std::vector<cv::Mat>> result_imgs;
    float scale_factor = 0.8;
    float rotate_angle = 15;
    getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, 4, rotate_angle, 24);
    end_time = clock();

    /// Get cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);

    // Keep minimum three costs (using definition MAP_3D) and their corresponding angles
    std::vector<MAP_3D> cost_maps;
    std::vector<MAP_3D> corresponding_angles;

    getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

    std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;


//    /// Display. Note the minimum display size for cv::imshow is 73x73, small image will be scaled
//    for(auto &imgs_i : result){
//        for(auto &img : imgs_i){
//            std::cout << "img_size=" << img.cols << std::endl;
//            cv::imshow("test", img);
//            cv::waitKey();
//        }
//    }

//    Eigen::MatrixXi matrix_in(img_in.rows, img_in.cols);
//    cv::cv2eigen(img_in, matrix_in);

    std::cout << "Bye" << std::endl;
    return 0;
}
