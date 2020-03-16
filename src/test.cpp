//
// Created by cc on 2020/3/12.
//

#include <ros/ros.h>
#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include "labelimg_xml_reader.h"
#include <dirent.h>
#include <sys/types.h>

#define KERNEL_X 13
#define KERNEL_Y 21

void getFileNames(std::string path, std::vector<std::string>& filenames, std::string required_type=".all")
{
    /// The required_type should be like ".jpg" or ".xml".
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }

    while((ptr = readdir(pDir)) != 0){
        std::string file_name_temp = ptr->d_name;
        if(required_type==".all"){
            filenames.push_back(file_name_temp);
        }else{
            std::string::size_type position;
            position = file_name_temp.find(required_type);
            if(position != file_name_temp.npos){
                std::string file_name_temp2 = file_name_temp.substr(0, position) + required_type;
                std::cout << file_name_temp2 << std::endl;
                filenames.push_back(file_name_temp2);
            }
        }

    }
    closedir(pDir);
}



void defineKernels(std::vector<Eigen::MatrixXf> &kernels)
{
    /// Kernel size should all be 13x21, if not, an edge adjustment is to be done when generating cost maps
    /*Kernel 1*/
    Eigen::MatrixXf kernel1(KERNEL_X, KERNEL_Y); //11 rows, 21 cols
    kernel1.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernel1.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel1.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6, 1.f);
    kernel1.block(7, 0, 6, 21) = Eigen::MatrixXf::Constant(6, 21, -0.5);
    std::cout << std::endl << kernel1 << std::endl;
    kernels.push_back(kernel1);

    /*Kernel 2*/
    Eigen::MatrixXf kernel2(KERNEL_X,KERNEL_Y); //11 rows, 21 cols
    kernel2.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernel2.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel2.block(0, 15, 13, 6) = Eigen::MatrixXf::Constant(13, 6, 1.f);
    kernel2.block(7, 0, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
    std::cout << std::endl << kernel2 << std::endl;
    kernels.push_back(kernel2);

    /*Kernel 3*/
    Eigen::MatrixXf kernel3(KERNEL_X, KERNEL_Y); //11 rows, 21 cols
    kernel3.block(0, 0, 13, 6) = Eigen::MatrixXf::Constant(13, 6,1.f);
    kernel3.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel3.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6, 1.f);
    kernel3.block(7, 6, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
    std::cout << std::endl << kernel3 << std::endl;
    kernels.push_back(kernel3);
}

int main(int argc, char** argv)
{
//    ros::init(argc, argv, "detect_test");
    clock_t start_time, end_time;

    /// Test to read an XML file
    std::vector<std::string> filenames;
    getFileNames("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2", filenames, ".xml");



    std::string xml_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811112644.xml";
    std::string img_path;
    int img_width, img_height, img_depth;
    std::vector<Object> objects;
    readLabelIMGObjectDetectionXML(xml_path, img_path, img_width, img_height, img_depth, objects);


    /// Read image
//    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811112649.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811114344.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Created1/pow7resolution1.000000T120281212722.png", cv::IMREAD_GRAYSCALE);

    cv::Mat img_in = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    start_time = clock();
    /// Scale and rotate
    std::vector<std::vector<cv::Mat>> result_imgs;
    float scale_factor = 0.8;
    float rotate_angle = 30;
    getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, 3, rotate_angle, 12); //4, 24

    /// Get cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);

    // Keep minimum three costs (using definition MAP_3D) and their corresponding angles
    std::vector<MAP_3D> cost_maps;
    std::vector<MAP_3D> corresponding_angles;

    getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

    end_time = clock();
    std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    /// Transform again to matrixs for display
    std::vector<Eigen::MatrixXf> cost_map_matrixs;

    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++)
    {
        Eigen::MatrixXf cost_map_matrix_this = Eigen::MatrixXf::Zero(result_imgs[0][0].rows, result_imgs[0][0].cols);
        for(int i=0; i<result_imgs[0][0].rows; i++){
            for(int j=0; j<result_imgs[0][0].cols; j++)
            {
                cost_map_matrix_this(i, j) = cost_maps[kernel_seq][i][j][0];
            }
        }
//        std::cout  << std::endl<<  cost_map_matrix_this << std::endl;
        cost_map_matrixs.push_back(cost_map_matrix_this);
    }

    for(int k=0; k<cost_map_matrixs.size(); k++){
        Eigen::Vector2i min_point;
        float min_value_k = cost_map_matrixs[k].minCoeff(&min_point(0), &min_point(1));
        std::cout << "rotate_angle for kernel " << k <<" is " <<  corresponding_angles[k][min_point(0)][min_point(1)][0] << std::endl;
        std::cout << "min_value for kernel " << k <<" is " << min_value_k << std::endl;
        cv::circle(img_in, cv::Point(min_point(1), min_point(0)), 2, cv::Scalar(0), 1);
    }

    cv::imshow("gateway", img_in);
    cv::waitKey();

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
