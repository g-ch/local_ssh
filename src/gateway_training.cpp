//
// Created by cc on 2020/3/16.
//

#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include "labelimg_xml_reader.h"
#include <dirent.h>
#include <fstream>
#include <cstdlib>
#include <ctime>

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

bool ifCloseToAnyPointInVector(Eigen::Vector2i p, const std::vector<Eigen::Vector2i>& v, const float threshold){
    for(const auto &point : v){
        float dist = sqrt( (p(0)-point(0))^2 + (p(1)-point(1))^2);
        if(dist <= threshold){
            return true;
        }
    }

    return false;
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
//    std::cout << std::endl << kernel1 << std::endl;
    kernels.push_back(kernel1);

    /*Kernel 2*/
    Eigen::MatrixXf kernel2(KERNEL_X,KERNEL_Y); //11 rows, 21 cols
    kernel2.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernel2.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel2.block(0, 15, 13, 6) = Eigen::MatrixXf::Constant(13, 6, 1.f);
    kernel2.block(7, 0, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
//    std::cout << std::endl << kernel2 << std::endl;
    kernels.push_back(kernel2);

    /*Kernel 3*/
    Eigen::MatrixXf kernel3(KERNEL_X, KERNEL_Y); //11 rows, 21 cols
    kernel3.block(0, 0, 13, 6) = Eigen::MatrixXf::Constant(13, 6,1.f);
    kernel3.block(0, 6, 7, 9) = Eigen::MatrixXf::Constant(7, 9,-0.5);
    kernel3.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6, 1.f);
    kernel3.block(7, 6, 6, 15) = Eigen::MatrixXf::Constant(6, 15, -0.5);
//    std::cout << std::endl << kernel3 << std::endl;
    kernels.push_back(kernel3);
}

int main(int argc, char** argv)
{
    clock_t start_time, end_time;

    /// Read an XML file names for training
    std::string data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";
    std::vector<std::string> filenames;
    getFileNames(data_dir, filenames, ".xml");

    /// Generate training dataset for gateways
    std::ofstream positive_data_file, negative_data_file;
    positive_data_file.open(data_dir +"positive_data.csv", std::ios::out);
    negative_data_file.open(data_dir +"negative_data.csv", std::ios::out);

    // define kernels for cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);

    unsigned seed;  // Random generator seed
    seed = time(0);
    srand(seed);

    for(const auto& filename : filenames){
        /// Read xml and image
        std::string xml_path = data_dir + filename;
        std::string img_path;
        int img_width, img_height, img_depth;
        std::vector<Object> objects;
        readLabelIMGObjectDetectionXML(xml_path, img_path, img_width, img_height, img_depth, objects);

        cv::Mat img_in = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        /// Scale and rotate
        std::vector<std::vector<cv::Mat>> result_imgs;
        float scale_factor = 0.8;
        float rotate_angle = 30;
        getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, 3, rotate_angle, 12); //4, 24

        /// Get cost maps
        // Keep minimum three costs (using definition MAP_3D) and their corresponding angles
        std::vector<MAP_3D> cost_maps;
        std::vector<MAP_3D> corresponding_angles;

        getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

        std::vector<Eigen::Vector2i> positive_sample_positions;

        for(const auto &object : objects){
            if(object.label == "gateway"){
                const int gateway_x = (object.y_min + object.y_max)/2; ///Note the x in a image is y in a matrix
                const int gateway_y = (object.x_min + object.x_max)/2;
                Eigen::Vector2i gateway_pos;
                gateway_pos << gateway_x, gateway_y;
                positive_sample_positions.push_back(gateway_pos);

//                cv::circle(img_in, cv::Point(gateway_pos(1), gateway_pos(0)), 2, cv::Scalar(0), 1);
//                cv::imshow("gateway", img_in);
//                cv::waitKey();

                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][gateway_x][gateway_y].size(); feature_seq++){
                        positive_data_file << cost_maps[kernel_seq][gateway_x][gateway_y][feature_seq] << ",";
                    }
                }
                positive_data_file << "\n";

            }else{
                const int other_label_x = (object.y_min + object.y_max)/2;
                const int other_label_y = (object.x_min + object.x_max)/2;

                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][other_label_x][other_label_y].size(); feature_seq++){
                        negative_data_file << cost_maps[kernel_seq][other_label_x][other_label_y][feature_seq] << ",";
                    }
                }
                negative_data_file << "\n";
            }
        }

        /// Add some more negative samples, two samples in one image
        for(int neg_extra_sample_seq=0; neg_extra_sample_seq<2; neg_extra_sample_seq++)
        {
            int pos_x = rand() % img_height;
            int pos_y = rand() % img_width;
            Eigen::Vector2i point;
            point << pos_x, pos_y;
            if(!ifCloseToAnyPointInVector(point, positive_sample_positions, 5)){
                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][pos_x][pos_y].size(); feature_seq++){
                        negative_data_file << cost_maps[kernel_seq][pos_x][pos_y][feature_seq] << ",";
                    }
                }
                negative_data_file << "\n";
            }else{
                neg_extra_sample_seq --;
            }

        }

    }

    positive_data_file.close();
    negative_data_file.close();


    std::cout << "Bye" << std::endl;
    return 0;
}
