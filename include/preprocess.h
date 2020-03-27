//
// Created by cc on 2020/3/12.
//

#ifndef LOCAL_SSH_PREPROCESS_H
#define LOCAL_SSH_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Eigen/Eigen"
#include <opencv2/core/eigen.hpp>
#include <dirent.h>


typedef std::vector<std::vector<Eigen::Vector3f>> MAP_3D;


void turnBlacktoGray(cv::Mat &src){
    int nr=src.rows;
    int nc=src.cols;

    for(int i=0; i<nr ;i++){
        auto* in_src = src.ptr<uchar>(i); // float
        for(int j=0; j<nc; j++){
            if(in_src[j] < 127){
                in_src[j] = 127;
            }
        }
    }
}

void turnGraytoBlack(cv::Mat &src){
    int nr=src.rows;
    int nc=src.cols;

    for(int i=0; i<nr ;i++){
        auto* in_src = src.ptr<uchar>(i); // float
        for(int j=0; j<nc; j++){
            if(in_src[j] < 150){
                in_src[j] = 0;
            }
        }
    }
}

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
//                std::cout << file_name_temp2 << std::endl;
                filenames.push_back(file_name_temp2);
            }
        }

    }
    closedir(pDir);
}

void imgRotateCutEdge(cv::Mat &src,cv::Mat &dst,float angle, cv::Scalar edge_color = cv::Scalar(127))
{
    float radian = (float) (angle /180.0 * CV_PI);

    //填充图像
    int maxBorder =(int) (std::max(src.cols, src.rows)* 1.414 ); //即为sqrt(2)*max
    int dx = (maxBorder - src.cols)/2;
    int dy = (maxBorder - src.rows)/2;
    cv::copyMakeBorder(src, dst, dy, dy, dx, dx, cv::BORDER_CONSTANT, edge_color);

    //旋转
    cv::Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    cv::Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );//求得旋转矩阵
    cv::warpAffine(dst, dst, affine_matrix, dst.size(), cv::INTER_NEAREST);

    //剪掉多余边框
    int x = (dst.cols - src.cols) / 2;
    int y = (dst.rows - src.rows) / 2;
    cv::Rect rect(x, y, src.cols, src.rows);
    dst = cv::Mat(dst,rect);
}


void getScaledAndRotatedImgs(cv::Mat src, std::vector<std::vector<cv::Mat>> &result, float scale_factor, int scale_times, float rotate_angle, int rotate_times)
{
    /// Note: rotate_times=12 and rotate_angle=30 would cover 360 degree
    /// The original image scale and rotate angle will be included

    auto img_width = (float)src.cols;
    auto img_height = (float)src.rows;

    // Rotate the unscaled image first. The first rotate angle is 0 degree
    std::vector<cv::Mat> imgs_ori_scale;
    imgs_ori_scale.push_back(src);
    for(int j=1; j<rotate_times; j++)
    {
        cv::Mat rotated_img;
        imgRotateCutEdge(src, rotated_img, rotate_angle*j);
        imgs_ori_scale.push_back(rotated_img);
    }
    result.push_back(imgs_ori_scale);

    // Scale the image and rotate
    for(int i=1; i<scale_times; i++)
    {
        std::vector<cv::Mat> imgs_this_scale;
        cv::Mat scaled_img;
        cv::resize(src, scaled_img, cv::Size((int)(img_width * scale_factor), (int)(img_height * scale_factor)), 0, 0, CV_INTER_AREA);
        imgs_this_scale.push_back(scaled_img);

        for(int j=1; j<rotate_times; j++)
        {
            cv::Mat rotated_img;
            imgRotateCutEdge(scaled_img, rotated_img, rotate_angle*j);
            imgs_this_scale.push_back(rotated_img);
        }
        result.push_back(imgs_this_scale);

        /// For next scale
        img_width = img_width * scale_factor;
        img_height = img_height * scale_factor;
    }
}

Eigen::Vector2i getPositionBeforeOrientationAndScale(int image_x, int image_y, float scale_factor, int scale_times,
        float rotate_angle, int rotate_times, Eigen::Vector2i position_src)
{
    /// The original operation is first to scale and then to orientate
    int position_x_matrix_center_coordinate = position_src[0] - image_x / 2;
    int position_y_matrix_center_coordinate = position_src[1] - image_y / 2;

    float orientation_angle = rotate_angle * rotate_times / 180.f * CV_PI;
    float scale_total_factor = pow(1.f/scale_factor, scale_times);
    float img_ori_x = image_x * scale_total_factor;
    float img_ori_y = image_y * scale_total_factor;

    float orientated_x = cos(orientation_angle)*position_x_matrix_center_coordinate + sin(orientation_angle)*position_y_matrix_center_coordinate;
    float orientated_y = -sin(orientation_angle)*position_x_matrix_center_coordinate + cos(orientation_angle)*position_y_matrix_center_coordinate;
    float scaled_x = orientated_x * scale_total_factor;
    float scaled_y = orientated_y * scale_total_factor;

    float ori_position_x = scaled_x + img_ori_x/2;
    float ori_position_y = scaled_y + img_ori_y/2;
    ori_position_x = std::min(ori_position_x, img_ori_x-1.f); //Eliminate the influence from calculation precision
    ori_position_x = std::max(ori_position_x, 0.f);
    ori_position_y = std::min(ori_position_y, img_ori_y-1.f);
    ori_position_y = std::max(ori_position_y, 0.f);

    Eigen::Vector2i original_position;
    original_position << (int)(ori_position_x),  (int)(ori_position_y);
    return original_position;
}


void exchangeValues(float &a, float &b){
    float c = a;
    a = b;
    b = c;
}

bool assertAndRankinOneVectorandChangeAnotherVectorCorrespondingly(float value_to_insert, float corresponding_value,
        Eigen::Vector3f &vector_to_rank, Eigen::Vector3f &vector_to_change_correspondingly)
{
    /// seq 0->2, small->large
    /// Suppose vector_to_rank is well ranked before this function.
    if(value_to_insert < vector_to_rank[2]){
        vector_to_rank[2] = value_to_insert;
        vector_to_change_correspondingly[2] = corresponding_value;
    }else{
        return false;
    }
    if(vector_to_rank[2] < vector_to_rank[1]){
        exchangeValues(vector_to_rank[2], vector_to_rank[1]);
        exchangeValues(vector_to_change_correspondingly[2], vector_to_change_correspondingly[1]);
    }else{
        return true;
    }

    if(vector_to_rank[1] < vector_to_rank[0]){
        exchangeValues(vector_to_rank[1], vector_to_rank[0]);
        exchangeValues(vector_to_change_correspondingly[1], vector_to_change_correspondingly[0]);
    }else{
        return true;
    }
}

void getCostMaps(std::vector<std::vector<cv::Mat>> &transformed_images, float scale_factor, float rotate_angle,
        std::vector<Eigen::MatrixXf> &kernels, std::vector<MAP_3D> &cost_maps, std::vector<MAP_3D> &corresponding_angles)
{
    /// Size of all kernels should be the same and should be odds
    /// The resulted cost map has the same size as the image with no transformation
    const int kernel_size_x = kernels[0].rows();
    const int kernel_size_y = kernels[0].cols();
    const int cost_map_size_x = transformed_images[0][0].rows; //transformed_images[0][0] is the original image
    const int cost_map_size_y = transformed_images[0][0].cols;
    const int kernel_edge_x = (kernel_size_x - 1) / 2;
    const int kernel_edge_y = (kernel_size_y - 1) / 2;

    // Initialize
    for(int i=0; i<kernels.size();i++){
        MAP_3D cost_map(cost_map_size_x, std::vector< Eigen::Vector3f >(cost_map_size_y, Eigen::Vector3f::Zero()));
        cost_maps.push_back(cost_map);
    }
    for(int j=0; j<kernels.size();j++){
        MAP_3D corresponding_angle(cost_map_size_x, std::vector< Eigen::Vector3f >(cost_map_size_y, Eigen::Vector3f::Zero()));
        corresponding_angles.push_back(corresponding_angle);
    }

    // Calculate cost maps
    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++)
    {
        for(int scaled_times = 0; scaled_times < transformed_images.size(); scaled_times++)
        {
            for(int rotate_times = 0; rotate_times < transformed_images[scaled_times].size(); rotate_times++)
            {
                // Transform image to Eigen matrix
                int image_this_size_x = transformed_images[scaled_times][rotate_times].rows;
                int image_this_size_y = transformed_images[scaled_times][rotate_times].cols;

                Eigen::MatrixXf img_matrix_this(image_this_size_x, image_this_size_y);
                cv::cv2eigen(transformed_images[scaled_times][rotate_times], img_matrix_this);
                // Start to do convolution operations for each image
                for(int i=0; i<=image_this_size_x-kernel_size_x; i++){
                    for(int j=0; j<=image_this_size_y-kernel_size_y;j++){
                        Eigen::ArrayXXf convolution_no_sum = img_matrix_this.block(i, j, kernel_size_x, kernel_size_y).array() * kernels[kernel_seq].array();
                        float convolution_result = convolution_no_sum.sum(); //chg

                        // Get corresponding position in cost map
                        int kernel_center_position_x = i + kernel_edge_x;
                        int kernel_center_position_y = j + kernel_edge_y;
                        Eigen::Vector2i position_src;
                        position_src << kernel_center_position_x, kernel_center_position_y;


                        Eigen::Vector2i position = getPositionBeforeOrientationAndScale(image_this_size_x, image_this_size_y,
                                 scale_factor, scaled_times, rotate_angle, rotate_times, position_src);


                        // store the cost and corresponding
                        assertAndRankinOneVectorandChangeAnotherVectorCorrespondingly(convolution_result, rotate_times * rotate_angle,
                                cost_maps[kernel_seq][position[0]][position[1]], corresponding_angles[kernel_seq][position[0]][position[1]]);

                    }
                }
            }
        }
    }

    std::cout << "cost map finished" << std::endl;
}


#endif //LOCAL_SSH_PREPROCESS_H
