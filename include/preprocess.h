//
// Created by cc on 2020/3/12.
//

#ifndef LOCAL_SSH_PREPROCESS_H
#define LOCAL_SSH_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Eigen/Eigen"
#include <opencv2/core/eigen.hpp>

typedef std::vector<std::vector<Eigen::Vector3f>> MAP_3D;

void imgRotateCutEdge(cv::Mat &src,cv::Mat &dst,float angle)
{
    float radian = (float) (angle /180.0 * CV_PI);

    //填充图像
    int maxBorder =(int) (std::max(src.cols, src.rows)* 1.414 ); //即为sqrt(2)*max
    int dx = (maxBorder - src.cols)/2;
    int dy = (maxBorder - src.rows)/2;
    cv::copyMakeBorder(src, dst, dy, dy, dx, dx, cv::BORDER_CONSTANT, cv::Scalar(127));

    //旋转
    cv::Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    cv::Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );//求得旋转矩阵
    cv::warpAffine(dst, dst, affine_matrix, dst.size());

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


void getCostMaps(std::vector<std::vector<cv::Mat>> &transformed_images, float scale_factor, float rotate_angle,
        std::vector<Eigen::MatrixXf> &kernels, std::vector<MAP_3D> &cost_maps, std::vector<MAP_3D> &corresponding_angles)
{
    /// Size of all kernels should be the same!
    int cost_map_size_x = transformed_images[0][0].rows - kernels[0].rows() + 1; //transformed_images[0][0] is the original image
    int cost_map_size_y = transformed_images[0][0].cols - kernels[0].cols() + 1;

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
                Eigen::MatrixXi img_matrix_this(transformed_images[scaled_times][rotate_times].rows, transformed_images[scaled_times][rotate_times].cols);
                cv::cv2eigen(transformed_images[scaled_times][rotate_times], img_matrix_this);
                // Start to do convolution operations for each image


            }
        }
    }

    std::cout << "done" << std::endl;
}


#endif //LOCAL_SSH_PREPROCESS_H
