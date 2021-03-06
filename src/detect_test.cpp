//
// Created by cc on 2020/3/9.
//

#include <iostream>
#include <string>
#include <cmath>
#include "Eigen/Eigen"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>


float kernel1Cal(Eigen::MatrixXi &region){
    /// Size of kernel 1 should be 11x19
    Eigen::MatrixXi block1 = region.block(0, 0, 6, 5);
    Eigen::MatrixXi block2 = region.block(0, 5, 6, 14);
    Eigen::MatrixXi block3 = region.block(0, 14, 6, 5);
    Eigen::MatrixXi block4 = region.block(6, 0, 5, 19);

    /// Convolution
    float obstacle_sum = block1.sum() + block3.sum();
    float freespace_sum = -0.5*(block2.sum() + block4.sum());

    return obstacle_sum + freespace_sum;
}

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


float kernel1MapGenerate(Eigen::MatrixXi &matrix_in, Eigen::Vector2i &min_value_position){

    Eigen::Vector2i kernel1;
    kernel1 << 11, 19;

    Eigen::MatrixXi kernel1_region(kernel1(0), kernel1(1));
    int kernel1_result_size_x = matrix_in.rows() - kernel1(0)+1;
    int kernel1_result_size_y = matrix_in.cols() - kernel1(1)+1;

    Eigen::MatrixXf kernel1_result_map(kernel1_result_size_x, kernel1_result_size_y);

    /// Calculate kernel map
    for(int i=0; i<kernel1_result_size_x; i++){
        for(int j=0; j<kernel1_result_size_y; j++){
            kernel1_region = matrix_in.block(i, j, kernel1(0), kernel1(1));
            kernel1_result_map(i, j) = kernel1Cal(kernel1_region);
        }
    }

    float min_value = kernel1_result_map.minCoeff(&min_value_position(0), &min_value_position(1));

    /// Show related matrix
    std::cout<<"----------------------------------"<< std::endl;
    std::cout<< matrix_in<< std::endl;
    std::cout<<"----------------------------------"<< std::endl;
    std::cout<< kernel1_result_map<< std::endl;
    std::cout<<"----------------------------------"<< std::endl;
    std::cout<< matrix_in.block(min_value_position(0), min_value_position(1), kernel1(0), kernel1(1)) << std::endl;
    std::cout<<"----------------------------------"<< std::endl;

    /// Correct min_value_position to center point
    min_value_position(0) += kernel1(0) / 2;
    min_value_position(1) += kernel1(1) / 2;

    return min_value;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detect_test");

    /// Read image
//    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811112649.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811114344.png", cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXi matrix_in(img_in.rows, img_in.cols);
    cv::cv2eigen(img_in, matrix_in);

    Eigen::Vector2i min_point;
    float min_value = kernel1MapGenerate(matrix_in, min_point);
    std::cout << "Min value=" << min_value << " Position=(" << min_point(1) <<", "<<min_point(0)<<")"<<std::endl;

    /// Shrink test
    float shrink_coefficient = 1.5;
    cv::Mat img_resize;
    cv::resize(img_in, img_resize, cv::Size((int)(img_in.cols / shrink_coefficient), (int)(img_in.rows / shrink_coefficient)), 0, 0, CV_INTER_AREA);

    Eigen::MatrixXi matrix_in2(img_resize.rows, img_resize.cols);
    cv::cv2eigen(img_resize, matrix_in2);

    Eigen::Vector2i min_point2;
    float min_value2 = kernel1MapGenerate(matrix_in2, min_point2);
    std::cout << "Min value2=" << min_value2 << " Position2=(" << min_point2(1) <<", "<<min_point2(0)<<")"<<std::endl;

    /// Rotate test
    cv::Mat img_rotate;
    imgRotateCutEdge(img_in, img_rotate, -30);
    cv::imshow("img_rotate", img_rotate);


    /// display
    cv::circle(img_in, cv::Point(min_point(1), min_point(0)), 2, cv::Scalar(0), 2);
    cv::imshow("test", img_in);

    cv::circle(img_resize, cv::Point(min_point2(1), min_point2(0)), 2, cv::Scalar(0), 2);
    cv::imshow("shrink", img_resize);
    cv::waitKey();

    return 0;

}

