//
// Created by cc on 2020/3/12.
//

#ifndef LOCAL_SSH_PREPROCESS_H
#define LOCAL_SSH_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Eigen/Eigen"
#include <opencv2/core/eigen.hpp>
#include <cstdlib>

typedef std::vector<std::vector<Eigen::VectorXf>> MAP_XD;

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

void addFixedValueNoise(cv::Mat& image, int value, int max_num = 60){
    int number = rand() % max_num;
    for(int k=0; k<number; k++){
        int i = rand()%image.cols;
        int j = rand()%image.rows;

        if(image.channels() == 1){
            image.at<uchar>(j,i) = value;
        }else{
            image.at<cv::Vec3b>(j,i)[0] = value;
            image.at<cv::Vec3b>(j,i)[1] = value;
            image.at<cv::Vec3b>(j,i)[2] = value;
        }
    }
}

void pepperAndSaltNoise(cv::Mat &src, cv::Mat &dist) {
    src.copyTo(dist);
    addFixedValueNoise(dist, 255, 50);
    addFixedValueNoise(dist, 127, 100);
//    addFixedValueNoise(dist, 0, 30);
//    cv::imshow("dist", dist);
//    cv::waitKey();
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

void getCostMaps(std::vector<std::vector<cv::Mat>> &transformed_images, float scale_factor, float rotate_angle,
        std::vector<Eigen::MatrixXf> &kernels, std::vector<MAP_XD> &cost_maps, std::vector<MAP_XD> &corresponding_angles)
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
    int feature_size;
    bool add_extra_differential_features = true; ///CHG
    int neighbor_pixels_num = 4;

    if(add_extra_differential_features){
        feature_size = transformed_images.size() * transformed_images[0].size() + neighbor_pixels_num; /// Add differential with 8 neighbourhood pixels
    }else{
        feature_size = transformed_images.size() * transformed_images[0].size();
    }

    for(int i=0; i<kernels.size();i++){
        MAP_XD cost_map(cost_map_size_x, std::vector< Eigen::VectorXf >(cost_map_size_y, Eigen::VectorXf::Zero(feature_size)));
        cost_maps.push_back(cost_map);
    }
    for(int j=0; j<kernels.size();j++){
        MAP_XD corresponding_angle(cost_map_size_x, std::vector< Eigen::VectorXf >(cost_map_size_y, Eigen::VectorXf::Zero(feature_size)));
        corresponding_angles.push_back(corresponding_angle);   ///CHG corresponding_angles has not been concerned in the following part
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

                int feature_seq = scaled_times * transformed_images[scaled_times].size() + rotate_times;

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

                        cost_maps[kernel_seq][position[0]][position[1]][feature_seq] = convolution_result;

                    }
                }
            }
        }
    }


    /// Add differential
    if(add_extra_differential_features)
    {
        std::vector<std::vector<std::vector<float>>> minimum_costs;
        for(int i=0; i<kernels.size(); i++){
            std::vector<std::vector<float>> minimum_cost_one_kernel(cost_map_size_x, std::vector<float>(cost_map_size_y, 0.f));
            minimum_costs.push_back(minimum_cost_one_kernel);
        }

        for(int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++){
            for(int map_x=0; map_x < cost_map_size_x; map_x++){
                for(int map_y=0; map_y < cost_map_size_y; map_y++){
                    minimum_costs[kernel_seq][map_x][map_y] = cost_maps[kernel_seq][map_x][map_y].minCoeff();
                }
            }
        }

        int extra_feature_start_num = transformed_images.size() * transformed_images[0].size();
        for(int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++){
            for(int map_x=0; map_x < cost_map_size_x; map_x++){
                for(int map_y=0; map_y < cost_map_size_y; map_y++){
                    if(map_x == 0 || map_x == cost_map_size_x-1 || map_y == 0 || map_y == cost_map_size_y-1){  //edge
                        for(int i=0; i<neighbor_pixels_num; i++){
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num + i] = 0.f;
                        }
                    }else{
                        if(neighbor_pixels_num == 8){
                            // Top left
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x-1][map_y-1];
                            // Top
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+1] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x-1][map_y];
                            // Top right
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+2] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x-1][map_y+1];
                            // Right
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+3] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x][map_y+1];
                            // Right bottom
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+4] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x+1][map_y+1];
                            // Bottom
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+5] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x+1][map_y];
                            // Left bottom
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+6] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x+1][map_y-1];
                            // Left
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+7] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x][map_y-1];
                        }else{
                            // Top
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x-1][map_y];
                            // Right
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+3] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x][map_y+1];
                            // Bottom
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+1] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x+1][map_y];
                            // Left
                            cost_maps[kernel_seq][map_x][map_y][extra_feature_start_num+2] = minimum_costs[kernel_seq][map_x][map_y] - minimum_costs[kernel_seq][map_x][map_y-1];

                        }
                    }
                }
            }
        }
    }

    static bool print_feature_dimension = true;
    if(print_feature_dimension){
        std::cout << "feature_dimension = " << cost_maps[0][0][0].size() * kernels.size() << std::endl;
        print_feature_dimension = false;
    }

    std::cout << "cost map finished" << std::endl;
}


#endif //LOCAL_SSH_PREPROCESS_H
