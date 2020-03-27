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
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <opencv2/ml.hpp>
#include "voronoi_skeleton_points.h"

#define KERNEL_X 13
#define KERNEL_Y 21

const float scale_factor = 0.9;
const int scale_times = 3;
const float rotate_angle = 45;
const int rotate_times = 8;



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

    /*Kernel 3*/
//    Eigen::MatrixXf kernel4 = Eigen::MatrixXf::Ones(KERNEL_X, KERNEL_Y); //11 rows, 21 cols
////    std::cout << std::endl << kernel4 << std::endl;
//    kernels.push_back(kernel4);
}


void defineSumKernels(std::vector<Eigen::MatrixXf> &kernels)
{
    /*Kernel 1*/
    Eigen::MatrixXf kernel1 = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel1.block(0, 0, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernels.push_back(kernel1);

    /*Kernel 2*/
    Eigen::MatrixXf kernel2 = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel2.block(7, 0, 6, 6) = Eigen::MatrixXf::Constant(6, 6,1.f);
    kernels.push_back(kernel2);

    /*Kernel 3*/
    Eigen::MatrixXf kernel3 = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel3.block(0, 15, 7, 6) = Eigen::MatrixXf::Constant(7, 6,1.f);
    kernels.push_back(kernel3);

    /*Kernel 4*/
    Eigen::MatrixXf kernel4 = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel4.block(7, 15, 6, 6) = Eigen::MatrixXf::Constant(6, 6,1.f);
    kernels.push_back(kernel4);

    /*Kernel 5*/
    Eigen::MatrixXf kernel5 = Eigen::MatrixXf::Zero(KERNEL_X, KERNEL_Y); //13 rows, 21 cols
    kernel5.block(0, 6, 13, 9) = Eigen::MatrixXf::Constant(13, 9,1.f);
    kernels.push_back(kernel5);

}

void generateTrainingData(std::string data_dir)
{
    clock_t start_time, end_time;

    /// Read an XML file names for training
    std::vector<std::string> filenames;
    getFileNames(data_dir, filenames, ".xml");

    /// Generate training dataset for gateways
    std::ofstream positive_data_file, negative_data_file;
    positive_data_file.open(data_dir +"positive_data.csv", std::ios::out);
    negative_data_file.open(data_dir +"negative_data.csv", std::ios::out);

    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);  // define kernels for cost maps
//    defineSumKernels(kernels);

    unsigned seed;  // Random generator seed for collecting extra negative samples
    seed = time(0);
    srand(seed);

    for(const auto& filename : filenames){
        /// Read xml and image
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


        /// Scale and rotate
        std::vector<std::vector<cv::Mat>> result_imgs;
        getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, scale_times, rotate_angle, rotate_times); //4, 24

        /// Get cost maps
        // Keep minimum three costs (using definition MAP_3D) and their corresponding angles
        std::vector<MAP_3D> cost_maps;
        std::vector<MAP_3D> corresponding_angles;

        getCostMaps(result_imgs, scale_factor, rotate_angle, kernels, cost_maps, corresponding_angles);

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

        /// Add some more negative samples, 120 samples in one image
        int extra_negative_sample_num = 50;
        int max_negative_sample_num = skeleton_points.size() - positive_sample_positions.size();
        extra_negative_sample_num = std::min(max_negative_sample_num, extra_negative_sample_num);

        int time_out_seq = 0;
        for(int neg_extra_sample_seq=0; neg_extra_sample_seq<extra_negative_sample_num; neg_extra_sample_seq++)
        {
            int random_seq = rand() % skeleton_points.size();

            Eigen::Vector2i point;
            point << skeleton_points[random_seq].y, skeleton_points[random_seq].x;

            if(!ifCloseToAnyPointInVector(point, positive_sample_positions, 3)) {
                // Add to feature csv file
                for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                    for (int feature_seq = 0;
                         feature_seq < cost_maps[kernel_seq][point(0)][point(1)].size(); feature_seq++) {
                        negative_data_file << cost_maps[kernel_seq][point(0)][point(1)][feature_seq] << ",";
                    }
                }
                negative_data_file << "\n";
            }else{
                neg_extra_sample_seq --;
                time_out_seq ++;
            }

            if(time_out_seq > 100) {
                std::cout << "Time out error. Can not generate more negative samples." <<std::endl;
                break;
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
    model->setKernel(cv::ml::SVM::RBF);  //核函数，这里使用线性核
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
//    defineSumKernels(kernels);

    /// Scale and rotate
    std::vector<std::vector<cv::Mat>> result_imgs;
    getScaledAndRotatedImgs(img_in, result_imgs, scale_factor, scale_times, rotate_angle, rotate_times); //4, 24

    /// Get cost maps
    // Keep minimum three costs (using definition MAP_3D) and their corresponding angles
    std::vector<MAP_3D> cost_maps;
    std::vector<MAP_3D> corresponding_angles;

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

    /// Generating training data
    std::string training_data_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";

    generateTrainingData(training_data_path);

    /// Read training data
    cv::Mat data;
    cv::Mat labels;
    readCSVstoMat(training_data_path+"positive_data.csv",
                  training_data_path+"negative_data.csv",
                  data, labels);

    /// Training
    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
    trainingSVM(model, data, labels);

    ///Validating
    validateSVM(model, data, labels);

    /// Testing with images
    std::string test_images_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";
    std::vector<std::string> test_image_names;
    getFileNames(test_images_path, test_image_names, ".png");
    clock_t start_time, end_time;

    for(const auto &image_to_test : test_image_names){

        cv::Mat img_in = cv::imread(test_images_path+image_to_test, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(img_in);

        start_time = clock();
        imgTest(model, img_in);
        end_time = clock();
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("img_in", img_in);
        cv::waitKey();
    }

    std::cout << "Bye" << std::endl;
    return 0;
}
