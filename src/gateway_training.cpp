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

#define KERNEL_X 13
#define KERNEL_Y 21

const float scale_factor = 0.8;
const int scale_times = 3;
const float rotate_angle = 15;
const int rotate_times = 24;

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

        cv::Mat img_in = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

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
                const int gateway_x = (object.y_min + object.y_max)/2; ///Note the x in a image is y in a matrix
                const int gateway_y = (object.x_min + object.x_max)/2;
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
        for(int neg_extra_sample_seq=0; neg_extra_sample_seq<100; neg_extra_sample_seq++)
        {
            int pos_x = rand() % img_height;
            int pos_y = rand() % img_width;
            Eigen::Vector2i point;
            point << pos_x, pos_y;
            if(!ifCloseToAnyPointInVector(point, positive_sample_positions, 3)){
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
    /// define kernels for cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);

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

    for(int cost_seq_x=0; cost_seq_x<cost_maps[0].size(); cost_seq_x++){
        for(int cost_seq_y=0; cost_seq_y<cost_maps[0][0].size(); cost_seq_y++){
            //Iterate every pixel
            int row_temp = cost_seq_x*cost_maps[0][0].size() + cost_seq_y;
            for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                for(int feature_seq=0; feature_seq<cost_maps[0][0][0].size(); feature_seq++){
                    int col_temp = kernel_seq*cost_maps[0][0][0].size()+feature_seq;
                    test_mat.at<float>(row_temp, col_temp) = cost_maps[kernel_seq][cost_seq_x][cost_seq_y][feature_seq];
                }
            }
        }
    }

    /// Predict
    cv::Mat result;
    float rst = model->predict(test_mat, result);

    /// Draw the result
    for(int i=0; i<img_in.rows;i+=1){
        for(int j=0; j<img_in.cols; j+=1){
            if(result.at<int>(i*img_in.cols +j,0) > 0){
                cv::circle(img_in, cv::Point(j,i), 2, cv::Scalar(0), 1);
            }
        }
    }
    cv::imshow("img_in", img_in);
    cv::waitKey();
}

int main(int argc, char** argv)
{

    generateTrainingData("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/");

    cv::Mat data;
    cv::Mat labels;
    readCSVstoMat("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/positive_data.csv",
                  "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/negative_data.csv",
                  data, labels);

    for(int i=0; i<data.rows; i++) {
        std::cout << data.at<float>(i, 0) << " ";
    }
    std::cout << std::endl;

    for(int i=0; i<labels.rows; i++) {
        std::cout << labels.at<int>(i, 0) << " ";
    }
    std::cout << std::endl;

    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
    trainingSVM(model, data, labels);

    validateSVM(model, data, labels);

    cv::Mat img_in = cv::imread("/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/pow7resolution1.000000T1202811112649.png", cv::IMREAD_GRAYSCALE);
    imgTest(model, img_in);

    std::cout << "Bye" << std::endl;
    return 0;
}
