//
// Created by cc on 2020/3/16.
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

#define KERNEL_X 13
#define KERNEL_Y 21
#define EDGE_X 6
#define EDGE_Y 10

const float scale_factor = 0.85;
const int scale_times = 4;
const float rotate_angle = 30;
const int rotate_times = 12;

bool use_noised_data = false;
bool use_rotated_data = false;

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


void generateTrainingData(std::string data_dir, int extra_negative_sample_num = 100)
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
        turnBlacktoGray(img_in); // CHG

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

        /// Get cost maps
        std::vector<MAP_XD> cost_maps;
        std::vector<MAP_XD> corresponding_angles;

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

                if(use_noised_data){  // Add samples from noised image
                    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                        for(int feature_seq=0; feature_seq < cost_maps[kernel_seq][gateway_x][gateway_y].size(); feature_seq++){
                            positive_data_file << noised_cost_maps[kernel_seq][gateway_x][gateway_y][feature_seq] << ",";
                        }
                    }
                    positive_data_file << "\n";
                }

            }else{
                const int other_label_x = (object.y_min + object.y_max)/2;
                const int other_label_y = (object.x_min + object.x_max)/2;

                for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                    for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][other_label_x][other_label_y].size(); feature_seq++){
                        negative_data_file << cost_maps[kernel_seq][other_label_x][other_label_y][feature_seq] << ",";
                    }
                }
                negative_data_file << "\n";

                if(use_noised_data){  // Add samples from noised image
                    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                        for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][other_label_x][other_label_y].size(); feature_seq++){
                            negative_data_file << noised_cost_maps[kernel_seq][other_label_x][other_label_y][feature_seq] << ",";
                        }
                    }
                    negative_data_file << "\n";
                }
            }
        }

        /// Add some more negative samples, "extra_negative_sample_num" samples in one image
        for(int neg_extra_sample_seq=0; neg_extra_sample_seq<extra_negative_sample_num; neg_extra_sample_seq++)
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

                if(use_noised_data){  // Add samples from noised image
                    for(int kernel_seq=0; kernel_seq<kernels.size(); kernel_seq++){
                        for(int feature_seq=0; feature_seq<cost_maps[kernel_seq][pos_x][pos_y].size(); feature_seq++){
                            negative_data_file << noised_cost_maps[kernel_seq][pos_x][pos_y][feature_seq] << ",";
                        }
                    }
                    negative_data_file << "\n";
                }

            }else{
                neg_extra_sample_seq --;
            }

        }

        /// Add samples from rotated images ***********************
        if(use_rotated_data){
            std::vector<std::vector<cv::Mat>> rotate_result_imgs;
            const float Rotate_Angle = 45;
            const int Rotate_Times = 8;
            getScaledAndRotatedImgs(img_in, rotate_result_imgs, 1, 1, Rotate_Angle, Rotate_Times); //4, 24

            const int img_center_x = img_in.cols / 2;
            const int img_center_y = img_in.rows / 2;

            for (int rotated_seq = 0; rotated_seq < rotate_result_imgs[0].size(); rotated_seq++) {

                std::vector<std::vector<cv::Mat>> rotate_result_imgs_rotate_scale_results;
                getScaledAndRotatedImgs(rotate_result_imgs[0][rotated_seq], rotate_result_imgs_rotate_scale_results, scale_factor, scale_times, rotate_angle, rotate_times);

                std::vector<MAP_XD> cost_maps_this_rotate;
                std::vector<MAP_XD> corresponding_angles_this_rotate;
                getCostMaps(rotate_result_imgs_rotate_scale_results, scale_factor, rotate_angle, kernels, cost_maps_this_rotate,
                            corresponding_angles_this_rotate);

                std::vector<Eigen::Vector2i> positive_sample_positions_this_rotate;
                for (const auto &object : objects) {

                    int gateway_x = (object.x_min + object.x_max) / 2;
                    int gateway_y = (object.y_min + object.y_max) / 2;

                    float rotate_angle_rad = Rotate_Angle / 180 * CV_PI;

                    float rotated_total_angle = rotate_angle_rad * rotated_seq;
                    int gateway_y_this = (gateway_x - img_center_x) * cos(rotated_total_angle) +
                                         (gateway_y - img_center_y) * sin(rotated_total_angle) + img_center_x;
                    int gateway_x_this = -(gateway_x - img_center_x) * sin(rotated_total_angle) +
                                         (gateway_y - img_center_y) * cos(rotated_total_angle) + img_center_y;

                    if(gateway_x_this < EDGE_X || gateway_x_this > cost_maps_this_rotate[0].size()-EDGE_X ||
                       gateway_y_this < EDGE_Y || gateway_y_this > cost_maps_this_rotate[0][0].size()-EDGE_Y){
                        continue;
                    }

//                cv::circle(rotate_result_imgs[0][rotated_seq], cv::Point(gateway_y_this, gateway_x_this), 2, cv::Scalar(0), 1);
//                cv::imshow("rotate_result_imgs[0][rotated_seq]", rotate_result_imgs[0][rotated_seq]);
//                cv::waitKey();


                    Eigen::Vector2i gateway_pos;
                    gateway_pos << gateway_x_this, gateway_y_this;
                    positive_sample_positions_this_rotate.push_back(gateway_pos);

                    if (object.label == "gateway") {
                        for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                            for (int feature_seq = 0; feature_seq <
                                                      cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this].size(); feature_seq++) {
                                positive_data_file
                                        << cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this][feature_seq]
                                        << ",";
                            }
                        }
                        positive_data_file << "\n";
                    } else {
                        for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                            for (int feature_seq = 0; feature_seq <
                                                      cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this].size(); feature_seq++) {
                                negative_data_file
                                        << cost_maps_this_rotate[kernel_seq][gateway_x_this][gateway_y_this][feature_seq]
                                        << ",";
                            }
                        }
                        negative_data_file << "\n";
                    }
                }

                /// Add some more negative samples, "extra_negative_sample_num" samples per rotated image
                for (int neg_extra_sample_seq = 0;
                     neg_extra_sample_seq < extra_negative_sample_num; neg_extra_sample_seq++) {
                    int pos_x = rand() % img_height;
                    int pos_y = rand() % img_width;
                    Eigen::Vector2i point;
                    point << pos_x, pos_y;
                    if (!ifCloseToAnyPointInVector(point, positive_sample_positions_this_rotate, 5)) {
                        for (int kernel_seq = 0; kernel_seq < kernels.size(); kernel_seq++) {
                            for (int feature_seq = 0;
                                 feature_seq < cost_maps_this_rotate[kernel_seq][pos_x][pos_y].size(); feature_seq++) {
                                negative_data_file << cost_maps_this_rotate[kernel_seq][pos_x][pos_y][feature_seq] << ",";
                            }
                        }
                        negative_data_file << "\n";
                    } else {
                        neg_extra_sample_seq--;
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
    /// define kernels for cost maps
    std::vector<Eigen::MatrixXf> kernels;
    defineKernels(kernels);

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
    clock_t start_time, end_time;
    cv::Mat result;
    start_time = clock();
    float rst = model->predict(test_mat, result);
    end_time = clock();
    std::cout << "Predict Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    /// Draw the result
    int gateway_num = 0;
    for(int i=0; i<img_in.rows;i+=1){
        for(int j=0; j<img_in.cols; j+=1){
            if(result.at<int>(i*img_in.cols +j,0) > 0){
                cv::circle(img_in, cv::Point(j,i), 2, cv::Scalar(0), 1);
                gateway_num ++;
            }
        }
    }
    std::cout << "Found " << gateway_num <<" gateways" << std::endl;
}

int main(int argc, char** argv)
{

    /// Generating training data
    std::string training_data_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";

    generateTrainingData(training_data_path, 8);

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
    generateTrainingData(validation_data_path, 5);
    cv::Mat validation_data;
    cv::Mat validation_labels;
    readCSVstoMat(validation_data_path+"positive_data.csv",
                  validation_data_path+"negative_data.csv",
                  validation_data, validation_labels);

    validateSVM(model, validation_data, validation_labels);
    validateSVM(model, data, labels);

    /// Load model
    cv::Ptr<cv::ml::SVM> model_loaded = cv::ml::StatModel::load<cv::ml::SVM>("svm_model.xml");

    /// Testing with images
    std::string test_images_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";  // Floor2_test
    std::vector<std::string> test_image_names;
    getFileNames(test_images_path, test_image_names, ".png");
    clock_t start_time, end_time;


    for(const auto &image_to_test : test_image_names){
        cv::Mat img_in = cv::imread(test_images_path+image_to_test, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(img_in);

        std::vector<std::vector<cv::Mat>> transformed_imgs;
        getScaledAndRotatedImgs(img_in, transformed_imgs, scale_factor, scale_times, rotate_angle, rotate_times);

        start_time = clock();
        imgTest(model_loaded,img_in);
        end_time = clock();
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("img_in", img_in);
        cv::waitKey();

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
    }

    std::cout << "Bye" << std::endl;


    return 0;
}
