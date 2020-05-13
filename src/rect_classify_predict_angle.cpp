//
// Created by cc on 2020/3/26.
//
#include "tiny_dnn/tiny_dnn.h"
#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <random>
#include <string>
#include <cmath>
#include "labelimg_xml_reader.h"
#include <dirent.h>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <algorithm>
#include "voronoi_skeleton_points.h"

using namespace tiny_dnn;

#define RECT_SIZE_X  30 //26
#define RECT_SIZE_Y 30 //26

template<typename T>
std::vector<T> vectorRangeCopy(std::vector<T> v,int startIndex, int count)
{
    return std::vector<T>(v.begin() + startIndex,v.begin() + startIndex + count);
}

bool readCSV(string filename, vector<vector<string>> &data)
{
    ifstream infile(filename);
    if(infile.fail()){
        cout << "Can not find this file" << endl;
        return false;
    }

    string line;
    while(getline(infile, line)){
        istringstream in_line(line);
        vector<string> wordsInLine;
        string word;
        while(getline(in_line, word, ',')){
            wordsInLine.push_back(word);
        }
        if(!wordsInLine.empty()) data.push_back(wordsInLine);
    }

    if(!data.empty()){
        return true;
    }else{
        cout << "Found nothing in the file" << endl;
        return false;
    }
}


bool ifCloseToAnyPointInVector(cv::Point p, const std::vector<cv::Point>& v, const float threshold){
    for(const auto &point : v){
        float dist = sqrt( (p.x-point.x)^2 + (p.y-point.y)^2);
        if(dist <= threshold){
            return true;
        }
    }

    return false;
}

void convert_image(const cv::Mat &img, vec_t &d)
{
    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(img.cols, img.rows));
    double minv = -1.0;
    double maxv = 1.0;

    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void convert_images(std::vector<cv::Mat> &rects, std::vector<vec_t>& data)
{
    for(auto &rect : rects){
        vec_t d;
        convert_image(rect, d);
        data.push_back(d);
    }
}

void convert_labels(std::vector<float> &labels_ori, std::vector<vec_t> &labels){
    for(const auto &label_ori : labels_ori){
        tiny_dnn::vec_t v = {label_ori};
        labels.push_back(v);
    }
//    std::cout << labels[0] << std::endl;
//    std::cout << labels[1000] << std::endl;
}


static void construct_net(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using max_pool = tiny_dnn::layers::max_pool;
    using tanh = tiny_dnn::activation::tanh;
    using relu = tiny_dnn::activation::relu;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

#define O true
#define X false
    static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
            O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
            O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
            X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
            X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    nn << conv(RECT_SIZE_X, RECT_SIZE_Y, 7, 1, 6,   // C1, 1@26x26-in, 6@20x20-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(20,20,6)
       << max_pool(20, 20, 6, 2)   // S2, 6@20x20-in, 6@10x10-out
       << relu(10,10,6)
       << conv(10, 10, 3, 6, 16,
               connection_table(tbl, 6, 16),// C3, 6@10x10-in, 16@8x8-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(8,8,16)
       << max_pool(8, 8, 16, 2)   // S4, 16@8x8-in, 16@4x4-out
       << relu(4,4,16)
       << conv(4, 4, 3, 16, 64,   // C4, 16@4x4-in, 64@2x2-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(2,2,64)
       << conv(2, 2, 2, 64, 128,   // C5, 64@2x2-in, 128@1x1-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(1,1,128)
       << fc(128, 1, true);  // F6, 128-in, 1-out
}


static void construct_net30(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using max_pool = tiny_dnn::layers::max_pool;
    using tanh = tiny_dnn::activation::tanh;
    using relu = tiny_dnn::activation::relu;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

#define O true
#define X false
    static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
            O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
            O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
            X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
            X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    nn << conv(RECT_SIZE_X, RECT_SIZE_Y, 7, 1, 6,   // C1, 1@30x30-in, 6@24x24-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(24,24,6)
       << max_pool(24, 24, 6, 2)   // S2, 6@24x24-in, 6@12x12-out
       << relu(12,12,6)
       << conv(12, 12, 3, 6, 16,
               connection_table(tbl, 6, 16),// C3, 6@12x12-in, 16@10x10-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(10,10,16)
       << max_pool(10, 10, 16, 2)   // S4, 16@10x10-in, 16@5x5-out
       << relu(5,5,16)
       << conv(5, 5, 3, 16, 64,   // C4, 16@5x5-in, 64@3x3-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(3,3,64)
       << conv(3, 3, 3, 64, 128,   // C5, 64@3x3-in, 128@1x1-out
               padding::valid, true, 1, 1, 1, 1)
       << relu(1,1,128)
       << fc(128, 1, true);  // F6, 128-in, 1-out
}


static void construct_vggnet(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using max_pool = tiny_dnn::layers::max_pool;
    using tanh = tiny_dnn::activation::tanh;
    using relu = tiny_dnn::activation::relu;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;


    nn  << conv(RECT_SIZE_X, RECT_SIZE_Y, 3, 1, 32,   // C1, 1@30x30-in, 64@28x28-out
                padding::valid, true, 1, 1, 1, 1)
        << relu(28,28,32)
        << max_pool(28, 28, 32, 2)   // S2, 64@28x28-in, 64@14x14-out
        << relu(14,14,32)
        << conv(14, 14, 3, 32, 64,  // C3-1, 64@14x14-in, 128@12x12-out
                padding::valid, true, 1, 1, 1, 1)
        << relu(12, 12, 64)
        << conv(12, 12, 3, 64, 64,  // C3-2, 128@12x12-in, 128@10x10-out
                padding::valid, true, 1, 1, 1, 1)
        << relu(10, 10, 64)
        << max_pool(10, 10, 64, 2)   // S4, 128@10x10-in, 128@5x5-out
        << relu(5,5,64)
        << conv(5, 5, 3, 64, 128,  // C5-1, 128@5x5-in, 256@3x3-out
                padding::valid, true, 1, 1, 1, 1)
        << relu(3,3,128)
        << conv(3, 3, 3, 128, 128,  // C5-2, 256@3x3-in, 256@1x1-out
                padding::valid, true, 1, 1, 1, 1)
        << relu(1,1,128)
        << fc(128, 1, true);  // F6, 128-in, 1-out
}

void shuffle(std::vector<vec_t>& images, std::vector<vec_t>& labels)
{
    unsigned seed;  // Random generator seed for collecting extra negative samples
    seed = time(0);
    srand(seed);

    int n = images.size();
    for (int i = n - 1; i>1; --i)
    {
        int r = rand() % i;
        std::swap(labels[i], labels[r]);
        std::swap(images[i], images[r]);
    }
}


static void train_lenet(tiny_dnn::network<tiny_dnn::sequential> &nn,
                        std::vector<vec_t> &trainging_data, std::vector<vec_t> &trainging_labels,
                        std::vector<vec_t> &testing_data, std::vector<vec_t> &testing_labels,
                        double learning_rate,  int n_train_epochs,  int n_minibatch, std::string model_name)
{
    shuffle(trainging_data, trainging_labels);

    // specify loss-function and learning strategy
//    tiny_dnn::adagrad optimizer;
//    tiny_dnn::adam optimizer;
    tiny_dnn::adam optimizer;

    /// Choose a network here
    //construct_net(nn);
    construct_net30(nn);
//    construct_vggnet(nn);

    tiny_dnn::progress_display disp(static_cast<unsigned long>(trainging_data.size()));
    tiny_dnn::timer t;

    optimizer.alpha *=
            std::min(tiny_dnn::float_t(4),
                     static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));


    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout <<std::endl << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                  << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        float loss = nn.get_loss<tiny_dnn::mse>(trainging_data, trainging_labels);
        std::cout << loss << "/" << std::endl;

        shuffle(trainging_data, trainging_labels);

        disp.restart(trainging_data.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.fit<tiny_dnn::mse>(optimizer, trainging_data, trainging_labels, n_minibatch,
                          n_train_epochs, []() {},  on_enumerate_epoch);
    std::cout << "end training." << std::endl;

    // save network model & trained weights
    nn.save(model_name);
}

void mergeAndConvertImages(std::vector<cv::Mat> positive_images, std::vector<cv::Mat> negative_images, std::vector<vec_t> &results)
{
    positive_images.insert(positive_images.end(), negative_images.begin(), negative_images.end());
    convert_images(positive_images, results);
}

void mergeAndConvertLabels(std::vector<float> postive_labels, std::vector<float> negative_labels, std::vector<vec_t> &results)
{
    postive_labels.insert(postive_labels.end(), negative_labels.begin(), negative_labels.end());
    convert_labels(postive_labels, results);
}

float pointSquareDistance(cv::Point p1, cv::Point p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

void rankFromLargeToSmallConfidence(std::vector<cv::Point> &p_in, std::vector<float> &confidence)
{
    for(int i=0;  i<p_in.size()-1; i++){
        for(int j=i+1; j<p_in.size(); j++){
            if(confidence[i] < confidence[j]){
                float temp_confidence = confidence[i];
                cv::Point temp_point = p_in[i];
                confidence[i] = confidence[j];
                p_in[i] = p_in[j];
                confidence[j] = temp_confidence;
                p_in[j] = temp_point;
            }
        }
    }
}

void nonMaximumSuppression(std::vector<cv::Point> &points_in, std::vector<float> &confidences, std::vector<cv::Point> &p_out, float dist_threshold = 4)
{
    if(points_in.size() == 0)
        return;
    else if(points_in.size() == 1){
        p_out = points_in;
    }

    float square_threshold = dist_threshold * dist_threshold;
    rankFromLargeToSmallConfidence(points_in, confidences);
    std::vector<int> merged_flag(points_in.size(), 0);
    for(int i=0; i<points_in.size()-1; i++){
        if(!merged_flag[i]){
            for(int j=i+1; j<points_in.size(); j++){
                if(!merged_flag[j]){
                    float square_dist = pointSquareDistance(points_in[i], points_in[j]);
                    if(square_dist <= square_threshold){
                        if(confidences[i]==confidences[j]){
                            points_in[i] = (points_in[i] + points_in[j]) / 2;
                        }
                        merged_flag[j] = 1;
                    }
                }
            }
            merged_flag[i] = 1;
            p_out.push_back(points_in[i]);
        }
    }
}

void nonMaximumSuppressionWithAngle(std::vector<cv::Point> &points_in, std::vector<float> &confidences, std::vector<float> &angles,
        std::vector<cv::Point> &p_out, std::vector<float> &angles_out, float dist_threshold = 4)
{
    if(points_in.size() == 0)
        return;
    else if(points_in.size() == 1){
        p_out = points_in;
        angles_out = angles;
    }

    float square_threshold = dist_threshold * dist_threshold;
    rankFromLargeToSmallConfidence(points_in, confidences);
    std::vector<int> merged_flag(points_in.size(), 0);
    for(int i=0; i<points_in.size()-1; i++){
        if(!merged_flag[i]){
            for(int j=i+1; j<points_in.size(); j++){
                if(!merged_flag[j]){
                    float square_dist = pointSquareDistance(points_in[i], points_in[j]);
                    if(square_dist <= square_threshold){
                        if(confidences[i]==confidences[j]){
                            points_in[i] = (points_in[i] + points_in[j]) / 2;
                            angles[i] = (angles[i] + angles[j]) / 2;
                        }
                        merged_flag[j] = 1;
                    }
                }
            }
            merged_flag[i] = 1;
            p_out.push_back(points_in[i]);
            angles_out.push_back(angles[i]);
        }
    }
}

float sumMatOneChannel(cv::Mat &src){
    float result = 0;
    for(int i=0; i<src.rows; i++){
        auto* row_i = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++){
            result += (int)row_i[j];
        }
    }
    return result;
}


int main(){
    std::vector<std::string> positive_sample_file_names, negative_sample_file_names;

    std::string positive_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/30combined/positive/";
    std::string negative_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/30combined/negative/";

    getFileNames(positive_data_dir, positive_sample_file_names, ".png");
    getFileNames(negative_data_dir, negative_sample_file_names, ".png");

    bool if_training_gateway_position = true;
    if(if_training_gateway_position){
        /// Read images
        std::vector<cv::Mat> positive_sample_rects, negative_sample_rects;
        for(const auto & file : positive_sample_file_names){
            cv::Mat temp_img = cv::imread(positive_data_dir + file, cv::IMREAD_GRAYSCALE);
            positive_sample_rects.push_back(temp_img);
        }

        for(const auto & file : negative_sample_file_names){
            cv::Mat temp_img = cv::imread(negative_data_dir + file, cv::IMREAD_GRAYSCALE);
            negative_sample_rects.push_back(temp_img);
        }

        /// Convert to required data form
        float validation_ratio = 0.0;
        int positive_data_training_num = positive_sample_rects.size() * (1.f-validation_ratio);
        int negative_data_training_num = negative_sample_rects.size() * (1.f-validation_ratio);
        int positive_data_validation_num = positive_sample_rects.size() - positive_data_training_num;
        int negative_data_validation_num = negative_sample_rects.size() - negative_data_training_num;

        std::cout << "positive_data_training_num = " << positive_data_training_num << std::endl;
        std::cout << "negative_data_training_num = " << negative_data_training_num << std::endl;
        std::cout << "positive_data_validation_num = " << positive_data_validation_num << std::endl;
        std::cout << "negative_data_validation_num = " << negative_data_validation_num << std::endl;

        std::vector<cv::Mat> positive_training_data = vectorRangeCopy(positive_sample_rects, 0, positive_data_training_num);
        std::vector<cv::Mat> negative_training_data = vectorRangeCopy(negative_sample_rects, 0, negative_data_training_num);
        std::vector<cv::Mat> positive_validation_data = vectorRangeCopy(positive_sample_rects, positive_data_training_num, positive_data_validation_num);
        std::vector<cv::Mat> negative_validation_data = vectorRangeCopy(negative_sample_rects, negative_data_training_num, negative_data_validation_num);

        std::vector<float> positive_training_data_labels(positive_data_training_num, 0);
        std::vector<float> negative_training_data_labels(negative_data_training_num, 1);
        std::vector<float> positive_validation_data_labels(positive_data_validation_num, 0);
        std::vector<float> negative_validation_data_labels(negative_data_validation_num, 1);

        std::vector<vec_t> training_data_desired_form, validation_data_desired_form;
        std::vector<vec_t> training_labels_desired_form, validation_labels_desired_form;

        mergeAndConvertImages(positive_training_data, negative_training_data, training_data_desired_form);
        mergeAndConvertImages(positive_validation_data, negative_validation_data, validation_data_desired_form);
        mergeAndConvertLabels(positive_training_data_labels, negative_training_data_labels, training_labels_desired_form);
        mergeAndConvertLabels(positive_validation_data_labels, negative_validation_data_labels, validation_labels_desired_form);

        std::cout << "training_data_num = " << training_data_desired_form.size() << std::endl;
        std::cout << "validation_data_num = " << validation_data_desired_form.size() << std::endl;


        /// Training
        tiny_dnn::network<tiny_dnn::sequential> nn;
        train_lenet(nn, training_data_desired_form, training_labels_desired_form, validation_data_desired_form,
                validation_labels_desired_form, 1.0, 200, 64, "LeNet-model-rects-regression-30combined-lenet");
    }

    /**----------------------------------------------**/

    std::vector<std::string> angle_training_file_names;
    string angle_training_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/30combined/positive/";
    getFileNames(angle_training_data_dir, angle_training_file_names, ".png");

    bool if_training_angle = false;

    if(if_training_angle){
        std::vector<float> positive_angle_output;
        std::vector<cv::Mat> positive_images_for_training_angle;

        for(const auto & file : angle_training_file_names){
            cv::Mat temp_img = cv::imread(angle_training_data_dir + file, cv::IMREAD_GRAYSCALE);
            positive_images_for_training_angle.push_back(temp_img);

            std::vector<std::vector<string>> csv_data;
            if(readCSV(angle_training_data_dir + file + ".csv", csv_data)){
                float angle = strtof(csv_data[0][0].c_str(), NULL);
                float angle_label = angle / 2 / CV_PI + 0.5;  //[-Pi, Pi] -> [0.0, 1.0]
                positive_angle_output.push_back(angle_label);
            }else{
                positive_angle_output.push_back(0);
            }
        }

        std::vector<vec_t> image_data_for_angle_training;
        std::vector<vec_t> angle_labels;
        convert_labels(positive_angle_output, angle_labels);
        convert_images(positive_images_for_training_angle, image_data_for_angle_training);

        tiny_dnn::network<tiny_dnn::sequential> nn2;
        train_lenet(nn2, image_data_for_angle_training, angle_labels, image_data_for_angle_training,
                angle_labels, 1.0, 200, 32, "LeNet-model-angles-regression-30combined-vgg");
    }


    /**---------------------------------------------**/

    /// Load model and testing
    tiny_dnn::network<tiny_dnn::sequential> model, angle_model;
    model.load("LeNet-model-rects-regression-30combined-lenet");
    angle_model.load("LeNet-model-angles-regression-30combined-vgg");

    std::cout << "Model loaded!" << std::endl;
    /// Testing on images and show
    std::string testing_images_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/sudoku2/";
    std::vector<std::string> image_names;
    getFileNames(testing_images_dir, image_names, ".png");

    if(image_names.empty()){
        std::cout << "Folder empty" <<std::endl;
        return -1;
    }
    clock_t start_time, end_time;

    for(const auto image_name : image_names){
        cv::Mat image_this = cv::imread(testing_images_dir+image_name, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(image_this);
//        cv::resize(image_this, image_this, cv::Size(RECT_SIZE_X, RECT_SIZE_Y));

        cv::Mat image_this_copy = image_this.clone();
        start_time = clock();

        /// Extract skeleton points
        std::vector<cv::Point> skeleton_points;
        findVoronoiSkeletonPoints(image_this, skeleton_points, true);  /// CHG
        if(skeleton_points.size() < 1){
            std::cout << "Found no skeleton_points in this image, skip!" << std::endl;
            continue;
        }

        /// Predict on skeleton points
        std::vector<float> confidences_vec;
        std::vector<cv::Point> valid_points;
//        std::vector<float> valid_angles;

        for(auto &sk_point : skeleton_points){
            if(sk_point.x < RECT_SIZE_X/2 || sk_point.y <RECT_SIZE_Y/2 || sk_point.x > image_this.cols-RECT_SIZE_X/2-1 || sk_point.y > image_this.rows-RECT_SIZE_Y/2-1){
                continue;
            }else{
                cv::Mat rect_this = image_this(cv::Rect(sk_point.x-RECT_SIZE_X/2, sk_point.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
                vec_t data_this;
                convert_image(rect_this, data_this);

                vec_t label_this = model.predict(data_this);
//                vec_t angle_label_this = angle_model.predict(data_this);

                if(label_this[0] < 0.5){
                    float confidence = 1.f - label_this[0];
                    confidences_vec.push_back(confidence);
                    valid_points.push_back(sk_point);
//                    float angle = (angle_label_this[0] - 0.5)*2*CV_PI;
//                    valid_angles.push_back(angle);
                }
            }
        }

        /// nonMaximumSuppression
        std::vector<cv::Point> result_points;
        std::vector<float> result_angles;

        nonMaximumSuppression(valid_points, confidences_vec, result_points, 5); //5 for 30, 4 for 26 image
//        nonMaximumSuppressionWithAngle(valid_points, confidences_vec, valid_angles, result_points, result_angles);

        /// Angle predicton
        for(auto &p : result_points){
            if(p.x < RECT_SIZE_X/2 || p.y <RECT_SIZE_Y/2 || p.x > image_this.cols-RECT_SIZE_X/2-1 || p.y > image_this.rows-RECT_SIZE_Y/2-1){
                continue;
            }else{
                cv::Mat rect_this = image_this(cv::Rect(p.x-RECT_SIZE_X/2, p.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
                vec_t data_this;
                convert_image(rect_this, data_this);

                vec_t angle_label_this = angle_model.predict(data_this);
                float angle = (angle_label_this[0] - 0.5)*2*CV_PI;
                result_angles.push_back(angle);
            }
        }


        /// Combine two results
//        std::cout << result_angles.size() <<", " << result_angles2.size() << std::endl;
//        if(result_angles.size() == result_angles2.size())
//        {
//            std::cout << "combine two angle results" << std::endl;
//            for(int i=0; i<result_angles.size();i++){
//                result_angles[i]= (result_angles[i] + result_angles2[i]) / 2;
//            }
//        }

        /// Correct directions for angle output in range[0, Pi]
        for(int i=0; i<result_points.size(); i++){
            auto p = result_points[i];
            float dy1 = p.x - image_this_copy.cols/2;
            float dx1 = p.y - image_this_copy.rows/2;
            float temp_direction = atan2(dy1, dx1);
            std::cout << "P "<< p.x<< " " << p.y << " temp_direction="<<temp_direction<<std::endl;
            temp_direction = -temp_direction + CV_PI / 2.f;
            if(temp_direction > CV_PI) temp_direction -= CV_2PI;
            else if(temp_direction < -CV_PI) temp_direction += CV_2PI;

            if( fabs(result_angles[i] - temp_direction) < CV_PI / 2.f || fabs(result_angles[i] - temp_direction) > CV_PI / 2.f * 3.f){
                continue;
            }else{
                std::cout << "angle corrected " << std::endl;
                if(result_angles[i] > 0){
                    result_angles[i] -= CV_PI;
                }else{
                    result_angles[i] += CV_PI;
                }
            }
        }

        end_time = clock();
        std::cout <<"gateway_num = " << result_points.size() <<" in image "<< image_name << std::endl;
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        /// Display
        cv::circle(image_this_copy, cv::Point(image_this_copy.cols/2, image_this_copy.rows/2), 1, cv::Scalar(150), 2);
        for(int i=0; i<result_points.size(); i++){
            auto p = result_points[i];
            cv::circle(image_this_copy, p, 2, cv::Scalar(0), 1);
            cv::Point direction_end_point;

            float direction_angle = result_angles[i];
            direction_end_point.x = 10 * cos(direction_angle) + p.x;
            direction_end_point.y = 10 * sin(direction_angle) + p.y;
            cv::line(image_this_copy, p, direction_end_point, cv::Scalar(160), 1);
        }

        cv::imshow("image_this_copy", image_this_copy);
        cv::waitKey();
    }

    return 0;
}
