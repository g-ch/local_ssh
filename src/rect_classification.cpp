//
// Created by cc on 2020/3/21.
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

#define RECT_SIZE_X 26
#define RECT_SIZE_Y 26

template<typename T>
std::vector<T> vectorRangeCopy(std::vector<T> v,int startIndex, int count)
{
    return std::vector<T>(v.begin() + startIndex,v.begin() + startIndex + count);
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

void convert_labels(std::vector<std::vector<float>> &labels_ori, std::vector<vec_t> &labels){
    for(const auto &label_ori_vec : labels_ori){
        tiny_dnn::vec_t v = {label_ori_vec[0], label_ori_vec[1]};
        labels.push_back(v);
    }
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
       << relu(1,1,128);
    nn << fc(128, 256, true)  // F6, 128-in, 1-out
       << relu(256);
    nn << fc(256, 2, true);  // F6, 128-in, 1-out
//       << relu(2);<< relu(256)
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
                        double learning_rate,  int n_train_epochs,  int n_minibatch)
{
    shuffle(trainging_data, trainging_labels);

    // specify loss-function and learning strategy
//    tiny_dnn::adagrad optimizer;
    tiny_dnn::adam optimizer;

    /// Choose a network here
    construct_net(nn);

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
    nn.save("LeNet-model-rects-regression");
}

void mergeAndConvertImages(std::vector<cv::Mat> positive_images, std::vector<cv::Mat> negative_images, std::vector<vec_t> &results)
{
    positive_images.insert(positive_images.end(), negative_images.begin(), negative_images.end());
    convert_images(positive_images, results);
}

void mergeAndConvertLabels(std::vector<std::vector<float>> &postive_labels, std::vector<std::vector<float>> &negative_labels, std::vector<vec_t> &results)
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


void nonMaximumSuppressionWithDirection(std::vector<cv::Point> &points_in, std::vector<float> &confidences, std::vector<float> &directions,
        std::vector<cv::Point> &p_out, std::vector<float> &directions_out, float dist_threshold = 4)
{
    if(points_in.size() == 0)
        return;
    else if(points_in.size() == 1){
        p_out = points_in;
    }

    /// nonMaximumSuppression by distance
    std::vector<cv::Point> p_suppressed;
    std::vector<float> direction_suppressed;

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
                            directions[i] = directions[i];
                        }
                        merged_flag[j] = 1;
                    }
                }
            }
            merged_flag[i] = 1;
            p_out.push_back(points_in[i]);
            directions_out.push_back(directions[i]);
        }
    }

//    /// Merge again by direction, when two points has the similar direction, use average position and average direction
//    std::vector<int> merged_flag2(p_suppressed.size(), 0);
//    for(int i=0; i<p_suppressed.size()-1; i++){
//        if(!merged_flag2[i]){
//            for(int j=i+1; j<p_suppressed.size(); j++){
//                if(!merged_flag2[j]){
//                    float dy1 = cos(direction_suppressed[i]); //Left hand coordinate
//                    float dx1 = sin(direction_suppressed[i]);
//                    float dy2 = cos(direction_suppressed[j]);
//                    float dx2 = sin(direction_suppressed[j]);
//                    float scalar_multiply = dx1 * dx2 + dy1 * dy2;
//                    if(scalar_multiply > 0.8 && pointSquareDistance(p_suppressed[i], p_suppressed[j]) < 2 * square_threshold){  // nearly 30 degree, 1.4 more dist threshold
//                        p_suppressed[i] = (p_suppressed[i] + p_suppressed[j]) / 2;
//                        direction_suppressed[i] = (direction_suppressed[i] + direction_suppressed[j]) / 2;
//                        merged_flag2[j] = 1;
//                    }
//                }
//            }
//            merged_flag2[i] = 1;
//            p_out.push_back(p_suppressed[i]);
//            directions_out.push_back(direction_suppressed[i]);
//        }
//    }
}


int main(){
//
    bool if_training = false;
    if(if_training){
        /// Read images
        std::string positive_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/data/positive/";
        std::string negative_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/data/negative/";

        std::vector<std::string> positive_sample_file_names, negative_sample_file_names;
        getFileNames(positive_data_dir, positive_sample_file_names, ".png");
        getFileNames(negative_data_dir, negative_sample_file_names, ".png");

        std::vector<cv::Mat> positive_sample_rects, negative_sample_rects;
        std::vector<std::vector<float>> positive_reference_output, negative_reference_output;

        for(const auto & file : positive_sample_file_names){
            cv::Mat temp_img = cv::imread(positive_data_dir + file, cv::IMREAD_GRAYSCALE);
            positive_sample_rects.push_back(temp_img);

            std::vector<float> output;
            output.push_back(0.0);
            std::vector<std::vector<string>> csv_data;
            if(readCSV(positive_data_dir + file + ".csv", csv_data)){
                float angle = strtof(csv_data[0][0].c_str(), NULL);
//                if(angle < 0) angle += CV_PI;
//                float angle_label = angle / CV_PI * 0.8 + 0.2; //[0, Pi] -> [0.2, 1.0]
                float angle_label = (angle / 2 / CV_PI + 0.5 )* 0.8 + 0.2;  //[-Pi, Pi] -> [0.2, 1.0]
                output.push_back(angle_label);
            }else{
                std::cout << "Can not find labeled angle in " << file + ".csv" << std::endl;
                float angle_label = 0.0;
                output.push_back(angle_label);
            }
            positive_reference_output.push_back(output);
        }

        for(const auto & file : negative_sample_file_names){
            cv::Mat temp_img = cv::imread(negative_data_dir + file, cv::IMREAD_GRAYSCALE);
            negative_sample_rects.push_back(temp_img);

            std::vector<float> output;
            output.push_back(1.0);
            output.push_back(0.0);
            negative_reference_output.push_back(output);
        }

        /// Convert to required data form
        std::vector<vec_t> training_data_desired_form;
        std::vector<vec_t> training_labels_desired_form;

        mergeAndConvertImages(positive_sample_rects, negative_sample_rects, training_data_desired_form);
        mergeAndConvertLabels(positive_reference_output, negative_reference_output, training_labels_desired_form);

        std::cout << "training_data_num = " << training_data_desired_form.size() << std::endl;

        /// Training
        tiny_dnn::network<tiny_dnn::sequential> nn;
        train_lenet(nn, training_data_desired_form, training_labels_desired_form, 1, 200, 64);
    }

    /**----------------------------------------------**/

    /// Load model and testing
    tiny_dnn::network<tiny_dnn::sequential> model;
    model.load("LeNet-model-rects-regression"); //LeNet-model-rects-regression
//    model.load("LeNet-model-rects-good-good");
    std::cout << "Model loaded!" << std::endl;

    /// Testing on images and show
    std::string testing_images_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";
    std::vector<std::string> image_names;
    getFileNames(testing_images_dir, image_names, ".png");
    clock_t start_time, end_time;

    for(const auto image_name : image_names){
        cv::Mat image_this = cv::imread(testing_images_dir+image_name, cv::IMREAD_GRAYSCALE);
        turnBlacktoGray(image_this);
//        cv::resize(image_this, image_this, cv::Size(RECT_SIZE_X, RECT_SIZE_Y));

        cv::Mat image_this_copy = image_this.clone();
        cv::Mat image_this_copy2 = image_this.clone();


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
        std::vector<float> directions;
        for(auto &sk_point : skeleton_points){
            if(sk_point.x < RECT_SIZE_X/2 || sk_point.y <RECT_SIZE_Y/2 || sk_point.x > image_this.cols-RECT_SIZE_X/2 || sk_point.y > image_this.rows-RECT_SIZE_Y/2){
                continue;
            }else{
                cv::Mat rect_this = image_this(cv::Rect(sk_point.x-RECT_SIZE_X/2, sk_point.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
                vec_t data_this;
                convert_image(rect_this, data_this);
                vec_t label_this = model.predict(data_this);

                if(label_this[0] < 0.5){
                    float confidence = 1.f - label_this[0];
                    confidences_vec.push_back(confidence);
                    valid_points.push_back(sk_point);
                    directions.push_back(((label_this[1]-0.2f) * 1.25 - 0.5)* 2*CV_PI);
                }
            }
        }

        for(int i=0; i<valid_points.size(); i++){
            auto p = valid_points[i];
            cv::circle(image_this_copy2, p, 2, cv::Scalar(0), 1);
            cv::imshow("image_this_copy2", image_this_copy2);

        }

        /// Correct directions for angle output in range[0, Pi]
//        for(int i=0; i<valid_points.size(); i++){
//            auto p = valid_points[i];
//            float dx1 = p.x - RECT_SIZE_X/2;
//            float dy1 = p.y - RECT_SIZE_Y/2;
//            float dx2 = cos(directions[i]);
//            float dy2 = sin(directions[i]);
//            float scalar_multiply = dx1 * dx2 + dy1 * dy2;
//            if(scalar_multiply < 0){
//                std::cout << "correct direction" << directions[i];
//                if(directions[i] < 0){
//                    directions[i] += CV_PI;
//                }else{
//                    directions[i] -= CV_PI;
//                }
//                std::cout <<" to " << directions[i] << std::endl;
//            }
//        }

        /// nonMaximumSuppression
//        nonMaximumSuppression(valid_points, confidences_vec, result_points, 4);
//
        std::vector<cv::Point> suppressed_points;
        std::vector<float> suppressed_directions;
        nonMaximumSuppressionWithDirection(valid_points, confidences_vec, directions,suppressed_points, suppressed_directions);

        for(int i=0; i<suppressed_points.size(); i++){
            auto p = suppressed_points[i];
            cv::circle(image_this_copy, p, 2, cv::Scalar(0), 1);
            cv::Point direction_end_point;

            float direction_angle = suppressed_directions[i];
            direction_end_point.x = 10 * cos(direction_angle) + p.x;
            direction_end_point.y = 10 * sin(direction_angle) + p.y;
            cv::line(image_this_copy, p, direction_end_point, cv::Scalar(160), 1);
        }

        end_time = clock();
        std::cout <<"gateway_num = " << valid_points.size() <<" in image "<< image_name << std::endl;
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("image_this_copy", image_this_copy);
        cv::waitKey();
    }

    return 0;
}

