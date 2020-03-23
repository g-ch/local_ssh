//
// Created by cc on 2020/3/18.
//

#include "tiny_dnn/tiny_dnn.h"
#include "preprocess.h"
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include "labelimg_xml_reader.h"
#include <dirent.h>
#include <cstdlib>
#include <fstream>
#include <ctime>

using namespace tiny_dnn;

#define RECT_SIZE_X 24
#define RECT_SIZE_Y 24

bool ifCloseToAnyPointInVector(cv::Point p, const std::vector<cv::Point>& v, const float threshold){
    for(const auto &point : v){
        float dist = sqrt( (p.x-point.x)^2 + (p.y-point.y)^2);
        if(dist <= threshold){
            return true;
        }
    }

    return false;
}

void generateTrainingData(std::string data_dir, std::vector<cv::Mat> &img_rects, std::vector<int> &labels, int extra_negative_sample_num = 10)
{
    /// Read an XML file names for training
    std::vector<std::string> filenames;
    getFileNames(data_dir, filenames, ".xml");

    int positive_tag = 1, negative_tag = 0;

    unsigned seed;  // Random generator seed for collecting extra negative samples
    seed = time(0);
    srand(seed);

    int positive_sample_num = 0, negative_sample_num = 0;

    /// Generate training dataset for gateways
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
        const float Rotate_Angle = 15;
        const int Rotate_Times = 24;
        getScaledAndRotatedImgs(img_in, result_imgs, 1, 1, Rotate_Angle, Rotate_Times); //4, 24

        /// Add samples corresponding to the hand-labeled gateway position in csv files
        const int img_center_x = img_in.cols / 2;
        const int img_center_y = img_in.rows / 2;

        for (int rotated_times = 0; rotated_times < result_imgs[0].size(); rotated_times++) {

            std::vector<cv::Point> positive_sample_positions;
            for(const auto &object : objects){

                int gateway_x = (object.x_min + object.x_max) / 2;
                int gateway_y = (object.y_min + object.y_max) / 2;

                float rotate_angle_rad = Rotate_Angle / 180 * CV_PI;

                float rotated_total_angle = rotate_angle_rad * rotated_times;
                int gateway_x_this = (gateway_x - img_center_x) * cos(rotated_total_angle) +
                                     (gateway_y - img_center_y) * sin(rotated_total_angle) + img_center_x;
                int gateway_y_this = -(gateway_x - img_center_x) * sin(rotated_total_angle) +
                                     (gateway_y - img_center_y) * cos(rotated_total_angle) + img_center_y;

                cv::Point gateway_pos(gateway_x_this, gateway_y_this);
                positive_sample_positions.push_back(gateway_pos);

                int rect_left_top_x = gateway_x_this - RECT_SIZE_X / 2;
                int rect_left_top_y = gateway_y_this - RECT_SIZE_Y / 2;
                if (rect_left_top_x < 0 || rect_left_top_y < 0 ||
                    rect_left_top_x + RECT_SIZE_X >= result_imgs[0][rotated_times].cols ||
                    rect_left_top_y + RECT_SIZE_Y >= result_imgs[0][rotated_times].rows) {
//                    std::cout << "Gateway too close to border. Abort." << std::endl;
                    break;
                }
                //Add the rect to dataset with positive label
                cv::Rect rect(rect_left_top_x, rect_left_top_y, RECT_SIZE_X, RECT_SIZE_Y);
                cv::Mat img_rect = result_imgs[0][rotated_times](rect);
                img_rects.push_back(img_rect);

                if(object.label == "gateway") {
                    labels.push_back(positive_tag);
                    positive_sample_num ++;
//                    cv::circle(result_imgs[0][rotated_times], cv::Point(gateway_x_this, gateway_y_this), 2, cv::Scalar(0), 1);
//                    cv::imshow("result_imgs",result_imgs[0][rotated_times]);
//                    cv::waitKey();
                }else{
                    labels.push_back(negative_tag);
                    negative_sample_num ++;
                }
            }

            /// Add some more negative samples, "extra_negative_sample_num" samples per rotated image
            for (int i=0; i< extra_negative_sample_num; i++) {
                int pos_x = rand() % (img_height - RECT_SIZE_X) + RECT_SIZE_X/2;
                int pos_y = rand() % (img_width - RECT_SIZE_Y) + RECT_SIZE_Y/2;
                cv::Point p(pos_x, pos_y);
                if(!ifCloseToAnyPointInVector(p, positive_sample_positions, 3)){
                    cv::Rect rect(pos_x - RECT_SIZE_X/2, pos_y - RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y);
                    cv::Mat img_rect = result_imgs[0][rotated_times](rect);
                    img_rects.push_back(img_rect);
                    labels.push_back(negative_tag);
                    negative_sample_num ++;
                }else{
                    i--;
                }
            }
        }
    }

    std::cout << "Data generation finished!" << std::endl;
    std::cout << "Samples number = " << img_rects.size() << std::endl;
    std::cout << "Labels number = " << labels.size() << ", where positive_sample_num = "<<positive_sample_num << ", negative_sample_num = "<< negative_sample_num << std::endl;
}

void convert_image(const cv::Mat &img, vec_t &data)
{
    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(img.cols, img.rows));
    std::transform(resized.begin(), resized.end(), std::back_inserter(data), [=](uint8_t c) { return c ; });
}

void convert_images(std::vector<cv::Mat> rects, std::vector<vec_t>& data)
{
    for(const auto &rect : rects){
        vec_t d;
        convert_image(rect, d);
        data.push_back(d);
    }
}

void convert_labels(std::vector<int> &labels_ori, std::vector<label_t> &labels){
    for(const auto &label_ori : labels_ori){
        labels.push_back(label_ori);
    }
}

static void construct_net(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
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

    core::backend_t backend_type = core::default_engine();
    nn << conv(RECT_SIZE_X, RECT_SIZE_Y, 5, 1, 6,   // C1, 1@24x24-in, 6@20x20-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << ave_pool(20, 20, 6, 2)   // S2, 6@20x20-in, 6@10x10-out
       << relu()
       << conv(10, 10, 3, 6, 16,
               connection_table(tbl, 6, 16),// C3, 6@10x10-in, 16@8x8-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << ave_pool(8, 8, 16, 2)   // S4, 16@8x8-in, 6@4x4-out
       << relu()
       << conv(4, 4, 3, 16, 64,   // C4, 16@4x4-in, 64@2x2-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << conv(2, 2, 2, 64, 128,   // C5, 64@2x2-in, 128@1x1-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << fc(128, 2, true, backend_type)  // F6, 128-in, 2-out
       << tanh();
}

static void construct_net_two_layers(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using tanh = tiny_dnn::activation::tanh;
    using relu = tiny_dnn::activation::relu;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

    core::backend_t backend_type = core::default_engine();
    nn << conv(16, 16, 9, 1, 32,  // C1, 1@16x16-in, 32@8x8-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << tanh()
       << ave_pool(8, 8, 32, 2)   // S2, 32@8x8-in, 32@4x4-out
       << tanh()
       << conv(4, 4, 3, 32, 64,  // C1, 32@4x4-in, 64@2x2-out
                padding::valid, true, 1, 1, 1, 1, backend_type)
       << tanh()
       << ave_pool(2, 2, 64, 2)   // S2, 64@2x2-in, 64@1x1-out
       << relu()
       << fc(64, 2, true, backend_type)  // F6, 64-in, 2-out
       << relu();
}

static void train_lenet(tiny_dnn::network<tiny_dnn::sequential> &nn,
                        std::vector<vec_t> &trainging_data, std::vector<label_t> &trainging_labels,
                        std::vector<vec_t> &testing_data, std::vector<label_t> &testing_labels,
                        double learning_rate, const int n_train_epochs, const int n_minibatch)
{
    // specify loss-function and learning strategy
    tiny_dnn::adagrad optimizer;

    /// Choose a network here
    construct_net(nn);
    //construct_net_two_layers(nn);

    tiny_dnn::progress_display disp(trainging_data.size());
    tiny_dnn::timer t;

    optimizer.alpha *=
            std::min(tiny_dnn::float_t(4),
                     static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                  << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = nn.test(testing_data, testing_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(trainging_data.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<tiny_dnn::mse>(optimizer, trainging_data, trainging_labels, n_minibatch,
                            n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(testing_data, testing_labels).print_detail(std::cout);
    // save network model & trained weights
    nn.save("LeNet-model");
}

int main(){

//    /// Generating training data
//    std::string data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2/";
//    std::vector<cv::Mat> img_rects;
//    std::vector<int> labels_ori;
//    generateTrainingData(data_dir, img_rects, labels_ori, 6);
//
//    std::vector<vec_t> trainging_data;
//    convert_images(img_rects, trainging_data);
//    std::vector<label_t> trainging_labels;
//    convert_labels(labels_ori, trainging_labels);
//
//    /// Generating Validation data
//    std::string testing_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";
//    std::vector<cv::Mat> testing_img_rects;
//    std::vector<int> testing_labels_ori;
//    generateTrainingData(testing_data_dir, testing_img_rects, testing_labels_ori, 3);
//
//    std::vector<vec_t> testing_data;
//    convert_images(testing_img_rects, testing_data);
//    std::vector<label_t> testing_labels;
//    convert_labels(testing_labels_ori, testing_labels);
//
//    /// Training
//    tiny_dnn::network<tiny_dnn::sequential> nn;
//    train_lenet(nn, trainging_data, trainging_labels, testing_data, testing_labels, 1.0, 100, 32);

    /**----------------------------------------------**/

    /// Load model and testing
    tiny_dnn::network<tiny_dnn::sequential> model;
    model.load("LeNet-model");
    std::cout << "Model loaded!" << std::endl;

    /// Testing on images and show
    std::string testing_images_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";
    std::vector<std::string> image_names;
    getFileNames(testing_images_dir, image_names, ".png");
    clock_t start_time, end_time;

    for(const auto image_name : image_names){
        cv::Mat image_this = cv::imread(testing_images_dir+image_name, cv::IMREAD_GRAYSCALE);
        cv::Mat image_this_copy = image_this.clone();
        start_time = clock();
        int gateway_num = 0;
        for(int x=0; x<=image_this.cols-RECT_SIZE_X; x+=1){
            for(int y=0; y<image_this.rows-RECT_SIZE_Y; y+=1)
            {
                cv::Mat rect_this = image_this(cv::Rect(x, y, RECT_SIZE_X, RECT_SIZE_Y));
                vec_t data_this;
                convert_image(rect_this, data_this);
                label_t label_this =  model.predict_label(data_this);

                if(label_this > 0){
                    cv::circle(image_this_copy, cv::Point(x+RECT_SIZE_X/2, y+RECT_SIZE_Y/2), 2, cv::Scalar(0), 1);
                    gateway_num ++;
                }
            }
        }

        end_time = clock();
        std::cout <<"gateway_num = " << gateway_num <<std::endl;
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("image_this_copy", image_this_copy);
        cv::waitKey();
    }

    return 0;
}

