//
// Created by cc on 2020/3/21.
//

//
// Created by cc on 2020/3/18.
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

void convert_labels(std::vector<size_t> &labels_ori, std::vector<label_t> &labels){
    for(const auto &label_ori : labels_ori){
        labels.push_back((label_t)label_ori);
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

    core::backend_t backend_type = core::default_engine();
    nn << conv(RECT_SIZE_X, RECT_SIZE_Y, 7, 1, 6,   // C1, 1@26x26-in, 6@20x20-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu(20,20,6)
       << max_pool(20, 20, 6, 2)   // S2, 6@20x20-in, 6@10x10-out
       << relu(10,10,6)
       << conv(10, 10, 3, 6, 16,
               connection_table(tbl, 6, 16),// C3, 6@10x10-in, 16@8x8-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu(8,8,16)
       << max_pool(8, 8, 16, 2)   // S4, 16@8x8-in, 16@4x4-out
       << relu(4,4,16)
       << conv(4, 4, 3, 16, 64,   // C4, 16@4x4-in, 64@2x2-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu(2,2,64)
       << conv(2, 2, 2, 64, 128,   // C5, 64@2x2-in, 128@1x1-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu(1,1,128)
       << fc(128, 2, true, backend_type)  // F6, 128-in, 2-out
       << tanh(2);
}


static void construct_deep_net(network<sequential>& nn) {
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using max_pool = tiny_dnn::layers::max_pool;
    using tanh = tiny_dnn::activation::tanh;
    using relu = tiny_dnn::activation::relu;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

    core::backend_t backend_type = core::default_engine();
    nn << conv(RECT_SIZE_X, RECT_SIZE_Y, 3, 1, 32,   // C1, 1@26x26-in, 16@24x24-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << max_pool(24, 24, 32, 2)   // S2, 16@24x24-in, 16@12x12-out
       << relu()
       << conv(12, 12, 3, 32, 64,  // C3, 16@12x12-in, 32@10x10-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << max_pool(10, 10, 64, 2)   // S4, 32@10x10-in, 32@5x5-out
       << relu()
       << conv(5, 5, 3, 64, 128,   // C5, 32@5x5-in, 64@3x3-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << conv(3, 3, 3, 128, 256,   // C6, 64@3x3-in, 128@1x1-out
               padding::valid, true, 1, 1, 1, 1, backend_type)
       << relu()
       << fc(128, 32, true, backend_type)  // F6, 128-in, 32-out
       << relu()
       << fc(32, 2, true, backend_type)  // F6, 128-in, 32-out
       << relu();

}


void shuffle(std::vector<vec_t>& images, std::vector<label_t>& labels)
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
                        std::vector<vec_t> &trainging_data, std::vector<label_t> &trainging_labels,
                        std::vector<vec_t> &testing_data, std::vector<label_t> &testing_labels,
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
        tiny_dnn::result res = nn.test(testing_data, testing_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        shuffle(trainging_data, trainging_labels);

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
    nn.save("LeNet-model-rects");
}

void mergeAndConvertImages(std::vector<cv::Mat> positive_images, std::vector<cv::Mat> negative_images, std::vector<vec_t> &results)
{
    positive_images.insert(positive_images.end(), negative_images.begin(), negative_images.end());
    convert_images(positive_images, results);
}

void mergeAndConvertLabels(std::vector<size_t> postive_labels, std::vector<size_t> negative_labels, std::vector<label_t> &results)
{
    postive_labels.insert(postive_labels.end(), negative_labels.begin(), negative_labels.end());
    convert_labels(postive_labels, results);
}

int main(){
//
    bool if_training = false;
    if(if_training){
        /// Read images
        std::string positive_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Rect/Positive_ori_2/";
        std::string negative_data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Rect/Negative_ori_2/";

//        std::string positive_data_dir = "/home/cc/Downloads/mnist/mnist_train/2/";
//        std::string negative_data_dir = "/home/cc/Downloads/mnist/mnist_train/3/";

        std::vector<std::string> positive_sample_file_names, negative_sample_file_names;
        getFileNames(positive_data_dir, positive_sample_file_names, ".png");
        getFileNames(negative_data_dir, negative_sample_file_names, ".png");

        std::vector<cv::Mat> positive_sample_rects, negative_sample_rects;
        for(const auto & file : positive_sample_file_names){
            cv::Mat temp_img = cv::imread(positive_data_dir + file, cv::IMREAD_GRAYSCALE);
            positive_sample_rects.push_back(temp_img);
        }

        std::shuffle(positive_sample_rects.begin(), positive_sample_rects.end(), std::mt19937(std::random_device()()));

        for(const auto & file : negative_sample_file_names){
            cv::Mat temp_img = cv::imread(negative_data_dir + file, cv::IMREAD_GRAYSCALE);
            negative_sample_rects.push_back(temp_img);
        }
        std::shuffle(negative_sample_rects.begin(), negative_sample_rects.end(), std::mt19937(std::random_device()()));

        /// Convert to required data form
        float validation_ratio = 0.05;
        int positive_data_training_num = positive_sample_rects.size() * (1.f-validation_ratio);
        int negative_data_training_num = negative_sample_rects.size() * (1.f-validation_ratio);
        int positive_data_validation_num = positive_sample_rects.size() - positive_data_training_num;
        int negative_data_validation_num = negative_sample_rects.size() - negative_data_training_num;

        std::cout << "positive_data_training_num = " << positive_data_training_num << std::endl;
        std::cout << "negative_data_training_num = " << negative_data_training_num << std::endl;
        std::cout << "positive_data_validation_num = " << positive_data_validation_num << std::endl;
        std::cout << "negative_data_validation_num = " << negative_data_validation_num << std::endl;

        std::vector<cv::Mat> positive_training_data = vectorRangeCopy(positive_sample_rects, 0, positive_data_training_num);
        std::cout << "positive_training_data_size = " << positive_training_data.size() <<std::endl;
        std::vector<cv::Mat> negative_training_data = vectorRangeCopy(negative_sample_rects, 0, negative_data_training_num);
        std::vector<cv::Mat> positive_validation_data = vectorRangeCopy(positive_sample_rects, positive_data_training_num, positive_data_validation_num);
        std::vector<cv::Mat> negative_validation_data = vectorRangeCopy(negative_sample_rects, negative_data_training_num, negative_data_validation_num);

        std::vector<size_t> positive_training_data_labels(positive_data_training_num, 0);
        std::vector<size_t> negative_training_data_labels(negative_data_training_num, 1);
        std::vector<size_t> positive_validation_data_labels(positive_data_validation_num, 0);
        std::vector<size_t> negative_validation_data_labels(negative_data_validation_num, 1);

        std::vector<vec_t> training_data_desired_form, validation_data_desired_form;
        std::vector<label_t> training_labels_desired_form, validation_labels_desired_form;

        mergeAndConvertImages(positive_training_data, negative_training_data, training_data_desired_form);
        mergeAndConvertImages(positive_validation_data, negative_validation_data, validation_data_desired_form);
        mergeAndConvertLabels(positive_training_data_labels, negative_training_data_labels, training_labels_desired_form);
        mergeAndConvertLabels(positive_validation_data_labels, negative_validation_data_labels, validation_labels_desired_form);

        std::cout << "training_data_num = " << training_data_desired_form.size() << std::endl;
        std::cout << "validation_data_num = " << validation_data_desired_form.size() << std::endl;


        /// Training
        tiny_dnn::network<tiny_dnn::sequential> nn;
        train_lenet(nn, training_data_desired_form, training_labels_desired_form, validation_data_desired_form, validation_labels_desired_form, 1.0, 100, 32);

    }

    /**----------------------------------------------**/

    /// Load model and testing
    tiny_dnn::network<tiny_dnn::sequential> model;
    model.load("LeNet-model-rects");
    std::cout << "Model loaded!" << std::endl;

    /// Testing on images and show
    std::string testing_images_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor2_test/";
    std::vector<std::string> image_names;
    getFileNames(testing_images_dir, image_names, ".png");
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

        int gateway_num = 0;

        for(auto &sk_point : skeleton_points){
            if(sk_point.x < RECT_SIZE_X/2 || sk_point.y <RECT_SIZE_Y/2 || sk_point.x > image_this.cols-RECT_SIZE_X/2 || sk_point.y > image_this.rows-RECT_SIZE_Y/2){
                continue;
            }else{
                cv::Mat rect_this = image_this(cv::Rect(sk_point.x-RECT_SIZE_X/2, sk_point.y-RECT_SIZE_Y/2, RECT_SIZE_X, RECT_SIZE_Y));
                vec_t data_this;
                convert_image(rect_this, data_this);
                label_t label_this =  model.predict_label(data_this);

                if(label_this < 0.5){
                    cv::circle(image_this_copy, cv::Point(sk_point.x, sk_point.y), 1, cv::Scalar(0), 1);
                    gateway_num ++;
                }
            }
        }

        end_time = clock();
        std::cout <<"gateway_num = " << gateway_num <<" in image "<< image_name << std::endl;
        std::cout << "Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

        cv::imshow("image_this_copy", image_this_copy);
        cv::waitKey();
    }

    return 0;
}

