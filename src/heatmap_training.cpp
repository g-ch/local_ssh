//
// Created by cc on 2020/5/13.
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


void construct_autoencoder(tiny_dnn::network<tiny_dnn::sequential> &nn) {
    // construct nets
    nn << tiny_dnn::convolutional_layer(32, 32, 5, 1, 6)
       << tiny_dnn::tanh_layer(28, 28, 6)
       << tiny_dnn::average_pooling_layer(28, 28, 6, 2)
       << tiny_dnn::convolutional_layer(14, 14, 5, 6, 16)
       << tiny_dnn::tanh_layer(10, 10, 16)
       << tiny_dnn::deconvolutional_layer(10, 10, 5, 16, 6)
       << tiny_dnn::tanh_layer(14, 14, 6)
       << tiny_dnn::average_unpooling_layer(14, 14, 6, 2)
       << tiny_dnn::deconvolutional_layer(28, 28, 5, 6, 1)
       << tiny_dnn::tanh_layer(32, 32, 1);
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


static void train(tiny_dnn::network<tiny_dnn::sequential> &nn,
                    std::vector<vec_t> &trainging_data, std::vector<vec_t> &trainging_labels,
                    double learning_rate,  int n_train_epochs,  int n_minibatch, std::string model_name)
{
    shuffle(trainging_data, trainging_labels);

    // specify loss-function and learning strategy
//    tiny_dnn::adagrad optimizer;
    tiny_dnn::adam optimizer;

//    construct_autoencoder(nn);
    construct_autoencoder(nn);

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
//        float loss = nn.get_loss<tiny_dnn::mse>(trainging_data, trainging_labels);
//        std::cout << loss << "/" << std::endl;

        shuffle(trainging_data, trainging_labels);

        disp.restart(trainging_data.size());
        t.restart();
    };

    // training
    nn.fit<tiny_dnn::mse>(optimizer, trainging_data, trainging_labels, n_minibatch,
                          n_train_epochs, []() {},  on_enumerate_epoch);
    std::cout << "end training." << std::endl;

    // save network model & trained weights
    nn.save(model_name);
}

int main(){

    std::string data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/sudoku2/";
    std::vector<std::string> sample_file_names;
    getFileNames(data_dir, sample_file_names, ".pcd", false);

    std::vector<cv::Mat> samples_imgs, samples_heatmaps;

    for(const auto & file : sample_file_names){
        cv::Mat img = cv::imread(data_dir + file + ".png", cv::IMREAD_GRAYSCALE);
        cv::Mat heatmap = cv::imread(data_dir + file + "_heatmap.png", cv::IMREAD_GRAYSCALE);

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(32, 32));

        samples_imgs.push_back(resized_img);
        samples_heatmaps.push_back(heatmap);
    }

    /// Convert to required data form
    std::vector<vec_t> training_data_desired_form, heatmap_data_desired_form;

    convert_images(samples_imgs, training_data_desired_form);
    convert_images(samples_heatmaps, heatmap_data_desired_form);

    tiny_dnn::network<tiny_dnn::sequential> nn;
    train(nn, training_data_desired_form, heatmap_data_desired_form, 1.0, 200, 32, "32_in_heatmap_model");

    std::cout << "------------- training finished! -----------------" << std::endl;

    /// Test
//    tiny_dnn::network<tiny_dnn::sequential> model;
//    model.load("32_in_heatmap_model");
//
//    for(auto & img_vec : training_data_desired_form)
//    {
//        auto output = model.predict_label(img_vec);
//
//    }

    return 0;
}