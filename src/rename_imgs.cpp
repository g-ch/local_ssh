//
// Created by cc on 2020/3/27.
//

#include <opencv2/opencv.hpp>
#include "preprocess.h"
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace cv;
using namespace std;

int main()
{
    string path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/data/negative/";
    std::vector<std::string> filenames;

    getFileNames(path, filenames, ".png");

    string add_symbol = "nn";

    for(int file_seq = 0; file_seq < filenames.size(); file_seq++){
        string file_name = filenames[file_seq];
        Mat img = imread(path + file_name, cv::IMREAD_GRAYSCALE);
        string new_name = add_symbol + file_name;
        cv::imwrite(path + new_name, img);
    }

    return 0;
}
