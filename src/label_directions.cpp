//
// Created by cc on 2020/3/25.
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

Mat img_resized;
string txt_filename;

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


void mouseEvent( int event, int x, int y, int flags, void* ustc)
{
    if((event == CV_EVENT_LBUTTONDOWN)&&(flags))
    {
        Mat cloned_img = img_resized.clone();
        Point pt = cvPoint(x,y);
        cv::circle(cloned_img, pt, 5, cv::Scalar(0), 2);
        imshow("img_resized", cloned_img);

        double direction = atan2(pt.y - 250, pt.x - 250);
        std::cout << "direction = " << direction << std::endl;

        ofstream outFile;
        outFile.open(txt_filename, ios::out);
        outFile << direction;
        outFile.close();
    }
}

int main(int argc, char** argv)
{
    string path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/30x30/positive/";
    std::vector<std::string> filenames;

    getFileNames(path, filenames, ".png");


    for(int file_seq = 0; file_seq < filenames.size(); ){
        string file = filenames[file_seq];

        txt_filename = path + file + ".csv";

        Mat img = imread(path + file, cv::IMREAD_GRAYSCALE);
        resize(img, img_resized, cv::Size(500, 500));

        Mat img_to_show = img_resized.clone();
        std::vector<std::vector<string>> data;

        if(readCSV(txt_filename, data)){
            double angle =  strtod(data[0][0].c_str(), NULL);
            cv::Point drawed_p;
            drawed_p.x = 250 + 100 * cos(angle);
            drawed_p.y = 250 + 100 * sin(angle);
            cv::circle(img_to_show, drawed_p, 5, cv::Scalar(0), 2);
        }

        imshow("img_resized", img_to_show);
        setMouseCallback("img_resized", mouseEvent, 0);
        char key = waitKey(0);
        if(key == 'd'){
            file_seq++;
        }else if(key == 'a'){
            if(file_seq > 0) file_seq--;
        }
    }



    return 0;
}