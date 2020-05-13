//
// Created by cc on 2020/5/13.
//


#include "preprocess.h"
#include <iostream>
#include <string>
#include "labelimg_xml_reader.h"

int main(){

    std::string data_dir = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/new/sudoku2/";
    std::vector<std::string> sample_file_names;
    getFileNames(data_dir, sample_file_names, ".pcd", false);

    for(const auto& filename : sample_file_names) {
        /// Read xml
        std::string xml_path = data_dir + filename + ".xml";
        std::string img_path;
        int img_width, img_height, img_depth;
        std::vector<Object> objects;
        bool xml_exist = readLabelIMGObjectDetectionXML(xml_path, img_path, img_width, img_height, img_depth, objects);

        int heatmap_width = 32;
        int heatmap_height = 32;
        cv::Mat heat_map = cv::Mat::zeros(heatmap_width, heatmap_height, CV_8UC1);
        float scale_width = img_width / heatmap_width;
        float scale_height = img_height / heatmap_height;

        if(xml_exist){
            for (const auto &object : objects) {
                if (object.label == "gateway") {
                    int row = (object.y_min + object.y_max)/ 2 / scale_height;
                    int col = (object.x_min + object.x_max)/ 2 / scale_width;
                    heat_map.at<uchar>(row, col) = 255;
                }
            }
        }

        cv::imwrite(data_dir + filename +"_heatmap.png", heat_map);
    }

    return 0;
}
