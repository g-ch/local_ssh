//
// Created by cc on 2020/3/20.
//

#ifndef LOCAL_SSH_VORONOI_SKELETON_POINTS_H
#define LOCAL_SSH_VORONOI_SKELETON_POINTS_H

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define SKELETON_POINT_COLOR 123

using namespace std;
using namespace cv;

bool inMatRange(cv::Point &p, cv::Mat &area_map, int shrink_size)
{
    if(p.x > shrink_size && p.x < area_map.cols - shrink_size && p.y > shrink_size && p.y < area_map.rows - shrink_size)
        return true;
    else
        return  false;
}

void drawVoronoi( Mat& img, Subdiv2D& subdiv )
{
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point> ifacet;

    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];

        Scalar color;
        color = SKELETON_POINT_COLOR; /// The color for voronoi skeleton

        int isize = ifacet.size();

        for(size_t k = 0; k < isize - 1; k++)
        {
            if(inMatRange(ifacet[k],img, 1) && img.ptr<unsigned char>(ifacet[k].y)[ifacet[k].x] > 200)
                line(img, ifacet[k], ifacet[k], color, 1);
        }
    }
}


void voronoiGenerate(cv::Mat &img, std::vector<cv::Point2f> &obstacle_points)
{
    /// Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    /// Remove points on the edge of the image, Keep obstacle points
    obstacle_points.clear();
    for(int i = 0; i < contours[0].size(); i++)
    {
        if(contours[0][i].y != 0 && contours[0][i].y != img.rows - 1 && contours[0][i].x != 0 && contours[0][i].x != img.cols - 1)
            obstacle_points.push_back(contours[0][i]);
    }

    /// Create a safe road area by large contour size
    cv::drawContours(img, contours, 0, cv::Scalar(50), 5);

    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);

    cv::Subdiv2D subdiv(rect);
    subdiv.insert(obstacle_points);

    drawVoronoi(img, subdiv);
}

void findVoronoiSkeletonPoints(cv::Mat map, std::vector<cv::Point> &skeleton_points){
    /// 1. Intensity filter
    cv::Mat map_eroded;
    map_eroded = map > 200;

    /// 2. Erode and dilate
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::erode(map, map_eroded, element); //Opening operation
    cv::dilate(map_eroded, map_eroded, element);

    /// 3. Flood fill to keep only one connected region
    cv::floodFill(map_eroded, cv::Point(map_eroded.cols/2, map_eroded.rows/2), cv::Scalar(100), 0, cv::Scalar(10), cv::Scalar(10), 8); /// Square area
    map_eroded = map_eroded == 100;

    /// 4. Remove small black pieces inside. Might be obstacles like pedestrians
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::dilate(map_eroded, map_eroded, element); /// Closing operation
    cv::erode(map_eroded, map_eroded, element);

    /// 5. Voronoi diagram
    std::vector<cv::Point2f> obstacle_points;
    voronoiGenerate(map_eroded, obstacle_points);
    for(int i=0; i<map_eroded.rows; i++){
        for(int j=0; j<map_eroded.cols; j++){
            if(map_eroded.ptr<unsigned char>(i)[j] == SKELETON_POINT_COLOR) {
                cv::Point p(j, i);
                skeleton_points.push_back(p);
            }
        }
    }

    /// 6. Visualization
//    cv::Mat image;
//    map.copyTo(image);
//    for(const auto &point : skeleton_points){
//        cv::circle(image, point, 1, cv::Scalar(0), 1);
//    }
//    cv::imshow("voronoi image", image);
//    cv::waitKey();

}

#endif //LOCAL_SSH_VORONOI_SKELETON_POINTS_H
