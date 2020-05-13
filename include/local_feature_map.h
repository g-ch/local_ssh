//
// Created by cc on 2020/3/29.
//

#ifndef LOCAL_SSH_LOCAL_FEATURE_MAP_H
#define LOCAL_SSH_LOCAL_FEATURE_MAP_H

#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>


#define NOT_CLOSE 0
#define DISTANCE_CLOSE 1
#define CLOSE 2

typedef struct poseInPlane{
    float x;
    float y;
    float yaw;
}Pose2D;

typedef struct featureStruct{
    std::string label;
    Pose2D pose;
    float exsitence_confidence;
    bool in_fov_flag;
}Feature;


typedef struct featureInStruct{
    std::string label;
    Pose2D pose;
}FeatureIN;


class LocalFeatureMap {
public:

    std::vector<Feature> feature_map;

    std::queue<Pose2D> recorded_path;

    Pose2D vehicle_current_pose;

    Feature gateway_just_passed;


    /** Coefficients **/
    float Path_Record_Distance_Interval;
    float Pass_Gateway_Check_Distance;
    float Gateway_Normal_Length;
    float In_Gateway_Distance_threshold;

    float Confidence_Init;
    float Confidence_High;  // If confidence is larger than this, the feature would be treated as a correct observation result.
    float Confidence_Low;   // If confidence is smaller than this, the feature would be treated as a false observation result and would be deleted.
    float Confidence_Add_Step;
    float Confidence_Reduce_Step;
    float Confidence_Max;
    float Confidence_Min;

    float Gateway_Detect_Range; // vehicle is the center

private:
    int path_queue_size;

public:

    LocalFeatureMap();

    ~LocalFeatureMap()= default;

    ///*** Use this function in feature detection callback to update feature map. Even if feature_new.size ==0, the function should still be excuted
    /// *** NOTE: This function can not be excuted at the same time with updateVehiclePosition() unless a data lock is added.
    int addToFeatureMap(std::vector<FeatureIN> &features_new){
        std::vector<bool> feature_updated(feature_map.size(), false);
        if(!features_new.empty()){
            for(auto &feature_in : features_new){
                bool feature_asserted = false;
                for(int seq = 0; seq < feature_map.size(); seq++)
                {
                    auto &stored_feature = feature_map[seq];

                    int feature_close_flag = featureInCloseToStoredFeature(feature_in, stored_feature);
                    if(feature_close_flag == CLOSE){ //For features that are not gateway. Angle is always zero for now.
                        stored_feature.pose.x = (stored_feature.pose.x + feature_in.pose.x) / 2.f;
                        stored_feature.pose.y = (stored_feature.pose.y + feature_in.pose.y) / 2.f;
                        stored_feature.pose.yaw = averageOfTwoAngles(stored_feature.pose.yaw, feature_in.pose.yaw);
                        stored_feature.exsitence_confidence += Confidence_Add_Step;  // Add confidence
                        stored_feature.in_fov_flag = true;
                        feature_asserted = true;
                        feature_updated[seq] = true;
                        break;
                    } else if (feature_close_flag == DISTANCE_CLOSE){  // In this case, the vehicle might be near a gateway which leads to instability on predicted gateway directions.
                        stored_feature.pose.x = (stored_feature.pose.x + feature_in.pose.x) / 2.f;
                        stored_feature.pose.y = (stored_feature.pose.y + feature_in.pose.y) / 2.f;
                        if( stored_feature.exsitence_confidence < Confidence_High){   // Use newly observed position but ignore direction.
                            stored_feature.pose.yaw = averageOfTwoAngles(stored_feature.pose.yaw, feature_in.pose.yaw);
                        }
                        stored_feature.exsitence_confidence += Confidence_Add_Step;   // Add confidence
                        stored_feature.in_fov_flag = true;
                        feature_asserted = true;
                        feature_updated[seq] = true;
                    }
                }

                if(feature_asserted){
                    continue;
                }else{
                    // Add as new feature
                    Feature temp_feature;
                    temp_feature.label = feature_in.label;
                    temp_feature.pose.x = feature_in.pose.x;
                    temp_feature.pose.y = feature_in.pose.y;
                    temp_feature.pose.yaw = feature_in.pose.yaw;
                    temp_feature.exsitence_confidence = Confidence_Init;
                    temp_feature.in_fov_flag = true;
                    feature_map.push_back(temp_feature);
//                    std::cout << "Add new feature (" << temp_feature.pose.x <<", " <<temp_feature.pose.y<<")"<<std::endl;
                }
            }
        }
        // Reduce the confidence of undetected features nearby, which should be detected. If the confidence is too low, delete the feature.
        for(int seq=0; seq < feature_map.size(); seq++){
            if(feature_map[seq].in_fov_flag && !feature_updated[seq]){
                feature_map[seq].exsitence_confidence -= Confidence_Reduce_Step;

                if(feature_map[seq].exsitence_confidence < Confidence_Low){ //delete
                    feature_map.erase(feature_map.begin() + seq);
                    seq --;
                }
            }
            // Set a limit
            if(feature_map[seq].exsitence_confidence > Confidence_Max) feature_map[seq].exsitence_confidence = Confidence_Max;
            else if(feature_map[seq].exsitence_confidence < Confidence_Min) feature_map[seq].exsitence_confidence = Confidence_Min;
        }
    }


    ///*** Use this function in a position callback to update recorded vehicle path and check if pass any gateway
    /// *** NOTE: This function can not be excuted at the same time with addToFeatureMap() unless a data lock is added.
    bool updateVehiclePosition(Pose2D &vehicle_pose){
        bool arrived_at_new_node = false;
        vehicle_current_pose = vehicle_pose;

        // Update flags to represent whether a feature is still in the neighborhood area
        for(auto &feature : feature_map){
            if(feature.label == "gateway")  //update gateway flag here
            {
                if(pointDistance(vehicle_pose.x, vehicle_pose.y, feature.pose.x, feature.pose.y) > Gateway_Detect_Range){
                    feature.in_fov_flag = false;
                }
            }
            else{
                /// TODO: update other features' in FOV flag
            }
        }

        // Update path track queue
        if(recorded_path.empty()){
            recorded_path.push(vehicle_pose);
        }else{
            if(pointDistance(vehicle_pose.x, vehicle_pose.y, recorded_path.back().x, recorded_path.back().y) > Path_Record_Distance_Interval){
                recorded_path.push(vehicle_pose);  //record
            }
            if(recorded_path.size() > path_queue_size){
                recorded_path.pop();
            }

            if(recorded_path.size() > 2){
                /* Check if pass a gateway. If pass, check and correct the direction of the passed gateway. */
                bool if_pass_any_gateway = checkIfPassedGateway(gateway_just_passed);
                if(if_pass_any_gateway){
                    arrived_at_new_node = true;
                    std::cout << "Passed Gateway (" << gateway_just_passed.pose.x << ", " <<gateway_just_passed.pose.y << ")"<<std::endl;
                    /* Check and correct the direction of this passed gateway */
                    for(auto &feature : feature_map){  // Find the gateway in map and correct its direction
                        if(feature.label == "gateway" && feature.pose.x == gateway_just_passed.pose.x && feature.pose.y == gateway_just_passed.pose.y){
                            float dx = feature.pose.x - vehicle_pose.x;
                            float dy = feature.pose.y - vehicle_pose.y;
                            float angle_vec_x = cos(feature.pose.yaw);
                            float angle_vec_y = sin(feature.pose.yaw);
                            if(dx*angle_vec_x + dy*angle_vec_y > 0){
                                reverseAngle(feature.pose.yaw);
                            }
                        }
                        feature.in_fov_flag = false; //Passed this gateway, all in_fov_flag should be false
                    }
                }
            }
        }

//        printMapInfo();

        return arrived_at_new_node;
    }

    bool deleteFromFeatureMap(std::vector<FeatureIN> &features_to_delete){
        /* Keep the gateway_just_passed and removed the newly detected gateways*/
        for(const auto &feature_d : features_to_delete){
            if(!featureInCloseToStoredFeature(feature_d, gateway_just_passed)){
                for(auto it = feature_map.begin(); it != feature_map.end(); it++)
                {
                    if(featureInCloseToStoredFeature(feature_d, *it)){
                        feature_map.erase(it);
                        break;
                    }
                }
            }
        }
    }

    void getGatewayBoundaryPoints(const Feature &gateway, float &boundary_point1_x, float &boundary_point1_y, float &boundary_point2_x,float &boundary_point2_y)
    {
        float boundary_point_direction1 = gateway.pose.yaw + M_PI_2;
        float boundary_point_direction2 = gateway.pose.yaw - M_PI_2;
        boundary_point1_x = Gateway_Normal_Length / 2 * cos(boundary_point_direction1) + gateway.pose.x;
        boundary_point1_y = Gateway_Normal_Length / 2 * sin(boundary_point_direction1) + gateway.pose.y;
        boundary_point2_x = Gateway_Normal_Length / 2 * cos(boundary_point_direction2) + gateway.pose.x;
        boundary_point2_y = Gateway_Normal_Length / 2 * sin(boundary_point_direction2) + gateway.pose.y;
    }

    void getGatewayBoundaryPixelPoints(int gateway_pose_x, int gateway_pose_y, float gateway_direction, float &boundary_point1_x,
            float &boundary_point1_y, float &boundary_point2_x,float &boundary_point2_y, int gateway_length = 16)
    {
        float boundary_point_direction1 = gateway_direction + M_PI_2;
        float boundary_point_direction2 = gateway_direction - M_PI_2;
        boundary_point1_x = gateway_length / 2 * cos(boundary_point_direction1) + gateway_pose_x;
        boundary_point1_y = gateway_length / 2 * sin(boundary_point_direction1) + gateway_pose_y;
        boundary_point2_x = gateway_length / 2 * cos(boundary_point_direction2) + gateway_pose_x;
        boundary_point2_y = gateway_length / 2 * sin(boundary_point_direction2) + gateway_pose_y;
    }

private:
    bool checkIfPassedGateway(Feature &gateway_passed){
        for(const auto &feature : feature_map){
            if(feature.label == "gateway" && feature.exsitence_confidence >= Confidence_High){

                float boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y;
                getGatewayBoundaryPoints(feature, boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y);

                float vehicle_gateway_distance = calculatePointToLineSegmentDistance(vehicle_current_pose.x, vehicle_current_pose.y,
                        boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y);

                // Skip check if in (very close to) this gateway or too far from the gateway
                if(vehicle_gateway_distance < In_Gateway_Distance_threshold || vehicle_gateway_distance > Pass_Gateway_Check_Distance){
//                    std::cout << "distance is too close or too far." << std::endl;
                    continue;
                }

                // If not too close to or too far from the gateway, judge if pass the gateway
                bool if_passed_this_gateway = judgeIfLinesABCross(boundary_point1_x, boundary_point1_y, boundary_point2_x, boundary_point2_y,
                                                                  recorded_path.front().x, recorded_path.front().y, recorded_path.back().x, recorded_path.back().y);
//                std::cout << "cross line = " << if_passed_this_gateway << std::endl;

                if(if_passed_this_gateway){
                    gateway_passed = feature;
                    return true;
                }
            }
        }

        // Checking finished and no gateway was passed through
        return false;
    }


    static float pointDistance(float p1_x, float p1_y, float p2_x, float p2_y) {
        return sqrt(pointSquareDistance(p1_x, p1_y, p2_x, p2_y));
    }

    static float pointSquareDistance(float p1_x, float p1_y, float p2_x, float p2_y) {
        return (p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y);
    }

    static float angleDifferenceABS(float yaw1, float yaw2){
        correctAngleToPiRange(yaw1);
        correctAngleToPiRange(yaw2);

        if(fabs(yaw1-yaw2) <= M_PI){
            return fabs(yaw1 -yaw2);
        } else{
            return M_PI *2 - fabs(yaw1 -yaw2);
        }
    }

    static float averageOfTwoAngles(float yaw1, float yaw2){
        float average;
        correctAngleToPiRange(yaw1);
        correctAngleToPiRange(yaw2);

        if(yaw1 * yaw2 < 0 && fabs(yaw1-yaw2) > M_PI){
            if(yaw1 < 0) yaw1 += M_PI *2;
            else yaw2 += M_PI *2;
        }

        average = (yaw1 + yaw2) / 2.f;
        correctAngleToPiRange(average);

        return average;
    }

    static void reverseAngle(float &yaw){
        yaw += M_PI;
        correctAngleToPiRange(yaw);
    }

    static void correctAngleToPiRange(float &yaw){
        if(yaw >= -M_PI && yaw <= M_PI){ return; }
        while(yaw < -M_PI){
            yaw += M_PI *2;
        }
        while(yaw > M_PI){
            yaw -= M_PI *2;
        }
    }

    static bool judgeIfLinesABCross(float Ax1,float Ay1,float Ax2,float Ay2,float Bx1,float By1,float Bx2,float By2)
    {
        if(( std::max(Ax1,Ax2)>=std::min(Bx1,Bx2)&&std::min(Ax1,Ax2)<=std::max(Bx1,Bx2) )&&
            (std::max(Ay1,Ay2)>=std::min(By1,By2)&&std::min(Ay1,Ay2)<=std::max(By1,By2)))
        {
            return ((Bx1 - Ax1) * (Ay2 - Ay1) - (By1 - Ay1) * (Ax2 - Ax1)) *
                   ((Bx2 - Ax1) * (Ay2 - Ay1) - (By2 - Ay1) * (Ax2 - Ax1)) <= 0 &&
                   ((Ax1 - Bx1) * (By2 - By1) - (Ay1 - By1) * (Bx2 - Bx1)) *
                   ((Ax2 - Bx1) * (By2 - By1) - (Ay2 - By1) * (Bx2 - Bx1)) <= 0;
        }
        else {return false;}
    }

    static float calculatePointToLineSegmentDistance(float Px, float Py, float Ax,float Ay,float Bx, float By)
    {
        float square_distance_ab = pointSquareDistance(Ax, Ay, Bx, By);
        float r = ( (Px-Ax)*(Bx-Ax) + (Py-Ay)*(By-Ay) ) / square_distance_ab;
        if(r < 0){
            return pointDistance(Px, Py, Ax, Ay);
        }else if(r > 1){
            return pointDistance(Px, Py, Bx, By);
        }else{
            float distance_ab = sqrt(square_distance_ab);
            float projection_length = r * distance_ab;
            return sqrt(pointSquareDistance(Px, Py, Ax, Ay) - projection_length*projection_length);
        }
    }

    int featureInCloseToStoredFeature(const FeatureIN &f1, const Feature &f2, float distance_threshold = 1.2, float direction_threshold = 3.14) // m ,rad
    {
        /** Return: 0: not close; 1: position close but yaw is very different; 2: position and yaw are both close **/
        if(f1.label != f2.label){
            return NOT_CLOSE;
        }

        if(pointSquareDistance(f1.pose.x, f1.pose.y, f2.pose.x, f2.pose.y) >  distance_threshold*distance_threshold) {
            return NOT_CLOSE;
        }else{
            if(angleDifferenceABS(f1.pose.yaw, f2.pose.yaw) > direction_threshold){
                return DISTANCE_CLOSE;
            }else{
                return CLOSE;
            }
        }
    }


    void printMapInfo(){
        std::cout << std::endl;
        std::cout << "current map:" << std::endl;
        for(const auto &feature : feature_map){
            std::cout<<"Feature label = " << feature.label << ", px=" << feature.pose.x << ", py=" << feature.pose.y << ", direction="
            << feature.pose.yaw << ", confidence=" << feature.exsitence_confidence << ", if nearby=" << feature.in_fov_flag << std::endl;
        }
        std::cout << std::endl;
    }


};


LocalFeatureMap::LocalFeatureMap() :
                    Path_Record_Distance_Interval(0.15f),
                    Pass_Gateway_Check_Distance(1.5f),
                    Gateway_Normal_Length(4.f),
                    Confidence_Init(0.5),
                    Confidence_High(0.65),
                    Confidence_Low(0.2),
                    Confidence_Add_Step(0.1),
                    Confidence_Reduce_Step(0.04),
                    Confidence_Max(1.2),
                    Confidence_Min(0),
                    In_Gateway_Distance_threshold(0.5),
                    Gateway_Detect_Range(3.2)
{
    path_queue_size = Pass_Gateway_Check_Distance / Path_Record_Distance_Interval;

    /** to test**/
//    std::vector<FeatureIN> features_in;
//    FeatureIN test_feature;
//    test_feature.label = "gateway";
//    test_feature.pose.x = 1.0;
//    test_feature.pose.y = 0.5;
//    test_feature.pose.yaw = 0.0;
//    features_in.push_back(test_feature);
//
//    test_feature.pose.x = 0.0;
//    test_feature.pose.y = 2.0;
//    test_feature.pose.yaw = 1.57;
//    features_in.push_back(test_feature);
//
//    addToFeatureMap(features_in);
//
//    Pose2D pose;
//    pose.x = -0.4;
//    pose.y = 0.0;
//    pose.yaw = 0.0;
//
//    for(int i=0; i<40; i++){
//        pose.x += 0.1;
//        pose.y += 0.01;
//        std::cout << "step = " << i << " position=(" <<pose.x<<", "<<pose.y <<")" << std::endl;
//
//        updateVehiclePosition(pose);
//
//        if(i>5 && i<20){
//            addToFeatureMap(features_in);
//        }
//    }
}


#endif //LOCAL_SSH_LOCAL_FEATURE_MAP_H
