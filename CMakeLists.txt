cmake_minimum_required(VERSION 2.8.3)
project(local_ssh)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O1 -ftree-vectorize -march=native)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_filters
  mav_msgs
  geometry_msgs
  sensor_msgs
  gazebo_msgs
  tf
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(${PCL_DEFINITIONS})

catkin_package()

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(save_pcd_jpgs src/save_pcd_jpgs.cpp)
target_link_libraries(save_pcd_jpgs ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(detect_test src/detect_test.cpp)
target_link_libraries(detect_test ${catkin_LIBRARIES} ${OpenCV_LIBS})

