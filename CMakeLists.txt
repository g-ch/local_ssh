cmake_minimum_required(VERSION 2.8.3)
project(local_ssh)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++14 -g -O1 -ftree-vectorize -march=native)

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
find_package(Threads)

catkin_package()

set(tinyxml2_LIB
        /usr/local/lib/libtinyxml2.so
        )

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(save_pcd_jpgs src/save_pcd_jpgs.cpp)
target_link_libraries(save_pcd_jpgs ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(detect_test src/detect_test.cpp)
target_link_libraries(detect_test ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test src/test.cpp)
target_link_libraries(test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${tinyxml2_LIB})

add_executable(gateway_training src/gateway_training.cpp)
target_link_libraries(gateway_training  ${OpenCV_LIBS} ${tinyxml2_LIB})

add_executable(gateway_training_gvg src/gateway_training_gvg.cpp)
target_link_libraries(gateway_training_gvg  ${OpenCV_LIBS} ${tinyxml2_LIB} ${CMAKE_THREAD_LIBS_INIT})

add_executable(gateway_cnn_training src/gateway_cnn_training.cpp)
target_link_libraries(gateway_cnn_training  ${OpenCV_LIBS} ${tinyxml2_LIB} ${CMAKE_THREAD_LIBS_INIT})

add_executable(rect_classification src/rect_classification.cpp)
target_link_libraries(rect_classification  ${OpenCV_LIBS} ${tinyxml2_LIB} ${CMAKE_THREAD_LIBS_INIT})

add_executable(test_svm src/test_svm.cpp)
target_link_libraries(test_svm  ${OpenCV_LIBS} ${tinyxml2_LIB})

