cmake_minimum_required(VERSION 3.0)
project(yolo_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# target
add_executable(yolov5_demo ${CMAKE_CURRENT_LIST_DIR}/yolov5_demo.cpp)
add_executable(yolov6_demo ${CMAKE_CURRENT_LIST_DIR}/yolov6_demo.cpp)
add_executable(yolov7_demo ${CMAKE_CURRENT_LIST_DIR}/yolov7_demo.cpp)
add_executable(yolov8_demo ${CMAKE_CURRENT_LIST_DIR}/yolov8_demo.cpp)
add_executable(yolov9_demo ${CMAKE_CURRENT_LIST_DIR}/yolov9_demo.cpp)

# link
if (MSVC)
    target_link_libraries(yolov5_demo MNN)
    target_link_libraries(yolov6_demo MNN)
    target_link_libraries(yolov7_demo MNN)
    target_link_libraries(yolov8_demo MNN)
    target_link_libraries(yolov9_demo MNN)
else()
    target_link_libraries(yolov5_demo MNN MNN_Express MNNOpenCV)
    target_link_libraries(yolov6_demo MNN MNN_Express MNNOpenCV)
    target_link_libraries(yolov7_demo MNN MNN_Express MNNOpenCV)
    target_link_libraries(yolov8_demo MNN MNN_Express MNNOpenCV)
    target_link_libraries(yolov9_demo MNN MNN_Express MNNOpenCV)
endif()
