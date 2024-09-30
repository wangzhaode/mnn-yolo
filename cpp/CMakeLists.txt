cmake_minimum_required(VERSION 3.0)
project(yolo_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

function(add_yolo_demo_target TARGET_NAME)
  add_executable(${TARGET_NAME} "${CMAKE_CURRENT_LIST_DIR}/${TARGET_NAME}.cpp")

  if (MSVC)
    target_link_libraries(${TARGET_NAME} PRIVATE MNN)
  else()
    target_link_libraries(${TARGET_NAME} PRIVATE MNN MNN_Express MNNOpenCV)
  endif()
endfunction()

# yolov5 - yolov10
foreach(DEMO_NUMBER RANGE 5 10)
  add_yolo_demo_target("yolov${DEMO_NUMBER}_demo")
endforeach()
add_yolo_demo_target("yolo11_demo")