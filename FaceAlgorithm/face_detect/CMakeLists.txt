cmake_minimum_required(VERSION 2.6)

project(FaceDetect)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -Wall -g -ggdb")

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib) 

find_package(CUDA REQUIRED)
find_package(OpenCV)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()
include_directories(${CMAKE_SOURCE_DIR}/include)
# tensorrt
include_directories(${TensorRT_INCLUDE})
link_directories(${TensorRT_LIB})

cuda_add_library(FaceDetect SHARED ${PROJECT_SOURCE_DIR}/decode.cu face_preprocess.cu)
target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/lib64/libcudart.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer.so")
include_directories(${OpenCV_INCLUDE_DIRS})

