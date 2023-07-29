cmake_minimum_required(VERSION 2.6)

project(FaceRecognition)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -Wall -g -ggdb")

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(CUDA REQUIRED)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

include_directories(${TensorRT_INCLUDE})
link_directories(${TensorRT_LIB})


cuda_add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/prelu.cu)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
else()
    target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/lib64/libcudart.so")
endif()
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})




