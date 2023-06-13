cmake_minimum_required(VERSION 2.6)

project(SilentFaceAntiSpoofing)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O0 -g -rdynamic -g2 -ggdb")
find_package(CUDA REQUIRED)


SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib) 

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/common)

file(GLOB lib_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB common_header "${CMAKE_SOURCE_DIR}/common/*.h")

file(GLOB lib_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB common_src "${CMAKE_SOURCE_DIR}/common/*.cpp")

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

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(${PROJECT_NAME} SHARED ${lib_src} ${common_src})

target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/lib64/libcudart.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvonnxparser.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer_plugin.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libcudnn.so.8.2.1")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} pthread)

