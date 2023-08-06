
cmake_minimum_required (VERSION 3.8)
project(FaceDetectRecognition)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")#-Wall-g -ggdb

#源文件
file(GLOB lib_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

#头文件
file(GLOB lib_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

#设置库文件输出目录
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib) 

#头文件目录
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/face_detect)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/face_detect_yolov5face)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/face_detect_yolov7face)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/face_detect_yolov8face)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/face_recognition)
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/mask_recognition)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/gender_age_recognition)
include_directories(${CMAKE_SOURCE_DIR}/FaceAlgorithm/silent_face_anti_spoofing)

find_package(CUDA REQUIRED)
# tensorrt
include_directories(${TensorRT_INCLUDE})
link_directories(${TensorRT_LIB})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()
include_directories(${NVINFER_INCLUDE_DIR})

#库文件目录
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#增加动态链接库
add_library(${PROJECT_NAME} SHARED ${lib_src})
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
else()
    target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/lib64/libcudart.so")
endif()
target_link_libraries(${PROJECT_NAME} FaceRecognition)
target_link_libraries(${PROJECT_NAME} FaceDetect)
target_link_libraries(${PROJECT_NAME} Yolov5FaceDetect)
target_link_libraries(${PROJECT_NAME} Yolov7FaceDetect)
target_link_libraries(${PROJECT_NAME} Yolov8FaceDetect)
target_link_libraries(${PROJECT_NAME} MaskRecognition)
target_link_libraries(${PROJECT_NAME} GenderAgeRecognition)
target_link_libraries(${PROJECT_NAME} SilentFaceAntiSpoofing)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

