
cmake_minimum_required (VERSION 2.6)

project (FaceAlgorithmDemo)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# tensorrt 目录
set(TensorRT_INCLUDE "/home/pcb/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/home/pcb/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")

#添加子目录
add_subdirectory("FaceAlgorithm")
add_subdirectory("FaceAlgorithm_Test")

