
cmake_minimum_required (VERSION 3.8)
project(FaceRecognition_Test)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -Wall -g -ggdb")

file(GLOB test_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB test_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

set(CMAKE_THREAD_LIBS_INIT -lpthred)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_executable (${PROJECT_NAME} ${test_src} ${test_header})

#第三方库路径
find_package(OpenCV)
IF(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
ELSE()
    MESSAGE(STATUS "Cannot find opencv libs")
ENDIF()

include_directories(${CMAKE_SOURCE_DIR})
include_directories("${CMAKE_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME} FaceDetectRecognition)

