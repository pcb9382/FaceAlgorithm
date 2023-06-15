#pragma once
#ifndef RECOGNITION_H
#define RECOGNITION_H
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "DataTypes_Face.h"
#include "prelu.h"
#define CHECK(status) \
do\
{\
    auto ret = (status);\
    if (ret != 0)\
    {\
        std::cerr << "Cuda failure: " << ret << std::endl;\
        abort();\
    }\
} while (0)

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1
using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
// static const int INPUT_H = 112;
// static const int INPUT_W = 112;
// static const int OUTPUT_SIZE = 512;
// const char* INPUT_BLOB_NAME = "data";
// const char* OUTPUT_BLOB_NAME = "prob";
//static Logger gLogger;

class Recognition
{
public:
    Recognition();
    ~Recognition();
    HZFLAG InitRecognition(Config&config);
    HZFLAG Extract_feature(cv::Mat&ImgVec,Feature&feature);                               //特征提取
    HZFLAG ReleaseRecognition();

public:
    int gpu_id;  // GPU id
    float conf_thresh;
    float nms_thresh;
    int INPUT_H;  // H, W must be able to  be divided by 32.
    int INPUT_W;
    int OUTPUT_SIZE;
    char* INPUT_BLOB_NAME;
    char* OUTPUT_BLOB_NAME;
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    float *data;
    float *prob;

public:

    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    std::map<std::string, Weights> loadWeights(const std::string file);
    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,std::string&wts_string);
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,std::string&wts_string);
    ILayer* resUnit(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int s, bool dim_match, std::string lname);
    ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname);

    //判断模型文件是否存在
    inline bool model_exists (const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
};

#endif 