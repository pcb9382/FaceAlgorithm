#pragma once
#ifndef DETECTOR_H
#define DETECTOR_H
#include <memory>
#include <vector>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "DataTypes_Face.h"
#include <dirent.h>
#include "NvInfer.h"
#include "decode.h"
#include "face_preprocess.h"
#define USE_FP16
#define MAX_IMAGE_INPUT_SIZE_THRESH 8000 * 6000
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


using namespace nvinfer1;

class Detector
{ 
public:
    Detector();
    ~Detector();
    HZRESULT InitDetector(Config&config);
    HZRESULT detect(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>&dets);
    HZRESULT ReleaseDetector();

public:
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* buffers[2];
    int inputIndex;
    int outputIndex;
    cudaStream_t stream;
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    int gpu_id;  // GPU id
    float conf_thresh;
    float nms_thresh;
    int INPUT_H;  // H, W must be able to  be divided by 32.
    int INPUT_W;
    int OUTPUT_SIZE;
    char* INPUT_BLOB_NAME;
    char* OUTPUT_BLOB_NAME;
    char *trtModelStream{nullptr};


public:
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers,float* output, int batchSize);
    cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
    int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
    void get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det);
    float iou(float lbox[4], float rbox[4]);
    static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b);
    void nms(std::vector<decodeplugin::Detection>& res, float *output); 
    // [type] [size] <data x size in hex>
    std::map<std::string, Weights> loadWeights(const std::string file);
    Weights getWeights(std::map<std::string, Weights>& weightMap, std::string key);
    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,std::string&wts_string);
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,std::string&wts_string);
    IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup);
    ILayer*conv_dw(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int s = 1, float leaky = 0.1);
    ILayer* conv_bn1X1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1, float leaky = 0.1);
    ILayer* conv_bn_no_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1);
    ILayer* conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s = 1, float leaky = 0.1);
    inline bool model_exists (const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
};
#endif 