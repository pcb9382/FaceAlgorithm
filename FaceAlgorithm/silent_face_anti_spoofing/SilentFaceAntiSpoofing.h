#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include "dirent.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "ONNX2TRT.h"
#include "DataTypes_Face.h"
using namespace nvinfer1;
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


class SilentFaceAntiSpoofing
{
public:
	SilentFaceAntiSpoofing();
	~SilentFaceAntiSpoofing();
	HZRESULT SilentFaceAntiSpoofingInit(Config&config);
	HZRESULT SilentFaceAntiSpoofingRun(cv::Mat&img, SilentFace&silentface);
	HZRESULT SilentFaceAntiSpoofingRelease();
private:
	char* INPUT_BLOB_NAME;
	char* OUTPUT_BLOB_NAME;
	int INPUT_H;
	int INPUT_W;
	int OUTPUT_SIZE; //class num
	float *data ;
	float *prob;
	Logger gLogger;
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;

private:
	cv::Mat CenterCrop(cv::Mat img);
	std::vector<float> softmax(float *prob);
	void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	bool model_exists(const std::string& name);
	int softmax(float *prob,SilentFace&silentface);

};


