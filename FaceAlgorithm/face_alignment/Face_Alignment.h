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


class Face_Alignment
{
public:
	Face_Alignment();
	~Face_Alignment();
	HZFLAG Face_AlignmentInit(Config&config);
	HZFLAG Face_AlignmentRun(cv::Mat&img, AlignmentFace&alignmentface);
	HZFLAG Face_AlignmentRelease();
private:
	char* Face_Alignment_INPUT_BLOB_NAME;
	char* Face_Alignment_OUTPUT_BLOB_NAME;
	int Face_Alignment_INPUT_H;
	int Face_Alignment_INPUT_W;
	int Face_Alignment_OUTPUT_SIZE; //class num
	float *Face_Alignment_data ;
	float *Face_Alignment_prob;
	Logger Face_Alignment_gLogger;
	IRuntime* Face_Alignment_runtime;
	ICudaEngine* Face_Alignment_engine;
	IExecutionContext* Face_Alignment_context;

private:
	void Face_Alignment_doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	bool Face_Alignment_model_exists(const std::string& name);
};


