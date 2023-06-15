#include "recognition.h"


Recognition::Recognition()
{


}
Recognition::~Recognition()
{

}

HZFLAG Recognition::InitRecognition(Config&config)
{
    INPUT_H = 112;
    INPUT_W = 112;
    OUTPUT_SIZE = 512;
    INPUT_BLOB_NAME = "data";
    OUTPUT_BLOB_NAME = "prob";

    //判断模型文件是否存在
    std::string directory; 
    const size_t last_slash_idx=config.FactReconitionModelPath.rfind(".wts");
    if (std::string::npos != last_slash_idx)
    {
        directory =config.FactReconitionModelPath.substr(0, last_slash_idx);
    }
    //engine文件名称
    std::string out_engine=directory+".engine";
    bool enginemodel=model_exists(out_engine);
    
    //engine文件不存在，创建符合符合batch的engine
    if (!enginemodel)
    {
        std::cout<<"the Engine of matching batch model not Exist!"<<std::endl;
        std::cout << "Building engine, please wait for a while..." << std::endl;
        //得到FaceDetector.wts文件，一般和需要加载的engine文件放在一起
        //首先得到engine的目录
        bool wts_model=model_exists(config.FactReconitionModelPath);
        //判断FaceDetectorTracker.wts文件是否存在
        if (!wts_model)
        {
           std::cout<<"FaceRecognition.wts is not Exist!!!Please Check!"<<std::endl;
           return HZ_WITHOUTMODEL;
        }

        //开始创建engine
        IHostMemory* modelStream{nullptr};
        APIToModel(config.face_recognition_bs, &modelStream,config.FactReconitionModelPath);
        assert(modelStream != nullptr);
        
        //生成engine的名字 
        std::ofstream p(out_engine, std::ios::binary);
        if (!p) 
        {
            std::cerr << "could not open plan output file" << std::endl;
            return HZ_WITHOUTMODEL;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }

    //cudaSetDevice(gpu_id);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(out_engine, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cout<<"face recognition model file not exist!"<<std::endl;
        return HZ_WITHOUTMODEL;
    }
    
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    //申请内存空间
    data=new float[3 * INPUT_H * INPUT_W];
    prob=new float[OUTPUT_SIZE];
    return HZ_SUCCESS;
}  
HZFLAG Recognition::Extract_feature(cv::Mat&img,Feature&feature)
{
    if (img.empty()||img.data==NULL)
    {
        std::cout<<"face recognition extract_feature img is empty,please check!"<<std::endl;
        return HZ_IMGEMPTY;
    }
    
    // prepare input data ---------------------------
    for (int i = 0; i < INPUT_H * INPUT_W; i++) 
    {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }

    doInference(*context, data, prob, BATCH_SIZE);
    //得到特征向量
    for (size_t i = 0; i < OUTPUT_SIZE; i++)
    {
        feature.feature[i]=prob[i];
    }
    return HZ_SUCCESS;
}
HZFLAG Recognition::ReleaseRecognition()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
    //清除申请过的内存空间
    delete []data;
    data=NULL;
    delete []prob;
    prob=NULL;
    return HZ_SUCCESS;
}

void Recognition::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> Recognition::loadWeights(const std::string file) 
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* Recognition::addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) 
{
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* Recognition::addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) 
{
	float *gamma = (float*)weightMap[lname + "_gamma"].values;
	int len = weightMap[lname + "_gamma"].count;

	float *scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	float *scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) 
    {
		scval_1[i] = -1.0;
		scval_2[i] = -gamma[i];
	}
	Weights scale_1{ DataType::kFLOAT, scval_1, len };
	Weights scale_2{ DataType::kFLOAT, scval_2, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) 
    {
		shval[i] = 0.0;
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) 
    {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	auto relu1 = network->addActivation(input, ActivationType::kRELU);
	assert(relu1);
	IScaleLayer* scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
	assert(scale1);
	auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
	assert(relu2);
	IScaleLayer* scale2 = network->addScale(*relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
	assert(scale2);
	IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
	assert(ew1);
	return ew1;
}

ILayer* Recognition::resUnit(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int s, bool dim_match, std::string lname) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    auto bn1 = addBatchNorm2d(network, weightMap, input, lname + "_bn1", 2e-5);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*bn1->getOutput(0), num_filters, DimsHW{3, 3}, weightMap[lname + "_conv1_weight"], emptywts);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{1, 1});
    auto bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn2", 2e-5);
    auto act1 = addPRelu(network, weightMap, *bn2->getOutput(0), lname + "_relu1");
    IConvolutionLayer* conv2 = network->addConvolutionNd(*act1->getOutput(0), num_filters, DimsHW{3, 3}, weightMap[lname + "_conv2_weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{1, 1});
    auto bn3 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "_bn3", 2e-5);

    IElementWiseLayer* ew1;
    if (dim_match) 
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } 
    else 
    {
        IConvolutionLayer* conv1sc = network->addConvolutionNd(input, num_filters, DimsHW{1, 1}, weightMap[lname + "_conv1sc_weight"], emptywts);
        assert(conv1sc);
        conv1sc->setStrideNd(DimsHW{s, s});
        auto bn1sc = addBatchNorm2d(network, weightMap, *conv1sc->getOutput(0), lname + "_sc", 2e-5);
        ew1 = network->addElementWise(*bn1sc->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    assert(ew1);
    return ew1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* Recognition::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,std::string&wts_string)
 {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_string);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv0 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["conv0_weight"], emptywts);
    assert(conv0);
    conv0->setPaddingNd(DimsHW{1, 1});
    auto bn0 = addBatchNorm2d(network, weightMap, *conv0->getOutput(0), "bn0", 2e-5);
    auto relu0 = addPRelu(network, weightMap, *bn0->getOutput(0), "relu0");

    auto s1u1 = resUnit(network, weightMap, *relu0->getOutput(0), 64, 2, false, "stage1_unit1");
    auto s1u2 = resUnit(network, weightMap, *s1u1->getOutput(0), 64, 1, true, "stage1_unit2");
    auto s1u3 = resUnit(network, weightMap, *s1u2->getOutput(0), 64, 1, true, "stage1_unit3");

    auto s2u1 = resUnit(network, weightMap, *s1u3->getOutput(0), 128, 2, false, "stage2_unit1");
    auto s2u2 = resUnit(network, weightMap, *s2u1->getOutput(0), 128, 1, true, "stage2_unit2");
    auto s2u3 = resUnit(network, weightMap, *s2u2->getOutput(0), 128, 1, true, "stage2_unit3");
    auto s2u4 = resUnit(network, weightMap, *s2u3->getOutput(0), 128, 1, true, "stage2_unit4");

    auto s3u1 = resUnit(network, weightMap, *s2u4->getOutput(0), 256, 2, false, "stage3_unit1");
    auto s3u2 = resUnit(network, weightMap, *s3u1->getOutput(0), 256, 1, true, "stage3_unit2");
    auto s3u3 = resUnit(network, weightMap, *s3u2->getOutput(0), 256, 1, true, "stage3_unit3");
    auto s3u4 = resUnit(network, weightMap, *s3u3->getOutput(0), 256, 1, true, "stage3_unit4");
    auto s3u5 = resUnit(network, weightMap, *s3u4->getOutput(0), 256, 1, true, "stage3_unit5");
    auto s3u6 = resUnit(network, weightMap, *s3u5->getOutput(0), 256, 1, true, "stage3_unit6");
    auto s3u7 = resUnit(network, weightMap, *s3u6->getOutput(0), 256, 1, true, "stage3_unit7");
    auto s3u8 = resUnit(network, weightMap, *s3u7->getOutput(0), 256, 1, true, "stage3_unit8");
    auto s3u9 = resUnit(network, weightMap, *s3u8->getOutput(0), 256, 1, true, "stage3_unit9");
    auto s3u10 = resUnit(network, weightMap, *s3u9->getOutput(0), 256, 1, true, "stage3_unit10");
    auto s3u11 = resUnit(network, weightMap, *s3u10->getOutput(0), 256, 1, true, "stage3_unit11");
    auto s3u12 = resUnit(network, weightMap, *s3u11->getOutput(0), 256, 1, true, "stage3_unit12");
    auto s3u13 = resUnit(network, weightMap, *s3u12->getOutput(0), 256, 1, true, "stage3_unit13");
    auto s3u14 = resUnit(network, weightMap, *s3u13->getOutput(0), 256, 1, true, "stage3_unit14");

    auto s4u1 = resUnit(network, weightMap, *s3u14->getOutput(0), 512, 2, false, "stage4_unit1");
    auto s4u2 = resUnit(network, weightMap, *s4u1->getOutput(0), 512, 1, true, "stage4_unit2");
    auto s4u3 = resUnit(network, weightMap, *s4u2->getOutput(0), 512, 1, true, "stage4_unit3");

    auto bn1 = addBatchNorm2d(network, weightMap, *s4u3->getOutput(0), "bn1", 2e-5);
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*bn1->getOutput(0), 512, weightMap["pre_fc1_weight"], weightMap["pre_fc1_bias"]);
    assert(fc1);
    auto bn2 = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);

    bn2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*bn2->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void Recognition::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,std::string&wts_string) 
{
    // Create builder
    IBuilder* builder123 = createInferBuilder(gLogger);
    IBuilderConfig* config123 = builder123->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine123 = createEngine(maxBatchSize, builder123, config123,wts_string);
    assert(engine123 != nullptr);

    // Serialize the engine
    (*modelStream) = engine123->serialize();

    // Close everything down
    engine123->destroy();
    builder123->destroy();
}