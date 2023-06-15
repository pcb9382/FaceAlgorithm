#include "detector.h"
using namespace std;

Detector::Detector()
{

}
Detector::~Detector()
{


}

HZFLAG Detector::InitDetector(Config& config)
{

    conf_thresh=config.confidence_thresh;
    nms_thresh=config.nms_thresh;
    // H, W must be able to  be divided by 32.
    INPUT_W = 640;
    INPUT_H = 480;  
   
    OUTPUT_SIZE =(INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;//189001;
    INPUT_BLOB_NAME = "data";
    OUTPUT_BLOB_NAME = "prob";
    std::string directory; 
    const size_t last_slash_idx=config.FactDetectModelPath.rfind(".wts");
    if (std::string::npos != last_slash_idx)
    {
        directory = config.FactDetectModelPath.substr(0, last_slash_idx);
    }
    std::string out_engine=directory+"_batch="+std::to_string(config.face_detect_bs)+".engine";
    bool enginemodel=model_exists(out_engine);
    if (!enginemodel)
    {
        std::cout << "Building engine, please wait for a while..." << std::endl;
        bool wts_model=model_exists(config.FactDetectModelPath);
        if (!wts_model)
        {
           std::cout<<"FaceDetector.wts is not Exist!!!Please Check!"<<std::endl;
           return HZ_WITHOUTMODEL;
        }
        IHostMemory* modelStream{nullptr};
        APIToModel(config.face_detect_bs, &modelStream,config.FactDetectModelPath);
        assert(modelStream != nullptr);
        std::ofstream p(out_engine, std::ios::binary);
        if (!p) 
        {
            std::cerr << "could not open plan output file" << std::endl;
            return HZ_WITHOUTMODEL;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }
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
        std::cout<<"model file not exist!"<<std::endl;
        return HZ_WITHOUTMODEL;
    }
    
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], config.face_detect_bs * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], config.face_detect_bs * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
     // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    return HZ_SUCCESS;
}

HZFLAG Detector::detect(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>&dets)
{
    
    // prepare input data ---------------------------
    int detector_batchsize=ImgVec.size();
    std::vector<cv::Mat> imgs_buffer(detector_batchsize);
    float* buffer_idx = (float*)buffers[inputIndex];
    for (int b = 0; b < detector_batchsize; b++)
    {
        if (ImgVec[b].empty()) 
        {
            continue;
        }
        imgs_buffer[b] = ImgVec[b].clone();
        size_t  size_image = imgs_buffer[b].cols * imgs_buffer[b].rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host,imgs_buffer[b].data,size_image);
        //copy data to device memory
        CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        mcv_preprocess_kernel_img(img_device, imgs_buffer[b].cols,imgs_buffer[b].rows, buffer_idx, INPUT_W, INPUT_H, stream);       
        buffer_idx += size_image_dst;
    }
    // Run inference
    float *prob=new float[detector_batchsize * OUTPUT_SIZE];
    doInference(*context,stream,(void**)buffers,prob,detector_batchsize);
    for (int b = 0; b < detector_batchsize; b++) 
    {
        std::vector<decodeplugin::Detection> res;
        nms(res, &prob[b * OUTPUT_SIZE]);
        std::vector<Det>Imgdet;
        for (size_t j = 0; j < res.size(); j++) 
        {
            if (res[j].class_confidence < conf_thresh) 
            {
                continue;
            }
            Det det;
            det.confidence=res[j].class_confidence;
            get_rect_adapt_landmark(imgs_buffer[b], INPUT_W, INPUT_H, res[j].bbox, res[j].landmark,det);
            Imgdet.push_back(det);
        }
        dets.push_back(Imgdet);
    }
    delete []prob;
    prob=NULL;
    return HZ_SUCCESS;

}
HZFLAG Detector::ReleaseDetector()
{
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return HZ_SUCCESS;
}
void Detector::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers,float* output, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


cv::Mat Detector::preprocess_img(cv::Mat& img, int input_w, int input_h) 
{
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w)
    {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } 
    else 
    {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}
int Detector::read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) 
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) 
    {
        return -1;
    }
    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) 
    {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) 
            {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

void Detector::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det) 
{
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) 
    {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) 
        {
            det.key_points.push_back(lmk[i]/r_w);
            det.key_points.push_back((lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w);
        }
    } 
    else 
    {
        l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) 
        {
            det.key_points.push_back((lmk[i] - (input_w - r_h * img.cols) / 2) / r_h);
            det.key_points.push_back(lmk[i + 1]/r_h);
        }
    }
    det.bbox.xmin=l>1?l:1;
    det.bbox.ymin=t>1?t:1;
    det.bbox.xmax=r>det.bbox.xmin?r:det.bbox.xmin+1;
    det.bbox.xmax=det.bbox.xmax<img.cols?det.bbox.xmax:img.cols-1;
    det.bbox.ymax=b>det.bbox.ymin?b:det.bbox.ymin+1;
    det.bbox.ymax=det.bbox.ymax<img.rows?det.bbox.ymax:img.rows-1;
    return ;
}

float Detector::iou(float lbox[4], float rbox[4]) 
{
    float interBox[] = 
    {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
    {
        return 0.0f;
    }
    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

bool Detector::cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) 
{
    return a.class_confidence > b.class_confidence;
}

void Detector::nms(std::vector<decodeplugin::Detection>& res, float *output) 
{
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) 
    {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) 
    {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) 
        {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) 
            {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}

// Load weights from files
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> Detector::loadWeights(const std::string file) 
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

Weights Detector::getWeights(std::map<std::string, Weights>& weightMap, std::string key) 
{
    if (weightMap.count(key) != 1) 
    {
        std::cerr << key << " not existed in weight map, fatal error!!!" << std::endl;
        exit(-1);
    }
    return weightMap[key];
}

IScaleLayer* Detector::addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) 
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) 
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) 
    {
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

void Detector::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,std::string&wts_string) 
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

// Creat the engine using only the API and not any parser.
ICudaEngine* Detector::createEngine(unsigned int maxBatchSize, IBuilder* builder123, IBuilderConfig* config123,std::string&wts_string)
 {
    INetworkDefinition* network = builder123->createNetworkV2(0U);

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor* data1 = network->addInput(INPUT_BLOB_NAME,DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
    assert(data1);

    std::map<std::string, Weights> weightMap = loadWeights(wts_string);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone mobilenet0.25  ---------------
    // stage 1
    auto x = conv_bn(network, weightMap, *data1, "body.stage1.0", 8, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.1", 8, 16);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.2", 16, 32, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.3", 32, 32);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.4", 32, 64, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage1.5", 64, 64);
    auto stage1 = x;

    // stage 2
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.0", 64, 128, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.1", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.2", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.3", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.4", 128, 128);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage2.5", 128, 128);
    auto stage2 = x;

    // stage 3
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage3.0", 128, 256, 2);
    x = conv_dw(network, weightMap, *x->getOutput(0), "body.stage3.1", 256, 256);
    auto stage3 = x;

    //Dims d1 = stage1->getOutput(0)->getDimensions();
    //std::cout << d1.d[0] << " " << d1.d[1] << " " << d1.d[2] << std::endl;
    // ------------- FPN ---------------
    auto output1 = conv_bn1X1(network, weightMap, *stage1->getOutput(0), "fpn.output1", 64);
    auto output2 = conv_bn1X1(network, weightMap, *stage2->getOutput(0), "fpn.output2", 64);
    auto output3 = conv_bn1X1(network, weightMap, *stage3->getOutput(0), "fpn.output3", 64);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) 
    {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolutionNd(*output3->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(64);
    weightMap["up3"] = deconvwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn(network, weightMap, *output2->getOutput(0), "fpn.merge2", 64);

    IDeconvolutionLayer* up2 = network->addDeconvolutionNd(*output2->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(64);
    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn(network, weightMap, *output1->getOutput(0), "fpn.merge1", 64);

    // ------------- SSH ---------------
    auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1", 64);
    auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2", 64);
    auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3", 64);

    //// ------------- Head ---------------
    auto bbox_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.0.conv1x1.weight"], weightMap["BboxHead.0.conv1x1.bias"]);
    auto bbox_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.1.conv1x1.weight"], weightMap["BboxHead.1.conv1x1.bias"]);
    auto bbox_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.2.conv1x1.weight"], weightMap["BboxHead.2.conv1x1.bias"]);

    auto cls_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.0.conv1x1.weight"], weightMap["ClassHead.0.conv1x1.bias"]);
    auto cls_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.1.conv1x1.weight"], weightMap["ClassHead.1.conv1x1.bias"]);
    auto cls_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.2.conv1x1.weight"], weightMap["ClassHead.2.conv1x1.bias"]);

    auto lmk_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.0.conv1x1.weight"], weightMap["LandmarkHead.0.conv1x1.bias"]);
    auto lmk_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.1.conv1x1.weight"], weightMap["LandmarkHead.1.conv1x1.bias"]);
    auto lmk_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.2.conv1x1.weight"], weightMap["LandmarkHead.2.conv1x1.bias"]);

    //// ------------- Decode bbox, conf, landmark ---------------
    ITensor* inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    ITensor* inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    ITensor* inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor* inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder123->setMaxBatchSize(maxBatchSize);
    config123->setMaxWorkspaceSize(1 << 20);
#if defined(USE_FP16)
    config123->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
    assert(builder->platformHasFastInt8());
    config123->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./widerface_calib/", "mnet_int8calib.table", INPUT_BLOB_NAME);
    config123->setInt8Calibrator(calibrator);
#endif

    
    ICudaEngine* engine123 = builder123->buildEngineWithConfig(*network, *config123);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
        mem.second.values = NULL;
    }

    return engine123;
}

ILayer* Detector::conv_bn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s, float leaky) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(leaky);
    assert(lr);
    return lr;
}

ILayer* Detector::conv_bn_no_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    return bn1;
}

ILayer* Detector::conv_bn1X1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup, int s, float leaky) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, oup, DimsHW{1, 1}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{0, 0});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(leaky);
    assert(lr);
    return lr;
}

ILayer* Detector::conv_dw(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int s, float leaky) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, inp, DimsHW{3, 3}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(inp);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    auto lr1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr1->setAlpha(leaky);
    assert(lr1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*lr1->getOutput(0), oup, DimsHW{1, 1}, getWeights(weightMap, lname + ".3.weight"), emptywts);
    assert(conv2);
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".4", 1e-5);
    auto lr2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    lr2->setAlpha(leaky);
    assert(lr2);
    return lr2;
}

IActivationLayer* Detector::ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int oup) 
{
    auto conv3x3 = conv_bn_no_relu(network, weightMap, input, lname + ".conv3X3", oup / 2);
    auto conv5x5_1 = conv_bn(network, weightMap, input, lname + ".conv5X5_1", oup / 4);
    auto conv5x5 = conv_bn_no_relu(network, weightMap, *conv5x5_1->getOutput(0), lname + ".conv5X5_2", oup / 4);
    auto conv7x7 = conv_bn(network, weightMap, *conv5x5_1->getOutput(0), lname + ".conv7X7_2", oup / 4);
    conv7x7 = conv_bn_no_relu(network, weightMap, *conv7x7->getOutput(0), lname + ".conv7x7_3", oup / 4);
    ITensor* inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0), conv7x7->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    IActivationLayer* relu1 = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}