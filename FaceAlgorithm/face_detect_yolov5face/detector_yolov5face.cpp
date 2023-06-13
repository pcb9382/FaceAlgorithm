#include "detector_yolov5face.h"
using namespace std;

Detector_Yolov5Face::Detector_Yolov5Face()
{

}
Detector_Yolov5Face::~Detector_Yolov5Face()
{


}

HZRESULT Detector_Yolov5Face::InitDetector_Yolov5Face(Config& config)
{

    this->conf_thresh=config.yolov5face_confidence_thresh;
    this->nms_thresh=config.yolov5face_nms_thresh;
    // H, W must be able to  be divided by 32.
    INPUT_W = 640;
    INPUT_H = 640;  
   
    OUTPUT_SIZE=(INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 3 * 16;
    INPUT_BLOB_NAME = "input";
    OUTPUT_BLOB_NAME = "output";
    std::string directory; 
    const size_t last_slash_idx=config.Yolov5FactDetectModelPath.rfind(".onnx");
    if (std::string::npos != last_slash_idx)
    {
        directory = config.Yolov5FactDetectModelPath.substr(0, last_slash_idx);
    }
    std::string out_engine=directory+"_batch="+std::to_string(config.yolov5face_detect_bs)+".engine";
    bool enginemodel=model_exists(out_engine);
    if (!enginemodel)
    {
        std::cout << "Building engine, please wait for a while..." << std::endl;
        bool wts_model=model_exists(config.Yolov5FactDetectModelPath);
        if (!wts_model)
        {
           std::cout<<"yolov5s-face.onnx is not Exist!!!Please Check!"<<std::endl;
           return HZ_WITHOUTMODEL;
        }
        Onnx2Ttr onnx2trt;
		//IHostMemory* modelStream{ nullptr };
		onnx2trt.onnxToTRTModel(config.Yolov5FactDetectModelPath.c_str(),config.yolov5face_detect_bs,out_engine.c_str());
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
        std::cout<<"yolov5s-face.engine model file not exist!"<<std::endl;
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
    CHECK(cudaMalloc(&buffers[inputIndex], config.yolov5face_detect_bs * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], config.yolov5face_detect_bs * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
     // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, config.yolov5face_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, config.yolov5face_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
    return HZ_SUCCESS;
}

HZRESULT Detector_Yolov5Face::Detect_Yolov5Face(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>& dets)
{
    
    // prepare input data ---------------------------
    int detector_batchsize=ImgVec.size();
    float* buffer_idx = (float*)buffers[inputIndex];
    std::vector<cv::Mat> imgs_buffer(detector_batchsize);
    for (int b = 0; b < detector_batchsize; b++)
    {
        if (ImgVec[b].empty()||ImgVec[b].data==NULL) 
        {
            continue;
        }
        imgs_buffer[b] = ImgVec[b].clone();
        size_t  size_image = imgs_buffer[b].cols * imgs_buffer[b].rows * 3*sizeof(uint8_t);
        size_t  size_image_dst = INPUT_H * INPUT_W * 3*sizeof(uint8_t);
        //copy data to pinned memory
        memcpy(img_host,imgs_buffer[b].data,size_image);
        //copy data to device memory
        CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        preprocess_kernel_img_yolov5_face(img_device,imgs_buffer[b].cols,imgs_buffer[b].rows, buffer_idx, INPUT_W, INPUT_H, stream);       
        buffer_idx += size_image_dst;
    }
    // Run inference
    float *prob=new float[detector_batchsize * OUTPUT_SIZE];
    doInference(*context,stream,(void**)buffers,prob,detector_batchsize);
    // std::fstream writetxt;
    // writetxt.open("12.txt",std::ios::out);
    // for (size_t k = 0; k < detector_batchsize * OUTPUT_SIZE; k++)
    // {
    //    writetxt<<prob[k]<<std::endl;
    // }
    // writetxt.close();

    for (int b = 0; b < detector_batchsize; b++) 
    {
        std::vector<decodeplugin_yolov5face::Detection> res;
        nms(res, &prob[b * OUTPUT_SIZE],this->conf_thresh,this->nms_thresh);
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
        // for (size_t j = 0; j <  Imgdet.size(); j++)
        // {
        //     cv::rectangle(ImgVec[b], cv::Point( Imgdet[j].bbox.xmin,  Imgdet[j].bbox.ymin),
        //     cv::Point( Imgdet[j].bbox.xmax,  Imgdet[j].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
        //     cv::circle(ImgVec[b], cv::Point2f( Imgdet[j].key_points[0],  Imgdet[j].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
        //     cv::circle(ImgVec[b], cv::Point2f( Imgdet[j].key_points[2],  Imgdet[j].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
        //     cv::circle(ImgVec[b], cv::Point2f( Imgdet[j].key_points[4],  Imgdet[j].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
        //     cv::circle(ImgVec[b], cv::Point2f( Imgdet[j].key_points[6],  Imgdet[j].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
        //     cv::circle(ImgVec[b], cv::Point2f( Imgdet[j].key_points[8],  Imgdet[j].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
        // }
        // cv::imshow("show", ImgVec[b]);
        // cv::waitKey(0);      
        dets.push_back(Imgdet);
    }
    delete []prob;
    prob=NULL;
    return HZ_SUCCESS;

}
HZRESULT Detector_Yolov5Face::ReleaseDetector_Yolov5Face()
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

void Detector_Yolov5Face::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) 
{
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
cv::Mat Detector_Yolov5Face::preprocess_img(cv::Mat& img, int input_w, int input_h) 
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
  cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
  return out;
}

bool cmp(const decodeplugin_yolov5face::Detection& a, const decodeplugin_yolov5face::Detection& b) 
{
  return a.class_confidence > b.class_confidence;
}

float Detector_Yolov5Face::iou(float lbox[4], float rbox[4]) 
{
  float interBox[] = 
  {
    std::max(lbox[0]-lbox[2]/2, rbox[0]-rbox[2]/2), //left
    std::min(lbox[0]+lbox[2]/2, rbox[0]+rbox[2]/2), //right
    std::max(lbox[1]-lbox[3]/2, rbox[1]-rbox[3]/2), //top
    std::min(lbox[1]+lbox[3]/2, rbox[1]+rbox[3]/2), //bottom
  };
  if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
  {
    return 0.0f;
  }
  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  //return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
  return interBoxS / ((lbox[2]) * (lbox[3]) + (rbox[2]) * (rbox[3]) -interBoxS + 0.000001f);
}
void Detector_Yolov5Face::nms(std::vector<decodeplugin_yolov5face::Detection>& res, float *output,float confidence,float nms_thresh) 
{
  std::vector<decodeplugin_yolov5face::Detection>  Imgdet;
  for (int i = 0; i < OUTPUT_SIZE/16; i++) 
  {
    //std::cout<<"confidence:"<<output[16* i +4]<<std::endl;
    if (output[16* i +4] <= confidence) 
    {
      continue;
    }
    decodeplugin_yolov5face::Detection det;
    det.bbox[0]=output[16* i];
    det.bbox[1]=output[16* i+1];
    det.bbox[2]=output[16* i+2];
    det.bbox[3]=output[16* i+3];
    det.class_confidence=output[16* i+4];
    det.landmark[0]=output[16* i+5];
    det.landmark[1]=output[16* i+6];
    det.landmark[2]=output[16* i+7];
    det.landmark[3]=output[16* i+8];
    det.landmark[4]=output[16* i+9];
    det.landmark[5]=output[16* i+10];
    det.landmark[6]=output[16* i+11];
    det.landmark[7]=output[16* i+12];
    det.landmark[8]=output[16* i+13];
    det.landmark[9]=output[16* i+14];
     Imgdet.push_back(det);
  }
  std::sort( Imgdet.begin(),  Imgdet.end(), cmp);
  for (size_t m = 0; m <  Imgdet.size(); ++m) 
  {
    auto& item =  Imgdet[m];
    res.push_back(item);
    //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
    for (size_t n = m + 1; n <  Imgdet.size(); ++n) 
    {
      if (iou(item.bbox,  Imgdet[n].bbox) > nms_thresh) 
      {
         Imgdet.erase( Imgdet.begin()+n);
        --n;
      }
    }
  }
}

// void Detector_Yolov5Face::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det) 
// {
//     int l, r, t, b;
//     float r_w = input_w / (img.cols * 1.0);
//     float r_h = input_h / (img.rows * 1.0);
//     if (r_h > r_w) 
//     {
//         l = bbox[0] / r_w;
//         r = bbox[2] / r_w;
//         t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
//         b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
//         for (int i = 0; i < 10; i += 2) 
//         {
//             det.key_points.push_back(lmk[i]/r_w);
//             det.key_points.push_back((lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w);
//         }
//     } 
//     else 
//     {
//         l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
//         r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
//         t = bbox[1] / r_h;
//         b = bbox[3] / r_h;
//         for (int i = 0; i < 10; i += 2) 
//         {
//             det.key_points.push_back((lmk[i] - (input_w - r_h * img.cols) / 2) / r_h);
//             det.key_points.push_back(lmk[i + 1]/r_h);
//         }
//     }
//     det.bbox.xmin=l>1?l:1;
//     det.bbox.ymin=t>1?t:1;
//     det.bbox.xmax=r>det.bbox.xmin?r:det.bbox.xmin+1;
//     det.bbox.xmax=det.bbox.xmax<img.cols?det.bbox.xmax:img.cols-1;
//     det.bbox.ymax=b>det.bbox.ymin?b:det.bbox.ymin+1;
//     det.bbox.ymax=det.bbox.ymax<img.rows?det.bbox.ymax:img.rows-1;
//     return ;
// }
void Detector_Yolov5Face::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det) 
{
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) 
    {
        l = (bbox[0]-bbox[2]/2)/r_w;
        r = (bbox[0]+bbox[2]/2)/r_w;
        t = (bbox[1]-bbox[3]/2 - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[1]+bbox[3]/2 - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) 
        {
            det.key_points.push_back(lmk[i]/r_w);
            det.key_points.push_back((lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w);
        }
    } 
    else 
    {
        l = (bbox[0]-bbox[2]/2 - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[0]+bbox[2]/2 - (input_w - r_h * img.cols) / 2) / r_h;
        t = (bbox[1]-bbox[3]/2) / r_h;
        b = (bbox[1]+bbox[3]/2) / r_h;
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
    return;
}