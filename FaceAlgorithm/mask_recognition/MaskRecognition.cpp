#include "MaskRecognition.h"

MaskRecognition::MaskRecognition()
{
}

MaskRecognition::~MaskRecognition()
{
}
HZFLAG MaskRecognition:: MaskRecognitionInit(Config&config)
{
	INPUT_BLOB_NAME = "images";
	OUTPUT_BLOB_NAME = "output";
	INPUT_H = 224;
	INPUT_W = 224;
	OUTPUT_SIZE = 2; //class num
	cudaSetDevice(config.gpu_id);
	// create a model using the API directly and serialize it to a stream
	char *trtModelStream{ nullptr };
	size_t size{0};
	std::string directory;
	const size_t last_slash_idx = config.MaskReconitionModelPath.rfind(".onnx");
	if (std::string::npos != last_slash_idx)
	{
		directory = config.MaskReconitionModelPath.substr(0, last_slash_idx);
	}
	std::string out_engine = directory +"_batch="+ std::to_string(1) + ".engine";
	bool enginemodel = model_exists(out_engine);
	if (!enginemodel)
	{
		std::cout << "Building engine, please wait for a while..." << std::endl;
		bool wts_model = model_exists(config.MaskReconitionModelPath);//config.classs_path
		if (!wts_model)
		{
			std::cout << "ONNX file is not Exist!Please Check!" << std::endl;
			return HZ_ERROR;
		}
		Onnx2Ttr onnx2trt;
		//IHostMemory* modelStream{ nullptr };
		onnx2trt.onnxToTRTModel(config.MaskReconitionModelPath.c_str(),1,out_engine.c_str());//config.classs_path
		//assert(modelStream != nullptr);
		//modelStream->destroy();
	}
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

	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;

	data = new float[3 * INPUT_H * INPUT_W];
	prob = new float[OUTPUT_SIZE];

	return HZ_SUCCESS;
}
HZFLAG MaskRecognition::MaskRecognitionRun(cv::Mat&img, float&pred)
{
	
	if (img.empty())
		return HZ_IMGEMPTY;
	cv::Mat pr_img = CenterCrop(img);
	int i = 0;
	for (int row = 0; row < INPUT_H; ++row)
	{
		uchar* uc_pixel = pr_img.data + row * pr_img.step;
		for (int col = 0; col < INPUT_W; ++col)
		{
			data[i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229; // R-0.485
			data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
			data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
			uc_pixel += 3;
			++i;
		}
	}
	// Run inference  
	auto start = std::chrono::system_clock::now();
	doInference(*context, data, prob, 1);
	auto end = std::chrono::system_clock::now();
	//softmax
	std::vector<float> preds;
	preds = softmax(prob);
	pred=preds[0];
	return HZ_SUCCESS;
}
HZFLAG MaskRecognition::MaskRecognitionRelease()
{
	context->destroy();
	engine->destroy();
	runtime->destroy();
	delete[]data;
	delete[]prob;
	data = NULL;
	prob = NULL;
	return HZ_SUCCESS;
}

cv::Mat  MaskRecognition::CenterCrop(cv::Mat img)
{
	int imh = img.rows;
	int imw = img.cols;
	int wh_min = imh < imw ? imh : imw;
	int top = floor((imh - wh_min) / 2);
	int left = floor((imw - wh_min) / 2);
	cv::Mat temp_img = img(cv::Rect(left, top, wh_min, wh_min));
	cv::Mat center_crop_img;
	cv::resize(temp_img, center_crop_img, cv::Size(224, 224), cv::INTER_LINEAR);
	return center_crop_img;
}
std::vector<float>  MaskRecognition::softmax(float *prob)
{
	float total = 0;
	float MAX = prob[0];
	for (int i = 1; i < OUTPUT_SIZE; i++)
	{
		MAX = std::max(prob[i], MAX);
		
	}
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		total += exp(prob[i] - MAX);
	}
	std::vector<float> result;
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		result.push_back(exp(prob[i] - MAX) / total);
	}
	return result;
}
void  MaskRecognition::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
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
bool  MaskRecognition::model_exists(const std::string& name)
{
	std::ifstream f(name.c_str());
	return f.good();
}