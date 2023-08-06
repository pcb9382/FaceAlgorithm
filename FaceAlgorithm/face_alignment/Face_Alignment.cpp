#include "Face_Alignment.h"

Face_Alignment::Face_Alignment()
{
}

Face_Alignment::~Face_Alignment()
{
}
HZFLAG Face_Alignment::Face_AlignmentInit(Config&config)
{
	Face_Alignment_INPUT_BLOB_NAME = "data";
	Face_Alignment_OUTPUT_BLOB_NAME = "fc1";
	Face_Alignment_INPUT_H = 192;
	Face_Alignment_INPUT_W = 192;
	Face_Alignment_OUTPUT_SIZE = 212;
	cudaSetDevice(config.gpu_id);
	char *trtModelStream{nullptr};
	size_t size{0};
	std::string directory;
	const size_t last_slash_idx = config.FaceAlignmentModelPath.rfind(".onnx");
	if (std::string::npos != last_slash_idx)
	{
		directory = config.FaceAlignmentModelPath.substr(0, last_slash_idx);
	}
	std::string out_engine = directory +"_batch="+ std::to_string(config.FaceAlignment_bs) + ".engine";
	bool enginemodel = Face_Alignment_model_exists(out_engine);
	if (!enginemodel)
	{
		std::cout << "Building engine, please wait for a while..." << std::endl;
		bool wts_model = Face_Alignment_model_exists(config.FaceAlignmentModelPath);
		if (!wts_model)
		{
			std::cout << "ONNX file is not Exist!Please Check!" << std::endl;
			return HZ_ERROR;
		}
		Onnx2Ttr onnx2trt;
		onnx2trt.onnxToTRTModel(Face_Alignment_gLogger,config.FaceAlignmentModelPath.c_str(),config.FaceAlignment_bs,out_engine.c_str());//config.classs_path
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
	Face_Alignment_runtime = createInferRuntime(Face_Alignment_gLogger);
	assert(Face_Alignment_runtime != nullptr);
	Face_Alignment_engine = Face_Alignment_runtime->deserializeCudaEngine(trtModelStream, size);
	assert(Face_Alignment_engine != nullptr);
	Face_Alignment_context =Face_Alignment_engine->createExecutionContext();
	assert(Face_Alignment_context != nullptr);
	delete[] trtModelStream;

	Face_Alignment_data = new float[3 * Face_Alignment_INPUT_H * Face_Alignment_INPUT_W];
	Face_Alignment_prob = new float[Face_Alignment_OUTPUT_SIZE];

	return HZ_SUCCESS;
}
HZFLAG Face_Alignment::Face_AlignmentRun(cv::Mat&img, AlignmentFace&alignmentface)
{
	if (img.empty())
	{
		std::cout<<"Face_AlignmentRun image is empty"<<std::endl;
		return HZ_IMGEMPTY;
	}
	cv::Mat pr_img;
	cv::resize(img,pr_img, cv::Size(Face_Alignment_INPUT_H, Face_Alignment_INPUT_W));
	int i = 0;
	for (int row = 0; row < Face_Alignment_INPUT_H; ++row)
	{
		uchar* uc_pixel = pr_img.data + row * pr_img.step;
		for (int col = 0; col < Face_Alignment_INPUT_W; ++col)
		{
			Face_Alignment_data[i] = (float)uc_pixel[2];
			Face_Alignment_data[i + Face_Alignment_INPUT_H * Face_Alignment_INPUT_W] = (float)uc_pixel[1];
			Face_Alignment_data[i + 2 * Face_Alignment_INPUT_H * Face_Alignment_INPUT_W] = (float)uc_pixel[0];
			uc_pixel += 3;
			++i;
		}
	}
	// Run inference  
	auto start = std::chrono::system_clock::now();
	Face_Alignment_doInference(*Face_Alignment_context,Face_Alignment_data,Face_Alignment_prob, 1);
	auto end = std::chrono::system_clock::now();
	for (int start = 0; start < 212; start += 2) 
	{
		alignmentface.landmarks[start]=int((Face_Alignment_prob[start]+1)*img.cols / 2);
		alignmentface.landmarks[start+1]=int((Face_Alignment_prob[start+1]+1)*img.cols / 2);
		//cv::circle(img, cv::Point(alignmentface.landmarks[start], alignmentface.landmarks[start+1]), 1, cv::Scalar(200, 160, 75), -1, cv::LINE_8, 0);
	}
	//106 keypoint
	return HZ_SUCCESS;
}
HZFLAG Face_Alignment::Face_AlignmentRelease()
{
	Face_Alignment_context->destroy();
	Face_Alignment_engine->destroy();
	Face_Alignment_runtime->destroy();
	delete[]Face_Alignment_data;
	delete[]Face_Alignment_prob;
	Face_Alignment_data = NULL;
	Face_Alignment_prob = NULL;
	return HZ_SUCCESS;
}
void Face_Alignment::Face_Alignment_doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(Face_Alignment_INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(Face_Alignment_OUTPUT_BLOB_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * Face_Alignment_INPUT_H * Face_Alignment_INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * Face_Alignment_OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * Face_Alignment_INPUT_H * Face_Alignment_INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * Face_Alignment_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}
bool  Face_Alignment::Face_Alignment_model_exists(const std::string& name)
{
	std::ifstream f(name.c_str());
	return f.good();
}