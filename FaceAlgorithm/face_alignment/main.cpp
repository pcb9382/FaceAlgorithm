#include <iostream>
#include <opencv2/opencv.hpp>
#include "Face_Alignment.h"

int main()
{
    //人脸检测的bbox可以按照中心点左右上下外扩1.2(左右上下各1.1)
    std::string data_model_path="/home/pcb/FaceRecognition_Linux_Release/FaceAlgorithm/";
	cv::Mat facealigenmentMat=cv::imread(data_model_path+"test5.jpg");
    Config config;
    config.FaceAlignment_bs=1;
    config.FaceAlignmentModelPath=data_model_path+"2d106det_bs=1.onnx";


    Face_Alignment face_alignment;
    face_alignment.Face_AlignmentInit(config);

	AlignmentFace alignmentface;
	face_alignment.Face_AlignmentRun(facealigenmentMat,alignmentface);
	for (size_t k = 0; k < 212; k+=2)
	{
		cv::circle(facealigenmentMat, cv::Point(alignmentface.landmarks[k], alignmentface.landmarks[k+1]), 1, cv::Scalar(200, 160, 75), -1, cv::LINE_8,0);
	}
    cv::imwrite("FaceAlignment_result.jpg",facealigenmentMat);
	std::cout<<"facealigenment test finash!"<<std::endl;
    face_alignment.Face_AlignmentRelease();
    return 0;
}