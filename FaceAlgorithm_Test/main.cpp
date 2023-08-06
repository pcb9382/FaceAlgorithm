#include<fstream>
#include<iostream>
#include<string>
#include "FaceRecognition.h"
#include "opencv2/opencv.hpp"
#include <chrono>

#define face_detect                       1           //人脸检测
#define yolov5face_detect				  1           //yolov5face 人脸检测
#define yolov7face_detect				  1           //yolov7face 人脸检测
#define yolov8face_detect				  1           //yolov8face 人脸检测
#define face_recognition                  1           //人脸识别（人脸特征提取）+相似度计算
#define face_detect_tracker               1           //人脸检测跟踪
#define face_detect_aligner_recognitiion  0           //人脸检测——矫正——识别(人脸特征提取)
#define mask_recognition                  1           //口罩识别
#define gender_age_recognition            1           //性别年龄识别
#define silnet_face_anti_spoofing         1           //静默活体检测
#define show							  1			  //显示

/*-----------------------------------------
	人脸识别流程

	1.人脸检测(图像、视频流)
			|
			|
	2.人脸矫正(根据5个人脸关键点)
			|
			|
	3.人脸特征特征提取（512维特征）
			|
			|
	4.人脸特征比对(人脸相似度计算)
------------------------------------------*/

struct FaceGroup
{ 
	cv::Mat face_img;                 //人脸检测得到的人脸图像(矩形框bbox外扩1.2倍)
	cv::Point2f key_points[5];        //人脸的5个关键点坐标
	cv::Mat face_aligner_img;         //矫正之后的人脸图像
	Feature face_feature;             //矫正之后的人脸特征
};


int main(int argc, char** argv) 
{
	double sumtime=0.0;
	std::string data_model_path="/home/pcb/FaceRecognition_Linux_Release/FaceAlgorithm/";
	/**********算法初始化**************************/
	Config config;
	
	//gpu_id
	config.gpu_id=0;

	//face detect params
	config.FactDetectModelPath=data_model_path+"FaceDetect.wts";           //人脸检测模型文件路径 
	config.face_detect_bs=1;                                               //表示一次最多输入几张图像（即batch_max，实际检测输入图像的张数可小于等于batch_max）用于检测，
	config.confidence_thresh=0.6;                                          //人脸检测置信度                     
	config.nms_thresh = 0.2;
	config.face_detect_enable=true;

#if yolov5face_detect
	config.Yolov5FactDetectModelPath=data_model_path+"yolov5s-face_bs=4.onnx";
	config.yolov5face_detect_bs=4;
	config.yolov5face_confidence_thresh=0.6;                               //人脸检测置信度                     
	config.yolov5face_nms_thresh = 0.2;
	config.yolov5face_detect_enable=true;
#endif

#if yolov7face_detect
	config.Yolov7FactDetectModelPath=data_model_path+"yolov7s-face_bs=4.onnx";
	config.yolov7face_detect_bs=4;
	config.yolov7face_confidence_thresh=0.1;                               //人脸检测置信度                     
	config.yolov7face_nms_thresh = 0.1;
	config.yolov7face_detect_enable=true;
#endif

#if yolov8face_detect
	config.Yolov8FactDetectModelPath=data_model_path+"yolov8n-face_bs=4.onnx";
	config.yolov8face_detect_bs=4;
	config.yolov8face_confidence_thresh=0.1;                               //人脸检测置信度                     
	config.yolov8face_nms_thresh = 0.4;
	config.yolov8face_detect_enable=true;
#endif

#if face_detect_aligner_recognitiion
	config.face_detect_bs=2;
#endif

	//face recognition params
	config.FactReconitionModelPath=data_model_path+"FaceRecognition.wts";                   //人脸识别文件路径
	config.face_recognition_bs=1;
#if face_recognition
	config.face_recognition_enable=true;
#endif

#if mask_recognition
	config.MaskReconitionModelPath=data_model_path+"MaskRecognition.onnx";
	config.face_mask_bs =1;
	config.face_mask_enable=true;
#endif

#if gender_age_recognition
	config.gender_age_bs=1;
	config.gender_age_enable=true;
	config.GenderAgeModelPath=data_model_path+"GenderAge.onnx";
#endif


#if silnet_face_anti_spoofing
	config.FaceSilentModelPath=data_model_path+"2.7_80x80_MiniFASNetV2.onnx";
	config.silent_face_anti_spoofing_bs=1;
	config.silent_face_anti_spoofing_enable=true;
#endif

	//初始化
	Initialize(config);

#if face_detect
	
	std::ifstream frontfin1;
	frontfin1.open(data_model_path+"b.txt");
	std::string str1;
	std::vector<std::string>ImageName1;
	while (!frontfin1.eof())
	{
		getline(frontfin1, str1);
		if (str1!="")
		{
			ImageName1.push_back(str1);
		}
		else
		{
			continue;
		}
	}
	frontfin1.close();
	for (int i=0;i< ImageName1.size()/config.face_detect_bs;i++)
	{
		std::vector<cv::Mat>RawImageVec;
		for (int j=0;j<config.face_detect_bs;j++)
		{
			cv::Mat img_detect = cv::imread(data_model_path+ImageName1[i*config.face_detect_bs + j]);
			if (img_detect.data==NULL)
			{
				continue;
			}
			RawImageVec.push_back(img_detect);
			
		}
		std::vector<std::vector<FaceDet>>dets;
		auto start = std::chrono::system_clock::now();
		Face_Detect(RawImageVec,dets);
		auto end = std::chrono::system_clock::now();
		sumtime+=(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		for (int i = 0; i < dets.size(); i++)
		{
			for (size_t j = 0; j < dets[i].size(); j++)
			{
				cv::rectangle(RawImageVec[i], cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin),
				cv::Point(dets[i][j].bbox.xmax, dets[i][j].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[0], dets[i][j].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[2], dets[i][j].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[4], dets[i][j].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[6], dets[i][j].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[8], dets[i][j].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
				std::string label3 = cv::format("%f", dets[i][j].confidence);
				cv::putText(RawImageVec[i], label3, cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 2);
				//std::cout << "yaw angle:" << dets[i][j].YawAngle<< " pitch angle:" << dets[i][j].PitchAngle<<" 瞳距:"<<dets[i][j].InterDis<< std::endl;
			}
#if show
			cv::imshow("show", RawImageVec[i]);
			cv::waitKey(1);
#endif      
		}	
	}
	std::cout<<"retinaface average time:"<<sumtime/(config.face_detect_bs*1.0*ImageName1.size()/config.face_detect_bs)<<"us"<<std::endl;
	std::cout<<"face_detect test finash!"<<std::endl;
#endif


#if yolov5face_detect
	sumtime=0.0;
	std::ifstream frontfin2;
	frontfin2.open(data_model_path+"b.txt");
	std::string str2;
	std::vector<std::string>ImageName2;
	while (!frontfin2.eof())
	{
		getline(frontfin2, str2);
		if (str2!="")
		{
			ImageName2.push_back(str2);
		}
		else
		{
			continue;
		}
	}
	frontfin2.close();
	for (int i=0;i< ImageName2.size()/config.yolov5face_detect_bs;i++)
	{
		std::vector<cv::Mat>RawImageVec;
		for (int j=0;j<config.yolov5face_detect_bs;j++)
		{
			cv::Mat img_detect = cv::imread(data_model_path+ImageName2[i*config.yolov5face_detect_bs + j]);
			if (img_detect.data==NULL)
			{
				continue;
			}
			RawImageVec.push_back(img_detect);
			
		}
		std::vector<std::vector<FaceDet>>dets;
		auto start = std::chrono::system_clock::now();
		Yolov5Face_Detect(RawImageVec,dets);
		auto end = std::chrono::system_clock::now();
		sumtime+=(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		//std::cout<<"yolov5 face detect average time:"<<(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/(config.yolov5face_detect_bs*1.0)<<"us"<<std::endl;
		for (int i = 0; i < dets.size(); i++)
		{
			for (size_t j = 0; j < dets[i].size(); j++)
			{
				cv::rectangle(RawImageVec[i], cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin),
				cv::Point(dets[i][j].bbox.xmax, dets[i][j].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[0], dets[i][j].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[2], dets[i][j].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[4], dets[i][j].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[6], dets[i][j].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
				cv::circle(RawImageVec[i], cv::Point2f(dets[i][j].key_points[8], dets[i][j].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
				std::string label3 = cv::format("%f", dets[i][j].confidence);
				cv::putText(RawImageVec[i], label3, cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 2);
				//std::cout << "yaw angle:" << dets[i][j].YawAngle<< " pitch angle:" << dets[i][j].PitchAngle<<" 瞳距:"<<dets[i][j].InterDis<< std::endl;
			}
#if show
			cv::imshow("show", RawImageVec[i]);
			cv::waitKey(1);
#endif      
		}
		
	}
	std::cout<<"yolov5face_detect average time:"<<sumtime/(config.yolov5face_detect_bs*1.0*ImageName1.size()/config.yolov5face_detect_bs)<<"us"<<std::endl;
	std::cout<<"yolov5face_detect test finash!"<<std::endl;
#endif

#if yolov7face_detect
	sumtime=0.0;
	std::ifstream frontfin3;
	frontfin3.open(data_model_path+"b.txt");
	std::string str3;
	std::vector<std::string>ImageName3;
	while (!frontfin3.eof())
	{
		getline(frontfin3, str3);
		if (str3!="")
		{
			ImageName3.push_back(str3);
		}
		else
		{
			continue;
		}
	}
	frontfin3.close();
	for (int i=0;i< ImageName3.size()/config.yolov7face_detect_bs;i++)
	{
		std::vector<cv::Mat>RawImageVec;
		for (int j=0;j<config.yolov7face_detect_bs;j++)
		{
			cv::Mat img_detect = cv::imread(data_model_path+ImageName3[i*config.yolov7face_detect_bs + j]);
			if (img_detect.data==NULL)
			{
				continue;
			}
			RawImageVec.push_back(img_detect);
			
		}
		std::vector<std::vector<FaceDet>>dets;
		auto start = std::chrono::system_clock::now();
		Yolov7Face_Detect(RawImageVec,dets);
		auto end = std::chrono::system_clock::now();
		sumtime+=(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		//std::cout<<"yolov7face average time:"<<(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/(config.yolov7face_detect_bs*1.0)<<"us"<<std::endl;
		for (int k = 0; k < dets.size(); k++)
		{
			for (size_t f = 0; f < dets[k].size(); f++)
			{
				cv::rectangle(RawImageVec[k], cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin),
				cv::Point(dets[k][f].bbox.xmax, dets[k][f].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[0], dets[k][f].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[2], dets[k][f].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[4], dets[k][f].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[6], dets[k][f].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[8], dets[k][f].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
				std::string label3 = cv::format("%f", dets[k][f].confidence);
				cv::putText(RawImageVec[k], label3, cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
				//std::cout << "yaw angle:" << dets[k][f].YawAngle<< " pitch angle:" << dets[k][f].PitchAngle<<" 瞳距:"<<dets[k][f].InterDis<< std::endl;
			}
#if show
			cv::imshow("show", RawImageVec[k]);
			cv::waitKey(1);
#endif
		}
		
	}
	std::cout<<"yolov7face_detect average time:"<<sumtime/(config.yolov7face_detect_bs*1.0*ImageName1.size()/config.yolov7face_detect_bs)<<"us"<<std::endl;
	std::cout<<"yolov7face_detect test finash!"<<std::endl;
#endif

#if yolov8face_detect
	sumtime=0.0;
	std::ifstream frontfin4;
	frontfin4.open(data_model_path+"b.txt");
	std::string str4;
	std::vector<std::string>ImageName4;
	while (!frontfin4.eof())
	{
		getline(frontfin4, str4);
		if (str4!="")
		{
			ImageName4.push_back(str4);
		}
		else
		{
			continue;
		}
	}
	frontfin4.close();
	for (int i=0;i< ImageName4.size()/config.yolov8face_detect_bs;i++)
	{
		std::vector<cv::Mat>RawImageVec;
		for (int j=0;j<config.yolov8face_detect_bs;j++)
		{
			cv::Mat img_detect = cv::imread(data_model_path+ImageName4[i*config.yolov8face_detect_bs + j]);
			if (img_detect.data==NULL)
			{
				continue;
			}
			RawImageVec.push_back(img_detect);
			
		}
		std::vector<std::vector<FaceDet>>dets;
		auto start = std::chrono::system_clock::now();
		Yolov8Face_Detect(RawImageVec,dets);
		auto end = std::chrono::system_clock::now();
		sumtime+=(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		//std::cout<<"yolov8face average time:"<<(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/(config.yolov8face_detect_bs*1.0)<<"us"<<std::endl;
		for (int k = 0; k < dets.size(); k++)
		{
			for (size_t f = 0; f < dets[k].size(); f++)
			{
				cv::rectangle(RawImageVec[k], cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin),
				cv::Point(dets[k][f].bbox.xmax, dets[k][f].bbox.ymax), cv::Scalar(0, 255, 0), 2, 8, 0);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[0], dets[k][f].key_points[1]), 2, cv::Scalar(255, 0, 0), -1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[2], dets[k][f].key_points[3]), 2, cv::Scalar(0, 0, 255), -1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[4], dets[k][f].key_points[5]), 2, cv::Scalar(0, 255, 0), -1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[6], dets[k][f].key_points[7]), 2, cv::Scalar(255, 0, 255), -1);
				cv::circle(RawImageVec[k], cv::Point2f(dets[k][f].key_points[8], dets[k][f].key_points[9]), 2, cv::Scalar(0, 255, 255), -1);
				std::string label4 = cv::format("%f", dets[k][f].confidence);
				cv::putText(RawImageVec[k], label4, cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
				//std::cout << "yaw angle:" << dets[k][f].YawAngle<< " pitch angle:" << dets[k][f].PitchAngle<<" 瞳距:"<<dets[k][f].InterDis<< std::endl;
			}
#if show
			cv::imshow("show", RawImageVec[k]);
			cv::waitKey(1);
#endif		
		}
		
	}
	std::cout<<"yolov8face_detect average time:"<<sumtime/(config.yolov8face_detect_bs*1.0*ImageName1.size()/config.yolov8face_detect_bs)<<"us"<<std::endl;
	std::cout<<"yolov8face_detect test finash!"<<std::endl;
#endif

# if face_detect_tracker
	
	/*-------------单独调用人脸检测跟踪接口------------*/
	for (size_t k = 0; k < 2; k++)
	{
		cv::VideoCapture cap(data_model_path+"video0.avi");
		if (!cap.isOpened())
		{
			std::cout<<"video is not exxist!"<<std::endl;
			return 0;
		}
		cv::Mat img;
		while (true)
		{
			if(!cap.read(img))
				break;
			std::vector<cv::Mat>RawImageVec;
			RawImageVec.push_back(img);
			std::vector<std::vector<FaceDet>>dets;
			auto start = std::chrono::system_clock::now();
			Face_Detect_Tracker(RawImageVec,dets);
			auto end = std::chrono::system_clock::now();
			//std::cout<<"time:"<<(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())<<"ms"<<std::endl;
			for (int i = 0; i < dets.size(); i++)
			{
				for (size_t j = 0; j < dets[i].size(); j++)
				{
					cv::rectangle(img, cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin),
					cv::Point(dets[i][j].bbox.xmax, dets[i][j].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
					cv::circle(img, cv::Point2f(dets[i][j].key_points[0], dets[i][j].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
					cv::circle(img, cv::Point2f(dets[i][j].key_points[2], dets[i][j].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
					cv::circle(img, cv::Point2f(dets[i][j].key_points[4], dets[i][j].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
					cv::circle(img, cv::Point2f(dets[i][j].key_points[6], dets[i][j].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
					cv::circle(img, cv::Point2f(dets[i][j].key_points[8], dets[i][j].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
					//std::string label3 = cv::format("%d", dets[i][j].idx);
					//cv::putText(img, label3, cv::Point(dets[i][j].bbox.xmin, dets[i][j].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
					//std::cout << "yaw angle:" << dets[i][j].YawAngle<< " pitch angle:" << dets[i][j].PitchAngle<<" 瞳距:"<<dets[i][j].InterDis<< std::endl;
				}        
			}
#if show
			cv::imshow("show", img);
			cv::waitKey(1);
#endif
		}
		
	}
	std::cout<<"face_detect_tracker test finash!"<<std::endl;	
	/*-------------单独调用人脸检测跟踪接口------------*/
#endif

#if face_recognition
	/*-------------单独调用识别和计算相似度接口------------*/
	cv::Mat image1=cv::imread(data_model_path+"joey0.ppm");   //图像已经经过矫正
	cv::Mat image2=cv::imread(data_model_path+"joey1.ppm");   //图像已经经过矫正
	//人脸特征特征提取
	Feature feature1,feature2;
	Face_Feature_Extraction(image1,feature1);
	Face_Feature_Extraction(image2,feature2);
	//计算特征的相似度
	float simi1=Cal_Score(feature1,feature2);
	std::cout<<"simi1:"<<simi1<<std::endl;
	std::cout<<"face_recognition test finash!"<<std::endl;
	/*-------------单独调用识别和计算相似度------------*/

#endif

#if face_detect_aligner_recognitiion

		std::vector<cv::Mat>MatVec;
		cv::Mat RealImg;
		cv::Mat img1=cv::imread(data_model_path+"3.jpg");
		cv::Mat img2=cv::imread(data_model_path+"4.jpg");
		MatVec.push_back(img1);
		MatVec.push_back(img2);
		std::vector<std::vector<FaceDet>>facedet;
		
		//step1.人脸检测
		Face_Detect(MatVec, facedet);
		std::vector<FaceGroup>Face_Grop;
		//显示结果

		for (int i = 0; i < facedet.size(); i++)
		{
			for (size_t j = 0; j < facedet[i].size(); j++)
			{
				
				cv::Mat face_draw=MatVec[i].clone();
				cv::rectangle(face_draw, cv::Point(facedet[i][j].bbox.xmin, facedet[i][j].bbox.ymin),
				cv::Point(facedet[i][j].bbox.xmax, facedet[i][j].bbox.ymax), cv::Scalar(0, 255, 255), 2, 8, 0);
				cv::circle(face_draw, cv::Point2f(facedet[i][j].key_points[0], facedet[i][j].key_points[1]), 2, cv::Scalar(255, 0, 0), 1);
				cv::circle(face_draw, cv::Point2f(facedet[i][j].key_points[2], facedet[i][j].key_points[3]), 2, cv::Scalar(0, 0, 255), 1);
				cv::circle(face_draw, cv::Point2f(facedet[i][j].key_points[4], facedet[i][j].key_points[5]), 2, cv::Scalar(0, 255, 0), 1);
				cv::circle(face_draw, cv::Point2f(facedet[i][j].key_points[6], facedet[i][j].key_points[7]), 2, cv::Scalar(255, 0, 255), 1);
				cv::circle(face_draw, cv::Point2f(facedet[i][j].key_points[8], facedet[i][j].key_points[9]), 2, cv::Scalar(0, 255, 255), 1);
				std::string label = cv::format("%d", facedet[i][j].confidence);
				std::cout << "偏航角:" << facedet[i][j].YawAngle<< " 俯仰角:" << facedet[i][j].PitchAngle << std::endl;
				cv::putText(face_draw, label, cv::Point(facedet[i][j].bbox.xmin, facedet[i][j].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
				
				FaceGroup face_grop;
				//step2.将人脸从原图中抠出来，并将bbox扩大1.2倍
				FaceRect face_rect;
				face_rect.xmin=facedet[i][j].bbox.xmin/1.1>0?facedet[i][j].bbox.xmin/1.1:1;
				face_rect.ymin=facedet[i][j].bbox.ymin/1.1>0?facedet[i][j].bbox.ymin/1.1:1;
				face_rect.xmax=facedet[i][j].bbox.xmax*1.1<MatVec[i].cols?facedet[i][j].bbox.xmax*1.1:MatVec[i].cols-1;
				face_rect.ymax=facedet[i][j].bbox.ymax*1.1<MatVec[i].rows?facedet[i][j].bbox.ymax*1.1:MatVec[i].rows-1;
				
				//外扩1.2倍的人脸图像
				face_grop.face_img=MatVec[i](cv::Rect(face_rect.xmin,face_rect.ymin,face_rect.xmax-face_rect.xmin,face_rect.ymax-face_rect.ymin)).clone();
				//step3.关键点坐标转换(原点坐标改变)
				for (size_t k = 0; k < 5; k++)
				{
					face_grop.key_points[k].x=facedet[i][j].key_points[2*k]-face_rect.xmin;
					face_grop.key_points[k].y=facedet[i][j].key_points[2*k+1]-face_rect.ymin;
				}
				//step4.调用人脸矫正接口
				Face_Aligner(face_grop.face_img,face_grop.key_points,face_grop.face_aligner_img);

				cv::imshow("aligner",face_grop.face_aligner_img);
				cv::waitKey(0);

				//step5.调用人脸识别算法接口,得到人脸的512维特征
				Face_Feature_Extraction(face_grop.face_aligner_img,face_grop.face_feature);
				Face_Grop.push_back(face_grop);
			}	
		}
		//step6.计算人脸的相似度
		float simi2=Cal_Score(Face_Grop[0].face_feature,Face_Grop[1].face_feature);
		std::cout<<"simi2:"<<simi2<<std::endl;
#if show
		cv::imshow("face",MatVec[0]);
		cv::waitKey(0);
#endif
#endif


#if mask_recognition
	cv::Mat mask_img=cv::imread(data_model_path+"1234.png");
	float mask_pred=0.0;
	Mask_Recognition(mask_img,mask_pred);
	std::cout<<"mask prob:"<<mask_pred<<std::endl;
	std::cout<<"mask_recognition test finash!"<<std::endl;
#endif
	

#if gender_age_recognition
	cv::Mat genderageMat=cv::imread(data_model_path+"1234.png");
	//gender=0 女性  gender=1 男性
	attribute gender_age;
	Gender_Age_Recognition(genderageMat,gender_age);
	std::cout<<"gender:"<<gender_age.gender<<" age:"<<gender_age.age<<std::endl;
	std::cout<<"gender_age_recognition test finash!"<<std::endl;
#endif


#if silnet_face_anti_spoofing
	//配合人脸检测使用算法使用时选择偏航角度 和俯仰角度都小于30°的人脸，并且宽高外扩2.7倍(左右各1.35倍，上下各1.35倍)
	cv::Mat silnetfaceMat=cv::imread(data_model_path+"antispoofing/4.jpg");
	//0 fake 
	//1 real 
	//2 fake
	SilentFace silentface;
	Silent_Face_Anti_Spoofing(silnetfaceMat,silentface);
	std::cout<<"classes:"<<silentface.classes<<" prob:"<<silentface.prob<<std::endl;
	std::cout<<"silnet_face_anti_spoofing test finash!"<<std::endl;
#endif

	Release(config);
	return 0;
}
