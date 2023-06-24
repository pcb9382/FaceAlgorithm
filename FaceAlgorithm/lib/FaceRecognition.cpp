#include <memory>
#include "FaceRecognition.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <fstream>
#include "DataTypes_Face.h"
#include "detector.h"
#include "detector_yolov5face.h"
#include "FaceAngle.h"
#include "aligner.h"
#include "recognition.h"
#include <fstream>
#include <time.h>
#include "BYTETracker.h"
#include "MaskRecognition.h"
#include "GenderAgeRecognition.h"
#include "SilentFaceAntiSpoofing.h"
#include "detector_yolov7face.h"

class FaceRecognition
{
public:
	Detector *detector;
	Detector_Yolov5Face *yolov5face;
	Detector_Yolov7Face *yolov7face;
	Aligner *aligner;
	Recognition*recognition;
	Tracking *tracker;
	MaskRecognition* maskrecognition;
	GenderAgeRecognition *genderage;
	SilentFaceAntiSpoofing*silnetface_antispoofing;
	/** 
	 * @brief                   人脸初始化函数
	 * @param config			模块配置参数结构体
	 * @return                  HZFLAG
	 */
	HZFLAG Initialize(Config& config);


	/** 
	 * @brief                   人脸检测
	 * @param img			    opencv　Mat格式
	 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
	 * @return                  HZFLAG
	 */		
	HZFLAG Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

	/** 
	 * @brief                   人脸检测(yolov5_face)
	 * @param img			    opencv　Mat格式
	 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
	 * @return                  HZFLAG
	 */		
	HZFLAG Yolov5Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

	/** 
	 * @brief                   人脸检测(yolov7_face)
	 * @param img			    opencv　Mat格式
	 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
	 * @return                  HZFLAG
	 */		
	HZFLAG Yolov7Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

	/** 
	 * @brief                   人脸检测跟踪(视频流)
	 * @param img			    opencv　Mat格式
	 * @param FaceDets		    FaceDets	人脸检测结果列表，包括人脸bbox，id,置信度，偏航角度，俯仰角度，五个关键点坐标
	 * @return                  HZFLAG
	 */	
	HZFLAG Face_Detect_Tracker(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);


	/** 
	 * @brief                   人脸矫正
	 * @param Faceimg           需要矫正的人脸图像(矩形框bbox外扩1.2倍得到的人脸图像然后进行矫正!!!!)
	 * @param KeyPoints         人脸关键点
	 * @param Face_Aligener		矫正之后的图像
	 * @return                  HZFLAG
	 */	
	HZFLAG Face_Aligner(cv::Mat&Face_image,cv::Point2f *KeyPoints,cv::Mat&Face_Aligener);

	/** 
	 * @brief                   人脸特征提取
	 * @param Face_Aligener     经过人脸矫正的人脸图像
	 * @param Face_Feature		人脸特征(512维特征)
	 * @return                  HZFLAG
	 */		
	HZFLAG Face_Feature_Extraction(cv::Mat&Face_Aligener,Feature&Face_Feature);


	/** 
	 * @brief               计算人脸特征的相似度
	 * @param Feature1      经过人脸矫正的人脸图像
	 * @param Feature2		人脸特征(512维特征)
	 * @return float		相似度得分               
	 */	
	float Cal_Score(Feature&Feature1,Feature&Feature2);

	/** 
	 * @brief                   人脸戴口罩识别
	 * @param img               需要识别的人脸戴口罩图像
	 * @param Result            人脸戴口罩识别结果
	 * @return                  HZFLAG
	 */
	HZFLAG Mask_Recognition(cv::Mat&img,float&pred);

	/** 
	 * @brief                   性别年龄识别
	 * @param img               需要识别的人脸图像
	 * @param Result            人脸戴口罩识别结果
	 * @return                  HZFLAG
	 */
	HZFLAG Gender_Age_Recognition(cv::Mat &img,attribute&gender_age);

	/** 
	 * @brief                   静默活体检测
	 * @param img               需要检测的人脸图像
	 * @param Result            静默活体检测识别结果
	 * @return                  HZFLAG
	 */
	HZFLAG Silent_Face_Anti_Spoofing(cv::Mat&img, SilentFace&silentface);

	/** 
	 * @brief               反初始化
	 * @param Config& config
	 * @return              
	 */		
	HZFLAG Release(Config& config);
};


FaceRecognition face_recognition;


/** 
 * @brief               初始化人脸参数结构体
 * @param facedet1      人脸参数结构体
 * @return int               
 */
static int InitFace(FaceDet&facedet)
{
	facedet.bbox.xmax = 0;
	facedet.bbox.xmin = 0;
	facedet.bbox.ymin = 0;
	facedet.bbox.ymax = 0;
	facedet.confidence = 0;
	facedet.YawAngle= 90;
	facedet.PitchAngle = 90;
	facedet.idx= -1;
	facedet.label = -1;
	for (int i=0;i<10;i++)
	{
		facedet.key_points[i] = 0;
	}
	return 0;

}
/** 
 * @brief                   人脸初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
HZFLAG FaceRecognition::Initialize(Config& config)
{
	//初始化人脸矫正算法；
	aligner=new Aligner();
	//人脸跟踪算法
	tracker=new Tracking(30,30,5);
	
	//初始化人脸检测算法
	if (config.face_detect_enable)
	{
		detector = new Detector();
		HZFLAG facedetect_flag=detector->InitDetector(config);
		if (facedetect_flag!=HZ_SUCCESS)
		{
			std::cout<<"face detect init failed"<<std::endl;
			return facedetect_flag;
		}
		std::cout<<"face detect init successed!"<<std::endl;
	}

	//yolov5face
	if (config.yolov5face_detect_enable)
	{
		yolov5face=new Detector_Yolov5Face();
		HZFLAG yolov5facedetect_flag=yolov5face->InitDetector_Yolov5Face(config);
		if (yolov5facedetect_flag!=HZ_SUCCESS)
		{
			std::cout<<"yolov5face detect init failed"<<std::endl;
			return yolov5facedetect_flag;
		}
		std::cout<<"yolov5face detect init successed!"<<std::endl;
	}
	
	//yolov7face
	if (config.yolov7face_detect_enable)
	{
		yolov7face=new Detector_Yolov7Face();
		HZFLAG yolov7facedetect_flag=yolov7face->InitDetector_Yolov7Face(config);
		if (yolov7facedetect_flag!=HZ_SUCCESS)
		{
			std::cout<<"yolov7face detect init failed"<<std::endl;
			return yolov7facedetect_flag;
		}
		std::cout<<"yolov7face detect init successed!"<<std::endl;
	}

	//初始化人脸识别算法
	if (config.face_recognition_enable)
	{
		recognition=new Recognition();
		HZFLAG facerecogniton_flag=recognition->InitRecognition(config);
		if (facerecogniton_flag!=HZ_SUCCESS)
		{
			std::cout<<"face recognition init failed"<<std::endl;
			return facerecogniton_flag;
		}
		std::cout<<"face recognition init successed!"<<std::endl;
	}
	
	if (config.face_mask_enable)
	{
		//戴口罩识别算法
		maskrecognition=new MaskRecognition();
		HZFLAG maskrecogniton_flag=maskrecognition->MaskRecognitionInit(config);
		if (maskrecogniton_flag!=HZ_SUCCESS)
		{
			std::cout<<"mask recognition init failed"<<std::endl;
			return maskrecogniton_flag;
		}
		std::cout<<"mask recognition init success!"<<std::endl;
	}

	if (config.gender_age_enable)
	{
		genderage=new GenderAgeRecognition();
		HZFLAG genderage_flag=genderage->GenderAgeRecognitionInit(config);
		if (genderage_flag!=HZ_SUCCESS)
		{
			std::cout<<"gender_age recognition init failed"<<std::endl;
			return genderage_flag;
		}
		std::cout<<"gender_age recognition init success!"<<std::endl;
	}
	
	if(config.silent_face_anti_spoofing_enable)
	{
		silnetface_antispoofing=new SilentFaceAntiSpoofing();
		HZFLAG silnetface_flag=silnetface_antispoofing->SilentFaceAntiSpoofingInit(config);
		if (silnetface_flag!=HZ_SUCCESS)
		{
			std::cout<<"Silent Face Anti Spoofing init failed"<<std::endl;
			return silnetface_flag;
		}
		std::cout<<"Silent Face Anti Spoofing init success!"<<std::endl;
	}

	std::cout<<"Initialize Finash!"<<std::endl;
	return HZ_SUCCESS;
}

/** 
 * @brief                   人脸检测
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */	
HZFLAG FaceRecognition::Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	//人脸检测
	std::vector<std::vector<Det>>temp_det;
	detector->detect(img,temp_det);
	
	//开始对检测结果进行解析
	//多帧图像
	for (size_t i = 0; i < temp_det.size(); i++)
	{
		//每一帧图像人脸
		std::vector<FaceDet> Temp_FaceDet;
		for (int j=0;j<temp_det[i].size();j++)
		{
			FaceDet facedet;
			InitFace(facedet);
			facedet.bbox = temp_det[i][j].bbox;
			facedet.confidence = temp_det[i][j].confidence;
			facedet.label = -1;
			cv::Point temp_keypoint[5];
			for (int k=0;k<5;k++)
			{
				cv::Point2f point111;
				point111.x = temp_det[i][j].key_points[2*k];
				point111.y = temp_det[i][j].key_points[2*k+1];
				facedet.key_points[2*k]=point111.x;
				facedet.key_points[2*k+1]=point111.y;
				temp_keypoint[k]=point111;
			}
			float face_params[3];    
			Face_Angle_Cal(cv::Rect(facedet.bbox.xmin, facedet.bbox.ymin,facedet.bbox.xmax - facedet.bbox.xmin, facedet.bbox.ymax - facedet.bbox.ymin), temp_keypoint,face_params);
			facedet.YawAngle=face_params[0];
			facedet.PitchAngle=face_params[1];  
			facedet.InterDis = face_params[2];
			Temp_FaceDet.push_back(facedet);
		}
		FaceDets.push_back(Temp_FaceDet);
	}
	return HZ_SUCCESS;
}

/** 
 * @brief                   人脸检测(yolov5_face)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */		
HZFLAG FaceRecognition::Yolov5Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	//人脸检测
	std::vector<std::vector<Det>>temp_det;
	yolov5face->Detect_Yolov5Face(img,temp_det);
	
	//开始对检测结果进行解析
	//多帧图像
	for (size_t i = 0; i < temp_det.size(); i++)
	{
		//每一帧图像人脸
		std::vector<FaceDet> Temp_FaceDet;
		for (int j=0;j<temp_det[i].size();j++)
		{
			FaceDet facedet;
			InitFace(facedet);
			facedet.bbox = temp_det[i][j].bbox;
			facedet.confidence = temp_det[i][j].confidence;
			facedet.label = -1;
			cv::Point temp_keypoint[5];
			for (int k=0;k<5;k++)
			{
				cv::Point2f point111;
				point111.x = temp_det[i][j].key_points[2*k];
				point111.y = temp_det[i][j].key_points[2*k+1];
				facedet.key_points[2*k]=point111.x;
				facedet.key_points[2*k+1]=point111.y;
				temp_keypoint[k]=point111;
			}
			float face_params[3];    
			Face_Angle_Cal(cv::Rect(facedet.bbox.xmin, facedet.bbox.ymin,facedet.bbox.xmax - facedet.bbox.xmin, facedet.bbox.ymax - facedet.bbox.ymin), temp_keypoint,face_params);
			facedet.YawAngle=face_params[0];
			facedet.PitchAngle=face_params[1];  
			facedet.InterDis = face_params[2];
			Temp_FaceDet.push_back(facedet);
		}
		FaceDets.push_back(Temp_FaceDet);
	}
	return HZ_SUCCESS;
}


/** 
 * @brief                   人脸检测(yolov7_face)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */		
HZFLAG FaceRecognition::Yolov7Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	//人脸检测
	std::vector<std::vector<Det>>temp_det;
	yolov7face->Detect_Yolov7Face(img,temp_det);
	
	//开始对检测结果进行解析
	//多帧图像
	for (size_t i = 0; i < temp_det.size(); i++)
	{
		//每一帧图像人脸
		std::vector<FaceDet> Temp_FaceDet;
		for (int j=0;j<temp_det[i].size();j++)
		{
			FaceDet facedet;
			InitFace(facedet);
			facedet.bbox = temp_det[i][j].bbox;
			facedet.confidence = temp_det[i][j].confidence;
			facedet.label = -1;
			cv::Point temp_keypoint[7];
			for (int k=0;k<7;k++)
			{
				cv::Point2f point111;
				point111.x = temp_det[i][j].key_points[2*k];
				point111.y = temp_det[i][j].key_points[2*k+1];
				facedet.key_points[2*k]=point111.x;
				facedet.key_points[2*k+1]=point111.y;
				temp_keypoint[k]=point111;
			}
			float face_params[3];    
			Face_Angle_Cal(cv::Rect(facedet.bbox.xmin, facedet.bbox.ymin,facedet.bbox.xmax - facedet.bbox.xmin, facedet.bbox.ymax - facedet.bbox.ymin), temp_keypoint,face_params);
			facedet.YawAngle=face_params[0];
			facedet.PitchAngle=face_params[1];  
			facedet.InterDis = face_params[2];
			Temp_FaceDet.push_back(facedet);
		}
		FaceDets.push_back(Temp_FaceDet);
	}
	return HZ_SUCCESS;
}

/** 
 * @brief                   人脸检测跟踪(视频流)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    FaceDets	人脸检测结果列表，包括人脸bbox，id,置信度，偏航角度，俯仰角度，五个关键点坐标
 * @return                  HZFLAG
 */	
HZFLAG FaceRecognition::Face_Detect_Tracker(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	//人脸检测
	std::vector<std::vector<Det>>temp_det;
	detector->detect(img,temp_det);
	//开始对检测结果进行解析
	//多帧图像
	for (size_t i = 0; i < temp_det.size(); i++)
	{
		std::vector<Object> objects;
		for (int j=0;j<temp_det[i].size();j++)
		{
			Object object;
			object.rect.x=temp_det[i][j].bbox.xmin;
			object.rect.y=temp_det[i][j].bbox.ymin;
			object.rect.width=temp_det[i][j].bbox.xmax-temp_det[i][j].bbox.xmin;
			object.rect.height=temp_det[i][j].bbox.ymax-temp_det[i][j].bbox.ymin;
			object.prob=temp_det[i][j].confidence;
			object.label=1;
			for (int k=0;k<5;k++)
			{
				object.key_points.push_back(cv::Point2f(temp_det[i][j].key_points[2*k],temp_det[i][j].key_points[2*k+1]));
			}
			objects.push_back(object);
		}

		//对该帧人脸进行跟踪
		std::vector<STrack> output_stracks=tracker->update(objects);              
		
		//解析跟踪结果
		std::vector<FaceDet>det;
		for (int i=0;i<output_stracks.size();i++)
		{
			FaceDet Dets1;
			InitFace(Dets1);
			std::vector<float> tlwh = output_stracks[i].tlwh;
			Dets1.idx=output_stracks[i].track_id;
			Dets1.confidence=1;
			Dets1.bbox.xmin=tlwh[0];
			Dets1.bbox.ymin=tlwh[1];
			Dets1.bbox.xmax=tlwh[0]+tlwh[2];
			Dets1.bbox.ymax=tlwh[1]+tlwh[3];
			cv::Point temp_keypoint[5];
			for (size_t k = 0; k <5; k++)
			{
				Dets1.key_points[2*k]=output_stracks[i].key_points[k].x;
				Dets1.key_points[2*k+1]=output_stracks[i].key_points[k].y;
				temp_keypoint[k]=cv::Point(output_stracks[i].key_points[k].x,output_stracks[i].key_points[k].y);
			}
			float face_params[3];    
			Face_Angle_Cal(cv::Rect(Dets1.bbox.xmin, Dets1.bbox.ymin,Dets1.bbox.xmax - Dets1.bbox.xmin, Dets1.bbox.ymax - Dets1.bbox.ymin), temp_keypoint,face_params);
			Dets1.YawAngle=face_params[0];
			Dets1.PitchAngle=face_params[1];  
			Dets1.InterDis = face_params[2];            
			det.push_back(Dets1);                 
		}
		FaceDets.push_back(det);
	}
	return HZ_SUCCESS;
}


/** 
 * @brief                   人脸特征提取
 * @param Face_Aligener     经过人脸矫正的人脸图像
 * @param Face_Feature		人脸特征(512维特征)
 * @return                  HZFLAG
 */	
HZFLAG FaceRecognition::Face_Aligner(cv::Mat&Face_image,cv::Point2f KeyPoints[5],cv::Mat&Face_Aligener)
{
	//将五点转换成cv::Point2f;
	std::vector<cv::Point2f>keypoints;
	for (size_t f = 0; f < 5; f+=1)
	{
		cv::Point2f keypoint;
		keypoint.x=KeyPoints[f].x;
		keypoint.y=KeyPoints[f].y;
		keypoints.push_back(keypoint);
	}
	//得到矫正之后的人脸
	aligner->AlignFace(Face_image,keypoints,&Face_Aligener);
	//人脸识别，进行人脸特征提取
    return HZ_SUCCESS;
}

/** 
 * @brief               计算人脸特征的相似度
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2		人脸特征(512维特征)
 * @return float		相似度得分               
 */	
 HZFLAG FaceRecognition::Face_Feature_Extraction(cv::Mat&Face_Aligener,Feature&Face_Feature)
{
	recognition->Extract_feature(Face_Aligener,Face_Feature);
	return HZ_SUCCESS;
}


/** 
 * @brief                   人脸戴口罩识别
 * @param img               需要识别的人脸戴口罩图像
 * @param Result            人脸戴口罩识别结果
 * @return                  HZFLAG
 */
 float FaceRecognition::Cal_Score(Feature&Feature1,Feature&Feature2)
{
	//特征1
    cv::Mat out1(512, 1, CV_32FC1, Feature1.feature);
    cv::Mat out_norm1;
    cv::normalize(out1, out_norm1);
    //特征2,待比对的特征,
    cv::Mat out2(1,512,CV_32FC1,Feature2.feature);
    cv::Mat out_norm2;
    cv::normalize(out2, out_norm2);
    cv::Mat res = out_norm2 * out_norm1;
    float Score=*(float*)res.data;
    return Score;
}

/** 
 * @brief                   人脸戴口罩识别
 * @param img               需要识别的人脸戴口罩图像
 * @param Result            人脸戴口罩识别结果
 * @return                  HZFLAG
 */
HZFLAG FaceRecognition::Mask_Recognition(cv::Mat &img,  float&pred)
{
	return maskrecognition->MaskRecognitionRun(img, pred);
}

/** 
 * @brief                   性别年龄识别
 * @param img               需要识别的人脸图像
 * @param Result            人脸戴口罩识别结果
 * @return                  HZFLAG
 */
HZFLAG FaceRecognition::Gender_Age_Recognition(cv::Mat &img,attribute&gender_age)
{
	return genderage->GenderAgeRecognitionRun(img,gender_age);
}

/** 
 * @brief                   静默活体检测
 * @param img               需要检测的人脸图像
 * @param Result            静默活体检测识别结果
 * @return                  HZFLAG
 */
HZFLAG FaceRecognition::Silent_Face_Anti_Spoofing(cv::Mat&img, SilentFace&silentface)
{
	return silnetface_antispoofing->SilentFaceAntiSpoofingRun(img,silentface);
}


/** 
 * @brief               反初始化
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2		人脸特征(512维特征)
 * @return float		相似度得分               
 */		
HZFLAG FaceRecognition::Release(Config& config)
{
	if (config.face_detect_enable)
	{
		detector-> ReleaseDetector();
		delete detector;
		detector = NULL;
	}
	if (config.yolov5face_detect_enable)
	{
		yolov5face->ReleaseDetector_Yolov5Face();
		delete yolov5face;
		yolov5face=NULL;
	}
	if (config.yolov7face_detect_enable)
	{
		yolov7face->ReleaseDetector_Yolov7Face();
		delete yolov7face;
		yolov7face=NULL;
	}

	if(config.face_recognition_enable)
	{
		recognition->ReleaseRecognition();
		delete recognition;
		recognition=NULL;
	}
	
	if (config.face_mask_enable)
	{
		maskrecognition->MaskRecognitionRelease();
		delete maskrecognition;
		maskrecognition=NULL;
	}
	
	if(config.gender_age_enable)
	{
		genderage->GenderAgeRecognitionRelease();
		delete genderage;
		genderage=NULL;
	}

	if (config.silent_face_anti_spoofing_enable)
	{
		silnetface_antispoofing->SilentFaceAntiSpoofingRelease();
		delete silnetface_antispoofing;
		silnetface_antispoofing=NULL;
	}
	
	delete aligner;
	aligner=NULL;
	
	return HZ_SUCCESS;
}


/** 
 * @brief                   人脸初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
HZFLAG Initialize(Config& config)
{
	return face_recognition.Initialize(config);
}
/** 
 * @brief                   人脸检测
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */		
HZFLAG Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	return face_recognition.Face_Detect(img,FaceDets);
}

/** 
 * @brief                   人脸检测(yolov5_face)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */		
HZFLAG Yolov5Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	return face_recognition.Yolov5Face_Detect(img,FaceDets);
}

/** 
 * @brief                   人脸检测(yolov7_face)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return                  HZFLAG
 */		
HZFLAG Yolov7Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	return face_recognition.Yolov7Face_Detect(img,FaceDets);
}

/** 
 * @brief                   人脸检测跟踪(视频流)
 * @param img			    opencv　Mat格式
 * @param FaceDets		    FaceDets	人脸检测结果列表，包括人脸bbox，id,置信度，偏航角度，俯仰角度，五个关键点坐标
 * @return                  HZFLAG
 */	
HZFLAG Face_Detect_Tracker(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets)
{
	return face_recognition.Face_Detect_Tracker(img,FaceDets);
}


/** 
 * @brief                   人脸矫正
 * @param Faceimg           需要矫正的人脸图像(矩形框bbox外扩1.2倍得到的人脸图像然后进行矫正!!!!)
 * @param KeyPoints         人脸关键点
 * @param Face_Aligener		矫正之后的图像
 * @return                  HZFLAG
 */	
HZFLAG Face_Aligner(cv::Mat&Face_image,cv::Point2f *KeyPoints,cv::Mat&Face_Aligener)
{
	return face_recognition.Face_Aligner(Face_image,KeyPoints,Face_Aligener);
}

/** 
 * @brief                   人脸特征提取
 * @param Face_Aligener     经过人脸矫正的人脸图像
 * @param Face_Feature		人脸特征(512维特征)
 * @return                  HZFLAG
 */		
HZFLAG Face_Feature_Extraction(cv::Mat&Face_Aligener,Feature&Face_Feature)
{
	return face_recognition.Face_Feature_Extraction(Face_Aligener,Face_Feature);
}
/** 
 * @brief               计算人脸特征的相似度
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2		人脸特征(512维特征)
 * @return float		相似度得分               
 */	
float Cal_Score(Feature&Feature1,Feature&Feature2)
{
	return face_recognition.Cal_Score(Feature1,Feature2);
}

/** 
 * @brief                   人脸戴口罩识别
 * @param img               需要识别的人脸戴口罩图像
 * @param Result            人脸戴口罩识别结果
 * @return                  HZFLAG
 */
HZFLAG Mask_Recognition(cv::Mat&img, float&pred)
{
	return face_recognition.Mask_Recognition(img,pred);
}

/** 
 * @brief                   性别年龄识别
 * @param img               需要识别的人脸图像
 * @param Result            人脸戴口罩识别结果
 * @return                  HZFLAG
 */
HZFLAG Gender_Age_Recognition(cv::Mat &img,attribute&gender_age)
{
	return face_recognition.Gender_Age_Recognition(img,gender_age);
}

/** 
 * @brief                   静默活体检测
 * @param img               需要检测的人脸图像
 * @param Result            静默活体检测识别结果
 * @return                  HZFLAG
 */
HZFLAG Silent_Face_Anti_Spoofing(cv::Mat&img, SilentFace&silentface)
{
	return face_recognition.Silent_Face_Anti_Spoofing(img,silentface);
}

/** 
 * @brief               反初始化
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2		人脸特征(512维特征)
 * @return float		相似度得分               
 */		
HZFLAG Release(Config& config)
{
	return face_recognition.Release(config);
}