#pragma once

#ifndef _FACERECOGNITION_
#define _FACERECOGNITION_

#include <iostream>
#include "DataTypes_Face.h"
#include <opencv2/opencv.hpp>

#ifdef __cplusplus 
extern "C" { 
#endif 


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
 * @brief                   人脸检测(yolov5_face)
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
HZFLAG Mask_Recognition(cv::Mat &img,float&pred);

/** 
 * @brief                   性别年龄识别
 * @param img               需要识别的人脸图像
 * @param Result            性别年龄识别结果
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
 * @return              HZFLAG 
 */		
HZFLAG Release(Config& config);

#ifdef __cplusplus 
} 
#endif

#endif



