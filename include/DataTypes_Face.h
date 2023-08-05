#ifndef _DATATYPES_
#define _DATATYPES_

#include <vector>
#include <utility>
#include <time.h>
#include <string>
enum HZFLAG
{
	HZ_FILEOPENFAILED,            //文件打开失败
	HZ_IMGEMPTY,                  //图像为空
	HZ_SUCCESS,                   //成功
	HZ_ERROR,                     //失败
	HZ_WITHOUTMODEL,               //模型不存在                                                     
	HZ_IMGFORMATERROR,            //图像格式错误
	HZ_CLASSEMPTY,                //类别文件为空
	HZ_LOGINITFAILED,             //日志初始化失败           
	HZ_CONFIGLOADFAILED,          //configi加载失败            
	HZ_INITFAILED,                //初始化i失败                                             
};

struct affineMatrix  //letter_box  仿射变换矩阵
{
    float i2d[6];       //仿射变换正变换
    float d2i[6];       //仿射变换逆变换
};
struct bbox 
{
    float x1,x2,y1,y2;
    float landmarks[10]; //5个关键点
    float score;
};
const float color_list[5][3] =
{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
    {255,255,0},
};

typedef struct 
{
	float xmin;
	float xmax;
	float ymin;
	float ymax;
}FaceRect;
typedef struct 
{
	FaceRect bbox;
	int label;
	int id;
	float confidence;
	std::vector<float> key_points;    //关键点
}Det;

typedef struct 
{
	FaceRect bbox;                    //bbox
	int label;
	int idx;                          //跟踪的ID
	float confidence;                 //置信度
	float key_points[10];             //关键点坐标
	float YawAngle;                   //偏航角度
	float PitchAngle;                 //俯仰角度
	float InterDis;                   //瞳距
}FaceDet;

typedef struct
{
	float feature[512];
}Feature;

typedef struct 
{
    int gender=-1;
    int age=0;
}attribute;

typedef struct
{
    int classes;
    float prob;
}SilentFace;

//初始化的参数
typedef struct 
{
	int gpu_id;

	//face detect params
	std::string FactDetectModelPath;
	float confidence_thresh;
	int face_detect_bs;        
	float nms_thresh;
	bool face_detect_enable=false;

	//yolov5face detect params
	std::string Yolov5FactDetectModelPath;
	float yolov5face_confidence_thresh;
	int yolov5face_detect_bs;        
	float yolov5face_nms_thresh;
	bool yolov5face_detect_enable=false;

	//yolov7face detect params
	std::string Yolov7FactDetectModelPath;
	float yolov7face_confidence_thresh;
	int yolov7face_detect_bs;        
	float yolov7face_nms_thresh;
	bool yolov7face_detect_enable=false;

	//yolov7face detect params
	std::string Yolov8FactDetectModelPath;
	float yolov8face_confidence_thresh;
	int yolov8face_detect_bs;        
	float yolov8face_nms_thresh;
	bool yolov8face_detect_enable=false;

	//face recogniton
	std::string FactReconitionModelPath;             
	int face_recognition_bs;                       //人脸识别batchsize
	bool face_recognition_enable=false;

	//mask
	std::string MaskReconitionModelPath;
	int face_mask_bs;                              //带口罩识别batchsize
	bool face_mask_enable=false;

	//gender_age
	std::string GenderAgeModelPath;
	int gender_age_bs;                             //年龄性别识别batchsize
	bool gender_age_enable=false;
	
	//silent
	std::string FaceSilentModelPath;
	int silent_face_anti_spoofing_bs;              //静默活体检测
	bool silent_face_anti_spoofing_enable=false;

	//tracker
	float max_cosine_distance;
	float max_iou_distance;
	int nn_budget;
	int max_age;
	int n_init;

	//face angle
	float YawAngle;                        //偏航角度阈值
	float PitchAngle;                      //俯仰角度阈值
	float InterDis;                        //瞳距阈值
}Config;
#endif
