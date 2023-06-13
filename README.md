# FaceAlgorithm
1. 人脸检测(retinaface,yolov5face),人脸矫正，人脸识别，带口罩识别，年龄性别识别，静默活体识别;
2. 使用C++ 和TensorRT加速;
3. 根据不同的显卡型号自动生成对应的engine(如果文件夹下有engine，则删除才能重新生成engien);
4. 人脸识别正常流程:
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

5. 条件编译测试
    #define face_detect                       1           //人脸检测
    #define yolov5face_detect				  1           //yolov5face 人脸检测
    #define face_recognition                  1           //人脸识别（人脸特征提取）+相似度计算
    #define face_detect_tracker               1           //人脸检测跟踪
    #define face_detect_aligner_recognitiion  0           //人脸检测——矫正——识别(人脸特征提取)
    #define mask_recognition                  1           //口罩识别
    #define gender_age_recognition            1           //性别年龄识别
    #define silnet_face_anti_spoofing         1           //静默活体检测


# 模型
百度云：链接: https://pan.baidu.com/s/1c8NQO2cZpAqwEMbfZxsJZg 提取码: 5xaa

# 人脸检测
1. retinaface(mobilenet0.25)
2. yolov5face

# 人脸识别
1. arcface(R50)

# 带口罩识别

# 年龄性别

# 静默活体识别

# 使用方法