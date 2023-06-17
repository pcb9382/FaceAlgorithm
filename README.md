简体中文 | [English](./FaceAlgorithm_en.md)

如果觉得有用，不妨给个Star⭐️🌟支持一下吧~ 谢谢！
# FaceAlgorithm
## 特性
1. 人脸检测(retinaface,yolov5face),人脸旋转角度计算(俯仰角，偏航角)，人脸矫正，人脸识别，带口罩识别，年龄性别识别，静默活体识别;
2. 使用C++和TensorRT加速;
3. 根据不同的显卡型号自动生成对应的engine(如果文件夹下有其他显卡适配engine，则删除engine才能重新生成使用中的显卡对应的engien);
4. 提供C/C++接口，可以直接移植在项目里;
5. 人脸识别流程:

	1)人脸检测(图像、视频流)
			
	2)根据每个人脸返回的角度，筛选出合适角度的人脸用于人脸矫正，人脸识别

	3)人脸矫正(根据5个人脸关键点)
			
	4)人脸特征特征提取（512维特征）
					
	5)人脸特征比对(人脸相似度计算)

6. 条件编译测试说明
	| 测试种类 |  启用    |  说明   |
	|:----------|:----------|:----------|
    |face_detect                       |1|           人脸检测                         |
    |yolov5face_detect				   |1|           yolov5face 人脸检测              |
    |face_recognition                  |1|           人脸识别（人脸特征提取）+相似度计算   |
    |face_detect_tracker               |1|           人脸检测跟踪                      |
    |face_detect_aligner_recognitiion  |0|           人脸检测——矫正——识别(人脸特征提取)   |
    |mask_recognition                  |1|           口罩识别                         |
    |gender_age_recognition            |1|           性别年龄识别                      |
    |silnet_face_anti_spoofing         |1|           静默活体检测                      |

## 算法说明
### 1.人脸检测
1. retinaface(mobilenet0.25，R50需要自己修改代码）
2. yolov5face(yolov5sface，n,m,l,x需要自己转换对应的onnx)
3. yolov7face(TO DO)
4. yolov8facee(TO DO))
   

### 2.人脸识别
1. arcface(R50)
2. arcface(R101,需要自己下载模型修改代码)

### 3.带口罩识别
1. 分类模型

### 4.年龄性别
1. InsightFace中的年龄和性别识别;

### 5.静默活体识别
1. Silent-Face-Anti-Spoofing

### 6.跟踪
1. ByteTracker(加上人脸bbox和人脸关键点作为跟踪的输入，修改Bug)

### 7.算法接口
```
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
 * @param Result            性别年龄识别别结果
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
```
# 使用方法
## 1.模型下载
([Baidu Drive](https://pan.baidu.com/s/1c8NQO2cZpAqwEMbfZxsJZg) code: 5xaa)
| 模型 |  作用    |  说明   |
|:----------|:----------|:----------|
|FaceDetect.wts                        |人脸检测|        
|FaceRecognition.wts				   |人脸识别|       
|GenderAge.onnx                        |年龄性别识别|          
|MaskRecognition.onnx                  |口罩识别|          
|yolov5s-face_bs=1.onnx                |yolov5s人脸检测|          
|yolov5s-face_bs=4.onnx                |yolov5s人脸检测|        
|2.7_80x80_MiniFASNetV2.onnx           |静默活体检测|           

## 2.环境
1. ubuntu20.04+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
2. ubuntu18.04+cuda10.2+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
3. Win10+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1      (测试通过)
4. 其他环境请自行尝试或者加群了解


## 3.编译

1. 更改根目录下的CMakeLists.txt,设置tensorrt的安装目录
```
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. 默认opencv已安装，cuda,cudnn已安装
3. 为了Debug默认编译 ```-g O0``` 版本,如果为了加快速度请编译Release版本

4. 使用Visual Studio Code
```
   ctrl+shift+B
```
5. 使用命令行编译:
```
   mkdir build
   cd build
   cmake ..
   make -j6
```
 

# References
1. https://github.com/deepcam-cn/yolov5-face
2. https://github.com/wang-xinyu/tensorrtx
3. https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
4. https://github.com/linghu8812/tensorrt_inference

# Acknowledgments & Contact 
## 1.QQ Group：517671804
![](resources/QQGroup.jpeg)
## 2.WeChat ID: cbp931126