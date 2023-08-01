 English|[简体中文](./README.md)

If you find it useful, you may wish to give a Star ⭐️🌟support~ Thank you!

# Acknowledgments & Contact 
### 1.WeChat ID: cbp931126
Add me WeChat# (Note: FaceAlgorithm) to pull you into the group
### 2.QQ Group：517671804


# FaceAlgorithm
## 特性
1. Face detection (retinaface, yolov5face, yolov7face), face rotation angle calculation (pitch angle, yaw angle), face correction, face recognition,mask recognition,age_gender recognition,silent living recognition;
2. All models use C++ and TensorRT to accelerate inference, and the preprocess and postprocess of yolov7face uses cuda acceleration,(other model acceleration optimizations can also be referred to);
3. All models use C++ and OnnxRuntime.OpenVINO, NCNN Accelerated Inference (TO DO);
4. Construct similar to NV Deepstream, support a variety of inference frameworks(TensorRT, OnnxRuntime, OpenVINO, NCNN), for multi-channel RTSP pull stream + hard decoding + Pipeline + push stream(TO DO);
5. Automatically generate the corresponding engine according to different graphics card models (if there are other graphics cards in the folder, delete the engine to regenerate the engine corresponding to the graphics card in use);
6. Provide C/C++ interface, which can be directly ported to the project;
7. General process of face recognition:

	1)Face detection (images, video streams)
			
	2)According to the angle by each face, the face at the appropriate angle is screened out for face correction and face recognition

	3)Face correction (based on 5 key points of the face)
			
	4)Face feature feature extraction (512-dimensional features)
					
	5)Face feature comparison (face similarity calculation)

8. Description of conditional compilation tests
	| Test category |  enable    |  description   |
	|:----------:|:----------:|:----------:|
   |face_detect                        |1|           Face detection                         |
   |yolov5face_detect				      |1|           yolov5face Face detection              |
   |yolov5face_detect				      |1|           yolov7face Face detection              |
   |face_recognition                   |1|           Face recognition (face feature extraction) + similarity calculation   |
   |face_detect_tracker                |1|           Face detection tracking                      |
   |face_detect_aligner_recognitiion   |0|           Face detection - correction - recognition (face feature extraction)   |
   |mask_recognition                   |1|           Mask identification                         |
   |gender_age_recognition             |1|           Gender age identification                      |
   |silnet_face_anti_spoofing          |1|           Silent living identification                      |

## 算法说明
### 1人脸检测
#### 1)人脸检测retinaface(mobilenet0.25,R50需要自己修改代码）
   ![demoimg1](https://insightface.ai/assets/img/github/11513D05.jpg)
#### 2)yolov5face(yolov5sface，n,m,l,x需要自己转换对应的onnx)
   <img src="./resources/yolov5face_test.jpg" alt="drawing" width="800"/> 

#### 3)yolov7face(yolov7sface,另外不同大小的模型需要自己转换)
   <img src="./resources/yolov7face_test.jpg" alt="drawing" width="800"/>

#### 4)yolov8facee(TO DO)
   

### 2.人脸识别

#### 1) arcface(R50)

#### 2)arcface(R101,需要自己下载模型修改代码)
<div align="left">
  <img src="https://insightface.ai/assets/img/github/facerecognitionfromvideo.PNG" width="800"/>
</div>


### 3.带口罩识别
#### 1)检测->裁剪->识别(分类)
![demoimg1](https://insightface.ai/assets/img/github/cov_test.jpg)

### 4.年龄性别
#### 1)人脸检测->裁剪->年龄和性别识别
<div align="left">
  <img src="https://insightface.ai/assets/img/github/t1_genderage.jpg" width="800"/>
</div>

### 5.静默活体识别
#### 1)Silent-Face-Anti-Spoofing
   
|name| sample| result |image| sample| result |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
0.jpg|<img src="./FaceAlgorithm_Test/antispoofing/0.jpg" width="300" height="300"/>|fake|1.jpg|<img src="./FaceAlgorithm_Test/antispoofing/1.jpg" width="300" height="300"/>|fake
2.jpg|<img src="./FaceAlgorithm_Test/antispoofing/2.jpg" width="300" height="300"/>|real|3.jpg|<img src="./FaceAlgorithm_Test/antispoofing/3.jpg" width="300" height="300"/>|real
4.jpg|<img src="./FaceAlgorithm_Test/antispoofing/4.jpg" width="300" height="300"/>|fake|5.jpg|<img src="./FaceAlgorithm_Test/antispoofing/5.jpg" width="300" height="300"/>|fake

### 6.跟踪
#### 1)ByteTracker(加上人脸bbox和人脸关键点作为跟踪的输入，修改Bug)

### 7.算法接口
```
/** 
 * @brief               人脸初始化函数
 * @param config        模块配置参数结构体
 * @return              HZFLAG
 */
HZFLAG Initialize(Config& config);


/** 
 * @brief               人脸检测
 * @param img           opencv　Mat格式
 * @param FaceDets      人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return              HZFLAG
 */		
HZFLAG Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);


/** 
 * @brief               人脸检测(yolov5_face)
 * @param img           opencv　Mat格式
 * @param FaceDets      人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return              HZFLAG
 */		
HZFLAG Yolov5Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
   * @brief             人脸检测(yolov7_face)
   * @param img         opencv　Mat格式
   * @param FaceDets    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
   * @return            HZFLAG
   */		
HZFLAG Yolov7Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
 * @brief               人脸检测跟踪(视频流)
 * @param img           opencv　Mat格式
 * @param FaceDets      FaceDets	人脸检测结果列表，包括人脸bbox，id,置信度，偏航角度，俯仰角度，五个关键点坐标
 * @return              HZFLAG
 */	
HZFLAG Face_Detect_Tracker(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);


/** 
 * @brief               人脸矫正
 * @param Faceimg       需要矫正的人脸图像(矩形框bbox外扩1.2倍得到的人脸图像然后进行矫正!!!!)
 * @param KeyPoints     人脸关键点
 * @param Face_Aligener 矫正之后的图像
 * @return              HZFLAG
 */	
HZFLAG Face_Aligner(cv::Mat&Face_image,cv::Point2f *KeyPoints,cv::Mat&Face_Aligener);

/** 
 * @brief               人脸特征提取
 * @param Face_Aligener 经过人脸矫正的人脸图像
 * @param Face_Feature  人脸特征(512维特征)
 * @return              HZFLAG
 */		
HZFLAG Face_Feature_Extraction(cv::Mat&Face_Aligener,Feature&Face_Feature);


/** 
 * @brief               计算人脸特征的相似度
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2      人脸特征(512维特征)
 * @return float        相似度得分               
 */	
float Cal_Score(Feature&Feature1,Feature&Feature2);

/** 
 * @brief               人脸戴口罩识别
 * @param img           需要识别的人脸戴口罩图像
 * @param Result        人脸戴口罩识别结果
 * @return              HZFLAG
 */
HZFLAG Mask_Recognition(cv::Mat &img,float&pred);

/** 
 * @brief               性别年龄识别
 * @param img           需要识别的人脸图像
 * @param Result        性别年龄识别别结果
 * @return              HZFLAG
 */
HZFLAG Gender_Age_Recognition(cv::Mat &img,attribute&gender_age);

/** 
 * @brief               静默活体检测
 * @param img           需要检测的人脸图像
 * @param Result        静默活体检测识别结果
 * @return              HZFLAG
 */
HZFLAG Silent_Face_Anti_Spoofing(cv::Mat&img, SilentFace&silentface);

/** 
 * @brief               反初始化
 * @return              HZFLAG
 */		
HZFLAG Release(Config& config);
```
# 使用方法
## 1.模型和测试数据下载
模型 ([Baidu Drive](https://pan.baidu.com/s/1c8NQO2cZpAqwEMbfZxsJZg) code: 5xaa)

测试数据 ([Baidu Drive](https://pan.baidu.com/s/1nNHUCFHza2JzAnMZhA_9gQ) code: bphn)
| name |  功能    |  说明   |
|:----------:|:----------:|:----------:|
|FaceDetect.wts                        |人脸检测|        
|FaceRecognition.wts				   |人脸识别|       
|GenderAge.onnx                        |年龄性别识别|          
|MaskRecognition.onnx                  |口罩识别|          
|yolov5s-face_bs=1.onnx                |yolov5s人脸检测|          
|yolov5s-face_bs=4.onnx                |yolov5s人脸检测| bs=4
|yolov7s-face_bs=1.onnx                |yolov7s人脸检测|          
|yolov7s-face_bs=4.onnx                |yolov7s人脸检测| bs=4       
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

4. 使用Visual Studio Code快捷键编译(4,5二选其一):
```
   ctrl+shift+B
```
5. 使用命令行编译(4,5二选其一):
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
5. https://github.com/derronqi/yolov7-face/tree/main
6. https://github.com/we0091234/yolov7-face-tensorrt
7. https://github.com/deepinsight/insightface   
