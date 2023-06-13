#include "FaceAngle.h"

/** 
 * @brief          点到直线的距离
 * @param point    点坐标
 * @param line     直线的坐标
 * @return         返回点到直线的距离
 */
float point2line(cv::Point point, cv::Point line[2])
{
	int  Ax, By, Cz;
	if (std::abs(line[0].x - line[1].x) < 3) 
	{
		return (float)(point.x - (line[0].x + line[1].x) / 2);
	}
	else if (std::abs(line[0].y - line[1].y) < 3) 
	{
		return (float)(point.y - (line[0].y + line[1].y) / 2);
	}
	Cz =  (Ax * line[0].x + By * line[0].y)*-1;
	Ax = line[1].y - line[0].y;
	By = line[0].x - line[1].x;
	float point2line_dis = (float)(Ax * point.x + By * point.y + Cz) / sqrt((float)(Ax*Ax + By*By));
	return point2line_dis;
}

/** 
 * @brief                   根据关键点计算人脸鼻子关键点和其他关键点(左眼关键点和嘴巴左关键点组成的直线、右眼关键点和嘴巴右关键点组成的直线)直线距离的比值
 * @param Face_keypoints    人脸的关键点坐标
 * @return                  返回人脸人脸鼻子关键点和其他关键点(左眼关键点和嘴巴左关键点、右眼关键点和嘴巴右关键点)直线距离的比值
 */
float Face_Horizontal_Ratio(cv::Point Face_keypoints[5])
{
	
	cv::Point right_line[2],left_line[2];
	left_line[0] = Face_keypoints[1];
	left_line[1] = Face_keypoints[4];
	right_line[0] = Face_keypoints[0];
	right_line[1] = Face_keypoints[3];
	cv::Point point = cv::Point(Face_keypoints[2].x, Face_keypoints[2].y);
	float distance_l = point2line(point, right_line);
	float distance_r =  point2line(point, left_line)*-1;
	float min_dis = std::abs(distance_l) < std::abs(distance_r) ? distance_l : distance_r;
	float max_dis = std::abs(distance_l) > std::abs(distance_r) ? distance_l : distance_r;
	if (max_dis <= 0)
	{
		return 0;
	}
	return min_dis / max_dis;
}
/** 
 * @brief                   根据关键点计算人脸鼻子关键点和其他关键点(眼睛关键点组成的直线、嘴巴关键定点组成的直线)直线距离的比值
 * @param Face_keypoints    人脸的关键点坐标
 * @return                  返回人脸鼻子关键点和其他关键点(眼睛关键点组成的直线、嘴巴关键定点组成的直线)直线距离的比值
 */
float Face_Vertical_Ratio(cv::Point Face_keypoints[5])
{
	
	cv::Point eye_line[2],mouth_line[2];
	mouth_line[0] = Face_keypoints[3];
	mouth_line[1] = Face_keypoints[4];
	eye_line[0] = Face_keypoints[0];
	eye_line[1] = Face_keypoints[1];
	cv::Point point = cv::Point(Face_keypoints[2].x, Face_keypoints[2].y);
	float eye_dis =std::abs(point2line(point, eye_line));
	float mouth_dis = std::abs(point2line(point, mouth_line));
	float min_dis = (eye_dis) < (mouth_dis) ? eye_dis : mouth_dis;
	float max_dis = (eye_dis) > (mouth_dis) ? eye_dis : mouth_dis;
	if (max_dis <=0)
	{
		return 0;
	}
	return min_dis / max_dis;
}
/** 
 * @brief                    计算人脸左右旋转的角度(偏航角)
 * @param Bbox               点坐标
 * @param Face_keypoints     直线的坐标
 * @return                   返回左右旋转的角度
 */
float Face_Yaw_Angle(cv::Point Bbox[2], cv::Point Face_keypoints[5])
{
	cv::Point nose_keypoint = cv::Point(Face_keypoints[2].x, Face_keypoints[2].y);
	//计算鼻子关键点到bbox的边界的距离
	float left_bbox_dis = std::abs(nose_keypoint.x - Bbox[0].x);
	float right_bbox_dis = std::abs(nose_keypoint.x - Bbox[1].x);
	//计算鼻子关键点到(左眼关键点和嘴巴左关键点组成的直线、右眼关键点和嘴巴右关键点组成的直线)直线中点距离
	float left_dis =(nose_keypoint.x - (Face_keypoints[0].x + Face_keypoints[3].x)/2);
	float right_dis =(nose_keypoint.x - (Face_keypoints[1].x + Face_keypoints[4].x)/2);
	float left_dis_abs = std::abs(nose_keypoint.x - (Face_keypoints[0].x + Face_keypoints[3].x)/2);
	float right_dis_abs = std::abs(nose_keypoint.x - (Face_keypoints[1].x + Face_keypoints[4].x)/2);
	
	float min_dis = left_dis_abs < right_dis_abs ? left_dis_abs : right_dis_abs;
	float max_dis = left_dis_abs > right_dis_abs ? left_dis_abs : right_dis_abs;
	if (left_dis*right_dis >= 0) 
	{
		min_dis =-1*min_dis;
	}
	float min_bbox_dis = left_bbox_dis < right_bbox_dis ? left_bbox_dis : right_bbox_dis;
	float max_bbox_dis = left_bbox_dis > right_bbox_dis ? left_bbox_dis : right_bbox_dis;
	
    if (max_dis == 0.0f || max_dis == 0.0f) 
	{
		return 0;
	}
	float face_yaw_angle= 0.49*(min_dis / max_dis) + 0.51*(min_bbox_dis / max_bbox_dis);
	face_yaw_angle = face_yaw_angle < 0 ? 0 : face_yaw_angle;
	face_yaw_angle = face_yaw_angle < 1 ? face_yaw_angle : 1;
	return face_yaw_angle;
}
/** 
 * @brief                    计算人脸抬头低头角度(俯仰角)
 * @param Bbox               点坐标
 * @param Face_keypoints     直线的坐标
 * @return                   返回人脸抬头低头角度
 */
float Face_Pitch_Angle(cv::Point Bbox[2], cv::Point Face_keypoints[5])
{
	cv::Point nose_keypoint = cv::Point(Face_keypoints[2].x, Face_keypoints[2].y);
	float top_bbox_dis = std::abs(nose_keypoint.y - Bbox[0].y);
	float bottom_bbox_dis = std::abs(nose_keypoint.y - Bbox[1].y);
	float min_bbox_dis = top_bbox_dis < bottom_bbox_dis ? top_bbox_dis : bottom_bbox_dis;
	float max_bbox_dis = top_bbox_dis > bottom_bbox_dis ? top_bbox_dis : bottom_bbox_dis;
	
	float top_dis = (nose_keypoint.y - (Face_keypoints[0].y + Face_keypoints[1].y)/2);
	float bottom_dis = (nose_keypoint.y - (Face_keypoints[3].y + Face_keypoints[4].y)/2);
	float top_dis_abs = std::abs(nose_keypoint.y - (Face_keypoints[0].y + Face_keypoints[1].y)/2);
	float bottom_dis_abs = std::abs(nose_keypoint.y - (Face_keypoints[3].y + Face_keypoints[4].y)/2);
	float min_dis = top_dis_abs < bottom_dis_abs ? top_dis_abs : bottom_dis_abs;
	float max_dis = top_dis_abs > bottom_dis_abs ? top_dis_abs : bottom_dis_abs;
	if (top_dis*bottom_dis >= 0) 
	{
		min_dis *=-1;
	}

	if (max_dis == 0.0f || max_bbox_dis == 0.0f) 
	{
		return 0;
	}
	float face_pitch_angle = 0.69*(min_dis / max_dis) + 0.31*(min_bbox_dis / max_bbox_dis);

	if (top_dis_abs < bottom_dis_abs)	
	{
		face_pitch_angle =face_pitch_angle* face_pitch_angle;
	}
	face_pitch_angle = face_pitch_angle < 0 ? 0 : face_pitch_angle;
	face_pitch_angle = face_pitch_angle > 1 ? 1 : face_pitch_angle;
	return face_pitch_angle;
}

/** 
 * @brief                    cal face angle
 * @param Face_Bbox          face bbox
 * @param Face_KeyPoints     Face_KeyPoints
 * @param Face_Params        Face_Params
 * @return                   void
 */
void Face_Angle_Cal(cv::Rect Face_Bbox, cv::Point Face_KeyPoints[5], float *Face_Params)
{
	cv::Point Bbox[2];
	Bbox[0].x = Face_Bbox.x;Bbox[0].y = Face_Bbox.y;
	Bbox[1].x = Face_Bbox.x+Face_Bbox.width;Bbox[1].y = Face_Bbox.y+Face_Bbox.height;
	
	float Yaw_Angle = Face_Yaw_Angle(Bbox, Face_KeyPoints);
	float Pitch_Angle = Face_Pitch_Angle(Bbox, Face_KeyPoints);
	Pitch_Angle = Pitch_Angle * (1/0.696) +(1- (1/0.696)); 
	Pitch_Angle = Pitch_Angle< 0?0:Pitch_Angle;
	float angle_Yaw=(1.498*(90.0 - asin(Yaw_Angle)*56.9896)- 45.0)-10;
	float angle_Pitch=(1.798*(90.0 - asin(Pitch_Angle)*56.9897)-72.0)-20;
	Face_Params[0] = angle_Yaw<0?0:angle_Yaw;                          
	Face_Params[1] = angle_Pitch<0?0:angle_Pitch;                       
	Face_Params[2] = sqrt((float)((Face_KeyPoints[0].x - Face_KeyPoints[1].x)
		*(Face_KeyPoints[0].x - Face_KeyPoints[1].x) + (Face_KeyPoints[0].y - Face_KeyPoints[1].y)*
		(Face_KeyPoints[0].y - Face_KeyPoints[1].y)));                     
	return;
}