#include "FaceAngle.h"
// 计算翻滚角
float getRoll(int leftEyeX, int leftEyeY, int rightEyeX, int rightEyeY)
{
	float dx = rightEyeX - leftEyeX;
	float dy = rightEyeY - leftEyeY;
	
	if (fabs(dx) < 0.0000001f)
		return 0.f;
	else
		return atanf(dx / dy)*180.0f / 3.1415926;	
}
 
// 计算水平角
float getYaw(int noseX, int faceX, int faceWidth)
{
	float dx = noseX - faceX;
	float rate = dx / (faceWidth * 0.5f) - 1;
	return asinf(rate)*180/3.1415926; 
}
 
// 计算俯仰角
float getPitch(int noseY, int faceY, int faceHigh)
{
	float dy = noseY - faceY;
	float rate = dy / (faceHigh*0.6f) - 1;
	return asinf(rate)*180/3.1415926;
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
	Face_Params[0] = getYaw(Face_KeyPoints[2].x,Face_Bbox.x, Face_Bbox.width);                        
	Face_Params[1] = getPitch(Face_KeyPoints[2].y,Face_Bbox.y, Face_Bbox.height);                        
	Face_Params[2] = sqrt((float)((Face_KeyPoints[0].x - Face_KeyPoints[1].x)
		*(Face_KeyPoints[0].x - Face_KeyPoints[1].x) + (Face_KeyPoints[0].y - Face_KeyPoints[1].y)*
		(Face_KeyPoints[0].y - Face_KeyPoints[1].y)));

	// float Roll_Angle1=getRoll(Face_KeyPoints[0].x, Face_KeyPoints[0].y,Face_KeyPoints[1].x,Face_KeyPoints[1].y);
	// float Yaw_Angle1=getYaw(Face_KeyPoints[2].x,Face_Bbox.x, Face_Bbox.width);
	// float Pitch_Angle1=getPitch(Face_KeyPoints[2].y,Face_Bbox.y, Face_Bbox.height);        
	return;
}