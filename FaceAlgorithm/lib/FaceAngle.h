#ifndef __FACEANGLE_H__
#define __FACEANGLE_H__

#include <opencv2/opencv.hpp>
#include <vector> 
#include <algorithm>
#include <iostream>

#ifdef __cplusplus 
extern "C" { 
#endif 
/** 
 * @brief                    cal face angle
 * @param Face_Bbox          face bbox
 * @param Face_KeyPoints     Face_KeyPoints
 * @param Face_Params        Face_Params
 * @return                   void
 */
void Face_Angle_Cal(cv::Rect Face_Bbox, cv::Point Face_KeyPoints[5], float *Face_Params);

#ifdef __cplusplus 
} 
#endif

#endif