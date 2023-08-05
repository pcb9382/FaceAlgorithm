#ifndef __YOLOV8_POSTPROCESS_H
#define __YOLOV8_POSTPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

// struct AffineMatrix{
//     float value[6];
// };

void yolov8face_decode_kernel_invoker
(
    float* predict,int NUM_BOX_ELEMENT, int num_bboxes, int num_classes, int ckpt,float confidence_threshold, 
    float* invert_affine_matrix, float* parray,
    int max_objects, cudaStream_t stream
);

void yolov8face_nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream,int NUM_BOX_ELEMENT);

void yolov8_transpose(float *src,int num_bboxes,int num_elements,float *dst,cudaStream_t stream);
#endif  // __POSTPROCESS_H