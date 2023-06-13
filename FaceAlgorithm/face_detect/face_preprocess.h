#ifndef __MCV_PREPROCESS_H
#define __MCV_PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct McvAffineMatrix
{
    float value[6];
};


void mcv_preprocess_kernel_img(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,
                            cudaStream_t stream);
#endif  // __PREPROCESS_H
