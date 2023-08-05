#include "yolov8face_postprocess.h"
#define MAX_LANDMARK 20
static __device__ void yolov8face_affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void yolov8face_decode_kernel(float* predict,int NUM_BOX_ELEMENT, int num_bboxes, int num_classes,int ckpt, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects)
{  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) //8400
    {
        return;
    }
    float*pitem= predict+(5+ckpt*3) * position;              //每个线程处理一个人脸的20个参数
    float confidence = pitem[4];                             //置信度
    if(confidence < confidence_threshold)                    //小于该置信度的舍弃
    {
        return;
    }
    //判断是否超过了最大人脸框的阈值,bbox个数存放在parray[0]
    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
    {
        return;
    }
    //center_x, center_y, width, height
    float cx         = pitem[0];
    float cy         = pitem[1];
    float width      = pitem[2];
    float height     = pitem[3];
    
    //五个关键点
    float *landmarks = pitem+5;
    float landmark_array[MAX_LANDMARK*2];
    for (int i = 0; i<ckpt; i++)
    {
        landmark_array[2*i]=landmarks[3*i];
        landmark_array[2*i+1]=landmarks[3*i+1];
    }

    //(center_x, center_y, width, height) to (x1, y1, x2, y2)
    float left   = cx - width * 0.5f;        
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    //bbox还原到原图的位置
    yolov8face_affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    yolov8face_affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    //landmark还原到原图的位置
    for(int i = 0; i<ckpt; i++)
    {
        yolov8face_affine_project(invert_affine_matrix, landmark_array[2*i],landmark_array[2*i+1],&landmark_array[2*i],&landmark_array[2*i+1]); 
    }
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;            //0
    *pout_item++ = top;             //1
    *pout_item++ = right;           //2
    *pout_item++= bottom;           //3
    *pout_item++ = confidence;      //4
    *pout_item++ = 0;               //5 label
    *pout_item++ = 1;               //6 1 = keep, 0 = ignore

    for(int i = 0; i<ckpt; i++)
    {
        *pout_item++=landmark_array[2*i];
        *pout_item++=landmark_array[2*i+1];
    }
}

//1*20*8400->1*8400*20
static __global__ void yolov8_transpose_kernel(float *src,int num_bboxes, int num_elements,float *dst,int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position>=edge)
        return;
    dst[position]=src[position/num_elements+(position%num_elements)*num_bboxes];
}

static __device__ float yolov8face_box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom)
{

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
    {
        return 0.0f;
    }
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void yolov8face_nms_kernel(float* bboxes, int max_objects, float threshold,int NUM_BOX_ELEMENT)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    //get bbox num
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
    {
        return;
    }
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i)
    {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        //是同一个或者不是同一类，则跳过
        if(i == position || pcurrent[5] != pitem[5]) 
        {
            continue;
        }
        //置信度大于本次并且iou大于阈值，则pcurrent[6] = 0
        if(pitem[4] >= pcurrent[4])
        {
            if(pitem[4] == pcurrent[4] && i < position)
            {
                continue;
            }
            //iou
            float iou = yolov8face_box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],pitem[0],pitem[1],pitem[2],pitem[3]);
            if(iou > threshold)
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void yolov8face_decode_kernel_invoker(float* predict, int  NUM_BOX_ELEMENT,int num_bboxes,int num_classes,int ckpt, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream)
{
    int block = 256;
    int  grid =  ceil((num_bboxes+block-1) / (float)block);
    
    yolov8face_decode_kernel<<<grid, block, 0, stream>>>(predict,NUM_BOX_ELEMENT, num_bboxes, num_classes,ckpt, confidence_threshold, invert_affine_matrix, parray, max_objects);
}

void yolov8face_nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream,int NUM_BOX_ELEMENT)
{
    int block = max_objects<256? max_objects:256;
    int grid = ceil((max_objects+block-1) / (float)block);
    yolov8face_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold,NUM_BOX_ELEMENT);
}
void yolov8_transpose(float *src,int num_bboxes,int num_elements,float *dst,cudaStream_t stream)
{
    int edge = num_bboxes*num_elements;
    int block =256;
    int gird = ceil(edge/(float)block);
    yolov8_transpose_kernel<<<gird,block,0,stream>>>(src,num_bboxes,num_elements,dst,edge);
}