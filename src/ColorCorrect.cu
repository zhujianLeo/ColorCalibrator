/**
    *****************************************************************************
    *@file          :ColorCorrect.cpp
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/2/6
    *****************************************************************************
**/

#include "ColorCorrect.h"


inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
__global__ void ColorCorrectKernel(uchar3 *data, int width, int height, double *ccm,double linearGamma,double unLinearGamma,double a,bool isRGB) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    double3 srcColor;
    if (isRGB) {
        srcColor = make_double3(data[idx].x / 255.0, data[idx].y / 255.0, data[idx].z / 255.0);
    } else {
        srcColor = make_double3(data[idx].z / 255.0, data[idx].y / 255.0, data[idx].x / 255.0);
    }

    //gamma correction line
    srcColor.x = srcColor.x >= 0 ? pow(srcColor.x, linearGamma) : -pow(-srcColor.x, linearGamma);
    srcColor.y = srcColor.y >= 0 ? pow(srcColor.y, linearGamma) : -pow(-srcColor.y, linearGamma);
    srcColor.z = srcColor.z >= 0 ? pow(srcColor.z, linearGamma) : -pow(-srcColor.z, linearGamma);


    double3 dstColor;
    dstColor.x = ccm[0] * srcColor.x + ccm[3] * srcColor.y + ccm[6] * srcColor.z;
    dstColor.y = ccm[1] * srcColor.x + ccm[4] * srcColor.y + ccm[7] * srcColor.z;
    dstColor.z = ccm[2] * srcColor.x + ccm[5] * srcColor.y + ccm[8] * srcColor.z;


    double alpha = a+1;
    double k0=a/(unLinearGamma-1);
    double phi = (pow(alpha,unLinearGamma)*pow(unLinearGamma-1,unLinearGamma-1))/(pow(a,unLinearGamma-1)*pow(unLinearGamma,unLinearGamma));
    double beta =k0/phi;
    dstColor.x = (dstColor.x>beta?(alpha*pow(dstColor.x,1/unLinearGamma)-(alpha-1)):dstColor.x*phi)*255.0;
    dstColor.y = (dstColor.y>beta?(alpha*pow(dstColor.y,1/unLinearGamma)-(alpha-1)):dstColor.y*phi)*255.0;
    dstColor.z = (dstColor.z>beta?(alpha*pow(dstColor.z,1/unLinearGamma)-(alpha-1)):dstColor.z*phi)*255.0;

    data[idx] = make_uchar3(min(max(dstColor.z, 0.0), 255.0), min(max(dstColor.y, 0.0), 255.0),
                            min(max(dstColor.x, 0.0), 255.0));
}

cudaError_t cudaColorCorrection(uchar3 *data, int width, int height, double *ccm,double linearGamma,double unLinearGamma,double a,bool isRGB, cudaStream_t stream) {
    const dim3 blockDim(32, 8, 1);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);
    ColorCorrectKernel<<<gridDim, blockDim, 0, stream>>>(data, width, height, ccm, linearGamma,unLinearGamma,a,isRGB);
    return cudaGetLastError();
}