/**
    *****************************************************************************
    *@file          :ColorCorrect.h
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/2/6
    *****************************************************************************
**/

#ifndef TOOLSPROJ_COLORCORRECT_H
#define TOOLSPROJ_COLORCORRECT_H

#include <cuda_runtime.h>

cudaError_t cudaColorCorrection(uchar3 *data, int width, int height, double *ccm, double linearGamma = 2.2,
                                double unLinearGamma = 2.4, double a = 0.055, bool isRGB = false,
                                cudaStream_t stream = nullptr);



#endif //TOOLSPROJ_COLORCORRECT_H
