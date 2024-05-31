/**
    *****************************************************************************
    *@file          :Utils.cpp
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/5/29
    *****************************************************************************
**/

#include "Utils.h"
#include <math.h>
#include <opencv2/core.hpp>
#include "MacbethColor.h"
void Utils::colorCorrect(uchar3 *data, int elementNum, double *ccm, double linearGamma, double unLinearGamma ,
                                       double a, bool isRGB){
    double alpha = a + 1;
    double k0 = a / (unLinearGamma - 1);
    double phi = (pow(alpha, unLinearGamma) * pow(unLinearGamma - 1, unLinearGamma - 1)) /
                 (pow(a, unLinearGamma - 1) * pow(unLinearGamma, unLinearGamma));
    double beta = k0 / phi;
    for (int i = 0; i < elementNum; ++i) {
        double3 srcColor;
        if (!isRGB) {
            srcColor.x = data[i].z / 255.0;
            srcColor.y = data[i].y / 255.0;
            srcColor.z = data[i].x / 255.0;
        } else {
            srcColor.x = data[i].x / 255.0;
            srcColor.y = data[i].y / 255.0;
            srcColor.z = data[i].z / 255.0;
        }
        srcColor.x = srcColor.x >= 0 ? pow(srcColor.x, linearGamma) : -pow(-srcColor.x, linearGamma);
        srcColor.y = srcColor.y >= 0 ? pow(srcColor.y, linearGamma) : -pow(-srcColor.y, linearGamma);
        srcColor.z = srcColor.z >= 0 ? pow(srcColor.z, linearGamma) : -pow(-srcColor.z, linearGamma);


        double3 dstColor;
        dstColor.x = ccm[0] * srcColor.x + ccm[3] * srcColor.y + ccm[6] * srcColor.z;
        dstColor.y = ccm[1] * srcColor.x + ccm[4] * srcColor.y + ccm[7] * srcColor.z;
        dstColor.z = ccm[2] * srcColor.x + ccm[5] * srcColor.y + ccm[8] * srcColor.z;

        dstColor.x =
                (dstColor.x > beta ? (alpha * pow(dstColor.x, 1 / unLinearGamma) - (alpha - 1)) : dstColor.x * phi) *
                255.0;
        dstColor.y =
                (dstColor.y > beta ? (alpha * pow(dstColor.y, 1 / unLinearGamma) - (alpha - 1)) : dstColor.y * phi) *
                255.0;
        dstColor.z =
                (dstColor.z > beta ? (alpha * pow(dstColor.z, 1 / unLinearGamma) - (alpha - 1)) : dstColor.z * phi) *
                255.0;

        data[i] = make_uchar3(std::min(std::max(dstColor.z, 0.0), 255.0), std::min(std::max(dstColor.y, 0.0), 255.0),
                              std::min(std::max(dstColor.x, 0.0), 255.0));

    }
}

QColor Utils::GetMacbethColor(int index){
    return MacbethColors[index];
}