/**
    *****************************************************************************
    *@file          :Utils.h
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/5/29
    *****************************************************************************
**/

#ifndef COLORCALIBRATOR_UTILS_H
#define COLORCALIBRATOR_UTILS_H

#include <cuda_runtime.h>
#include <QColor>

class Utils {
public:
    static void
    Utils::colorCorrect(uchar3 *data, int elementNum, double *ccm, double linearGamma = 2.2, double unLinearGamma = 2.4,
                        double a = 0.055, bool isRGB = false);
    static QColor GetMacbethColor(int index);

};


#endif //COLORCALIBRATOR_UTILS_H
