/**
    *****************************************************************************
    *@file          :MacbethColorChecker.cpp
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/5/17
    *****************************************************************************
**/
#define NOMINMAX

#include "MacbethColorChecker.h"
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace gala {
    MacbethColorChecker::MacbethColorChecker() {}

    MacbethColorChecker::MacbethColorChecker(Parameters parameters) {
        parameters_ = parameters;
    }

    MacbethColorChecker::~MacbethColorChecker() {}

    std::vector<cv::Ptr<cv::mcc::CChecker>> MacbethColorChecker::Detect(cv::Mat image, int nChart) {
        cv::Ptr<cv::mcc::CCheckerDetector> detector = cv::mcc::CCheckerDetector::create();
        if (!detector->process(image, cv::mcc::TYPECHART(0), nChart)) {
            return std::vector<cv::Ptr<cv::mcc::CChecker>>();
        } else {
            return detector->getListColorChecker();
        }
    }

    cv::ccm::ColorCorrectionModel &MacbethColorChecker::CheckColor(cv::Ptr<cv::mcc::CChecker> pChecker) {
        cv::Mat chartsRGB = pChecker->getChartsRGB();
        cv::Mat src = chartsRGB.col(1).clone().reshape(3, chartsRGB.rows / 3);
        src /= 255.0;
        cv::ccm::ColorCorrectionModel model1(src, cv::ccm::COLORCHECKER_Macbeth);
        model1.run();
        ccm_ = model1.getCCM();
        loss_ = model1.getLoss();
        return model1;
    }

    bool MacbethColorChecker::CheckColor(std::vector<std::pair<MacbethColor, cv::Mat>> colorBlocks) {
        CV_Assert(colorBlocks.size() >= parameters_.minColorBlockNum);
        cv::Mat macbethBlock = cv::Mat::zeros(MacbethColor::ColorNum, 1, CV_64FC3);
        for (int i = 0; i < colorBlocks.size(); ++i) {
            std::pair<MacbethColor, cv::Mat> &colorBlock = colorBlocks[i];
            cv::Mat rbgBlock;
            if (colorBlock.second.channels() == 3) {
                cv::cvtColor(colorBlock.second, rbgBlock, cv::COLOR_BGR2RGB);
            } else if (colorBlock.second.channels() == 4) {
                cv::cvtColor(colorBlock.second, rbgBlock, cv::COLOR_BGRA2BGR);
            }
            MacbethColor colorType = colorBlock.first;
            cv::Mat chartRgb;
            calculate_cell_rgb(rbgBlock, chartRgb);
            cv::Mat color = chartRgb.col(1).clone().reshape(3, chartRgb.rows / 3);
            cv::Point3d colorPt = color.at<cv::Point3d>(0);
            macbethBlock.at<cv::Point3d>(colorType) = colorPt;
        }

        macbethBlock /= 255.0;
        cv::ccm::ColorCorrectionModel model(macbethBlock, cv::ccm::COLORCHECKER_Macbeth);
        model.run();
        ccm_ = model.getCCM();
        loss_ = model.getLoss();
        return true;
    }

    cv::Mat MacbethColorChecker::Infer(cv::Mat image){
        colorCorrect((uchar3*)image.data, image.cols*image.rows, (double*)ccm_.data, parameters_.gamma, parameters_.unGamma, parameters_.alp,
                     false);
        return image;
    }

    void MacbethColorChecker::calculate_cell_rgb(cv::InputArray rgb, cv::OutputArray chartRgb) {
        std::vector<cv::Mat> rgb_planes;
        cv::split(rgb, rgb_planes);
        cv::Scalar mu_rgb, st_rgb, p_size;
        p_size = rgb.rows() * rgb.cols();
        double max_rgb[3], min_rgb[3];
        cv::meanStdDev(rgb, mu_rgb, st_rgb);
        cv::minMaxLoc(rgb_planes[0], &min_rgb[0], &max_rgb[0], NULL, NULL);
        cv::minMaxLoc(rgb_planes[1], &min_rgb[1], &max_rgb[1], NULL, NULL);
        cv::minMaxLoc(rgb_planes[2], &min_rgb[2], &max_rgb[2], NULL, NULL);

        cv::Mat _charts_rgb = cv::Mat(cv::Size(5, 3 * (int) 1), CV_64F);

        _charts_rgb.at<double>(0, 0) = p_size(0);
        _charts_rgb.at<double>(0, 1) = mu_rgb(0);
        _charts_rgb.at<double>(0, 2) = st_rgb(0);
        _charts_rgb.at<double>(0, 3) = min_rgb[0];
        _charts_rgb.at<double>(0, 4) = max_rgb[0];
        // raw_g
        _charts_rgb.at<double>(1, 0) = p_size(0);
        _charts_rgb.at<double>(1, 1) = mu_rgb(1);
        _charts_rgb.at<double>(1, 2) = st_rgb(1);
        _charts_rgb.at<double>(1, 3) = min_rgb[1];
        _charts_rgb.at<double>(1, 4) = max_rgb[1];
        // raw_b
        _charts_rgb.at<double>(2, 0) = p_size(0);
        _charts_rgb.at<double>(2, 1) = mu_rgb(2);
        _charts_rgb.at<double>(2, 2) = st_rgb(2);
        _charts_rgb.at<double>(2, 3) = min_rgb[2];
        _charts_rgb.at<double>(2, 4) = max_rgb[2];
        chartRgb.assign(_charts_rgb);
    }

    void MacbethColorChecker::colorCorrect(uchar3 *data, int elementNum, double *ccm, double linearGamma, double unLinearGamma ,
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

}