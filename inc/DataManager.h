#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <opencv2/opencv.hpp>
#include <opencv2/mcc.hpp>
class DataManager
{
public:
    DataManager();

    void Clear();

public:
    std::vector<std::string> image_fns_;
    std::vector<cv::Mat> image_mats_;
    std::vector<cv::Ptr<cv::mcc::CChecker>> checkers_;
    std::vector<cv::Mat> image_mats_with_checkers_;
    std::vector<cv::Mat> image_mats_corrected_;

    std::vector<std::pair<int,cv::Rect2f>> color_rects_; //[colorIdx:[frameIdx:rect]]
    double loss_;
    cv::Mat ccm_;
    int current_image_index_;

};

#endif // DATAMANAGER_H
