#include "DataManager.h"

DataManager::DataManager()
{
    Clear();
}

void DataManager::Clear(){
    image_fns_.clear();
    image_mats_.clear();
    checkers_.clear();
    color_rects_.clear();
    image_mats_with_checkers_.clear();
    image_mats_corrected_.clear();
    loss_ = 0;
    ccm_ = cv::Mat();
    current_image_index_ = 0;
}
