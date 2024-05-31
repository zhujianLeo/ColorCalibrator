/**
    *****************************************************************************
    *@file          :MacbethColorChecker.h
    *@author        :Leo
    *@brief         :None
    *@attention     :None
    *@email         :leo.zhu@galasports.com
    *@date          :2024/5/17
    *****************************************************************************
**/

#ifndef TOOLSPROJ_MACBETHCOLORCHECKER_H
#define TOOLSPROJ_MACBETHCOLORCHECKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/mcc.hpp>
#include <cuda_runtime.h>
namespace gala {
    enum MacbethColor {
        DarkSkin,
        LightSkin,
        BlueSky,
        Foliage,
        BlueFlower,
        BluishGreen,
        Orange,
        PurplishBlue,
        ModerateRed,
        Purple,
        YellowGreen,
        OrangeYellow,
        Blue,
        Green,
        Red,
        Yellow,
        Magenta,
        Cyan,
        White,
        Neutral8,
        Neutral6Dot5,
        Neutral5,
        Neutral3Dot5,
        Black,
        ColorNum
    };

    struct Parameters {
        float gamma = 2.2;
        float unGamma = 2.4;
        float alp = 0.055;
        int minColorBlockNum = 10;
    };

    /**
     * @brief MacbethColorChecker
     */
    class MacbethColorChecker {
    public:
        MacbethColorChecker();

        MacbethColorChecker(Parameters parameters);

        ~MacbethColorChecker();

        ///
        /// \param image     bgr image
        /// \param nChart    color chart num default 1
        /// \return
        std::vector<cv::Ptr<cv::mcc::CChecker>> Detect(cv::Mat image, int nChart = 1);

        ///
        /// \param pChecker detected checker
        cv::ccm::ColorCorrectionModel& CheckColor(cv::Ptr<cv::mcc::CChecker> pChecker);

        ///
        /// \param colorBlocks bgr color chart blocks
        /// \return
        bool CheckColor(std::vector<std::pair<MacbethColor, cv::Mat>> colorBlocks);

        cv::Mat Infer(cv::Mat image);

        double GetLoss() { return loss_; }

        cv::Mat GetCCM() { return ccm_; }

        cv::Vec3d GetGammaParam() { return cv::Vec3d(parameters_.gamma, parameters_.unGamma, parameters_.alp); }

    private:

        void calculate_cell_rgb(cv::InputArray rgb,cv::OutputArray chartRgb);
        void colorCorrect(uchar3 *data, int elementNum, double *ccm, double linearGamma = 2.2, double unLinearGamma = 2.4,
                          double a = 0.055, bool isRGB = false);
    private:
        double loss_;
        cv::Mat ccm_;
        Parameters parameters_;
    };

}


#endif //TOOLSPROJ_MACBETHCOLORCHECKER_H
