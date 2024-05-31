#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include "ColorSpace.h"
#include "ColorCorrect.h"
#include <qdebug.h>
#include <QPainter>
#include <qmath.h>
#include <qdebug.h>
#include <QFileDialog>
#include <QPalette>
#include <fstream>
#include "Utils.h"
MainWindow::MainWindow(QWidget *parent)
        : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    ccm_data_ = nullptr;
    frame_data_ = nullptr;
    frame_corrected_data_ = nullptr;
    jop_status_ = 0;
    dataManager_ = new DataManager();
    colorChecker_ = new gala::MacbethColorChecker();
    ui->widget_4->SetHandleMode(true);

    ui->splitter_2->setStretchFactor(0, 1);
    ui->splitter_2->setStretchFactor(1, 3);
    ui->splitter->setStretchFactor(0, 10);
    colorGroup_ = new QButtonGroup();
    colorGroup_->addButton(ui->darkSkin, 0);
    colorGroup_->addButton(ui->lightSkin, 1);
    colorGroup_->addButton(ui->blueSky, 2);
    colorGroup_->addButton(ui->foliage, 3);
    colorGroup_->addButton(ui->blueFlower, 4);
    colorGroup_->addButton(ui->bluishGreen, 5);
    colorGroup_->addButton(ui->orange, 6);
    colorGroup_->addButton(ui->purplishBlue, 7);
    colorGroup_->addButton(ui->moderateRed, 8);
    colorGroup_->addButton(ui->purple, 9);
    colorGroup_->addButton(ui->yellowGreen, 10);
    colorGroup_->addButton(ui->orangeYellow, 11);
    colorGroup_->addButton(ui->blue, 12);
    colorGroup_->addButton(ui->green, 13);
    colorGroup_->addButton(ui->red, 14);
    colorGroup_->addButton(ui->yellow, 15);
    colorGroup_->addButton(ui->magenta, 16);
    colorGroup_->addButton(ui->cyan, 17);
    colorGroup_->addButton(ui->white, 18);
    colorGroup_->addButton(ui->neutral8, 19);
    colorGroup_->addButton(ui->neutral65, 20);
    colorGroup_->addButton(ui->neutral5, 21);
    colorGroup_->addButton(ui->neutral35, 22);
    colorGroup_->addButton(ui->black, 23);
    ui->darkSkin->setChecked(true);
    currentColorIdx_ = gala::MacbethColor::DarkSkin;
    connect(colorGroup_, SIGNAL(buttonClicked(int)), this, SLOT(color_index_changed(int)));

    previewGroup_ = new QButtonGroup();
    previewGroup_->addButton(ui->radioButton_4, 0);
    previewGroup_->addButton(ui->radioButton_5, 1);
    previewGroup_->addButton(ui->radioButton_6, 2);
    connect(previewGroup_, SIGNAL(buttonClicked(int)), this, SLOT(preview_index_changed(int)));

    manualPreviewGroup_ = new QButtonGroup();
    manualPreviewGroup_->addButton(ui->radioButton, 3);
    manualPreviewGroup_->addButton(ui->radioButton_2, 4);
    manualPreviewGroup_->addButton(ui->radioButton_3, 5);
    connect(manualPreviewGroup_, SIGNAL(buttonClicked(int)), this, SLOT(preview_index_changed(int)));


    ui->radioButton_4->setChecked(true);


    ui->label->setAutoFillBackground(true);
    ui->label_2->setAutoFillBackground(true);
    ui->label_3->setAutoFillBackground(true);

    QPalette p;
    p.setColor(QPalette::Window, QColor(103, 189, 170, 0));
    ui->label->setPalette(p);

    ui->label_2->setPalette(p);

    ui->label_3->setPalette(p);


    ui->widget_6->setHidden(true);

    connect(ui->actionOpen, SIGNAL(triggered()), this, SLOT(open_slot()));

    connect(ui->actionNew, SIGNAL(triggered()), this, SLOT(new_slot()));

    connect(ui->actionExist, SIGNAL(triggered()), this, SLOT(exit_slot()));

    connect(ui->pushButton_2, SIGNAL(clicked()), this, SLOT(previous_slot()));
    connect(ui->pushButton_3, SIGNAL(clicked()), this, SLOT(next_slot()));
    connect(ui->pushButton_7, SIGNAL(clicked()), this, SLOT(detect_slot()));
    connect(ui->pushButton_8, SIGNAL(clicked()), this, SLOT(calibrate_slot()));
    connect(ui->pushButton_10, SIGNAL(clicked()), this, SLOT(export_slot()));
    connect(ui->pushButton, SIGNAL(clicked()), this, SLOT(color_area_apply_slot()));
    connect(ui->pushButton_4, SIGNAL(clicked()), this, SLOT(manual_calibrate_slot()));
    connect(ui->pushButton_6, SIGNAL(clicked()), this, SLOT(export_slot()));

    connect(ui->tabWidget_2, SIGNAL(currentChanged(int)), this, SLOT(tabWidget_index_changed(int)));


}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::jobStatusChanged(int status) {
    jop_status_ = status;
    if (ui->tabWidget_2->currentIndex() <= 0) {
        switch (jop_status_) {
            case 0: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(0);
                ui->label->setPalette(QPalette(color));
                ui->label_2->setPalette(QPalette(color));
                ui->label_3->setPalette(QPalette(color));
                ui->radioButton_4->setChecked(true);
                break;
            }
            case 1: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(255);
                ui->label->setPalette(QPalette(color));
                color.setAlpha(0);
                ui->label_2->setPalette(QPalette(color));
                ui->label_3->setPalette(QPalette(color));
                ui->radioButton_5->setChecked(true);
                break;
            }
            case 2: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(255);
                ui->label->setPalette(QPalette(color));
                ui->label_2->setPalette(QPalette(color));
                color.setAlpha(0);
                ui->label_3->setPalette(QPalette(color));
                ui->radioButton_6->setChecked(true);
                break;
            }

        }
    }else{
        switch (jop_status_) {
            case 0: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(0);
                ui->label_5->setPalette(QPalette(color));
                ui->label_6->setPalette(QPalette(color));
                ui->label_7->setPalette(QPalette(color));
                ui->radioButton->setChecked(true);
                break;
            }
            case 1: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(255);
                ui->label_5->setPalette(QPalette(color));
                color.setAlpha(0);
                ui->label_6->setPalette(QPalette(color));
                ui->label_7->setPalette(QPalette(color));
                ui->radioButton_2->setChecked(true);
                break;
            }
            case 2: {
                QColor color(ui->label->palette().color(QPalette::Window));
                color.setAlpha(255);
                ui->label_5->setPalette(QPalette(color));
                ui->label_6->setPalette(QPalette(color));
                color.setAlpha(0);
                ui->label_7->setPalette(QPalette(color));
                ui->radioButton_3->setChecked(true);
                break;
            }

        }
    }


}


void MainWindow::open_slot() {
    qDebug() << "open_slot:";
    if (ui->tabWidget_2->currentIndex() <= 0) {
        QString fn = QFileDialog::getOpenFileName(this, QObject::tr("Open images"), "./",
                                                  QObject::tr("*.jpg *.png *.JPG *.PNG"));
        if (fn.isEmpty()) {
            qDebug() << "No file selected";
            ui->textBrowser->append("No file selected");
        } else {
            dataManager_->Clear();
            std::vector<std::string> stdFiles;
            stdFiles.push_back(fn.toStdString());
            cv::Mat frame = cv::imread(fn.toStdString());
            if (!frame.empty()) {
                dataManager_->image_fns_ = stdFiles;
                dataManager_->image_mats_.push_back(frame);
                ui->widget_4->Update(frame);
            }
            jobStatusChanged(0);
            ui->textBrowser->append("open fn:" +fn);
        }
    } else {
        QStringList files = QFileDialog::getOpenFileNames(this, QObject::tr("Open images"), "./",
                                                          QObject::tr("*.jpg *.png *.JPG *.PNG"));
        if (files.empty()) {
            qDebug() << "No file selected";
            ui->textBrowser_2->append("No file selected");
        } else {
            dataManager_->Clear();
            for (int i = 0; i < files.size(); i++) {
                std::string fn = files[i].toStdString();
                dataManager_->image_fns_.push_back(fn);
                cv::Mat frame = cv::imread(fn);
                dataManager_->image_mats_.push_back(frame);
                ui->textBrowser_2->append("open fn:" + files[i]);
            }
            dataManager_->current_image_index_ = 0;
            ui->widget_4->Update(dataManager_->image_mats_[0]);
            jobStatusChanged(0);
        }

    }


}

void MainWindow::new_slot() {
    dataManager_->Clear();
    ui->widget_4->Reset();
}

void MainWindow::exit_slot() {

}

void MainWindow::detect_slot() {
    if (dataManager_->image_mats_.empty()) {
        qDebug() << "No image found";
        ui->textBrowser->append("No image found");
        return;
    }

    cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_];
    std::vector<cv::Ptr<cv::mcc::CChecker>> checkers = colorChecker_->Detect(frame);
    if (!checkers.empty()) {
        cv::Mat checkedFrame = frame.clone();
        cv::Ptr<cv::mcc::CCheckerDraw> cdraw = cv::mcc::CCheckerDraw::create(checkers[0]);
        cdraw->draw(checkedFrame);
        dataManager_->image_mats_with_checkers_.push_back(checkedFrame);
        dataManager_->checkers_.push_back(checkers[0]);
        jobStatusChanged(1);
        ui->widget_4->Update(checkedFrame);
        ui->textBrowser->append("found "+QString::number(checkers.size())+"color checker");
    }else{
        ui->textBrowser->append("No color checker found");
    }
    jobStatusChanged(1);

}


void MainWindow::calibrate_slot() {
    if (dataManager_->checkers_.empty()) {
        qDebug() << "No checker found";
        ui->textBrowser->append("No checker found");
        return;
    }
    cv::ccm::ColorCorrectionModel &model = colorChecker_->CheckColor(dataManager_->checkers_[0]);
    dataManager_->ccm_ = colorChecker_->GetCCM();
    dataManager_->loss_ = colorChecker_->GetLoss();

    cv::Mat inferFrame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
    colorChecker_->Infer(inferFrame);

    dataManager_->image_mats_corrected_.push_back(inferFrame);
    ui->widget_4->Update(inferFrame);
    jobStatusChanged(2);
    std::cout<<"ccm:"<<dataManager_->ccm_<<std::endl;
    std::cout<<"loss:"<<dataManager_->loss_<<std::endl;
    std::ostringstream ss;
    ss<<"ccm:"<<dataManager_->ccm_<<"\nloss:"<<dataManager_->loss_;
    ui->textBrowser->append(QString::fromStdString(ss.str()));

//    cv::imwrite("./infer.jpg", inferFrame);
}

void MainWindow::export_slot() {
    if (dataManager_->ccm_.empty()) {
        return;
    }
    QString fn = QFileDialog::getSaveFileName(this, tr("Export"), "./", tr("Text Files (*.txt)"));
    if (fn.isEmpty()) {
        qDebug() << "No directory selected";
        return;
    }

    std::ofstream ofs(fn.toStdString());
    ofs << "ccm:" << dataManager_->ccm_ << "\n";
    ofs << "loss:" << dataManager_->loss_;
    ofs.close();
    if (ui->tabWidget_2->currentIndex() <= 0) {
        ui->textBrowser->append("export ccm and loss to " + fn);
    }else{
        ui->textBrowser_2->append("export ccm and loss to " + fn);
    }
}

void MainWindow::next_slot() {
    if (dataManager_->image_mats_.empty()) { return; }
    int index = dataManager_->current_image_index_ + 1;
    if (index >= dataManager_->image_mats_.size()) { return; }
    dataManager_->current_image_index_ = index;
    if (manualPreviewGroup_->checkedId() == 3) {
        ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
    }else if (manualPreviewGroup_->checkedId() == 4){
        cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
        for (int i = 0; i < dataManager_->color_rects_.size(); ++i) {
            std::pair<int,cv::Rect2f>& pair = dataManager_->color_rects_[i];
            if (pair.first == dataManager_->current_image_index_ && !pair.second.empty()) {
                cv::Scalar white = cv::Scalar(255,255,255);
                cv::rectangle(frame,pair.second,white,3);
                cv::Scalar color = cv::Scalar(Utils::GetMacbethColor(i).blue(),Utils::GetMacbethColor(i).green(),Utils::GetMacbethColor(i).red());
                cv::rectangle(frame,pair.second,color,2);
            }
        }
        ui->widget_4->Update(frame);
    }else{
        if (!dataManager_->ccm_.empty()){
            cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
            Utils::colorCorrect((uchar3*)frame.data,frame.cols*frame.rows,(double *)dataManager_->ccm_.data);
            ui->widget_4->Update(frame);
        }else{
            ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
        }
    }

}

void MainWindow::previous_slot() {
    if (dataManager_->image_mats_.empty()) { return; }
    int index = dataManager_->current_image_index_ - 1;
    if (index >= dataManager_->image_mats_.size()) { return; }
    dataManager_->current_image_index_ = index;
    if (manualPreviewGroup_->checkedId() == 3) {
        ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
    }else if (manualPreviewGroup_->checkedId() == 4){
        cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
        for (int i = 0; i < dataManager_->color_rects_.size(); ++i) {
            std::pair<int,cv::Rect2f>& pair = dataManager_->color_rects_[i];
            if (pair.first == dataManager_->current_image_index_ && !pair.second.empty()) {
                cv::Scalar white = cv::Scalar(255,255,255);
                cv::rectangle(frame,pair.second,white,3);
                cv::Scalar color = cv::Scalar(Utils::GetMacbethColor(i).blue(),Utils::GetMacbethColor(i).green(),Utils::GetMacbethColor(i).red());
                cv::rectangle(frame,pair.second,color,2);
            }
        }
        ui->widget_4->Update(frame);
    }else{
        if (!dataManager_->ccm_.empty()){
            cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
            Utils::colorCorrect((uchar3*)frame.data,frame.cols*frame.rows,(double *)dataManager_->ccm_.data);
            ui->widget_4->Update(frame);
        }else{
            ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
        }
    }
}

void MainWindow::color_area_apply_slot() {
    if (dataManager_->color_rects_.empty()) {
        dataManager_->color_rects_.resize(gala::ColorNum);
    }

    int colorIndex = colorGroup_->checkedId();
    dataManager_->color_rects_[colorIndex] = std::make_pair(dataManager_->current_image_index_,
                                                            ui->widget_4->GetRect());
    cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
    for (int i = 0; i < dataManager_->color_rects_.size(); ++i) {
        std::pair<int,cv::Rect2f>& pair = dataManager_->color_rects_[i];
        if (pair.first == dataManager_->current_image_index_ && !pair.second.empty()) {
            cv::Scalar white = cv::Scalar(255,255,255);
            cv::rectangle(frame,pair.second,white,3);
            cv::Scalar color = cv::Scalar(Utils::GetMacbethColor(i).blue(),Utils::GetMacbethColor(i).green(),Utils::GetMacbethColor(i).red());
            cv::rectangle(frame,pair.second,color,2);
        }
    }
    ui->widget_4->Update(frame);
    jobStatusChanged(1);
    ui->textBrowser_2->append("color block index: "+QString::number(colorIndex)+" selected");

}

void MainWindow::manual_calibrate_slot() {
    if (dataManager_->color_rects_.size()<=10){
        qDebug()<<dataManager_->color_rects_.size()<<" color block are selected,but for calibrate need at least 10 ";
        return;
    }
    std::vector<std::pair<gala::MacbethColor, cv::Mat>> colorBlocks;
    for (int i = 0; i < dataManager_->color_rects_.size(); ++i) {
        gala::MacbethColor colorIdx = (gala::MacbethColor)i;
        int frameIdx = dataManager_->color_rects_[i].first;
        cv::Rect2f rect = dataManager_->color_rects_[i].second;
        if (rect.empty()) {continue;}
        cv::Mat& frame = dataManager_->image_mats_[frameIdx];
        cv::Mat colorBlock = frame(rect);
        colorBlocks.push_back(std::make_pair(colorIdx, colorBlock));
    }
    colorChecker_->CheckColor(colorBlocks);
    dataManager_->ccm_ = colorChecker_->GetCCM();
    dataManager_->loss_ = colorChecker_->GetLoss();
    cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
    Utils::colorCorrect((uchar3*)frame.data,frame.cols*frame.rows,(double *)dataManager_->ccm_.data);
    ui->widget_4->Update(frame);
    jobStatusChanged(2);
    std::cout<<"ccm:"<<dataManager_->ccm_<<std::endl;
    std::cout<<"loss:"<<dataManager_->loss_<<std::endl;
    std::ostringstream ss;
    ss<<"ccm:"<<dataManager_->ccm_<<"\nloss:"<<dataManager_->loss_;
    ui->textBrowser_2->append(QString::fromStdString(ss.str()));
}

void MainWindow::tabWidget_index_changed(int index) {
    bool isAutomatic = index == 0;
    ui->widget_4->SetHandleMode(isAutomatic);
    ui->widget_6->setHidden(isAutomatic);

    dataManager_->Clear();
    ui->widget_4->Reset();

    QColor color(ui->label->palette().color(QPalette::Window));
    color.setAlpha(0);
    ui->label->setPalette(QPalette(color));
    ui->label_2->setPalette(QPalette(color));
    ui->label_3->setPalette(QPalette(color));
    ui->label_5->setPalette(QPalette(color));
    ui->label_6->setPalette(QPalette(color));
    ui->label_7->setPalette(QPalette(color));
    ui->radioButton_4->setChecked(true);
    ui->radioButton->setChecked(true);
    ui->textBrowser->clear();
    ui->textBrowser_2->clear();

}

void MainWindow::color_index_changed(int index) {
    switch (index) {
        case gala::DarkSkin: {

            break;
        }
    }
}

void MainWindow::preview_index_changed(int index) {
    switch (index) {
        case 0: {
            if (!dataManager_->image_mats_.empty()) {
                ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
            }
            break;
        }

        case 1: {
            if (!dataManager_->image_mats_with_checkers_.empty()) {
                ui->widget_4->Update(dataManager_->image_mats_with_checkers_[dataManager_->current_image_index_]);
            }
            break;
        }
        case 2: {
            if (!dataManager_->image_mats_corrected_.empty()) {
                ui->widget_4->Update(dataManager_->image_mats_corrected_[dataManager_->current_image_index_]);
            }
            break;
        }
        case 3: {
            if (!dataManager_->image_mats_.empty()) {
                ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
            }
            break;
        }
        case 4: {
            cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
            for (int i = 0; i < dataManager_->color_rects_.size(); ++i) {
                std::pair<int,cv::Rect2f>& pair = dataManager_->color_rects_[i];
                if (pair.first == dataManager_->current_image_index_ && !pair.second.empty()) {
                    cv::Scalar white = cv::Scalar(255,255,255);
                    cv::rectangle(frame,pair.second,white,3);
                    cv::Scalar color = cv::Scalar(Utils::GetMacbethColor(i).blue(),Utils::GetMacbethColor(i).green(),Utils::GetMacbethColor(i).red());
                    cv::rectangle(frame,pair.second,color,2);
                }
            }
            ui->widget_4->Update(frame);
            break;
        }
        case 5: {
            if (!dataManager_->ccm_.empty()){
                cv::Mat frame = dataManager_->image_mats_[dataManager_->current_image_index_].clone();
                Utils::colorCorrect((uchar3*)frame.data,frame.cols*frame.rows,(double *)dataManager_->ccm_.data);
                ui->widget_4->Update(frame);
            }else{
                ui->widget_4->Update(dataManager_->image_mats_[dataManager_->current_image_index_]);
            }
            break;
        }

    }
}

//void MainWindow::tabWidget_index_changed(int index) {
//    bool isAutomatic = index == 0;
//    ui->widget_4->SetHandleMode(isAutomatic);
//    ui->widget_6->setHidden(isAutomatic);
//}

