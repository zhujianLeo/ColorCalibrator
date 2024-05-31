#include "Widget.h"
#include "ui_Widget.h"
#include <QPainter>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QDebug>
#include "Utils.h"

#if 1
Widget::Widget(QWidget *parent) :
        QWidget(parent),
        ui(new Ui::Widget)
{
    ui->setupUi(this);
    mousePressStatus_ = 0;
    max_magnification_= 8;
    Reset();
}

Widget::~Widget()
{
    delete ui;
}

void Widget::Reset(){
    image_ = cv::Mat();
    if (!image_.empty()){
        sourceRect_ = QRect(0,0,image_.cols,image_.rows);
        calculateTargetRect();
    }
    
    ori_mouse_pos_ = QPoint(0,0);
    cur_mouse_pos_ = QPoint(0,0);
    scale_= 1.0;
    update();
}

void Widget::Update(cv::Mat frame){
    image_ = frame;
    update();
}

//void Widget::Update(cv::Mat frame, std::vector<std::pair<int,cv::Rect2f>> rects){
//    image_ = frame;
//    drawRects_ = rects;
//    update();
//}

cv::Rect2f Widget::GetRect(){
    return rect_;
}

void Widget::mousePressEvent(QMouseEvent *event) {
    QWidget::mousePressEvent(event);
    if (event->button() == Qt::LeftButton){
        if (isAutomatic_){ return;}
        mousePressStatus_ = 1;
        mouseStartpoint_ = event->pos();
        mouseEndpoint_ = event->pos();
    }else if (event->button() == Qt::RightButton){
        mousePressStatus_ = 2;
        ori_mouse_pos_ = event->pos();
        cur_mouse_pos_ = event->pos();
    }



}

void Widget::mouseReleaseEvent(QMouseEvent *event) {
    QWidget::mouseReleaseEvent(event);
    if (event->button() == Qt::LeftButton){
        if (isAutomatic_){ return;}
        mouseEndpoint_ = event->pos();
        int wndRectW = std::abs(mouseStartpoint_.x()-mouseEndpoint_.x())*2;
        int wndRectH = std::abs(mouseStartpoint_.y()-mouseEndpoint_.y())*2;
        if (wndRectH > image_.rows){wndRectH = image_.rows;}
        if (wndRectW > image_.cols){wndRectW = image_.cols;}
        int wndTopLeftX = mouseStartpoint_.x()-wndRectW/2;
        int wndTopLeftY = mouseStartpoint_.y()-wndRectH/2;
        if (wndTopLeftX<=0){wndTopLeftX = 0;}
        if (wndTopLeftY<=0){wndTopLeftY = 0;}



        float source2TargetWRatio = (float)sourceRect_.width()/(float)targetRect_.width();
        float source2TargetHRatio = (float)sourceRect_.height()/(float)targetRect_.height();

        float sourceX = wndTopLeftX*source2TargetWRatio;
        float sourceY = wndTopLeftY*source2TargetHRatio;
        float sourceW = wndRectW*source2TargetWRatio;
        float sourceH = wndRectH*source2TargetHRatio;


        float imageX = sourceX+sourceRect_.x();
        float imageY = sourceY+sourceRect_.y();




        rect_ = cv::Rect2f(imageX,imageY,sourceW,sourceH);










//        int rect_w = std::abs(mouseStartpoint_.x()-mouseEndpoint_.x())*2;
//        int rect_h = std::abs(mouseStartpoint_.y()-mouseEndpoint_.y())*2;
//        int x = mouseStartpoint_.x()-rect_w/2;
//        int y = mouseStartpoint_.y()-rect_h/2;
//        if (x<=0){x = 0;}
//        if (y<=0){y = 0;}
//        float fx = ((float)x/(QWidget::width()-1))*(image_.cols-1);
//        float fy = ((float)y/(QWidget::height()-1))*(image_.rows-1);
//        float frect_w = ((float)rect_w/QWidget::width())*(image_.cols);
//        float frect_h = ((float)rect_h/QWidget::height())*(image_.rows);
//        rect_ =  cv::Rect2f(fx,fy,frect_w,frect_h);
    }
    mousePressStatus_ = 0;

}

void Widget::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
    if (mousePressStatus_ == 1) {
        mouseEndpoint_ = event->pos();
        update();
    }else if (mousePressStatus_ == 2) {
        cur_mouse_pos_ = event->pos();
        int deltaX = cur_mouse_pos_.x() - ori_mouse_pos_.x();
        int deltaY = cur_mouse_pos_.y() - ori_mouse_pos_.y();
        int viewDeltaX = deltaX * sourceRect_.width() / width();
        int viewDeltaY = deltaY * sourceRect_.height() / height();
        sourceRect_.moveCenter(QPoint(sourceRect_.center().x() - viewDeltaX, sourceRect_.center().y() - viewDeltaY));
        calculateTargetRect();
        ori_mouse_pos_ = cur_mouse_pos_;
        update();
    }
}

void Widget::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);
    if(image_.empty()){return;}
    if (sourceRect_.isEmpty()) {
        sourceRect_ = QRect(0,0,image_.cols,image_.rows);
        calculateTargetRect();
    }
    QImage image = QImage((uchar*)image_.data, image_.cols, image_.rows, QImage::Format_BGR888);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.drawImage(targetRect_, image, sourceRect_);
    if (isAutomatic_){ return;}
    switch (mousePressStatus_) {
        case 0:{
            break;
        }
        case 1:{
            QColor color(100,126,175);
            painter.setPen(color);
            painter.drawEllipse(mouseStartpoint_,2,2);
            int rect_w = std::abs(mouseStartpoint_.x()-mouseEndpoint_.x())*2;
            int rect_h = std::abs(mouseStartpoint_.y()-mouseEndpoint_.y())*2;
            painter.drawRect(mouseStartpoint_.x()-rect_w/2,mouseStartpoint_.y()-rect_h/2,rect_w,rect_h);
            break;
        }
        case 2:{

        }
    }

}

void Widget::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
}

void Widget::mouseDoubleClickEvent(QMouseEvent *event) {
    QWidget::mouseDoubleClickEvent(event);
    scale_ = 1.0;
    if (!image_.empty()) {
        sourceRect_ = QRect(0,0,image_.cols,image_.rows);
        calculateTargetRect();
    }

    update();
}

void Widget::wheelEvent(QWheelEvent *event) {
    QWidget::wheelEvent(event);
    if (image_.empty())return;
    QPoint theCenter = sourceRect_.center();
    double zoomFactor = 1.0;
    if (event->delta() > 0) {
        zoomFactor = 0.9;
    } else {
        zoomFactor = 1.1;
    }
    QRect tempRect = sourceRect_;      // Has pixmap coordinates
    int www = round(zoomFactor * (double) tempRect.width());
    int hhh = round(zoomFactor * (double) tempRect.height());
    if ((www < (this->width() / max_magnification_)) || (hhh < (this->height() / max_magnification_))) {  // If too small
        www = this->width() / max_magnification_;
        hhh = this->height() / max_magnification_;
    }
    if ((www > (max_magnification_ * image_.cols)) || (hhh > (max_magnification_ * image_.rows))) {
        www = max_magnification_ *image_.cols;
        hhh = max_magnification_ * image_.rows;
    }
    tempRect.setX(sourceRect_.x());
    tempRect.setY(sourceRect_.y());
    tempRect.setWidth(www);
    tempRect.setHeight(hhh);
    tempRect.moveCenter(theCenter);
    sourceRect_ = tempRect;
    calculateTargetRect();
    update();
}

void Widget::calculateTargetRect() {
    if (image_.empty()){ return;}

#if 1

    targetRect_ = QRect(0,0,QWidget::width(), QWidget::height());

    return;
#endif

    QRect rect(0,0,image_.cols,image_.rows);
    if(rect.contains(sourceRect_,true)){
        float wgtRatio = (float)width() / height();
        float viewRatio = (float)sourceRect_.width() / sourceRect_.height();
        int x,y,w,h;
        if(wgtRatio>=viewRatio){
            w =  height()*viewRatio;
            h =  height();
            x = (width()-w)/2;
            y =0;
        }else{
            w =  width();
            h =  width()/viewRatio;
            x = 0;
            y =(height()-h)/2;
        }
        targetRect_ = QRect(x,y,w,h);
    }else {
        float wgtRatio = (float)width() / height();
        float viewRatio = (float)sourceRect_.width() / sourceRect_.height();
        int x,y,w,h;
        if(wgtRatio>=viewRatio){
            w =  height()*viewRatio;
            h =  height();
            x = (width()-w)/2;
            y =0;
        }else{
            w =  width();
            h =  width()/viewRatio;
            x = 0;
            y =(height()-h)/2;
        }
        targetRect_ = QRect(x,y,w,h);
    }
}
#else


Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    mousePressStatus_ = 0;
    Reset();
}

Widget::~Widget()
{
    delete ui;
}

void Widget::Reset(){
    image_ = cv::Mat();
    drawRects_.clear();
    transform_.reset();
    sourceRect_ = QRect(0,0,QWidget::width(),QWidget::height());
    if (!image_.empty()){
        sourceRect_ = QRect(0,0,image_.cols,image_.rows);
    }
    scale_= 1.0;
    update();
}

void Widget::Update(cv::Mat frame){
    image_ = frame;
    drawRects_.clear();
    sourceRect_ = QRect(0,0,QWidget::width(),QWidget::height());
    sourceRect_ = QRect(0,0,image_.cols,image_.rows);
    update();
}

void Widget::Update(cv::Mat frame, std::vector<std::pair<int,cv::Rect2f>> rects){
    image_ = frame;
    drawRects_ = rects;
    sourceRect_ = QRect(0,0,QWidget::width(),QWidget::height());
    sourceRect_ = QRect(0,0,image_.cols,image_.rows);
//    transform_.reset();
//    transform_.translate(QWidget::width()/2, QWidget::height()/2);
    update();
}

cv::Rect2f Widget::GetRect(){
    return rect_;
}

void Widget::mousePressEvent(QMouseEvent *event) {
    QWidget::mousePressEvent(event);
    if (event->button() == Qt::LeftButton){
        if (isAutomatic_){ return;}
        if (event->button() == Qt::LeftButton){
            mousePressStatus_ = 1;
        }else if (event->button() == Qt::RightButton){
            mousePressStatus_ = 2;
        }
        mouseStartpoint_ = event->pos();
        mouseEndpoint_ = event->pos();
    }else if(event->button() == Qt::RightButton){

    }


}

void Widget::mouseReleaseEvent(QMouseEvent *event) {
    QWidget::mouseReleaseEvent(event);
    if (event->button() == Qt::LeftButton){
        if (isAutomatic_){ return;}
        mouseEndpoint_ = event->pos();
        mousePressStatus_ = 0;
        int rect_w = std::abs(mouseStartpoint_.x()-mouseEndpoint_.x())*2;
        int rect_h = std::abs(mouseStartpoint_.y()-mouseEndpoint_.y())*2;
        int x = mouseStartpoint_.x()-rect_w/2;
        int y = mouseStartpoint_.y()-rect_h/2;
        if (x<=0){x = 0;}
        if (y<=0){y = 0;}
        float fx = ((float)x/(QWidget::width()-1))*(image_.cols-1);
        float fy = ((float)y/(QWidget::height()-1))*(image_.rows-1);
        float frect_w = ((float)rect_w/QWidget::width())*(image_.cols);
        float frect_h = ((float)rect_h/QWidget::height())*(image_.rows);
        rect_ =  cv::Rect2f(fx,fy,frect_w,frect_h);
    }else if(event->button() == Qt::RightButton){

    }

}

void Widget::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
    if (event->button() == Qt::LeftButton){

    }else if(event->button() == Qt::RightButton){

    }

    if (isAutomatic_){ return;}
    mouseEndpoint_ = event->pos();
    update();
}

void Widget::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);
    if(image_.empty()){return;}

    QImage image = QImage((uchar*)image_.data, image_.cols, image_.rows, QImage::Format_BGR888);
    QPainter painter(this);

    painter.setTransform(transform_);
    painter.drawImage(0, 0, image);
//    painter.drawImage(sourceRect_, image,sourceRect_);

    if (isAutomatic_){ return;}
    if (!drawRects_.empty()){
        for (int i = 0; i < drawRects_.size(); ++i) {
            QPen pen(Utils::GetMacbethColor(drawRects_[i].first),3);
            painter.setPen(pen);
            int x = ((float)drawRects_[i].second.x/(image_.cols-1))*(QWidget::width()-1);
            int y = ((float)drawRects_[i].second.y/(image_.rows-1))*(QWidget::height()-1);
            int w = ((float)drawRects_[i].second.width/(image_.cols))*(QWidget::width());
            int h = ((float)drawRects_[i].second.height/(image_.rows))*(QWidget::height());
            painter.drawRect(x,y,w,h);
        }
    }
    switch (mousePressStatus_) {
        case 0:{
            break;
        }
        case 1:{
            QColor color(100,126,175);
            painter.setPen(color);
            painter.drawEllipse(mouseStartpoint_,2,2);
            int rect_w = std::abs(mouseStartpoint_.x()-mouseEndpoint_.x())*2;
            int rect_h = std::abs(mouseStartpoint_.y()-mouseEndpoint_.y())*2;
            painter.drawRect(mouseStartpoint_.x()-rect_w/2,mouseStartpoint_.y()-rect_h/2,rect_w,rect_h);

            break;
        }
        case 2:{

        }
    }

}

void Widget::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
}

void Widget::mouseDoubleClickEvent(QMouseEvent *event) {
    QWidget::mouseDoubleClickEvent(event);
    scale_ = 1.0;
    sourceRect_ = QRect(0,0,QWidget::width(),QWidget::height());
    update();
}

void Widget::wheelEvent(QWheelEvent *event) {
    QWidget::wheelEvent(event);
    if (event->angleDelta().y() > 0){
        scale_ *= 1.1;
    }else{
        scale_ /= 1.1;
    }
    transform_.scale(scale_,scale_);
    QPoint pos = event->pos();
    int w = QWidget::width()*scale_;
    int h = QWidget::height()*scale_;



//    float fx = ((float)pos.x()/(QWidget::width()-1))*(image_.cols-1);
//    float fy = ((float)pos.y()/(QWidget::height()-1))*(image_.rows-1);
//    rect_ =  cv::Rect2f(fx,fy,w,h);
//    sourceRect_ = QRect(0,0,image_.cols*scale_,image_.rows*scale_);

    int offsetX = (((float)pos.x()/(QWidget::width()-1))*w)-pos.x();
    int offsetY = (((float)pos.y()/(QWidget::height()-1))*h)-pos.y();;
//
//    int offsetX = (w-QWidget::width())/2;
//    int offsetY = (h-QWidget::height())/2;
    sourceRect_ = QRect(-offsetX,-offsetY,w,h);
    update();
}
#endif