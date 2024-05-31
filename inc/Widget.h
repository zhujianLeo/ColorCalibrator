#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QPoint>
#include <opencv2/opencv.hpp>
namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget();

    void Reset();
    void Update(cv::Mat frame);
//    void Update(cv::Mat frame, std::vector<std::pair<int,cv::Rect2f>> rects);
    cv::Rect2f GetRect();
    void SetHandleMode(bool isAutomatic){isAutomatic_ = isAutomatic;};
protected:
    void mousePressEvent(QMouseEvent *event) override;

    void mouseReleaseEvent(QMouseEvent *event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

    void paintEvent(QPaintEvent *event) override;

    void resizeEvent(QResizeEvent *event) override;

    void mouseDoubleClickEvent(QMouseEvent *event) override;

    void wheelEvent(QWheelEvent *event) override;


private:
    void calculateTargetRect();

private:
    Ui::Widget *ui;
    cv::Rect2f rect_;
    QRect sourceRect_,targetRect_;
    QPoint cur_mouse_pos_,ori_mouse_pos_;
    int max_magnification_;

    float scale_ = 1.0f;
    bool isAutomatic_ = false;
    cv::Mat image_;
    QPoint mouseStartpoint_;
    QPoint mouseEndpoint_;
    int mousePressStatus_;//0 not pressed;1 left pressed 2 right pressed

};

#endif // WIDGET_H
