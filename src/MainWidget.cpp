#include "MainWidget.h"
#include <qdebug.h>
#include <QPainter>
#include <qmath.h>
MainWidget::MainWidget(QWidget *parent) : QWidget(parent)
{

}

void MainWidget::mousePressEvent(QMouseEvent *event) {
    QWidget::mousePressEvent(event);
}

void MainWidget::mouseReleaseEvent(QMouseEvent *event) {
    QWidget::mouseReleaseEvent(event);
}

void MainWidget::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
}

void MainWidget::paintEvent(QPaintEvent *event) {
//    QPainter painter(this);
//    QPen pen(Qt::yellow, 5);
//    painter.setPen(pen);
//
//    QPoint start(50, 50);
//    QPoint end(400, 400);
//    painter.drawLine(start, end);
//
//    QWidget::paintEvent(event);


//    double angle = std::atan2(-1.0 * (end.y() - start.y()), end.x() - start.x());
//    QPointF arrowP1 = end + QPointF(sin(angle - M_PI / 6) * 10, cos(angle - M_PI / 6) * 10);
//    QPointF arrowP2 = end + QPointF(sin(angle + M_PI / 6) * 10, cos(angle + M_PI / 6) * 10);
//
//    QPolygonF arrowHead;
//    arrowHead << end << arrowP1 << arrowP2;
//    painter.drawPolygon(arrowHead);
}

void MainWidget::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    const QObjectList& childrenList = children();
    for (QObject* child : childrenList) {
        QWidget* widgetChild = qobject_cast<QWidget*>(child);
        if (widgetChild) {
            widgetChild->resize(QWidget::width(),QWidget::height());
        }
    }

}
