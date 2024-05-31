#include "OverlayWidget.h"
#include <QPainter>
#include <QtMath>
#include <QMouseEvent>
OverlayWidget::OverlayWidget(QWidget *parent) : QWidget(parent)
{

}

void OverlayWidget::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);

        QPainter painter(this);
    QPen pen(Qt::yellow, 5);
    painter.setPen(pen);

    QPoint start(50, 50);
    QPoint end(400, 400);
    painter.drawLine(start, end);

    QWidget::paintEvent(event);


    double angle = std::atan2(-1.0 * (end.y() - start.y()), end.x() - start.x());
    QPointF arrowP1 = end + QPointF(sin(angle - M_PI / 6) * 10, cos(angle - M_PI / 6) * 10);
    QPointF arrowP2 = end + QPointF(sin(angle + M_PI / 6) * 10, cos(angle + M_PI / 6) * 10);

    QPolygonF arrowHead;
    arrowHead << end << arrowP1 << arrowP2;
    painter.drawPolygon(arrowHead);
}

void OverlayWidget::mousePressEvent(QMouseEvent *event) {
    event->ignore();
    event->globalPos();
}

void OverlayWidget::mouseReleaseEvent(QMouseEvent *event) {
    event->ignore();
}

void OverlayWidget::mouseMoveEvent(QMouseEvent *event) {
    event->ignore();
}
