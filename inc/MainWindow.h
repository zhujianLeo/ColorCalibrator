#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QButtonGroup>
#include "DataManager.h"
#include "MacbethColorChecker.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


protected:

    void jobStatusChanged(int status);
private slots:
    void open_slot();
    void new_slot();
    void exit_slot();

    void detect_slot();
    void calibrate_slot();
    void export_slot();


    void next_slot();
    void previous_slot();

    void color_area_apply_slot();
    void manual_calibrate_slot();



    void color_index_changed(int index);
    void preview_index_changed(int index);
    void tabWidget_index_changed(int index);



private:
    Ui::MainWindow *ui;
    QButtonGroup* previewGroup_;
    QButtonGroup* manualPreviewGroup_;
    QButtonGroup* colorGroup_;
    int currentColorIdx_;
    DataManager* dataManager_;
    gala::MacbethColorChecker* colorChecker_;
    int jop_status_;
    int8_t* ccm_data_;
    int8_t* frame_data_;
    int8_t* frame_corrected_data_;
};
#endif // MAINWINDOW_H
