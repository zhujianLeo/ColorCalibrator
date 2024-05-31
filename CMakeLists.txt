cmake_minimum_required(VERSION 3.5)

project(ColorCalibrator LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        $ENV{THIRDPARTY}/opencv490
        )

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

if(ANDROID)
else()
  set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_AUTOUIC_SEARCH_PATHS} ./ui)
  qt5_add_resources(QRCS
          res.qrc
          )
  include_directories(./inc)
  cuda_add_executable(ColorCalibrator
    main.cpp
          ./src/MainWindow.cpp
          ./src/Widget.cpp
          ./src/MainWidget.cpp
          ./src/MacbethColorChecker.cpp
          ./src/DataManager.cpp
          ./src/ColorCorrect.cu

          ./inc/MainWindow.h
          ./inc/Widget.h
          ./inc/MainWidget.h
          ./inc/DataManager.h

          ./ui/MainWindow.ui
          ./ui/Widget.ui

          ${QRCS}
          inc/ColorSpace.h
          src/Utils.cpp
          inc/Utils.h
          inc/MacbethColor.h)
  target_link_libraries(ColorCalibrator
          Qt5::Widgets
          ${OpenCV_LIBS}
          ${CUDA_LIBRARIES}
          ${CUDA_CUDA_LIBRARY})
endif()


