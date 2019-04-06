TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

include(eigen3.pri)
include(opencv.pri)

SOURCES += \
        main.cpp
