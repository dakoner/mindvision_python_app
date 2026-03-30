QT += core gui widgets

TARGET = camera_gui
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH += Include
INCLUDEPATH += src

CONFIG += c++2a

CONFIG(debug, debug|release) {
    DESTDIR = $$OUT_PWD/debug
    OBJECTS_DIR = $$OUT_PWD/debug/app
    MOC_DIR = $$OUT_PWD/debug/app
} else {
    DESTDIR = $$OUT_PWD/release
    OBJECTS_DIR = $$OUT_PWD/release/app
    MOC_DIR = $$OUT_PWD/release/app
}

SOURCES += \
    src/CameraMainWindow.cpp \
    src/main.cpp

HEADERS += \
    src/CameraMainWindow.h

CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/debug
    QMAKE_RPATHDIR += $$OUT_PWD/debug
} else {
    LIBS += -L$$PWD/release
    QMAKE_RPATHDIR += $$OUT_PWD/release
}

LIBS += -lmindvision_qobject
LIBS += -L$$PWD/Lib -lMVSDK

unix:!macx {
    QMAKE_RPATHDIR += $$PWD/Lib
}