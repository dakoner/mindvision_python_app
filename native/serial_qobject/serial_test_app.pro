TEMPLATE = app
CONFIG += c++17 console
QT += core widgets serialport

TARGET = serial_test_app
DESTDIR = release

SOURCES += \
    src/SerialWorker.cpp \
    src/serial_test_app.cpp

HEADERS += \
    src/SerialWorker.h