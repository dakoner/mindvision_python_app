QT       += core gui

TARGET = _mindvision_qobject_py
TEMPLATE = lib
CONFIG += shared

DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH += Include
INCLUDEPATH += src

# Python and pybind11 include paths
PYTHON_INCLUDES = $$system(python3-config --includes 2>/dev/null)
isEmpty(PYTHON_INCLUDES) {
    PYTHON_INCLUDES = -I/usr/include/python3.12
}
QMAKE_CXXFLAGS += $$PYTHON_INCLUDES

PYBIND11_INCLUDES = $$system(python3 -m pybind11 --includes 2>/dev/null)
isEmpty(PYBIND11_INCLUDES) {
    INCLUDEPATH += /usr/include/pybind11
} else {
    QMAKE_CXXFLAGS += $$PYBIND11_INCLUDES
}

CONFIG += c++17

# Directory where the Python module will be placed
CONFIG(debug, debug|release) {
    DESTDIR = $$OUT_PWD/debug
    OBJECTS_DIR = $$OUT_PWD/debug
    MOC_DIR = $$OUT_PWD/debug
} else {
    DESTDIR = $$OUT_PWD/release
    OBJECTS_DIR = $$OUT_PWD/release
    MOC_DIR = $$OUT_PWD/release
}

SOURCES += src/mindvision_qobject_python.cpp

# Link against the main C++ library
CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/debug
} else {
    LIBS += -L$$PWD/release
}
LIBS += -lmindvision_qobject

# Link against the MindVision SDK
LIBS += -L$$PWD/Lib -lMVSDK

# Link against the Python library
win32: {
    PYTHON_LIB_DIR = C:/Users/davidek/scoop/apps/python313/current/libs
    LIBS += -L$$PYTHON_LIB_DIR -lpython3.11
} else: {
    PYTHON_LDFLAGS = $$system(python3-config --embed --ldflags 2>/dev/null)
    isEmpty(PYTHON_LDFLAGS) {
        LIBS += -lpython3.12
    } else {
        LIBS += $$PYTHON_LDFLAGS
    }
}

unix:!macx: {
    QMAKE_POST_LINK += $$QMAKE_MOVE $$shell_path($$DESTDIR/lib_mindvision_qobject_py.so) $$shell_path($$DESTDIR/_mindvision_qobject_py.so)
}
