TEMPLATE = lib
CONFIG += plugin c++17
CONFIG += no_plugin_name_prefix
QT += core serialport

TARGET = _serial_qobject_py
DESTDIR = release

PYBIND11_CFLAGS = $$system(uv run python3 -m pybind11 --includes)
QMAKE_CXXFLAGS += $$PYBIND11_CFLAGS

# Set correct extension for Python modules
macx: QMAKE_EXTENSION_PLUGIN = so
else:linux: QMAKE_EXTENSION_PLUGIN = so
else:win32: QMAKE_EXTENSION_PLUGIN = pyd

SOURCES += \
    src/SerialWorker.cpp \
    src/bindings.cpp

HEADERS += \
    src/SerialWorker.h