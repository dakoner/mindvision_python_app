# MindVision QObject

This project builds three qmake targets:

1. `mindvision_qobject`: shared Qt/C++ camera library
2. `camera_gui`: native Qt Widgets camera application
3. `_mindvision_qobject_py`: Python extension module

## Requirements

- Qt 6 (or compatible) with MinGW or MSVC.
- MindVision SDK (MVCAMSDK).

## Setup

The project relies on the MindVision SDK headers and library files.
Please place the SDK files in the following directories within the project root:

1.  **Include/**: Copy `CameraApi.h` and other SDK headers here.
    -   Source: Usually `C:/Program Files (x86)/MindVision/MindVision-Gige/SDK/Include`
2.  **Lib/**: Copy the library file `MVCAMSDK_X64.lib` (for MSVC) or `libMVCAMSDK_X64.a` / `MVCAMSDK_X64.dll` (for MinGW) here.
    -   Ensure the library name matches `LIBS += -L$$PWD/Lib -lMVCAMSDK_X64` in the `.pro` file.
    -   If you are using 32-bit build, update the `.pro` file to link against the 32-bit library (e.g. `MVCAMSDK.lib`).

## Building

1.  Open a terminal in the project directory.
2.  Run `qmake`.
3.  Run `mingw32-make` (or `nmake` for MSVC).

## Output

The build places artifacts in the `release` or `debug` folder:

1. `release/libmindvision_qobject.so` or platform equivalent
2. `release/camera_gui`
3. `release/_mindvision_qobject_py.so`
