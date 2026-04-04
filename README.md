# MindVision Python App

This project now contains the code that previously lived in `mindvision_qobject`.

## Layout

- `src/mindvision_python_app/`: Python package and Qt application code
- `src/mindvision_python_app/camera_gui.py`: migrated simple camera/recording GUI
- `src/mindvision_python_app/test_mindvision.py`: migrated wrapper smoke test
- `native/mindvision_qobject/`: migrated C++ wrapper sources, qmake files, SDK headers/libs, and release artifacts

## Native Wrapper Location

The Python package loads `_mindvision_qobject_py` from:

- `native/mindvision_qobject/release`

You can override that location with `MINDVISION_QOBJECT_RELEASE_DIR` if needed.

## Build The Native Module

From the project root:

```bash
cd native/mindvision_qobject
qmake6 mindvision_qobject.pro
make
```

If you only need the Python extension target:

```bash
cd native/mindvision_qobject
qmake6 mindvision_py.pro
make
```

The expected output is `native/mindvision_qobject/release/_mindvision_qobject_py.so`.

## Run

Main application:

```bash
python -m mindvision_python_app.main
```

Simple migrated camera GUI:

```bash
python -m mindvision_python_app.camera_gui
```
