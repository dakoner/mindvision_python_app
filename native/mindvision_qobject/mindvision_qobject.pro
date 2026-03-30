TEMPLATE = subdirs

SUBDIRS += \
    mindvision_lib \
    camera_gui \
    mindvision_py

mindvision_lib.file = mindvision_lib.pro
camera_gui.file = camera_gui.pro
mindvision_py.file = mindvision_py.pro

camera_gui.depends = mindvision_lib
mindvision_py.depends = mindvision_lib