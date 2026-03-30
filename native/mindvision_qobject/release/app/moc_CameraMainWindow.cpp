/****************************************************************************
** Meta object code from reading C++ file 'CameraMainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../src/CameraMainWindow.h"
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'CameraMainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.4.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
namespace {
struct qt_meta_stringdata_CameraMainWindow_t {
    uint offsetsAndSizes[36];
    char stringdata0[17];
    char stringdata1[11];
    char stringdata2[1];
    char stringdata3[12];
    char stringdata4[16];
    char stringdata5[17];
    char stringdata6[18];
    char stringdata7[17];
    char stringdata8[6];
    char stringdata9[12];
    char stringdata10[17];
    char stringdata11[4];
    char stringdata12[24];
    char stringdata13[10];
    char stringdata14[14];
    char stringdata15[25];
    char stringdata16[9];
    char stringdata17[26];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CameraMainWindow_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CameraMainWindow_t qt_meta_stringdata_CameraMainWindow = {
    {
        QT_MOC_LITERAL(0, 16),  // "CameraMainWindow"
        QT_MOC_LITERAL(17, 10),  // "openCamera"
        QT_MOC_LITERAL(28, 0),  // ""
        QT_MOC_LITERAL(29, 11),  // "closeCamera"
        QT_MOC_LITERAL(41, 15),  // "toggleRecording"
        QT_MOC_LITERAL(57, 16),  // "updateFpsDisplay"
        QT_MOC_LITERAL(74, 17),  // "renderLatestFrame"
        QT_MOC_LITERAL(92, 16),  // "handleFrameReady"
        QT_MOC_LITERAL(109, 5),  // "image"
        QT_MOC_LITERAL(115, 11),  // "timestampMs"
        QT_MOC_LITERAL(127, 16),  // "handleFpsChanged"
        QT_MOC_LITERAL(144, 3),  // "fps"
        QT_MOC_LITERAL(148, 23),  // "handleQueueStatsChanged"
        QT_MOC_LITERAL(172, 9),  // "queueSize"
        QT_MOC_LITERAL(182, 13),  // "droppedFrames"
        QT_MOC_LITERAL(196, 24),  // "handleRecordFileSelected"
        QT_MOC_LITERAL(221, 8),  // "filename"
        QT_MOC_LITERAL(230, 25)   // "handleRecordFileCancelled"
    },
    "CameraMainWindow",
    "openCamera",
    "",
    "closeCamera",
    "toggleRecording",
    "updateFpsDisplay",
    "renderLatestFrame",
    "handleFrameReady",
    "image",
    "timestampMs",
    "handleFpsChanged",
    "fps",
    "handleQueueStatsChanged",
    "queueSize",
    "droppedFrames",
    "handleRecordFileSelected",
    "filename",
    "handleRecordFileCancelled"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CameraMainWindow[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   74,    2, 0x08,    1 /* Private */,
       3,    0,   75,    2, 0x08,    2 /* Private */,
       4,    0,   76,    2, 0x08,    3 /* Private */,
       5,    0,   77,    2, 0x08,    4 /* Private */,
       6,    0,   78,    2, 0x08,    5 /* Private */,
       7,    2,   79,    2, 0x08,    6 /* Private */,
      10,    1,   84,    2, 0x08,    9 /* Private */,
      12,    2,   87,    2, 0x08,   11 /* Private */,
      15,    1,   92,    2, 0x08,   14 /* Private */,
      17,    0,   95,    2, 0x08,   16 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QImage, QMetaType::LongLong,    8,    9,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void, QMetaType::ULongLong, QMetaType::ULongLong,   13,   14,
    QMetaType::Void, QMetaType::QString,   16,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject CameraMainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_CameraMainWindow.offsetsAndSizes,
    qt_meta_data_CameraMainWindow,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CameraMainWindow_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<CameraMainWindow, std::true_type>,
        // method 'openCamera'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'closeCamera'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'toggleRecording'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'updateFpsDisplay'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'renderLatestFrame'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'handleFrameReady'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QImage &, std::false_type>,
        QtPrivate::TypeAndForceComplete<qint64, std::false_type>,
        // method 'handleFpsChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<double, std::false_type>,
        // method 'handleQueueStatsChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong, std::false_type>,
        // method 'handleRecordFileSelected'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'handleRecordFileCancelled'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void CameraMainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<CameraMainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->openCamera(); break;
        case 1: _t->closeCamera(); break;
        case 2: _t->toggleRecording(); break;
        case 3: _t->updateFpsDisplay(); break;
        case 4: _t->renderLatestFrame(); break;
        case 5: _t->handleFrameReady((*reinterpret_cast< std::add_pointer_t<QImage>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<qint64>>(_a[2]))); break;
        case 6: _t->handleFpsChanged((*reinterpret_cast< std::add_pointer_t<double>>(_a[1]))); break;
        case 7: _t->handleQueueStatsChanged((*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[2]))); break;
        case 8: _t->handleRecordFileSelected((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 9: _t->handleRecordFileCancelled(); break;
        default: ;
        }
    }
}

const QMetaObject *CameraMainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CameraMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CameraMainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int CameraMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 10;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
