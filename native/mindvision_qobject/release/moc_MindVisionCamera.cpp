/****************************************************************************
** Meta object code from reading C++ file 'MindVisionCamera.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../src/MindVisionCamera.h"
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MindVisionCamera.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_MindVisionCamera_t {
    uint offsetsAndSizes[128];
    char stringdata0[17];
    char stringdata1[11];
    char stringdata2[1];
    char stringdata3[6];
    char stringdata4[12];
    char stringdata5[18];
    char stringdata6[10];
    char stringdata7[14];
    char stringdata8[11];
    char stringdata9[4];
    char stringdata10[14];
    char stringdata11[8];
    char stringdata12[5];
    char stringdata13[6];
    char stringdata14[6];
    char stringdata15[5];
    char stringdata16[16];
    char stringdata17[8];
    char stringdata18[16];
    char stringdata19[15];
    char stringdata20[14];
    char stringdata21[5];
    char stringdata22[12];
    char stringdata23[7];
    char stringdata24[16];
    char stringdata25[16];
    char stringdata26[14];
    char stringdata27[12];
    char stringdata28[21];
    char stringdata29[8];
    char stringdata30[6];
    char stringdata31[6];
    char stringdata32[20];
    char stringdata33[19];
    char stringdata34[5];
    char stringdata35[4];
    char stringdata36[4];
    char stringdata37[22];
    char stringdata38[12];
    char stringdata39[9];
    char stringdata40[8];
    char stringdata41[8];
    char stringdata42[7];
    char stringdata43[7];
    char stringdata44[15];
    char stringdata45[5];
    char stringdata46[16];
    char stringdata47[6];
    char stringdata48[16];
    char stringdata49[9];
    char stringdata50[19];
    char stringdata51[12];
    char stringdata52[29];
    char stringdata53[5];
    char stringdata54[29];
    char stringdata55[8];
    char stringdata56[30];
    char stringdata57[14];
    char stringdata58[18];
    char stringdata59[9];
    char stringdata60[19];
    char stringdata61[20];
    char stringdata62[9];
    char stringdata63[16];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_MindVisionCamera_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_MindVisionCamera_t qt_meta_stringdata_MindVisionCamera = {
    {
        QT_MOC_LITERAL(0, 16),  // "MindVisionCamera"
        QT_MOC_LITERAL(17, 10),  // "frameReady"
        QT_MOC_LITERAL(28, 0),  // ""
        QT_MOC_LITERAL(29, 5),  // "image"
        QT_MOC_LITERAL(35, 11),  // "timestampMs"
        QT_MOC_LITERAL(47, 17),  // "queueStatsChanged"
        QT_MOC_LITERAL(65, 9),  // "queueSize"
        QT_MOC_LITERAL(75, 13),  // "droppedFrames"
        QT_MOC_LITERAL(89, 10),  // "fpsChanged"
        QT_MOC_LITERAL(100, 3),  // "fps"
        QT_MOC_LITERAL(104, 13),  // "errorOccurred"
        QT_MOC_LITERAL(118, 7),  // "message"
        QT_MOC_LITERAL(126, 4),  // "open"
        QT_MOC_LITERAL(131, 5),  // "close"
        QT_MOC_LITERAL(137, 5),  // "start"
        QT_MOC_LITERAL(143, 4),  // "stop"
        QT_MOC_LITERAL(148, 15),  // "setAutoExposure"
        QT_MOC_LITERAL(164, 7),  // "enabled"
        QT_MOC_LITERAL(172, 15),  // "setExposureTime"
        QT_MOC_LITERAL(188, 14),  // "exposureTimeMs"
        QT_MOC_LITERAL(203, 13),  // "setAnalogGain"
        QT_MOC_LITERAL(217, 4),  // "gain"
        QT_MOC_LITERAL(222, 11),  // "setAeTarget"
        QT_MOC_LITERAL(234, 6),  // "target"
        QT_MOC_LITERAL(241, 15),  // "getAutoExposure"
        QT_MOC_LITERAL(257, 15),  // "getExposureTime"
        QT_MOC_LITERAL(273, 13),  // "getAnalogGain"
        QT_MOC_LITERAL(287, 11),  // "getAeTarget"
        QT_MOC_LITERAL(299, 20),  // "getExposureTimeRange"
        QT_MOC_LITERAL(320, 7),  // "double&"
        QT_MOC_LITERAL(328, 5),  // "minMs"
        QT_MOC_LITERAL(334, 5),  // "maxMs"
        QT_MOC_LITERAL(340, 19),  // "getExposureTimeStep"
        QT_MOC_LITERAL(360, 18),  // "getAnalogGainRange"
        QT_MOC_LITERAL(379, 4),  // "int&"
        QT_MOC_LITERAL(384, 3),  // "min"
        QT_MOC_LITERAL(388, 3),  // "max"
        QT_MOC_LITERAL(392, 21),  // "getFrameCallbackStats"
        QT_MOC_LITERAL(414, 11),  // "qulonglong&"
        QT_MOC_LITERAL(426, 8),  // "received"
        QT_MOC_LITERAL(435, 7),  // "emitted"
        QT_MOC_LITERAL(443, 7),  // "dropped"
        QT_MOC_LITERAL(451, 6),  // "setRoi"
        QT_MOC_LITERAL(458, 6),  // "enable"
        QT_MOC_LITERAL(465, 14),  // "setTriggerMode"
        QT_MOC_LITERAL(480, 4),  // "mode"
        QT_MOC_LITERAL(485, 15),  // "setTriggerCount"
        QT_MOC_LITERAL(501, 5),  // "count"
        QT_MOC_LITERAL(507, 15),  // "setTriggerDelay"
        QT_MOC_LITERAL(523, 8),  // "delay_us"
        QT_MOC_LITERAL(532, 18),  // "setTriggerInterval"
        QT_MOC_LITERAL(551, 11),  // "interval_us"
        QT_MOC_LITERAL(563, 28),  // "setExternalTriggerSignalType"
        QT_MOC_LITERAL(592, 4),  // "type"
        QT_MOC_LITERAL(597, 28),  // "setExternalTriggerJitterTime"
        QT_MOC_LITERAL(626, 7),  // "time_us"
        QT_MOC_LITERAL(634, 29),  // "setExternalTriggerShutterMode"
        QT_MOC_LITERAL(664, 13),  // "setStrobeMode"
        QT_MOC_LITERAL(678, 17),  // "setStrobePolarity"
        QT_MOC_LITERAL(696, 8),  // "polarity"
        QT_MOC_LITERAL(705, 18),  // "setStrobeDelayTime"
        QT_MOC_LITERAL(724, 19),  // "setStrobePulseWidth"
        QT_MOC_LITERAL(744, 8),  // "width_us"
        QT_MOC_LITERAL(753, 15)   // "triggerSoftware"
    },
    "MindVisionCamera",
    "frameReady",
    "",
    "image",
    "timestampMs",
    "queueStatsChanged",
    "queueSize",
    "droppedFrames",
    "fpsChanged",
    "fps",
    "errorOccurred",
    "message",
    "open",
    "close",
    "start",
    "stop",
    "setAutoExposure",
    "enabled",
    "setExposureTime",
    "exposureTimeMs",
    "setAnalogGain",
    "gain",
    "setAeTarget",
    "target",
    "getAutoExposure",
    "getExposureTime",
    "getAnalogGain",
    "getAeTarget",
    "getExposureTimeRange",
    "double&",
    "minMs",
    "maxMs",
    "getExposureTimeStep",
    "getAnalogGainRange",
    "int&",
    "min",
    "max",
    "getFrameCallbackStats",
    "qulonglong&",
    "received",
    "emitted",
    "dropped",
    "setRoi",
    "enable",
    "setTriggerMode",
    "mode",
    "setTriggerCount",
    "count",
    "setTriggerDelay",
    "delay_us",
    "setTriggerInterval",
    "interval_us",
    "setExternalTriggerSignalType",
    "type",
    "setExternalTriggerJitterTime",
    "time_us",
    "setExternalTriggerShutterMode",
    "setStrobeMode",
    "setStrobePolarity",
    "polarity",
    "setStrobeDelayTime",
    "setStrobePulseWidth",
    "width_us",
    "triggerSoftware"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_MindVisionCamera[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      33,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    2,  212,    2, 0x06,    1 /* Public */,
       5,    2,  217,    2, 0x06,    4 /* Public */,
       8,    1,  222,    2, 0x06,    7 /* Public */,
      10,    1,  225,    2, 0x06,    9 /* Public */,

 // methods: name, argc, parameters, tag, flags, initial metatype offsets
      12,    0,  228,    2, 0x02,   11 /* Public */,
      13,    0,  229,    2, 0x02,   12 /* Public */,
      14,    0,  230,    2, 0x02,   13 /* Public */,
      15,    0,  231,    2, 0x02,   14 /* Public */,
      16,    1,  232,    2, 0x02,   15 /* Public */,
      18,    1,  235,    2, 0x02,   17 /* Public */,
      20,    1,  238,    2, 0x02,   19 /* Public */,
      22,    1,  241,    2, 0x02,   21 /* Public */,
      24,    0,  244,    2, 0x02,   23 /* Public */,
      25,    0,  245,    2, 0x02,   24 /* Public */,
      26,    0,  246,    2, 0x02,   25 /* Public */,
      27,    0,  247,    2, 0x02,   26 /* Public */,
      28,    2,  248,    2, 0x02,   27 /* Public */,
      32,    0,  253,    2, 0x02,   30 /* Public */,
      33,    2,  254,    2, 0x02,   31 /* Public */,
      37,    3,  259,    2, 0x02,   34 /* Public */,
      42,    1,  266,    2, 0x02,   38 /* Public */,
      44,    1,  269,    2, 0x02,   40 /* Public */,
      46,    1,  272,    2, 0x02,   42 /* Public */,
      48,    1,  275,    2, 0x02,   44 /* Public */,
      50,    1,  278,    2, 0x02,   46 /* Public */,
      52,    1,  281,    2, 0x02,   48 /* Public */,
      54,    1,  284,    2, 0x02,   50 /* Public */,
      56,    1,  287,    2, 0x02,   52 /* Public */,
      57,    1,  290,    2, 0x02,   54 /* Public */,
      58,    1,  293,    2, 0x02,   56 /* Public */,
      60,    1,  296,    2, 0x02,   58 /* Public */,
      61,    1,  299,    2, 0x02,   60 /* Public */,
      63,    0,  302,    2, 0x02,   62 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QImage, QMetaType::LongLong,    3,    4,
    QMetaType::Void, QMetaType::ULongLong, QMetaType::ULongLong,    6,    7,
    QMetaType::Void, QMetaType::Double,    9,
    QMetaType::Void, QMetaType::QString,   11,

 // methods: parameters
    QMetaType::Bool,
    QMetaType::Void,
    QMetaType::Bool,
    QMetaType::Void,
    QMetaType::Bool, QMetaType::Bool,   17,
    QMetaType::Bool, QMetaType::Double,   19,
    QMetaType::Bool, QMetaType::Int,   21,
    QMetaType::Bool, QMetaType::Int,   23,
    QMetaType::Bool,
    QMetaType::Double,
    QMetaType::Int,
    QMetaType::Int,
    QMetaType::Void, 0x80000000 | 29, 0x80000000 | 29,   30,   31,
    QMetaType::Double,
    QMetaType::Void, 0x80000000 | 34, 0x80000000 | 34,   35,   36,
    QMetaType::Void, 0x80000000 | 38, 0x80000000 | 38, 0x80000000 | 38,   39,   40,   41,
    QMetaType::Bool, QMetaType::Bool,   43,
    QMetaType::Bool, QMetaType::Int,   45,
    QMetaType::Bool, QMetaType::Int,   47,
    QMetaType::Bool, QMetaType::Int,   49,
    QMetaType::Bool, QMetaType::Int,   51,
    QMetaType::Bool, QMetaType::Int,   53,
    QMetaType::Bool, QMetaType::Int,   55,
    QMetaType::Bool, QMetaType::Int,   45,
    QMetaType::Bool, QMetaType::Int,   45,
    QMetaType::Bool, QMetaType::Int,   59,
    QMetaType::Bool, QMetaType::Int,   49,
    QMetaType::Bool, QMetaType::Int,   62,
    QMetaType::Bool,

       0        // eod
};

Q_CONSTINIT const QMetaObject MindVisionCamera::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_MindVisionCamera.offsetsAndSizes,
    qt_meta_data_MindVisionCamera,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_MindVisionCamera_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MindVisionCamera, std::true_type>,
        // method 'frameReady'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QImage, std::false_type>,
        QtPrivate::TypeAndForceComplete<qint64, std::false_type>,
        // method 'queueStatsChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong, std::false_type>,
        // method 'fpsChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<double, std::false_type>,
        // method 'errorOccurred'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QString, std::false_type>,
        // method 'open'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'close'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'start'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'stop'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'setAutoExposure'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'setExposureTime'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<double, std::false_type>,
        // method 'setAnalogGain'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setAeTarget'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'getAutoExposure'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'getExposureTime'
        QtPrivate::TypeAndForceComplete<double, std::false_type>,
        // method 'getAnalogGain'
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'getAeTarget'
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'getExposureTimeRange'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<double &, std::false_type>,
        QtPrivate::TypeAndForceComplete<double &, std::false_type>,
        // method 'getExposureTimeStep'
        QtPrivate::TypeAndForceComplete<double, std::false_type>,
        // method 'getAnalogGainRange'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int &, std::false_type>,
        QtPrivate::TypeAndForceComplete<int &, std::false_type>,
        // method 'getFrameCallbackStats'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong &, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong &, std::false_type>,
        QtPrivate::TypeAndForceComplete<qulonglong &, std::false_type>,
        // method 'setRoi'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'setTriggerMode'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setTriggerCount'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setTriggerDelay'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setTriggerInterval'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setExternalTriggerSignalType'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setExternalTriggerJitterTime'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setExternalTriggerShutterMode'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setStrobeMode'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setStrobePolarity'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setStrobeDelayTime'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setStrobePulseWidth'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'triggerSoftware'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>
    >,
    nullptr
} };

void MindVisionCamera::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MindVisionCamera *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->frameReady((*reinterpret_cast< std::add_pointer_t<QImage>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<qint64>>(_a[2]))); break;
        case 1: _t->queueStatsChanged((*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<qulonglong>>(_a[2]))); break;
        case 2: _t->fpsChanged((*reinterpret_cast< std::add_pointer_t<double>>(_a[1]))); break;
        case 3: _t->errorOccurred((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 4: { bool _r = _t->open();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 5: _t->close(); break;
        case 6: { bool _r = _t->start();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 7: _t->stop(); break;
        case 8: { bool _r = _t->setAutoExposure((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 9: { bool _r = _t->setExposureTime((*reinterpret_cast< std::add_pointer_t<double>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 10: { bool _r = _t->setAnalogGain((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 11: { bool _r = _t->setAeTarget((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 12: { bool _r = _t->getAutoExposure();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 13: { double _r = _t->getExposureTime();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 14: { int _r = _t->getAnalogGain();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 15: { int _r = _t->getAeTarget();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 16: _t->getExposureTimeRange((*reinterpret_cast< std::add_pointer_t<double&>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<double&>>(_a[2]))); break;
        case 17: { double _r = _t->getExposureTimeStep();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 18: _t->getAnalogGainRange((*reinterpret_cast< std::add_pointer_t<int&>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<int&>>(_a[2]))); break;
        case 19: _t->getFrameCallbackStats((*reinterpret_cast< std::add_pointer_t<qulonglong&>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<qulonglong&>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<qulonglong&>>(_a[3]))); break;
        case 20: { bool _r = _t->setRoi((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 21: { bool _r = _t->setTriggerMode((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 22: { bool _r = _t->setTriggerCount((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 23: { bool _r = _t->setTriggerDelay((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 24: { bool _r = _t->setTriggerInterval((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 25: { bool _r = _t->setExternalTriggerSignalType((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 26: { bool _r = _t->setExternalTriggerJitterTime((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 27: { bool _r = _t->setExternalTriggerShutterMode((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 28: { bool _r = _t->setStrobeMode((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 29: { bool _r = _t->setStrobePolarity((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 30: { bool _r = _t->setStrobeDelayTime((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 31: { bool _r = _t->setStrobePulseWidth((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 32: { bool _r = _t->triggerSoftware();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MindVisionCamera::*)(QImage , qint64 );
            if (_t _q_method = &MindVisionCamera::frameReady; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (MindVisionCamera::*)(qulonglong , qulonglong );
            if (_t _q_method = &MindVisionCamera::queueStatsChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (MindVisionCamera::*)(double );
            if (_t _q_method = &MindVisionCamera::fpsChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (MindVisionCamera::*)(QString );
            if (_t _q_method = &MindVisionCamera::errorOccurred; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 3;
                return;
            }
        }
    }
}

const QMetaObject *MindVisionCamera::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MindVisionCamera::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MindVisionCamera.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int MindVisionCamera::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 33)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 33;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 33)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 33;
    }
    return _id;
}

// SIGNAL 0
void MindVisionCamera::frameReady(QImage _t1, qint64 _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MindVisionCamera::queueStatsChanged(qulonglong _t1, qulonglong _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MindVisionCamera::fpsChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void MindVisionCamera::errorOccurred(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
