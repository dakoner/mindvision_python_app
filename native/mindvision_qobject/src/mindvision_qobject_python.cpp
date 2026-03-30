#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h> 

#include "MindVisionCamera.h"
#include "VideoThread.h"
#include <QObject>
#include <QImage>
#include <QBuffer>
#include <QDebug>

namespace py = pybind11;

namespace pybind11 { namespace detail {
    template <> struct type_caster<QString> {
    public:
        PYBIND11_TYPE_CASTER(QString, _("str"));
        bool load(handle src, bool) {
            if (!src) return false;
            PyObject *source = src.ptr();
            if (PyUnicode_Check(source)) {
                 Py_ssize_t size;
                 const char *ptr = PyUnicode_AsUTF8AndSize(source, &size);
                 if (!ptr) return false;
                 value = QString::fromUtf8(ptr, (int)size);
                 return true;
            }
            return false;
        }
        static handle cast(QString const &src, return_value_policy /* policy */, handle /* parent */) {
            QByteArray utf8 = src.toUtf8();
            return PyUnicode_FromStringAndSize(utf8.data(), utf8.size());
        }
    };
}}

// Wrapper for MindVisionCamera to support callbacks
class PyMindVisionCamera : public MindVisionCamera {
public:
    using MindVisionCamera::MindVisionCamera;

    void registerFrameViewCallback(py::function callback) {
        m_frameViewCallback = callback;
        if (m_frameViewCallbackConnection) {
            QObject::disconnect(m_frameViewCallbackConnection);
            m_frameViewCallbackConnection = QMetaObject::Connection();
        }

        m_frameViewCallbackConnection = connect(this, &MindVisionCamera::frameReady, this, [this](QImage img, qint64 timestampMs) {
            if (m_frameViewCallback) {
                py::gil_scoped_acquire acquire;
                try {
                    auto frameView = py::memoryview::from_memory(
                        reinterpret_cast<const char*>(img.bits()),
                        static_cast<py::ssize_t>(img.sizeInBytes()));
                    m_frameViewCallback(img.width(), img.height(), img.bytesPerLine(), static_cast<int>(img.format()), frameView, timestampMs);
                } catch (py::error_already_set &e) {
                    qDebug() << "Python error in frame view callback:" << e.what();
                }
            }
        });
    }

    void registerFrameCallback(py::function callback) {
        m_frameCallback = callback;
        if (m_frameCallbackConnection) {
            QObject::disconnect(m_frameCallbackConnection);
            m_frameCallbackConnection = QMetaObject::Connection();
        }
        
        m_frameCallbackConnection = connect(this, &MindVisionCamera::frameReady, this, [this](QImage img, qint64 timestampMs) {
            if (m_frameCallback) {
                py::gil_scoped_acquire acquire;
                try {
                    // Extract data: width, height, bytesPerLine, format, data
                    py::bytes data((const char*)img.bits(), img.sizeInBytes());
                    m_frameCallback(img.width(), img.height(), img.bytesPerLine(), (int)img.format(), data, timestampMs);
                } catch (py::error_already_set &e) {
                    qDebug() << "Python error in frame callback:" << e.what();
                }
            }
        });
    }

    void registerFpsCallback(py::function callback) {
        m_fpsCallback = callback;
        if (m_fpsCallbackConnection) {
            QObject::disconnect(m_fpsCallbackConnection);
            m_fpsCallbackConnection = QMetaObject::Connection();
        }
        
        m_fpsCallbackConnection = connect(this, &MindVisionCamera::fpsChanged, this, [this](double fps) {
             py::gil_scoped_acquire acquire;
             if (m_fpsCallback) {
                 try {
                     m_fpsCallback(fps);
                 } catch (py::error_already_set &e) {
                     qDebug() << "Python error in fps callback:" << e.what();
                 }
             }
        });
    }

    void registerQueueStatsCallback(py::function callback) {
        m_queueStatsCallback = callback;
        if (m_queueStatsCallbackConnection) {
            QObject::disconnect(m_queueStatsCallbackConnection);
            m_queueStatsCallbackConnection = QMetaObject::Connection();
        }

        m_queueStatsCallbackConnection = connect(this, &MindVisionCamera::queueStatsChanged, this, [this](qulonglong queueSize, qulonglong droppedFrames) {
            py::gil_scoped_acquire acquire;
            if (m_queueStatsCallback) {
                try {
                    m_queueStatsCallback(queueSize, droppedFrames);
                } catch (py::error_already_set &e) {
                    qDebug() << "Python error in queue stats callback:" << e.what();
                }
            }
        });
    }
    
    void registerErrorCallback(py::function callback) {
        m_errorCallback = callback;
        if (m_errorCallbackConnection) {
            QObject::disconnect(m_errorCallbackConnection);
            m_errorCallbackConnection = QMetaObject::Connection();
        }
        
        m_errorCallbackConnection = connect(this, &MindVisionCamera::errorOccurred, this, [this](QString msg) {
             py::gil_scoped_acquire acquire;
             if (m_errorCallback) {
                 try {
                     m_errorCallback(msg);
                 } catch (py::error_already_set &e) {
                     qDebug() << "Python error in error callback:" << e.what();
                 }
             }
        });
    }

private:
    py::function m_frameCallback;
    py::function m_frameViewCallback;
    py::function m_fpsCallback;
    py::function m_queueStatsCallback;
    py::function m_errorCallback;
    QMetaObject::Connection m_frameCallbackConnection;
    QMetaObject::Connection m_frameViewCallbackConnection;
    QMetaObject::Connection m_fpsCallbackConnection;
    QMetaObject::Connection m_queueStatsCallbackConnection;
    QMetaObject::Connection m_errorCallbackConnection;
};

// Wrapper for VideoThread
class PyVideoThread : public VideoThread {
public:
    using VideoThread::VideoThread;

        void setFrameSource(PyMindVisionCamera &camera) {
            VideoThread::setFrameSource(&camera);
        }

        void clearFrameSource() {
            VideoThread::clearFrameSource();
        }
    
    void addFrameBytes(int width, int height, int bytesPerLine, int format, py::bytes data) {
         char *buffer = nullptr;
         Py_ssize_t size = 0;
         if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &size) != 0 || buffer == nullptr) {
             throw py::error_already_set();
         }

         const qsizetype expectedSize = static_cast<qsizetype>(bytesPerLine) * static_cast<qsizetype>(height);
         if (size < expectedSize) {
             throw std::runtime_error("Frame buffer is smaller than expected image size");
         }

         QImage img(reinterpret_cast<const uchar*>(buffer), width, height, bytesPerLine, static_cast<QImage::Format>(format));
         addFrame(img.copy());
    }
};

PYBIND11_MODULE(_mindvision_qobject_py, m) {
    m.doc() = "pybind11 wrapper for MindVision QObject library"; 

    py::class_<QObject>(m, "QObject");

    // Bind PyMindVisionCamera but expose as MindVisionCamera
    py::class_<PyMindVisionCamera, QObject>(m, "MindVisionCamera")
        .def(py::init<QObject *>(), py::arg("parent") = nullptr)
        .def("open", &MindVisionCamera::open)
        .def("close", &MindVisionCamera::close)
        .def("start", &MindVisionCamera::start)
        .def("stop", &MindVisionCamera::stop)
        .def("setAutoExposure", &MindVisionCamera::setAutoExposure)
        .def("setExposureTime", &MindVisionCamera::setExposureTime)
        .def("setAnalogGain", &MindVisionCamera::setAnalogGain)
        .def("setAeTarget", &MindVisionCamera::setAeTarget)
        .def("getAutoExposure", &MindVisionCamera::getAutoExposure)
        .def("getExposureTime", &MindVisionCamera::getExposureTime)
        .def("getAnalogGain", &MindVisionCamera::getAnalogGain)
        .def("getAeTarget", &MindVisionCamera::getAeTarget)
        .def("getExposureTimeStep", &MindVisionCamera::getExposureTimeStep)
        .def("getExposureTimeRange", [](PyMindVisionCamera &self) {
            double minMs, maxMs;
            self.getExposureTimeRange(minMs, maxMs);
            return py::make_tuple(minMs, maxMs);
        })
        .def("getFrameCallbackStats", [](PyMindVisionCamera &self) {
            qulonglong received = 0;
            qulonglong emitted = 0;
            qulonglong dropped = 0;
            self.getFrameCallbackStats(received, emitted, dropped);
            return py::make_tuple(received, emitted, dropped);
        })
        .def("getAnalogGainRange", [](PyMindVisionCamera &self) {
            int min, max;
            self.getAnalogGainRange(min, max);
            return py::make_tuple(min, max);
        })
        .def("setRoi", &MindVisionCamera::setRoi)
        .def("setTriggerMode", &MindVisionCamera::setTriggerMode)
        .def("setTriggerCount", &MindVisionCamera::setTriggerCount)
        .def("setTriggerDelay", &MindVisionCamera::setTriggerDelay)
        .def("setTriggerInterval", &MindVisionCamera::setTriggerInterval)
        .def("setExternalTriggerSignalType", &MindVisionCamera::setExternalTriggerSignalType)
        .def("setExternalTriggerJitterTime", &MindVisionCamera::setExternalTriggerJitterTime)
        .def("setExternalTriggerShutterMode", &MindVisionCamera::setExternalTriggerShutterMode)
        .def("setStrobeMode", &MindVisionCamera::setStrobeMode)
        .def("setStrobePolarity", &MindVisionCamera::setStrobePolarity)
        .def("setStrobeDelayTime", &MindVisionCamera::setStrobeDelayTime)
        .def("setStrobePulseWidth", &MindVisionCamera::setStrobePulseWidth)
        .def("triggerSoftware", &MindVisionCamera::triggerSoftware)
        .def("registerFrameViewCallback", &PyMindVisionCamera::registerFrameViewCallback)
        .def("registerFrameCallback", &PyMindVisionCamera::registerFrameCallback)
        .def("registerFpsCallback", &PyMindVisionCamera::registerFpsCallback)
        .def("registerQueueStatsCallback", &PyMindVisionCamera::registerQueueStatsCallback)
        .def("registerErrorCallback", &PyMindVisionCamera::registerErrorCallback)
        ;

    py::class_<PyVideoThread, QObject>(m, "VideoThread")
        .def(py::init<QObject *>(), py::arg("parent") = nullptr)
        .def("startRecording", &VideoThread::startRecording,
             py::arg("width"), py::arg("height"), py::arg("fps"), py::arg("filename"))
        .def("stopRecording", &VideoThread::stopRecording)
        .def("setFrameSource", &PyVideoThread::setFrameSource)
        .def("clearFrameSource", &PyVideoThread::clearFrameSource)
        .def("addFrameBytes", &PyVideoThread::addFrameBytes)
        .def("isRunning", &VideoThread::isRunning)
        ;
}
