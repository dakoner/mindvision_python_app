#include <pybind11/pybind11.h>
#include <pybind11/functional.h> 

#include "SerialWorker.h"
#include <QObject>
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

class PySerialWorker : public SerialWorker {
public:
    using SerialWorker::SerialWorker;

    void registerLogCallback(py::function callback) {
        m_logCallback = callback;
        if (m_logCallbackConnection) {
            QObject::disconnect(m_logCallbackConnection);
            m_logCallbackConnection = QMetaObject::Connection();
        }
        
        m_logCallbackConnection = QObject::connect(this, &SerialWorker::log_signal, this, [this](const QString& msg) {
            if (m_logCallback) {
                py::gil_scoped_acquire acquire;
                try {
                    m_logCallback(msg);
                } catch (py::error_already_set &e) {
                    qDebug() << "Python error in log callback:" << e.what();
                }
            }
        });
    }

    void registerStatusCallback(py::function callback) {
        m_statusCallback = callback;
        if (m_statusCallbackConnection) {
            QObject::disconnect(m_statusCallbackConnection);
            m_statusCallbackConnection = QMetaObject::Connection();
        }
        
        m_statusCallbackConnection = QObject::connect(this, &SerialWorker::connection_status, this, [this](bool connected) {
            if (m_statusCallback) {
                py::gil_scoped_acquire acquire;
                try {
                    m_statusCallback(connected);
                } catch (py::error_already_set &e) {
                    qDebug() << "Python error in status callback:" << e.what();
                }
            }
        });
    }

private:
    py::function m_logCallback;
    QMetaObject::Connection m_logCallbackConnection;

    py::function m_statusCallback;
    QMetaObject::Connection m_statusCallbackConnection;
};

PYBIND11_MODULE(_serial_qobject_py, m) {
    m.doc() = "Native QSerialPort worker module for mindvision_python_app";

    py::class_<PySerialWorker>(m, "SerialWorker")
        .def(py::init<>())
        .def("connect_serial", &PySerialWorker::connect_serial, py::arg("port_name"), py::arg("baud_rate") = 115200)
        .def("disconnect_serial", &PySerialWorker::disconnect_serial)
        .def("send_command", &PySerialWorker::send_command, py::arg("cmd"))
        .def("send_raw_command", &PySerialWorker::send_raw_command, py::arg("cmd"))
        .def("poll_serial", &PySerialWorker::poll_serial)
        .def("register_log_callback", &PySerialWorker::registerLogCallback)
        .def("register_status_callback", &PySerialWorker::registerStatusCallback);
}
