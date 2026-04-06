#include "SerialWorker.h"

SerialWorker::SerialWorker(QObject *parent)
    : QObject(parent)
{
    connect(&m_serial, &QSerialPort::readyRead, this, &SerialWorker::handleReadyRead);
    connect(&m_serial, &QSerialPort::errorOccurred, this, &SerialWorker::handleError);
}

SerialWorker::~SerialWorker()
{
    disconnect_serial();
}

void SerialWorker::register_log_callback(std::function<void(std::string)> cb)
{
    m_log_cb = std::move(cb);
}

void SerialWorker::register_status_callback(std::function<void(bool)> cb)
{
    m_status_cb = std::move(cb);
}

void SerialWorker::connect_serial(const std::string& port_name, int baud_rate)
{
    if (m_serial.isOpen()) {
        m_serial.close();
    }
    
    m_serial.setPortName(QString::fromStdString(port_name));
    m_serial.setBaudRate(baud_rate);
    m_serial.setDataBits(QSerialPort::Data8);
    m_serial.setParity(QSerialPort::NoParity);
    m_serial.setStopBits(QSerialPort::OneStop);
    m_serial.setFlowControl(QSerialPort::NoFlowControl);

    if (m_serial.open(QIODevice::ReadWrite)) {
        if (m_log_cb) {
            m_log_cb("Connected to " + port_name + " at " + std::to_string(baud_rate) + " baud.");
        }
        emit log_signal(QString::fromStdString("Connected to " + port_name + " at " + std::to_string(baud_rate) + " baud."));
        if (m_status_cb) {
            m_status_cb(true);
        }
        emit connection_status(true);
    } else {
        if (m_log_cb) {
            m_log_cb("Failed to connect to " + port_name + ": " + m_serial.errorString().toStdString());
        }
        emit log_signal("Failed to connect to " + QString::fromStdString(port_name) + ": " + m_serial.errorString());
        if (m_status_cb) {
            m_status_cb(false);
        }
        emit connection_status(false);
    }
}

void SerialWorker::disconnect_serial()
{
    if (m_serial.isOpen()) {
        m_serial.close();
        if (m_log_cb) {
            m_log_cb("Disconnected from serial port.");
        }
        emit log_signal("Disconnected from serial port.");
        if (m_status_cb) {
            m_status_cb(false);
        }
        emit connection_status(false);
    }
}

void SerialWorker::send_command(const std::string& cmd)
{
    if (m_serial.isOpen()) {
        std::string full_cmd = cmd + "\n";
        m_serial.write(full_cmd.c_str(), full_cmd.size());
        if (m_log_cb) {
            m_log_cb("> " + cmd);
        }
        emit log_signal("> " + QString::fromStdString(cmd));
    } else {
        if (m_log_cb) {
            m_log_cb("Error: Serial port not open.");
        }
        emit log_signal("Error: Serial port not open.");
    }
}

void SerialWorker::send_raw_command(const std::string& cmd)
{
    if (m_serial.isOpen()) {
        m_serial.write(cmd.c_str(), cmd.size());
    }
}

void SerialWorker::poll_serial()
{
    // Intentionally left blank. QSerialPort operates asynchronously via the Qt event loop natively.
}

void SerialWorker::handleReadyRead()
{
    while (m_serial.canReadLine()) {
        QByteArray line = m_serial.readLine();
        if (m_log_cb) {
            QString s = QString::fromLatin1(line).trimmed();
            if (!s.isEmpty()) {
                m_log_cb("Rx: " + s.toStdString());
            }
        }
        emit log_signal("Rx: " + QString::fromLatin1(line).trimmed());
    }
}

void SerialWorker::handleError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::ResourceError) {
        if (m_log_cb) {
            m_log_cb("Serial port resource error: " + m_serial.errorString().toStdString());
        }
        emit log_signal("Serial port resource error: " + m_serial.errorString());
        disconnect_serial();
    }
}