#ifndef SERIALWORKER_H
#define SERIALWORKER_H

#include <QObject>
#include <QSerialPort>
#include <QString>
#include <QByteArray>
#include <functional>
#include <string>

class SerialWorker : public QObject {
    Q_OBJECT
public:
    explicit SerialWorker(QObject *parent = nullptr);
    ~SerialWorker() override;

    void register_log_callback(std::function<void(std::string)> cb);
    void register_status_callback(std::function<void(bool)> cb);

    void connect_serial(const std::string& port_name, int baud_rate);
    void disconnect_serial();
    void send_command(const std::string& cmd);
    void send_raw_command(const std::string& cmd);
    void poll_serial();

signals:
    void log_signal(const QString& msg);
    void connection_status(bool connected);

private slots:
    void handleReadyRead();
    void handleError(QSerialPort::SerialPortError error);

private:
    QSerialPort m_serial;
    std::function<void(std::string)> m_log_cb;
    std::function<void(bool)> m_status_cb;
};

#endif // SERIALWORKER_H