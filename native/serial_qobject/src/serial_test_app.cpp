#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QPushButton>
#include <QTextEdit>
#include <QLineEdit>
#include <QSerialPortInfo>
#include "SerialWorker.h"

class SerialTestApp : public QMainWindow {
    Q_OBJECT
public:
    SerialTestApp(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Serial Worker Test App");
        
        QWidget *centralWidget = new QWidget(this);
        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

        // --- Top Bar: Controls ---
        QHBoxLayout *topLayout = new QHBoxLayout();
        portComboBox = new QComboBox();
        
        // Populate available ports
        const auto ports = QSerialPortInfo::availablePorts();
        for (const QSerialPortInfo &info : ports) {
            portComboBox->addItem(info.portName());
        }

        baudComboBox = new QComboBox();
        baudComboBox->addItem("9600", 9600);
        baudComboBox->addItem("19200", 19200);
        baudComboBox->addItem("38400", 38400);
        baudComboBox->addItem("57600", 57600);
        baudComboBox->addItem("115200", 115200);
        baudComboBox->setCurrentText("115200");

        connectButton = new QPushButton("Connect");

        topLayout->addWidget(portComboBox);
        topLayout->addWidget(baudComboBox);
        topLayout->addWidget(connectButton);

        // --- Middle: Console Output ---
        console = new QTextEdit();
        console->setReadOnly(true);
        console->setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;");

        // --- Bottom Bar: Command Input ---
        QHBoxLayout *bottomLayout = new QHBoxLayout();
        input = new QLineEdit();
        sendButton = new QPushButton("Send");
        bottomLayout->addWidget(input);
        bottomLayout->addWidget(sendButton);

        mainLayout->addLayout(topLayout);
        mainLayout->addWidget(console);
        mainLayout->addLayout(bottomLayout);

        setCentralWidget(centralWidget);
        resize(600, 400);

        // --- Initialize Worker & Callbacks ---
        worker = new SerialWorker(this);

        worker->register_log_callback([this](std::string msg) {
            QMetaObject::invokeMethod(this, [this, msg]() {
                console->append(QString::fromStdString(msg));
            });
        });

        worker->register_status_callback([this](bool connected) {
            QMetaObject::invokeMethod(this, [this, connected]() {
                isConnected = connected;
                connectButton->setText(connected ? "Disconnect" : "Connect");
                portComboBox->setEnabled(!connected);
                baudComboBox->setEnabled(!connected);
            });
        });

        connect(connectButton, &QPushButton::clicked, this, [this]() {
            if (isConnected) {
                worker->disconnect_serial();
            } else {
                worker->connect_serial(portComboBox->currentText().toStdString(),
                                       baudComboBox->currentData().toInt());
            }
        });

        connect(sendButton, &QPushButton::clicked, this, &SerialTestApp::sendCommand);
        connect(input, &QLineEdit::returnPressed, this, &SerialTestApp::sendCommand);
    }

private:
    void sendCommand() {
        QString text = input->text();
        if (!text.isEmpty()) {
            // Worker automatically appends \n in send_command
            worker->send_command(text.toStdString());
            input->clear();
        }
    }

    QComboBox *portComboBox;
    QComboBox *baudComboBox;
    QPushButton *connectButton;
    QTextEdit *console;
    QLineEdit *input;
    QPushButton *sendButton;
    SerialWorker *worker;
    bool isConnected = false;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    SerialTestApp win;
    win.show();
    return app.exec();
}

#include "serial_test_app.moc"