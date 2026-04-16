#include "CameraMainWindow.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QScreen>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("MindVision Camera Recorder"));
    app.setOrganizationName(QStringLiteral("MicroTools"));

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("MindVision Camera Recorder"));
    parser.addHelpOption();
    const QCommandLineOption startRecordingOption(
        QStringLiteral("start-recording"),
        QStringLiteral("Automatically begin recording when the application starts."));
    parser.addOption(startRecordingOption);
    parser.process(app);

    const bool autoStartRecording = parser.isSet(startRecordingOption);

    CameraMainWindow window(autoStartRecording);

    const auto applyKioskGeometry = [&window]() {
        QScreen *screen = window.screen();
        if (screen == nullptr) {
            screen = QGuiApplication::primaryScreen();
        }
        if (screen == nullptr) {
            return;
        }

        const QRect screenRect = screen->geometry();
        window.setGeometry(screenRect);
        window.showFullScreen();
        window.raise();
        window.activateWindow();
    };

    window.show();
    QTimer::singleShot(0, &window, applyKioskGeometry);
    QTimer::singleShot(250, &window, applyKioskGeometry);

    return app.exec();
}