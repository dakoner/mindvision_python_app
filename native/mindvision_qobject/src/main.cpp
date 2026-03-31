#include "CameraMainWindow.h"

#include <QApplication>
#include <QScreen>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("MindVision Camera Recorder"));
    app.setOrganizationName(QStringLiteral("MicroTools"));

    CameraMainWindow window;

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