#include "CameraMainWindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("MindVision Camera Recorder"));
    app.setOrganizationName(QStringLiteral("MicroTools"));

    CameraMainWindow window;
    window.show();

    return app.exec();
}