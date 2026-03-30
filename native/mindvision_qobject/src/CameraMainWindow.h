#ifndef CAMERAMAINWINDOW_H
#define CAMERAMAINWINDOW_H

#include <QImage>
#include <QMainWindow>
#include <QPointer>
#include <QSize>

class QFileDialog;
class QLabel;
class QPushButton;
class QStatusBar;
class QTimer;

class MindVisionCamera;
class VideoThread;

class CameraMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit CameraMainWindow(QWidget *parent = nullptr);
    ~CameraMainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void openCamera();
    void closeCamera();
    void toggleRecording();
    void updateFpsDisplay();
    void renderLatestFrame();
    void handleFrameReady(const QImage &image, qint64 timestampMs);
    void handleFpsChanged(double fps);
    void handleQueueStatsChanged(qulonglong queueSize, qulonglong droppedFrames);
    void handleRecordFileSelected(const QString &filename);
    void handleRecordFileCancelled();

private:
    void initUi();
    void resetPreview();

    MindVisionCamera *m_camera;
    VideoThread *m_videoThread;
    QLabel *m_cameraLabel;
    QPushButton *m_openButton;
    QPushButton *m_recordButton;
    QPushButton *m_closeButton;
    QStatusBar *m_statusBar;
    QTimer *m_fpsTimer;
    QTimer *m_renderTimer;
    QPointer<QFileDialog> m_recordFileDialog;
    bool m_isRecording;
    double m_currentFps;
    QSize m_lastFrameSize;
    qulonglong m_lastQueueSize;
    qulonglong m_lastDroppedFrames;
    qulonglong m_frameCount;
    QImage m_latestFrame;
    quint64 m_latestFrameSeq;
    quint64 m_renderedFrameSeq;
};

#endif