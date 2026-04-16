#include "CameraMainWindow.h"
#include "MindVisionCamera.h"
#include "VideoThread.h"

#include <QCloseEvent>
#include <QDateTime>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QResizeEvent>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

CameraMainWindow::CameraMainWindow(bool autoStartRecording, QWidget *parent)
    : QMainWindow(parent),
      m_camera(new MindVisionCamera(this)),
      m_videoThread(new VideoThread(this)),
      m_cameraLabel(nullptr),
      m_recordButton(nullptr),
      m_statusBar(nullptr),
      m_fpsTimer(new QTimer(this)),
      m_renderTimer(new QTimer(this)),
      m_isRecording(false),
      m_autoStartRecording(autoStartRecording),
      m_currentFps(0.0),
      m_lastQueueSize(0),
      m_lastDroppedFrames(0),
      m_frameCount(0),
      m_latestFrameSeq(0),
      m_renderedFrameSeq(0)
{
    initUi();
    setWindowFlag(Qt::FramelessWindowHint, true);

    connect(m_recordButton, &QPushButton::clicked, this, &CameraMainWindow::toggleRecording);

    connect(m_fpsTimer, &QTimer::timeout, this, &CameraMainWindow::updateFpsDisplay);
    connect(m_renderTimer, &QTimer::timeout, this, &CameraMainWindow::renderLatestFrame);

    connect(m_camera, &MindVisionCamera::frameReady, this, &CameraMainWindow::handleFrameReady);
    connect(m_camera, &MindVisionCamera::fpsChanged, this, &CameraMainWindow::handleFpsChanged);
    connect(m_camera, &MindVisionCamera::queueStatsChanged, this, &CameraMainWindow::handleQueueStatsChanged);

    m_videoThread->setFrameSource(m_camera);

    m_fpsTimer->start(100);
    m_renderTimer->start(33);

    QTimer::singleShot(0, this, &CameraMainWindow::openCamera);
}

CameraMainWindow::~CameraMainWindow()
{
    closeCamera();
}

void CameraMainWindow::initUi()
{
    setWindowTitle(tr("MindVision Camera - Video Recorder"));
    resize(1280, 900);

    QWidget *centralWidget = new QWidget(this);
    auto *mainLayout = new QVBoxLayout(centralWidget);
    auto *buttonLayout = new QHBoxLayout();

    m_cameraLabel = new QLabel(tr("Camera feed will appear here"), centralWidget);
    m_cameraLabel->setAlignment(Qt::AlignCenter);
    m_cameraLabel->setStyleSheet(QStringLiteral("QLabel { background-color: black; color: white; }"));
    m_cameraLabel->setMinimumSize(100, 100);
    mainLayout->addWidget(m_cameraLabel, 1);

    m_recordButton = new QPushButton(tr("Start Recording"), centralWidget);
    m_recordButton->setEnabled(false);
    buttonLayout->addWidget(m_recordButton);

    mainLayout->addLayout(buttonLayout);
    setCentralWidget(centralWidget);

    m_statusBar = new QStatusBar(this);
    setStatusBar(m_statusBar);
    m_statusBar->showMessage(tr("Opening camera..."));
}

void CameraMainWindow::openCamera()
{
    if (m_recordButton->isEnabled()) {
        return;
    }

    if (!m_camera->open()) {
        QMessageBox::critical(this, tr("Error"), tr("Failed to open camera. Check if camera is connected."));
        m_statusBar->showMessage(tr("Failed to open camera."));
        return;
    }

    if (!m_camera->start()) {
        QMessageBox::critical(this, tr("Error"), tr("Failed to start camera capture."));
        m_camera->close();
        m_statusBar->showMessage(tr("Failed to start camera capture."));
        return;
    }

    m_recordButton->setEnabled(true);
    m_statusBar->showMessage(tr("Camera opened and capturing frames."));
}

void CameraMainWindow::closeCamera()
{
    if (m_isRecording) {
        toggleRecording();
    }

    if (m_recordFileDialog != nullptr) {
        m_recordFileDialog->close();
    }

    m_camera->stop();
    m_camera->close();
    resetPreview();

    m_recordButton->setEnabled(false);
    m_statusBar->showMessage(tr("Camera closed."));
}

void CameraMainWindow::toggleRecording()
{
    if (!m_isRecording) {
        if (!m_lastFrameSize.isValid()) {
            QMessageBox::warning(this, tr("No Frame"), tr("Wait for the first camera frame before starting recording."));
            return;
        }

        if (m_recordFileDialog != nullptr && m_recordFileDialog->isVisible()) {
            m_recordFileDialog->raise();
            m_recordFileDialog->activateWindow();
            return;
        }

        auto *dialog = new QFileDialog(this);
        dialog->setFileMode(QFileDialog::AnyFile);
        dialog->setAcceptMode(QFileDialog::AcceptSave);
        dialog->setNameFilter(tr("Matroska video files (*.mkv);;All files (*)"));
        dialog->setAttribute(Qt::WA_DeleteOnClose);

        const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
        dialog->selectFile(QStringLiteral("recording_%1.mkv").arg(timestamp));

        m_recordFileDialog = dialog;

        connect(dialog, &QFileDialog::fileSelected, this, &CameraMainWindow::handleRecordFileSelected);
        connect(dialog, &QFileDialog::rejected, this, &CameraMainWindow::handleRecordFileCancelled);
        dialog->open();
        return;
    }

    m_videoThread->stopRecording();
    m_isRecording = false;
    m_recordButton->setText(tr("Start Recording"));
    m_statusBar->showMessage(tr("Recording stopped."));
}

void CameraMainWindow::updateFpsDisplay()
{
    QString statusText = tr("FPS: %1 | Frames: %2 | Queue: %3 | Dropped: %4")
                             .arg(m_currentFps, 0, 'f', 1)
                             .arg(m_frameCount)
                             .arg(m_lastQueueSize)
                             .arg(m_lastDroppedFrames);
    if (m_isRecording) {
        statusText += tr(" | RECORDING");
    }
    m_statusBar->showMessage(statusText);
}

void CameraMainWindow::renderLatestFrame()
{
    if (m_latestFrame.isNull()) {
        return;
    }

    if (m_renderedFrameSeq == m_latestFrameSeq) {
        return;
    }

    QPixmap pixmap = QPixmap::fromImage(m_latestFrame);
    if (m_cameraLabel->size().isValid()) {
        pixmap = pixmap.scaled(m_cameraLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    m_cameraLabel->setPixmap(pixmap);
    m_renderedFrameSeq = m_latestFrameSeq;
}

void CameraMainWindow::handleFrameReady(const QImage &image, qint64)
{
    if (image.isNull()) {
        return;
    }

    m_lastFrameSize = image.size();

    if (m_autoStartRecording && !m_isRecording) {
        m_autoStartRecording = false;
        const QString timestamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
        handleRecordFileSelected(QStringLiteral("recording_%1.mkv").arg(timestamp));
    }

    ++m_frameCount;
    m_latestFrame = image;
    ++m_latestFrameSeq;
}

void CameraMainWindow::handleFpsChanged(double fps)
{
    m_currentFps = fps;
}

void CameraMainWindow::handleQueueStatsChanged(qulonglong queueSize, qulonglong droppedFrames)
{
    m_lastQueueSize = queueSize;
    m_lastDroppedFrames = droppedFrames;
}

void CameraMainWindow::handleRecordFileSelected(const QString &filename)
{
    QString outputPath = filename;
    if (!outputPath.endsWith(QStringLiteral(".mkv"), Qt::CaseInsensitive)) {
        outputPath += QStringLiteral(".mkv");
    }

    if (!m_lastFrameSize.isValid()) {
        QMessageBox::warning(this, tr("No Frame"), tr("Wait for the first camera frame before starting recording."));
        return;
    }

    const double fps = m_currentFps > 0.0 ? m_currentFps : 30.0;
    m_videoThread->startRecording(m_lastFrameSize.width(), m_lastFrameSize.height(), fps, outputPath);
    m_isRecording = true;
    m_recordButton->setText(tr("Stop Recording"));
    m_statusBar->showMessage(tr("Recording started: %1").arg(outputPath));
}

void CameraMainWindow::handleRecordFileCancelled()
{
    m_statusBar->showMessage(tr("Recording cancelled."));
}

void CameraMainWindow::resetPreview()
{
    m_cameraLabel->clear();
    m_cameraLabel->setText(tr("Camera feed will appear here"));
    m_cameraLabel->setStyleSheet(QStringLiteral("QLabel { background-color: black; color: white; }"));
    m_cameraLabel->setMinimumSize(100, 100);
    m_cameraLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

    m_currentFps = 0.0;
    m_lastFrameSize = QSize();
    m_lastQueueSize = 0;
    m_lastDroppedFrames = 0;
    m_frameCount = 0;
    m_latestFrame = QImage();
    m_latestFrameSeq = 0;
    m_renderedFrameSeq = 0;
    m_recordButton->setText(tr("Start Recording"));
}

void CameraMainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    if (!m_latestFrame.isNull()) {
        m_renderedFrameSeq = 0;
        renderLatestFrame();
    }
}

void CameraMainWindow::closeEvent(QCloseEvent *event)
{
    closeCamera();
    event->accept();
}