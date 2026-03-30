#include "VideoThread.h"
#include "MindVisionCamera.h"
#include <QDebug>
#include <QFile>

namespace {

bool writeAll(QFile &file, const char *data, qint64 bytesToWrite)
{
    qint64 totalWritten = 0;
    while (totalWritten < bytesToWrite) {
        const qint64 written = file.write(data + totalWritten, bytesToWrite - totalWritten);
        if (written <= 0) {
            return false;
        }
        totalWritten += written;
    }
    return true;
}

} // namespace

VideoThread::VideoThread(QObject *parent) 
    : QThread(parent),
      m_abort(false),
      m_isRecording(false),
      m_droppedFrames(0),
      m_width(0),
      m_height(0),
    m_fps(30.0),
    m_frameSource(nullptr)
{
}

VideoThread::~VideoThread()
{
    clearFrameSource();

    m_mutex.lock();
    m_abort = true;
    m_condition.wakeOne();
    m_mutex.unlock();
    wait();
}

void VideoThread::startRecording(int width, int height, double fps, const QString &filename)
{
    QMutexLocker locker(&m_mutex);
    m_width = width;
    m_height = height;
    m_fps = fps;
    m_filename = filename;
    m_isRecording = true;
    m_abort = false;
    m_droppedFrames = 0;
    m_queue.clear();
    
    if (!isRunning()) {
        start();
    }
}

void VideoThread::stopRecording()
{
    QMutexLocker locker(&m_mutex);
    m_isRecording = false;
    m_condition.wakeOne();
}

void VideoThread::addFrame(const QImage &image)
{
    QMutexLocker locker(&m_mutex);
    if (m_isRecording) {
        while (m_queue.size() >= kMaxQueuedFrames) {
            m_queue.dequeue();
            ++m_droppedFrames;
        }

        m_queue.enqueue(image);
        m_condition.wakeOne();
    }
}

void VideoThread::setFrameSource(MindVisionCamera *camera)
{
    if (m_frameSource == camera) {
        return;
    }

    clearFrameSource();
    if (camera == nullptr) {
        return;
    }

    camera->setRecordingTarget(this);
    m_frameSource = camera;
}

void VideoThread::clearFrameSource()
{
    if (m_frameSource != nullptr) {
        m_frameSource->clearRecordingTarget();
        m_frameSource = nullptr;
    }
}

void VideoThread::run()
{
    QFile outputFile;
    
    m_mutex.lock();
    int width = m_width;
    int height = m_height;
    double fps = m_fps;
    QString filename = m_filename;
    m_mutex.unlock();

    qDebug() << "VideoThread: Writing rawvideo rgb24 to" << filename
             << "size" << width << "x" << height << "fps" << fps;

    outputFile.setFileName(filename);
    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        qDebug() << "VideoThread: Failed to open output file:" << outputFile.errorString();
        return;
    }

    while (true) {
        m_mutex.lock();
        
        if (m_abort) {
            m_mutex.unlock();
            break;
        }

        if (m_queue.isEmpty()) {
            // If stopped and empty, we are done
            if (!m_isRecording) {
                m_mutex.unlock();
                break;
            }
            // Wait for more frames
            m_condition.wait(&m_mutex);
        }

        // Check again after wake
        if (m_abort) {
            m_mutex.unlock();
            break;
        }
        
        if (m_queue.isEmpty() && !m_isRecording) {
            m_mutex.unlock();
            break;
        }

        if (!m_queue.isEmpty()) {
            QImage img = m_queue.dequeue();
            m_mutex.unlock();
            
            QImage convertedImg = img.convertToFormat(QImage::Format_RGB888);

            if (convertedImg.width() != width || convertedImg.height() != height) {
                qDebug() << "VideoThread: Dropping frame with unexpected size"
                         << convertedImg.width() << "x" << convertedImg.height()
                         << "expected" << width << "x" << height;
                continue;
            }

            const qint64 rowBytes = static_cast<qint64>(convertedImg.width()) * 3;
            bool writeOk = true;
            for (int y = 0; y < convertedImg.height(); ++y) {
                if (!writeAll(outputFile, reinterpret_cast<const char *>(convertedImg.constScanLine(y)), rowBytes)) {
                    qDebug() << "VideoThread: Failed to write raw frame:" << outputFile.errorString();
                    writeOk = false;
                    break;
                }
            }

            if (!writeOk) {
                break;
            }
            
            static int frameCount = 0;
            if (++frameCount % 30 == 0) {
                m_mutex.lock();
                const int queuedFrames = m_queue.size();
                const int droppedFrames = m_droppedFrames;
                m_mutex.unlock();

                qDebug() << "VideoThread: Processed frame" << frameCount
                         << "Queue size:" << queuedFrames
                         << "Dropped frames:" << droppedFrames;
            }
        } else {
            m_mutex.unlock();
        }
    }

    outputFile.close();
    
    qDebug() << "VideoThread: Finished.";
}
