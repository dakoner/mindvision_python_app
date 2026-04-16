#include "VideoThread.h"
#include "MindVisionCamera.h"
#include <QDebug>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace {

// Shell-quote a filename so it is safe to embed in a popen() command string.
// Uses POSIX single-quote escaping: 'text' with any ' replaced by '\''
QByteArray shellQuote(const QString &s)
{
    QByteArray quoted = "'";
    for (const QChar c : s) {
        if (c == QLatin1Char('\'')) {
            quoted += "'\\''";
        } else {
            quoted += c.toLatin1();
        }
    }
    quoted += "'";
    return quoted;
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
    m_mutex.lock();
    int width = m_width;
    int height = m_height;
    double fps = m_fps;
    QString filename = m_filename;
    m_mutex.unlock();

    qDebug() << "VideoThread: Starting ffmpeg to write mkv to" << filename
             << "size" << width << "x" << height << "fps" << fps;

    const QByteArray cmd =
        "ffmpeg -y"
        " -f rawvideo -vcodec rawvideo"
        " -s " + QByteArray::number(width) + "x" + QByteArray::number(height) +
        " -pix_fmt rgb24"
        " -r " + QByteArray::number(fps, 'f', 6) +
        " -i pipe:0"
        " -c:v libx264 -preset veryfast -crf 12"
        " " + shellQuote(filename);

    FILE *const pipe = ::popen(cmd.constData(), "w");
    if (!pipe) {
        qDebug() << "VideoThread: Failed to popen ffmpeg:" << ::strerror(errno);
        return;
    }

    bool failed = false;
    while (!failed) {
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

            const QImage converted = img.convertToFormat(QImage::Format_RGB888);

            if (converted.width() != width || converted.height() != height) {
                qDebug() << "VideoThread: Dropping frame with unexpected size"
                         << converted.width() << "x" << converted.height()
                         << "expected" << width << "x" << height;
                continue;
            }

            const std::size_t rowBytes = static_cast<std::size_t>(converted.width()) * 3;
            for (int y = 0; y < converted.height(); ++y) {
                if (::fwrite(converted.constScanLine(y), 1, rowBytes, pipe) != rowBytes) {
                    qDebug() << "VideoThread: fwrite failed:" << ::strerror(errno);
                    failed = true;
                    break;
                }
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

    ::pclose(pipe);
    qDebug() << "VideoThread: Finished recording to" << filename;
}
