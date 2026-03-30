#ifndef VIDEOTHREAD_H
#define VIDEOTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <QImage>
#include "mindvision_qobject_global.h"

class MindVisionCamera;

class MINDVISION_QOBJECT_EXPORT VideoThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoThread(QObject *parent = nullptr);
    ~VideoThread();

    // Configure and start the recording thread
    void startRecording(int width, int height, double fps, const QString &filename);
    
    // Signal the thread to stop accepting new frames and finish writing the queue
    void stopRecording();
    
    // Add a frame to the queue
    void addFrame(const QImage &image);

    void setFrameSource(MindVisionCamera *camera);
    void clearFrameSource();

protected:
    void run() override;

private:
    static constexpr int kMaxQueuedFrames = 1024;

    QMutex m_mutex;
    QWaitCondition m_condition;
    QQueue<QImage> m_queue;
    bool m_abort;
    bool m_isRecording; // True while actively recording, False when stopping but draining queue
    int m_droppedFrames;
    
    // Recording parameters
    int m_width;
    int m_height;
    double m_fps;
    QString m_filename;
    MindVisionCamera *m_frameSource;
};

#endif // VIDEOTHREAD_H
