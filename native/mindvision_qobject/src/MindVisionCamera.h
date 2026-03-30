#ifndef MINDVISIONCAMERA_H
#define MINDVISIONCAMERA_H

#include <QObject>
#include <QImage>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QtGlobal>

#ifdef _WIN32
#include <windows.h>
#endif

#include "mindvision_qobject_global.h"

// Forward declaration of private implementation class
class MindVisionCameraPrivate;
class VideoThread;

// Main interface class
class MINDVISION_QOBJECT_EXPORT MindVisionCamera : public QObject
{
    Q_OBJECT
public:
    explicit MindVisionCamera(QObject *parent = nullptr);
    ~MindVisionCamera();

    // Initializes the SDK and opens the first available camera
    Q_INVOKABLE bool open();
    
    // Closes the camera and releases resources
    Q_INVOKABLE void close();

    // Starts the capture thread
    Q_INVOKABLE bool start();

    // Stops the capture thread
    Q_INVOKABLE void stop();

    // Settings
    Q_INVOKABLE bool setAutoExposure(bool enabled);
    Q_INVOKABLE bool setExposureTime(double exposureTimeMs); // Milliseconds
    Q_INVOKABLE bool setAnalogGain(int gain);
    Q_INVOKABLE bool setAeTarget(int target);
    
    Q_INVOKABLE bool getAutoExposure();
    Q_INVOKABLE double getExposureTime(); // Milliseconds
    Q_INVOKABLE int getAnalogGain();
    Q_INVOKABLE int getAeTarget();

    // Capabilities
    Q_INVOKABLE void getExposureTimeRange(double &minMs, double &maxMs);
    Q_INVOKABLE double getExposureTimeStep(); // Milliseconds
    Q_INVOKABLE void getAnalogGainRange(int &min, int &max);

    // Preview callback stats
    // received: frames processed by the camera worker
    // emitted: frames emitted via MindVisionCamera::frameReady
    // dropped: frames overwritten before callback emission
    Q_INVOKABLE void getFrameCallbackStats(qulonglong &received, qulonglong &emitted, qulonglong &dropped);

    // ROI
    Q_INVOKABLE bool setRoi(bool enable);

    // Trigger Mode
    // mode: 0 = Continuous, 1 = Software, 2 = Hardware
    Q_INVOKABLE bool setTriggerMode(int mode);
    
    // Trigger Parameters
    Q_INVOKABLE bool setTriggerCount(int count);
    Q_INVOKABLE bool setTriggerDelay(int delay_us);
    Q_INVOKABLE bool setTriggerInterval(int interval_us);

    // External Trigger Parameters
    Q_INVOKABLE bool setExternalTriggerSignalType(int type);
    Q_INVOKABLE bool setExternalTriggerJitterTime(int time_us);
    Q_INVOKABLE bool setExternalTriggerShutterMode(int mode);

    // Strobe Parameters
    Q_INVOKABLE bool setStrobeMode(int mode);
    Q_INVOKABLE bool setStrobePolarity(int polarity);
    Q_INVOKABLE bool setStrobeDelayTime(int delay_us);
    Q_INVOKABLE bool setStrobePulseWidth(int width_us);
    
    // Software Trigger
    Q_INVOKABLE bool triggerSoftware();

    void setRecordingTarget(VideoThread *target);
    void clearRecordingTarget();

signals:
    // Signal for UI to update (connect this to your QML or Widget)
    void frameReady(QImage image, qint64 timestampMs);
    // Signal emitted once per second while capturing.
    void queueStatsChanged(qulonglong queueSize, qulonglong droppedFrames);
    // Signal for current FPS
    void fpsChanged(double fps);
    // Signal for error reporting
    void errorOccurred(QString message);

private:
    void deliverLatestFrame();

    MindVisionCameraPrivate *d;
};

#endif // MINDVISIONCAMERA_H
