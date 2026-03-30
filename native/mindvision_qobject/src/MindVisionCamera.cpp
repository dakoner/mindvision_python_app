#include "MindVisionCamera.h"
#include "VideoThread.h"
#include <QDebug>
#include <QElapsedTimer>
#include <QDateTime>
#include <cstdlib>
#include <iostream>

// SDK Header
#include "CameraApi.h"

// =============================================================================
// CameraWorker Declaration & Definition
// =============================================================================

class CameraWorker : public QObject
{
    Q_OBJECT
public:
    explicit CameraWorker(CameraHandle handle, int width, int height);
    ~CameraWorker();

    QImage takeLatestPreviewFrame(qint64 &timestampMs);
    void getPreviewStats(qulonglong &received, qulonglong &emitted, qulonglong &dropped);

public slots:
    // Main capture loop
    void process();
    // Request the loop to stop
    void stop();

signals:
    // Emitted when a frame is processed and converted to QImage
    void frameReady(QImage image, qint64 timestampMs);
    void previewFrameAvailable();
    void queueStatsChanged(qulonglong queueSize, qulonglong droppedFrames);
    // Emitted to report FPS
    void fpsChanged(double fps);
    // Emitted when the capture loop finishes
    void finished();

private:
    CameraHandle m_hCamera;
    int m_width;
    int m_height;
    bool m_stopRequested;
    unsigned char* m_pRgbBuffer; // Buffer for RGB conversion
    tSdkFrameHead m_frameHead;
    QMutex m_previewMutex;
    QImage m_latestPreviewFrame;
    qint64 m_latestPreviewTimestampMs;
    qulonglong m_previewFramesReceived;
    qulonglong m_previewFramesEmitted;
    qulonglong m_previewFramesDropped;
    bool m_previewPending;
};

CameraWorker::CameraWorker(CameraHandle handle, int width, int height)
    : m_hCamera(handle), m_width(width), m_height(height), m_stopRequested(false), m_pRgbBuffer(nullptr), m_latestPreviewTimestampMs(0), m_previewFramesReceived(0), m_previewFramesEmitted(0), m_previewFramesDropped(0), m_previewPending(false)
{
    // Allocate buffer for RGB conversion (Width * Height * 3 bytes for RGB24)
    // Note: The SDK documentation recommends CameraAlignMalloc for better performance,
    // but standard malloc is used here for standard C++ compatibility.
    // Ensure you link against MVCAMSDK.lib/dll.
    int size = m_width * m_height * 3;
    m_pRgbBuffer = (unsigned char*)malloc(size);
}

CameraWorker::~CameraWorker()
{
    if (m_pRgbBuffer) {
        free(m_pRgbBuffer);
        m_pRgbBuffer = nullptr;
    }
}

void CameraWorker::stop()
{
    m_stopRequested = true;
}

QImage CameraWorker::takeLatestPreviewFrame(qint64 &timestampMs)
{
    QMutexLocker locker(&m_previewMutex);
    m_previewPending = false;
    timestampMs = m_latestPreviewTimestampMs;
    if (!m_latestPreviewFrame.isNull()) {
        ++m_previewFramesEmitted;
    }
    return m_latestPreviewFrame;
}

void CameraWorker::getPreviewStats(qulonglong &received, qulonglong &emitted, qulonglong &dropped)
{
    QMutexLocker locker(&m_previewMutex);
    received = m_previewFramesReceived;
    emitted = m_previewFramesEmitted;
    dropped = m_previewFramesDropped;
}

void CameraWorker::process()
{
    unsigned char* pRawBuffer = nullptr;
    CameraSdkStatus status;
    
    QElapsedTimer fpsTimer;
    fpsTimer.start();
    QElapsedTimer statsTimer;
    statsTimer.start();
    int frameCount = 0;

    while (!m_stopRequested) {
        // 1. Get raw image buffer with a timeout (e.g., 1000ms)
        // CameraGetImageBuffer retrieves a pointer to the internal SDK buffer (Zero-Copy)
        status = CameraGetImageBuffer(m_hCamera, &m_frameHead, &pRawBuffer, 1000);

        if (status == CAMERA_STATUS_SUCCESS) {
            // Calculate FPS
            frameCount++;
            if (fpsTimer.elapsed() >= 1000) {
                double fps = frameCount * 1000.0 / fpsTimer.elapsed();
                emit fpsChanged(fps);
                
                fpsTimer.restart();
                frameCount = 0;
            }

            // 2. Process raw data to RGB
            // CameraImageProcess converts the RAW data (pRawBuffer) to the target format (m_pRgbBuffer).
            // We previously configured the camera to output RGB24 via CameraSetIspOutFormat.
            status = CameraImageProcess(m_hCamera, pRawBuffer, m_pRgbBuffer, &m_frameHead);

            if (status == CAMERA_STATUS_SUCCESS) {
                qint64 timestampMs = QDateTime::currentMSecsSinceEpoch();

                // 3. Create QImage from the processed RGB buffer
                // QImage::Format_RGB888 expects 3 bytes per pixel (R, G, B)
                QImage img(m_pRgbBuffer, m_frameHead.iWidth, m_frameHead.iHeight, QImage::Format_RGB888);
                QImage frameCopy = img.copy();
                
                // Emit to direct consumers such as the recorder from the capture thread.
                emit frameReady(frameCopy, timestampMs);

                // Coalesce preview updates so the main-thread event queue cannot accumulate frames.
                bool shouldNotifyPreview = false;
                {
                    QMutexLocker locker(&m_previewMutex);
                    ++m_previewFramesReceived;
                    m_latestPreviewFrame = frameCopy;
                    m_latestPreviewTimestampMs = timestampMs;
                    if (!m_previewPending) {
                        m_previewPending = true;
                        shouldNotifyPreview = true;
                    } else {
                        ++m_previewFramesDropped;
                    }
                }
                if (shouldNotifyPreview) {
                    emit previewFrameAvailable();
                }
            }

            // 5. Release the raw buffer back to the SDK so it can be reused for new frames
            CameraReleaseImageBuffer(m_hCamera, pRawBuffer);

            if (statsTimer.elapsed() >= 1000) {
                qulonglong queueSize = 0;
                qulonglong droppedFrames = 0;
                {
                    QMutexLocker locker(&m_previewMutex);
                    queueSize = m_previewPending ? 1ULL : 0ULL;
                    droppedFrames = m_previewFramesDropped;
                }
                emit queueStatsChanged(queueSize, droppedFrames);
                statsTimer.restart();
            }
        } else {
            // Handle timeout or error. 
            // status == CAMERA_STATUS_TIME_OUT if no frame arrived in 1000ms.
        }
    }
    
    emit finished();
}

// =============================================================================
// MindVisionCameraPrivate
// =============================================================================

class MindVisionCameraPrivate
{
public:
    MindVisionCameraPrivate()
        : m_hCamera(0),
          m_isOpen(false),
          m_workerThread(nullptr),
          m_worker(nullptr),
          m_recordingTarget(nullptr)
    {}
    
    CameraHandle m_hCamera;
    tSdkCameraDevInfo m_devInfo;
    tSdkCameraCapbility m_capInfo; 
    bool m_isOpen;
    
    QThread* m_workerThread;
    CameraWorker* m_worker;
    VideoThread* m_recordingTarget;
    QMetaObject::Connection m_recordingConnection;
};

// =============================================================================
// MindVisionCamera Implementation
// =============================================================================

MindVisionCamera::MindVisionCamera(QObject *parent)
    : QObject(parent), d(new MindVisionCameraPrivate)
{
    // Initialize the SDK
    // 0: English, 1: Chinese
    CameraSdkInit(0); 
}

MindVisionCamera::~MindVisionCamera()
{
    stop();
    close();
    delete d;
}

bool MindVisionCamera::open()
{
    if (d->m_isOpen) return true;

    int iCameraCounts = 1;
    tSdkCameraDevInfo tCameraList[1]; // Buffer for 1 camera info
    
    // Enumerate devices to find connected cameras
    if (CameraEnumerateDevice(tCameraList, &iCameraCounts) != CAMERA_STATUS_SUCCESS || iCameraCounts == 0) {
        std::cout << "No MindVision camera found." << std::endl;
        emit errorOccurred("No MindVision camera found.");
        return false;
    }

    std::cout << "Found camera: " << tCameraList[0].acFriendlyName << " (" << tCameraList[0].acProductName << ")" << std::endl;

    // Initialize the first available camera found
    int status = CameraInit(&tCameraList[0], -1, -1, &d->m_hCamera);
    if (status != CAMERA_STATUS_SUCCESS) {
        std::cout << "Failed to initialize camera. Error code: " << status << std::endl;
        emit errorOccurred("Failed to initialize camera.");
        return false;
    }

    d->m_devInfo = tCameraList[0];
    d->m_isOpen = true;
    
    // Get Camera Capabilities
    CameraGetCapability(d->m_hCamera, &d->m_capInfo);

    // Check for Level Trigger (Bulb Mode) Support
    UINT uTrigCapability = 0;
    if (CameraGetExtTrigCapability(d->m_hCamera, &uTrigCapability) == CAMERA_STATUS_SUCCESS) {
        bool supportsLevelTrigger = (uTrigCapability & EXT_TRIG_MASK_LEVEL_MODE);
        std::cout << "External Trigger Capability Mask: " << uTrigCapability << std::endl;
        std::cout << "Supports Level Trigger (Bulb Mode): " << (supportsLevelTrigger ? "YES" : "NO") << std::endl;
    } else {
        std::cout << "Failed to get External Trigger Capability." << std::endl;
    }

    // Set the ISP output format to RGB24. This ensures CameraImageProcess produces
    // data compatible with QImage::Format_RGB888.
    CameraSetIspOutFormat(d->m_hCamera, CAMERA_MEDIA_TYPE_RGB8);
    
    // Default to Auto Exposure
    setAutoExposure(true);

    return true;
}

void MindVisionCamera::close()
{
    if (d->m_isOpen) {
        stop(); // Stop capturing first
        
        // Release SDK resources for this camera
        CameraUnInit(d->m_hCamera);
        
        d->m_hCamera = 0;
        d->m_isOpen = false;
    }
}

bool MindVisionCamera::start()
{
    if (!d->m_isOpen) {
        emit errorOccurred("Camera is not open.");
        return false;
    }

    // Start the camera video stream
    if (CameraPlay(d->m_hCamera) != CAMERA_STATUS_SUCCESS) {
        emit errorOccurred("Failed to start camera play.");
        return false;
    }

    // Get current resolution to allocate the correct buffer size in the worker
    tSdkImageResolution tResolution;
    if (CameraGetImageResolution(d->m_hCamera, &tResolution) != CAMERA_STATUS_SUCCESS) {
         emit errorOccurred("Failed to get camera resolution.");
         return false;
    }

    // Create the worker thread
    d->m_workerThread = new QThread;
    d->m_worker = new CameraWorker(d->m_hCamera, tResolution.iWidth, tResolution.iHeight);
    d->m_worker->moveToThread(d->m_workerThread);

    // Connect signals and slots for thread management
    connect(d->m_workerThread, &QThread::started, d->m_worker, &CameraWorker::process);
    connect(d->m_worker, &CameraWorker::previewFrameAvailable, this, &MindVisionCamera::deliverLatestFrame);
    connect(d->m_worker, &CameraWorker::queueStatsChanged, this, &MindVisionCamera::queueStatsChanged);
    connect(d->m_worker, &CameraWorker::fpsChanged, this, &MindVisionCamera::fpsChanged);

    if (d->m_recordingTarget != nullptr) {
        d->m_recordingConnection = connect(
            d->m_worker,
            &CameraWorker::frameReady,
            d->m_recordingTarget,
            &VideoThread::addFrame,
            Qt::DirectConnection);
    }
    
    // Clean up worker and thread when finished
    connect(d->m_worker, &CameraWorker::finished, d->m_workerThread, &QThread::quit);
    connect(d->m_worker, &CameraWorker::finished, d->m_worker, &CameraWorker::deleteLater);
    connect(d->m_workerThread, &QThread::finished, d->m_workerThread, &QThread::deleteLater);

    // Start the thread
    d->m_workerThread->start();
    return true;
}

void MindVisionCamera::stop()
{
    // Stop the worker thread
    if (d->m_worker) {
        if (d->m_recordingConnection) {
            QObject::disconnect(d->m_recordingConnection);
            d->m_recordingConnection = QMetaObject::Connection();
        }
        d->m_worker->stop(); // Tell the loop to break
        if (d->m_workerThread) {
            d->m_workerThread->quit();
            d->m_workerThread->wait(); // Wait for the thread to actually finish
        }
        d->m_worker = nullptr;
        d->m_workerThread = nullptr;
    }
    
    // Stop the camera SDK stream
    if (d->m_isOpen) {
        CameraStop(d->m_hCamera); 
    }
}

void MindVisionCamera::setRecordingTarget(VideoThread *target)
{
    if (d->m_recordingConnection) {
        QObject::disconnect(d->m_recordingConnection);
        d->m_recordingConnection = QMetaObject::Connection();
    }

    d->m_recordingTarget = target;
    if (d->m_worker != nullptr && d->m_recordingTarget != nullptr) {
        d->m_recordingConnection = connect(
            d->m_worker,
            &CameraWorker::frameReady,
            d->m_recordingTarget,
            &VideoThread::addFrame,
            Qt::DirectConnection);
    }
}

void MindVisionCamera::clearRecordingTarget()
{
    if (d->m_recordingConnection) {
        QObject::disconnect(d->m_recordingConnection);
        d->m_recordingConnection = QMetaObject::Connection();
    }
    d->m_recordingTarget = nullptr;
}

void MindVisionCamera::deliverLatestFrame()
{
    if (d->m_worker == nullptr) {
        return;
    }

    qint64 timestampMs = 0;
    QImage frame = d->m_worker->takeLatestPreviewFrame(timestampMs);
    if (!frame.isNull()) {
        emit frameReady(frame, timestampMs);
    }
}

void MindVisionCamera::getFrameCallbackStats(qulonglong &received, qulonglong &emitted, qulonglong &dropped)
{
    if (d->m_worker == nullptr) {
        received = 0;
        emitted = 0;
        dropped = 0;
        return;
    }

    d->m_worker->getPreviewStats(received, emitted, dropped);
}

bool MindVisionCamera::setAutoExposure(bool enabled)
{
    if (!d->m_isOpen) return false;
    return CameraSetAeState(d->m_hCamera, enabled ? TRUE : FALSE) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setExposureTime(double exposureTimeMs)
{
    if (!d->m_isOpen) return false;
    // SDK takes microseconds
    double exposureTimeUs = exposureTimeMs * 1000.0;
    return CameraSetExposureTime(d->m_hCamera, exposureTimeUs) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setAnalogGain(int gain)
{
    if (!d->m_isOpen) return false;
    return CameraSetAnalogGain(d->m_hCamera, gain) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setAeTarget(int target)
{
    if (!d->m_isOpen) return false;
    return CameraSetAeTarget(d->m_hCamera, target) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::getAutoExposure()
{
    if (!d->m_isOpen) return false;
    BOOL state = FALSE;
    CameraGetAeState(d->m_hCamera, &state);
    return state == TRUE;
}

double MindVisionCamera::getExposureTime()
{
    if (!d->m_isOpen) return 0.0;
    double timeUs = 0;
    CameraGetExposureTime(d->m_hCamera, &timeUs);
    return timeUs / 1000.0; // Convert back to ms
}

int MindVisionCamera::getAnalogGain()
{
    if (!d->m_isOpen) return 0;
    int gain = 0;
    CameraGetAnalogGain(d->m_hCamera, &gain);
    return gain;
}

int MindVisionCamera::getAeTarget()
{
    if (!d->m_isOpen) return 0;
    int target = 0;
    CameraGetAeTarget(d->m_hCamera, &target);
    return target;
}

void MindVisionCamera::getExposureTimeRange(double &minMs, double &maxMs)
{
    if (!d->m_isOpen) {
        minMs = 0;
        maxMs = 0;
        return;
    }
    
    double minUs = 0, maxUs = 0, stepUs = 0;
    if (CameraGetExposureTimeRange(d->m_hCamera, &minUs, &maxUs, &stepUs) == CAMERA_STATUS_SUCCESS) {
        minMs = minUs / 1000.0;
        maxMs = maxUs / 1000.0;
        return;
    }

    // Fallback
    double lineTime = 0;
    CameraGetExposureLineTime(d->m_hCamera, &lineTime);
    
    // If lineTime is valid
    if (lineTime > 0) {
        minMs = d->m_capInfo.sExposeDesc.uiExposeTimeMin * lineTime / 1000.0;
        maxMs = d->m_capInfo.sExposeDesc.uiExposeTimeMax * lineTime / 1000.0;
    } else {
        // Fallback
        minMs = 0.1;
        maxMs = 1000.0;
    }
}

double MindVisionCamera::getExposureTimeStep()
{
    if (!d->m_isOpen) return 0.0;

    double minUs = 0, maxUs = 0, stepUs = 0;
    if (CameraGetExposureTimeRange(d->m_hCamera, &minUs, &maxUs, &stepUs) == CAMERA_STATUS_SUCCESS) {
        return stepUs / 1000.0;
    }

    double lineTime = 0;
    CameraGetExposureLineTime(d->m_hCamera, &lineTime);
    return lineTime / 1000.0;
}

void MindVisionCamera::getAnalogGainRange(int &min, int &max)
{
    if (!d->m_isOpen) {
        min = 0;
        max = 0;
        return;
    }
    min = d->m_capInfo.sExposeDesc.uiAnalogGainMin;
    max = d->m_capInfo.sExposeDesc.uiAnalogGainMax;
}

bool MindVisionCamera::setRoi(bool enable)
{
    if (!d->m_isOpen) return false;

    // We must restart the stream if it's running because the worker buffer size depends on resolution
    bool wasRunning = (d->m_workerThread != nullptr);
    if (wasRunning) {
        stop();
    }

    tSdkImageResolution tResolution;
    // Initialize with current just in case, though we will overwrite
    CameraGetImageResolution(d->m_hCamera, &tResolution);

    if (enable) {
        // Set to 640x480 Custom ROI
        tResolution.iIndex = 0xFF; // Custom resolution index
        tResolution.iWidth = 640;
        tResolution.iHeight = 480;
        
        // Try to center the ROI
        int maxWidth = d->m_capInfo.pImageSizeDesc[0].iWidth;
        int maxHeight = d->m_capInfo.pImageSizeDesc[0].iHeight;
        
        if (maxWidth > 640 && maxHeight > 480) {
            tResolution.iHOffsetFOV = (maxWidth - 640) / 2;
            tResolution.iVOffsetFOV = (maxHeight - 480) / 2;
        } else {
            tResolution.iHOffsetFOV = 0;
            tResolution.iVOffsetFOV = 0;
        }
    } else {
        // Restore to Full Resolution (Preset 0)
        tResolution.iIndex = 0;
        tResolution.iWidth = d->m_capInfo.pImageSizeDesc[0].iWidth;
        tResolution.iHeight = d->m_capInfo.pImageSizeDesc[0].iHeight;
        tResolution.iHOffsetFOV = 0;
        tResolution.iVOffsetFOV = 0;
    }

    if (CameraSetImageResolution(d->m_hCamera, &tResolution) != CAMERA_STATUS_SUCCESS) {
        // If failed, try to restore state (restart if was running)
        if (wasRunning) start();
        return false;
    }

    if (wasRunning) {
        return start();
    }
    return true;
}

bool MindVisionCamera::setTriggerMode(int mode)
{
    if (!d->m_isOpen) return false;
    
    // Set trigger count to 1 frame per trigger
    CameraSetTriggerCount(d->m_hCamera, 1);
    
    // 0: Continuous, 1: Soft Trigger, 2: Hardware Trigger
    return CameraSetTriggerMode(d->m_hCamera, mode) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setTriggerCount(int count)
{
    if (!d->m_isOpen) return false;
    return CameraSetTriggerCount(d->m_hCamera, count) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setTriggerDelay(int delay_us)
{
    if (!d->m_isOpen) return false;
    return CameraSetTriggerDelayTime(d->m_hCamera, delay_us) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setTriggerInterval(int interval_us)
{
    if (!d->m_isOpen) return false;
    return CameraSetExtTrigIntervalTime(d->m_hCamera, interval_us) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setExternalTriggerSignalType(int type)
{
    if (!d->m_isOpen) return false;
    return CameraSetExtTrigSignalType(d->m_hCamera, type) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setExternalTriggerJitterTime(int time_us)
{
    if (!d->m_isOpen) return false;
    return CameraSetExtTrigJitterTime(d->m_hCamera, time_us) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setExternalTriggerShutterMode(int mode)
{
    if (!d->m_isOpen) return false;
    return CameraSetExtTrigShutterType(d->m_hCamera, mode) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setStrobeMode(int mode)
{
    if (!d->m_isOpen) return false;
    return CameraSetStrobeMode(d->m_hCamera, mode) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setStrobePolarity(int polarity)
{
    if (!d->m_isOpen) return false;
    return CameraSetStrobePolarity(d->m_hCamera, polarity) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setStrobeDelayTime(int delay_us)
{
    if (!d->m_isOpen) return false;
    return CameraSetStrobeDelayTime(d->m_hCamera, delay_us) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::setStrobePulseWidth(int width_us)
{
    if (!d->m_isOpen) return false;
    return CameraSetStrobePulseWidth(d->m_hCamera, width_us) == CAMERA_STATUS_SUCCESS;
}

bool MindVisionCamera::triggerSoftware()
{
    if (!d->m_isOpen) return false;
    return CameraSoftTrigger(d->m_hCamera) == CAMERA_STATUS_SUCCESS;
}

#include "MindVisionCamera.moc"
