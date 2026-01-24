import sys
import os
import platform
import time
import signal
import json

# Add the directory containing the generated module to sys.path
script_dir = os.path.dirname(__file__)
release_dir = os.path.join(r"c:\users\davidek\microtools\mindvision_qobject", "release")
sys.path.insert(0, release_dir)

os.add_dll_directory(release_dir)
# Add Qt bin directory
os.add_dll_directory(r"C:\Qt\6.10.1\msvc2022_64\bin")
# Add MindVision SDK directory
os.add_dll_directory(r"C:\Program Files (x86)\MindVision\SDK\X64")

import _mindvision_qobject_py
import PySide6.QtWidgets
import cv2
import numpy as np

try:
    import serial
    import serial.tools.list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("Warning: pyserial not installed. Serial features disabled.")

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QButtonGroup,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QSlider,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
    Slot,
    QFile,
    QObject,
    QEvent,
    QThread,
    QMutex,
    QPointF,
    QLineF,
)
from PySide6.QtGui import QImage, QPixmap, QAction, QPainter, QPen, QColor, QIcon
from PySide6.QtUiTools import QUiLoader
from range_slider import RangeSlider
from intensity_chart import IntensityChart

try:
    from _mindvision_qobject_py import MindVisionCamera, VideoThread
except ImportError as e:
    print(f"Failed to import _mindvision_qobject_py: {e}")
    sys.exit(1)

def precompute_ssim_constants(img):
    img = img.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu_sq = mu ** 2
    sigma_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu_sq
    
    return {
        "img": img,
        "mu": mu,
        "mu_sq": mu_sq,
        "sigma_sq": sigma_sq,
        "window": window
    }

def compute_ssim_cached(img1, ref_stats):
    C1 = 6.5025
    C2 = 58.5225
    
    img1 = img1.astype(np.float64)
    window = ref_stats["window"]
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    
    mu2 = ref_stats["mu"]
    mu2_sq = ref_stats["mu_sq"]
    sigma2_sq = ref_stats["sigma_sq"]
    img2 = ref_stats["img"]
    
    mu1_mu2 = mu1 * mu2
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_ssim(img1, img2):
    # Legacy wrapper for non-cached usage if needed, or we can just compute on the fly
    stats = precompute_ssim_constants(img2)
    return compute_ssim_cached(img1, stats)


class SerialWorker(QObject):
    log_signal = Signal(str)
    connection_status = Signal(bool)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.mutex = QMutex()

    @Slot(str, int)
    def connect_serial(self, port_name, baud_rate=115200):
        if not HAS_SERIAL:
            self.log_signal.emit("Error: pyserial not installed.")
            return

        with QMutexLocker(self.mutex):
            if self.is_connected:
                self.disconnect_serial()

            try:
                self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
                self.is_connected = True
                self.connection_status.emit(True)
                self.log_signal.emit(f"Connected to {port_name} at {baud_rate}")
            except Exception as e:
                self.log_signal.emit(f"Failed to connect to {port_name}: {e}")
                self.connection_status.emit(False)

    @Slot()
    def disconnect_serial(self):
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.close()
                except Exception as e:
                    self.log_signal.emit(f"Error closing port: {e}")
            self.serial_port = None
            self.is_connected = False
            self.connection_status.emit(False)
            self.log_signal.emit("Disconnected.")

    @Slot(str)
    def send_command(self, cmd):
        with QMutexLocker(self.mutex):
            if not self.is_connected or not self.serial_port:
                self.log_signal.emit("Error: Not connected to serial port.")
                return

            try:
                full_cmd = cmd.strip() + "\n"
                self.serial_port.write(full_cmd.encode("utf-8"))
                self.log_signal.emit(f"Tx: {cmd}")
            except Exception as e:
                self.log_signal.emit(f"Send error: {e}")
                self.disconnect_serial()

    @Slot()
    def read_loop(self):
        # This is intended to be called periodically or run in a loop if we had a dedicated thread loop
        # For simplicity with QThread/moveToThread, we can use a QTimer in the worker or just rely on manual reads
        # But a robust way is a loop. However, since we are using standard QThread, let's just expose a read method
        # and let the main thread trigger it or use a timer in this worker.
        # BETTER: Use a timer inside this worker to poll.
        pass

    @Slot()
    def poll_serial(self):
        # Called by a timer in the worker thread
        with QMutexLocker(self.mutex):
            if self.is_connected and self.serial_port and self.serial_port.in_waiting:
                try:
                    # Read up to 10 lines to catch up but not freeze UI
                    for _ in range(10):
                        if self.serial_port.in_waiting:
                            line = (
                                self.serial_port.readline()
                                .decode("utf-8", errors="replace")
                                .strip()
                            )
                            if line:
                                self.log_signal.emit(f"Rx: {line}")
                        else:
                            break
                except Exception as e:
                    self.log_signal.emit(f"Read error: {e}")


class MatchingWorker(QObject):
    result_ready = Signal(QImage)
    log_signal = Signal(str)
    qr_found_signal = Signal(str)
    ssim_score_signal = Signal(float)

    def __init__(self):
        super().__init__()
        self.detector = None
        self.bf = None
        self.template_img = None
        self.template_kp = None
        self.template_des = None
        self.is_matching_enabled = False
        self.mutex = QMutex()

        # ArUco
        self.aruco_dict = None
        self.aruco_params = None
        self.aruco_obj = None
        self.aruco_display = {"ids": True, "rejected": False}
        self.contour_params = {
            "mode": "Canny",
            "thresh_min": 50,
            "thresh_max": 150,
            "threshold": 127,
            "min_area": 100,
            "max_area": 100000,
            "fill": False,
            "box": False,
        }
        self.current_algo = "ORB"
        self.is_contours_enabled = False

        # QR Code
        self.qr_detector = cv2.QRCodeDetector()
        
        # SSIM
        self.ssim_ref_img = None
        self.ssim_cache = None
        
        self.last_params = {}

    @Slot(dict)
    def update_params(self, params):
        # Update detector based on params
        with QMutexLocker(self.mutex):
            # Check if params actually changed
            if params == self.last_params:
                return
            self.last_params = params.copy()

            try:
                # If 'algo' is provided, we are updating the Matching Algo
                if "algo" in params:
                    self.current_algo = params["algo"]

                    if self.current_algo == "ORB":
                        self.detector = cv2.ORB_create(
                            nfeatures=params.get("nfeatures", 500),
                            scaleFactor=params.get("scaleFactor", 1.2),
                            nlevels=params.get("nlevels", 8),
                            edgeThreshold=params.get("edgeThreshold", 31),
                            firstLevel=params.get("firstLevel", 0),
                            WTA_K=params.get("WTA_K", 2),
                            scoreType=params.get("scoreType", 0),
                            patchSize=params.get("patchSize", 31),
                            fastThreshold=params.get("fastThreshold", 20),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    elif self.current_algo == "SIFT":
                        self.detector = cv2.SIFT_create(
                            nfeatures=params.get("nfeatures", 0),
                            nOctaveLayers=params.get("nOctaveLayers", 3),
                            contrastThreshold=params.get("contrastThreshold", 0.04),
                            edgeThreshold=params.get("edgeThreshold", 10),
                            sigma=params.get("sigma", 1.6),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                    elif self.current_algo == "AKAZE":
                        self.detector = cv2.AKAZE_create(
                            descriptor_type=params.get("descriptor_type", 5),
                            threshold=params.get("threshold", 0.0012),
                            nOctaves=params.get("nOctaves", 4),
                            nOctaveLayers=params.get("nOctaveLayers", 4),
                        )
                        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    elif self.current_algo == "ARUCO":
                        dict_name = params.get("dict", "DICT_4X4_50")
                        if hasattr(cv2.aruco, dict_name):
                            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                                getattr(cv2.aruco, dict_name)
                            )
                            if self.aruco_params is None:
                                self.aruco_params = cv2.aruco.DetectorParameters()

                            # Update ArUco Params
                            if "markerBorderBits" in params:
                                self.aruco_params.markerBorderBits = params[
                                    "markerBorderBits"
                                ]

                            self.aruco_display["ids"] = params.get("show_ids", True)
                            self.aruco_display["rejected"] = params.get(
                                "show_rejected", False
                            )

                            # Create ArucoDetector
                            try:
                                self.aruco_obj = cv2.aruco.ArucoDetector(
                                    self.aruco_dict, self.aruco_params
                                )
                            except AttributeError:
                                self.aruco_obj = None
                                self.log_signal.emit(
                                    "Error: cv2.aruco.ArucoDetector not found"
                                )
                        else:
                            self.log_signal.emit(f"Unknown ArUco dict: {dict_name}")
                            self.aruco_dict = None
                            self.aruco_obj = None
                    elif self.current_algo == "QRCODE":
                        # QR Code detector is already initialized
                        pass
                    elif self.current_algo == "SSIM":
                        pass

                    # Recompute template if exists (feature matching)
                    if (
                        self.current_algo in ["ORB", "SIFT", "AKAZE"]
                        and self.template_img is not None
                    ):
                        self.template_kp, self.template_des = (
                            self.detector.detectAndCompute(self.template_img, None)
                        )

            except Exception as e:
                self.log_signal.emit(f"Worker update error: {e}")

    @Slot(dict)
    def update_contour_params(self, params):
        with QMutexLocker(self.mutex):
            self.contour_params.update(params)

    @Slot(str)
    def set_ssim_reference(self, file_path):
        with QMutexLocker(self.mutex):
            if not file_path:
                self.ssim_ref_img = None
                self.ssim_cache = None
                if self.current_algo == 'SSIM':
                    self.is_matching_enabled = False
                return
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.ssim_ref_img = img
                self.ssim_cache = None
                self.log_signal.emit(f"Worker: SSIM reference loaded {file_path}")
                if self.current_algo == 'SSIM':
                    self.is_matching_enabled = True
            else:
                self.log_signal.emit("Worker: Failed to load SSIM reference")

    @Slot(str)
    def set_template(self, file_path):
        with QMutexLocker(self.mutex):
            if not file_path:
                if self.current_algo != "ARUCO" and self.current_algo != "QRCODE" and self.current_algo != "SSIM":
                    self.is_matching_enabled = False
                self.template_img = None
                return

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.template_img = img
                if self.detector is not None and self.current_algo not in [
                    "ARUCO",
                    "QRCODE",
                    "SSIM",
                ]:
                    self.template_kp, self.template_des = (
                        self.detector.detectAndCompute(img, None)
                    )
                    if self.template_des is not None:
                        self.is_matching_enabled = True
                        self.log_signal.emit(f"Worker: Template loaded {file_path}")
                    else:
                        self.log_signal.emit("Worker: No features in template")
                else:
                    self.log_signal.emit(
                        f"Worker: Template loaded (not used for {self.current_algo})"
                    )
            else:
                self.log_signal.emit("Worker: Failed to load template")

    @Slot(bool)
    def toggle_matching(self, enabled):
        with QMutexLocker(self.mutex):
            self.is_matching_enabled = enabled

    @Slot(bool)
    def toggle_contours(self, enabled):
        with QMutexLocker(self.mutex):
            self.is_contours_enabled = enabled

    @Slot(int, int, int, int, bytes)
    def process_frame(self, width, height, bytes_per_line, fmt, data_bytes):
        # This runs in the worker thread
        with QMutexLocker(self.mutex):
            matching_active = self.is_matching_enabled
            contours_active = self.is_contours_enabled
            algo = self.current_algo

            # Local refs
            local_detector = self.detector
            local_bf = self.bf
            local_template_des = self.template_des
            local_template_kp = self.template_kp
            local_template_img = self.template_img
            local_aruco_dict = self.aruco_dict
            local_aruco_params = self.aruco_params
            local_aruco_obj = self.aruco_obj
            local_aruco_display = self.aruco_display.copy()
            local_contour_params = self.contour_params.copy()
            local_qr_detector = self.qr_detector
            local_ssim_ref = self.ssim_ref_img
            local_ssim_cache = self.ssim_cache

        try:
            channels = bytes_per_line // width
            img_np = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                (height, width, channels)
            )

            # If no processing is needed, return original
            if not matching_active and not contours_active:
                qimg = QImage(
                    img_np.data, width, height, bytes_per_line, QImage.Format(fmt)
                ).copy()
                self.result_ready.emit(qimg)
                return

            # Prepare visualization image (BGR for OpenCV drawing)
            if channels == 1:
                vis_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                gray_frame = img_np
            else:
                vis_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # --- 1. Contours Processing ---
            if contours_active:
                # Blur to reduce noise
                blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

                # Binarize based on mode
                if local_contour_params.get("mode") == "Threshold":
                    # Binary Threshold
                    _, binary_img = cv2.threshold(
                        blurred,
                        local_contour_params.get("threshold", 127),
                        255,
                        cv2.THRESH_BINARY,
                    )
                    edges = binary_img  # Treat binary result as input for findContours
                else:
                    # Default: Canny Edges
                    edges = cv2.Canny(
                        blurred,
                        local_contour_params["thresh_min"],
                        local_contour_params["thresh_max"],
                    )

                # Find Contours
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Draw on vis_img
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if (
                        local_contour_params["min_area"]
                        < area
                        < local_contour_params["max_area"]
                    ):
                        thickness = -1 if local_contour_params["fill"] else 2
                        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), thickness)
                        if local_contour_params["box"]:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(
                                vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2
                            )

            # --- 2. Matching Processing ---
            if matching_active:
                if algo == 'SSIM' and local_ssim_ref is not None:
                    # Resize reference if dimensions don't match or cache is missing
                    use_cache = local_ssim_cache
                    
                    if use_cache is None or use_cache["img"].shape != gray_frame.shape:
                        if local_ssim_ref.shape != gray_frame.shape:
                             resized_ref = cv2.resize(local_ssim_ref, (gray_frame.shape[1], gray_frame.shape[0]))
                        else:
                             resized_ref = local_ssim_ref
                        
                        use_cache = precompute_ssim_constants(resized_ref)
                        
                        # Update cache if ref hasn't changed
                        with QMutexLocker(self.mutex):
                            if self.ssim_ref_img is local_ssim_ref:
                                self.ssim_cache = use_cache
                    
                    score = compute_ssim_cached(gray_frame, use_cache)
                    self.ssim_score_signal.emit(score)
                    
                    cv2.putText(vis_img, f"SSIM: {score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif algo == "QRCODE" and local_qr_detector:
                    retval, decoded_info, points, straight_qrcode = (
                        local_qr_detector.detectAndDecodeMulti(gray_frame)
                    )
                    if retval:
                        # points is a list of points for each QR code
                        for i in range(len(decoded_info)):
                            text = decoded_info[i]
                            pts = points[i].astype(int)

                            # Draw bounding box
                            for j in range(4):
                                cv2.line(
                                    vis_img,
                                    tuple(pts[j]),
                                    tuple(pts[(j + 1) % 4]),
                                    (255, 0, 0),
                                    2,
                                )

                            # Draw text
                            cv2.putText(
                                vis_img,
                                text,
                                tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2,
                            )

                            if text:
                                self.qr_found_signal.emit(text)

                elif algo == "ARUCO" and local_aruco_obj:
                    corners, ids, rejected = local_aruco_obj.detectMarkers(gray_frame)
                    if local_aruco_display["rejected"] and rejected:
                        cv2.aruco.drawDetectedMarkers(
                            vis_img, rejected, borderColor=(100, 0, 255)
                        )
                    if ids is not None and len(ids) > 0:
                        display_ids = ids if local_aruco_display["ids"] else None
                        cv2.aruco.drawDetectedMarkers(vis_img, corners, display_ids)

                elif (
                    algo in ["ORB", "SIFT", "AKAZE"]
                    and local_template_img is not None
                    and local_detector is not None
                ):
                    # Always attempt detection
                    kp_frame, des_frame = local_detector.detectAndCompute(
                        gray_frame, None
                    )
                    
                    # Ensure iterable if None
                    if kp_frame is None: kp_frame = []

                    good_matches = []
                    if local_template_des is not None and des_frame is not None:
                        matches = local_bf.match(local_template_des, des_frame)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = matches[:20]

                    # Always draw, even if no matches, to show template and keypoints
                    # Using flags=0 allows drawing single points (unmatched keypoints)
                    vis_img = cv2.drawMatches(
                        local_template_img,
                        local_template_kp if local_template_kp else [],
                        vis_img,
                        kp_frame,
                        good_matches,
                        None,
                        flags=0 
                    )

            # Convert final result to QImage
            # vis_img is BGR (or BGR-like output from drawMatches)
            # Need RGB for QImage
            res_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            h, w, c = res_img_rgb.shape
            qimg = QImage(res_img_rgb.data, w, h, w * c, QImage.Format_RGB888).copy()
            self.result_ready.emit(qimg)

        except Exception as e:
            self.log_signal.emit(f"Worker processing error: {e}")


# Helper for MutexLocker context manager style
class QMutexLocker:
    def __init__(self, mutex):
        self.mutex = mutex

    def __enter__(self):
        self.mutex.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.mutex.unlock()


class MainWindow(QObject):
    update_fps_signal = Signal(float)
    error_signal = Signal(str)

    # Serial Signals
    connect_serial_signal = Signal(str, int)
    disconnect_serial_signal = Signal()
    send_serial_cmd_signal = Signal(str)
    poll_serial_signal = Signal()

    # Signal to send frame to worker
    process_frame_signal = Signal(int, int, int, int, bytes)
    # Signal to update worker params
    update_worker_params_signal = Signal(dict)
    # Signal to set template
    set_worker_template_signal = Signal(str)
    # Signal to set SSIM reference
    set_worker_ssim_ref_signal = Signal(str)
    # Signal to toggle matching
    toggle_worker_matching_signal = Signal(bool)

    # Signal to toggle contours
    toggle_worker_contours_signal = Signal(bool)
    # Signal to update contour params
    update_worker_contour_params_signal = Signal(dict)

    def __init__(self):
        super().__init__()

        # Load UI from file
        loader = QUiLoader()
        loader.registerCustomWidget(RangeSlider)
        ui_file_path = os.path.join(script_dir, "mainwindow.ui")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file_path}: {ui_file.errorString()}")
            sys.exit(-1)

        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            print(loader.errorString())
            sys.exit(-1)

        # Install event filter to handle close event
        self.ui.installEventFilter(self)

        self.ui.video_label.installEventFilter(self)

        self.current_pixmap = None

        # Setup Worker Thread
        self.matching_thread = QThread()
        self.worker = MatchingWorker()
        self.worker.moveToThread(self.matching_thread)

        # Connect Signals
        self.process_frame_signal.connect(self.worker.process_frame)
        self.update_worker_params_signal.connect(self.worker.update_params)
        self.set_worker_template_signal.connect(self.worker.set_template)
        self.set_worker_ssim_ref_signal.connect(self.worker.set_ssim_reference)
        self.toggle_worker_matching_signal.connect(self.worker.toggle_matching)
        
        self.toggle_worker_contours_signal.connect(self.worker.toggle_contours)
        self.update_worker_contour_params_signal.connect(
            self.worker.update_contour_params
        )

        self.worker.result_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log)
        self.worker.qr_found_signal.connect(self.handle_qr_found)
        self.worker.ssim_score_signal.connect(self.handle_ssim_score)
        
        self.matching_thread.start()
        # --- Serial Worker Setup ---
        self.serial_thread = QThread()
        self.serial_worker = SerialWorker()
        self.serial_worker.moveToThread(self.serial_thread)

        # Connect signals
        self.connect_serial_signal.connect(self.serial_worker.connect_serial)
        self.disconnect_serial_signal.connect(self.serial_worker.disconnect_serial)
        self.send_serial_cmd_signal.connect(self.serial_worker.send_command)
        self.poll_serial_signal.connect(self.serial_worker.poll_serial)

        self.serial_worker.log_signal.connect(self.log)
        self.serial_worker.connection_status.connect(self.on_serial_status_changed)

        self.serial_thread.start()

        # Timer for polling serial read
        self.serial_poll_timer = QTimer()
        self.serial_poll_timer.timeout.connect(lambda: self.poll_serial_signal.emit())
        self.serial_poll_timer.start(50) # Poll every 50ms

        # Initial Detector config
        
        if hasattr(self.ui, 'chk_qrcode_enable'):
            self.ui.chk_qrcode_enable.toggled.connect(self.on_qrcode_enable_toggled)

        # SSIM UI
        if hasattr(self.ui, 'btn_load_ssim_ref'):
            self.ui.btn_load_ssim_ref.clicked.connect(self.on_load_ssim_ref_clicked)
        if hasattr(self.ui, 'chk_ssim_enable'):
            self.ui.chk_ssim_enable.toggled.connect(self.on_ssim_enable_toggled)
            
        self.ssim_ref_loaded = False

        # --- Ruler / Measurement Tool Init ---
        self.ruler_active = False
        self.ruler_start = None
        self.ruler_end = None
        self.ruler_calibration = None # None or float (px per mm)

        # Ruler Toolbar Action
        self.action_ruler = QAction(QIcon(os.path.join(script_dir, "icons", "ruler.png")), "Ruler Tool", self)
        self.action_ruler.setCheckable(True)
        self.ui.main_toolbar.addAction(self.action_ruler)
        self.action_ruler.toggled.connect(self.on_ruler_toggled)

        # Ruler GroupBox
        self.group_ruler = QGroupBox("Measurement / Ruler")
        self.layout_ruler = QFormLayout()

        self.spin_ruler_len = QSpinBox() # Using SpinBox for int or DoubleSpinBox? User said "specify length".
        # Let's use QDoubleSpinBox for precision
        self.spin_ruler_len = PySide6.QtWidgets.QDoubleSpinBox()
        self.spin_ruler_len.setRange(0.0, 9999.0)
        self.spin_ruler_len.setValue(10.0) # Default 10mm
        self.spin_ruler_len.setSuffix(" mm")
        self.layout_ruler.addRow("Known Length:", self.spin_ruler_len)

        self.lbl_ruler_px = QLabel("0.0 px")
        self.layout_ruler.addRow("Pixel Dist:", self.lbl_ruler_px)

        self.lbl_ruler_calib = QLabel("Not Calibrated")
        self.layout_ruler.addRow("Scale:", self.lbl_ruler_calib)

        self.lbl_ruler_meas = QLabel("0.00 mm")
        self.lbl_ruler_meas.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        self.layout_ruler.addRow("Measured:", self.lbl_ruler_meas)

        self.btn_ruler_calibrate = QPushButton("Calibrate from Line")
        self.btn_ruler_calibrate.clicked.connect(self.calibrate_ruler)
        self.layout_ruler.addRow(self.btn_ruler_calibrate)
        
        self.chk_show_profile = QCheckBox("Show Intensity Profile")
        self.chk_show_profile.toggled.connect(self.on_show_profile_toggled)
        self.layout_ruler.addRow(self.chk_show_profile)
        
        self.intensity_chart = IntensityChart()

        self.group_ruler.setLayout(self.layout_ruler)
        # Insert into Right Scroll Area at top
        self.ui.scroll_layout_right.insertWidget(0, self.group_ruler)
        self.group_ruler.setVisible(False) # Hide initially until tool is active? Or just keep visible. 
        # User said "Allows user to click-drag... There is a panel". Better to show panel when tool is active or always.
        # I'll keep it visible but maybe disabled? No, let's just leave it visible.
        self.group_ruler.setVisible(False)

        self.update_detector()

        # UI State for Matching
        self.is_matching_ui_active = False
        self.worker_busy = False

        # Status Bar (add permanent widget manually)
        self.fps_label = QLabel("FPS: 0.0")
        self.ui.statusBar().addPermanentWidget(self.fps_label)

        # Log Window Setup
        self.ui.log_text_edit.setMaximumBlockCount(1000)
        self.ui.log_text_edit.setCenterOnScroll(True)
        self.log("Application started.")



        # --- Status Widgets Setup (Programmatic) ---
        self.has_initialized_settings = False
        self.status_items_map = {}  # Map pin -> QListWidgetItem
        self.interrupt_items_map = {}  # Map pin -> QListWidgetItem

        # Status List - already in UI
        # self.ui.list_status
        # self.ui.list_interrupts

        # Serial UI Init
        self.refresh_serial_ports()
        self.ui.btn_serial_refresh.clicked.connect(self.refresh_serial_ports)
        self.ui.btn_serial_connect.clicked.connect(self.on_btn_serial_connect_clicked)
        self.ui.btn_cmd_pulse.clicked.connect(self.on_cmd_pulse)
        # self.ui.btn_cmd_level connection removed
        self.ui.btn_cmd_pwm.clicked.connect(self.on_cmd_pwm)
        self.ui.btn_cmd_stoppwm.clicked.connect(self.on_cmd_stoppwm)
        self.ui.btn_cmd_repeat.clicked.connect(self.on_cmd_repeat)
        self.ui.btn_cmd_stoprepeat.clicked.connect(self.on_cmd_stoprepeat)
        self.ui.btn_cmd_interrupt.clicked.connect(self.on_cmd_interrupt)
        self.ui.btn_cmd_stopinterrupt.clicked.connect(self.on_cmd_stopinterrupt)
        self.ui.btn_cmd_throb.clicked.connect(self.on_cmd_throb)
        self.ui.btn_cmd_stopthrob.clicked.connect(self.on_cmd_stopthrob)
        self.ui.btn_cmd_info.clicked.connect(
            lambda: self.send_serial_cmd_signal.emit("info")
        )
        self.ui.btn_cmd_wifi.clicked.connect(
            lambda: self.send_serial_cmd_signal.emit("wifi")
        )
        self.ui.btn_cmd_mem.clicked.connect(self.on_cmd_mem)

        # --- Setup Pin Level Buttons ---
        self.pin_buttons = {}
        valid_pins = [4, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33]
        
        for pin in valid_pins:
            btn_name = f"btn_pin_{pin}"
            if hasattr(self.ui, btn_name):
                btn = getattr(self.ui, btn_name)
                btn.clicked.connect(lambda checked=False, p=pin: self.toggle_pin_level(p))
                self.pin_buttons[pin] = btn

        # Disable serial tabs initially
        self.ui.tabs_serial_cmds.setEnabled(False)

        # Connections
        self.ui.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        self.ui.chk_roi.toggled.connect(self.on_roi_toggled)

        self.ui.spin_exposure_time.valueChanged.connect(self.on_exposure_time_changed)
        self.ui.slider_exposure.valueChanged.connect(self.on_exposure_slider_changed)

        self.ui.spin_gain.valueChanged.connect(self.on_gain_changed)
        self.ui.slider_gain.valueChanged.connect(self.on_gain_slider_changed)

        self.ui.spin_ae_target.valueChanged.connect(self.on_ae_target_changed)
        self.ui.slider_ae_target.valueChanged.connect(self.on_ae_slider_changed)

        # Recreate ButtonGroup for logic
        self.trigger_bg = QButtonGroup(self.ui)
        self.trigger_bg.addButton(self.ui.rb_continuous, 0)
        self.trigger_bg.addButton(self.ui.rb_software, 1)
        self.trigger_bg.addButton(self.ui.rb_hardware, 2)
        self.trigger_bg.idToggled.connect(self.on_trigger_mode_changed)

        self.ui.btn_soft_trigger.clicked.connect(self.on_soft_trigger_clicked)

        # New Connections for Trigger Params
        self.ui.spin_trigger_count.valueChanged.connect(self.on_trigger_count_changed)
        self.ui.spin_trigger_delay.valueChanged.connect(self.on_trigger_delay_changed)
        self.ui.spin_trigger_interval.valueChanged.connect(
            self.on_trigger_interval_changed
        )

        # New Connections for External Trigger Params
        self.ui.combo_ext_mode.currentIndexChanged.connect(self.on_ext_mode_changed)
        self.ui.spin_ext_jitter.valueChanged.connect(self.on_ext_jitter_changed)
        self.ui.combo_ext_shutter.currentIndexChanged.connect(
            self.on_ext_shutter_changed
        )

        # New Connections for Strobe Params
        self.ui.combo_strobe_mode.currentIndexChanged.connect(self.on_strobe_mode_changed)
        self.ui.combo_strobe_polarity.currentIndexChanged.connect(self.on_strobe_polarity_changed)
        self.ui.spin_strobe_delay.valueChanged.connect(self.on_strobe_delay_changed)
        self.ui.spin_strobe_width.valueChanged.connect(self.on_strobe_width_changed)
        
        # Retrieve Actions
        self.ui.action_start_camera.triggered.connect(self.on_start_clicked)
        self.ui.action_stop_camera.triggered.connect(self.on_stop_clicked)
        self.ui.action_record.triggered.connect(self.on_record_clicked)
        self.ui.action_snapshot.triggered.connect(self.on_snapshot_clicked)

        # Template Matching UI
        if hasattr(self.ui, 'btn_load_template'):
            self.ui.btn_load_template.clicked.connect(self.on_load_template_clicked)
        if hasattr(self.ui, 'chk_match_enable'):
            self.ui.chk_match_enable.toggled.connect(self.on_match_enable_toggled)
            
        self.template_loaded = False
        
        # --- Contour Controls ---
        self.btn_toggle_contours = self.ui.btn_toggle_contours
        self.combo_contour_mode = self.ui.combo_contour_mode
        self.slider_threshold = self.ui.slider_threshold
        self.lbl_threshold_val = self.ui.lbl_threshold_val
        self.slider_canny = self.ui.slider_canny
        self.lbl_canny_val = self.ui.lbl_canny_val
        self.spin_min_area = self.ui.spin_min_area
        self.spin_max_area = self.ui.spin_max_area
        self.chk_fill_contours = self.ui.chk_fill_contours
        self.chk_show_box = self.ui.chk_show_box
        if self.slider_canny:
            self.slider_canny.setRange(0, 255)
            self.slider_canny.setValues(50, 150)

        # Connect Contour Signals
        self.btn_toggle_contours.toggled.connect(self.on_toggle_contours_toggled)
        self.combo_contour_mode.currentTextChanged.connect(
            self.on_contour_params_changed
        )
        self.slider_threshold.valueChanged.connect(self.on_contour_params_changed)
        self.slider_canny.valuesChanged.connect(self.on_contour_params_changed)
        self.spin_min_area.valueChanged.connect(self.on_contour_params_changed)
        self.spin_max_area.valueChanged.connect(self.on_contour_params_changed)
        self.chk_fill_contours.toggled.connect(self.on_contour_params_changed)
        self.chk_show_box.toggled.connect(self.on_contour_params_changed)
        # -----------------------------------------

        # Matching Tabs Connection
        self.ui.tabs_matching.currentChanged.connect(self.on_detector_params_changed)

        # ORB Parameter Connections
        self.ui.orb_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_scaleFactor.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_nlevels.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_firstLevel.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_wta_k.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_scoreType.currentIndexChanged.connect(
            self.on_detector_params_changed
        )
        self.ui.orb_patchSize.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_fastThreshold.valueChanged.connect(self.on_detector_params_changed)

        # SIFT Parameter Connections
        self.ui.sift_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_nOctaveLayers.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_contrastThreshold.valueChanged.connect(
            self.on_detector_params_changed
        )
        self.ui.sift_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_sigma.valueChanged.connect(self.on_detector_params_changed)

        # AKAZE Parameter Connections
        self.ui.akaze_descriptor_type.currentIndexChanged.connect(
            self.on_detector_params_changed
        )
        self.ui.akaze_threshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.akaze_nOctaves.valueChanged.connect(self.on_detector_params_changed)
        self.ui.akaze_nOctaveLayers.valueChanged.connect(
            self.on_detector_params_changed
        )

        # ArUco Parameter Connections
        self.ui.aruco_dict.currentTextChanged.connect(self.on_detector_params_changed)
        self.ui.chk_aruco_show_ids.toggled.connect(self.on_detector_params_changed)
        self.ui.chk_aruco_show_rejected.toggled.connect(self.on_detector_params_changed)
        self.ui.spin_aruco_border_bits.valueChanged.connect(
            self.on_detector_params_changed
        )

        # New ArUco Enable Checkbox
        self.ui.chk_aruco_enable.toggled.connect(self.on_aruco_enable_toggled)

        # Camera Setup
        self.camera = MindVisionCamera()

        # Register Callbacks
        self.camera.registerFrameCallback(self.frame_callback)
        self.camera.registerFpsCallback(self.fps_callback)
        self.camera.registerErrorCallback(self.error_callback)

        # Signals
        self.update_fps_signal.connect(self.update_fps)
        self.error_signal.connect(self.handle_error)

        self.video_thread = VideoThread()
        self.recording_requested = False
        self.current_fps = 30.0
        self.last_ui_update_time = 0

        self.load_settings()

    def show(self):
        self.ui.showMaximized()

    def close(self):
        # Triggers closeEvent via the widget
        self.ui.close()

    def eventFilter(self, watched, event):
        try:
            if watched is self.ui and event.type() == QEvent.Close:
                self.save_settings()
                self.on_stop_clicked()

                # Cleanup Matching Thread
                if self.matching_thread.isRunning():
                    self.matching_thread.quit()
                    self.matching_thread.wait()

                # Cleanup Serial Thread
                if self.serial_thread.isRunning():
                    self.serial_thread.quit()
                    self.serial_thread.wait()

                self.ui.removeEventFilter(self)
                try:
                    self.ui.video_label.removeEventFilter(self)
                except RuntimeError:
                    pass
            elif watched == self.ui.video_label:
                if event.type() == QEvent.Resize:
                    self.refresh_video_label()
                elif self.ruler_active:
                    if event.type() == QEvent.MouseButtonPress:
                        if event.button() == Qt.LeftButton:
                            pos = self.get_image_coords(event.position())
                            if pos:
                                self.ruler_start = pos
                                self.ruler_end = pos
                                self.refresh_video_label()
                    elif event.type() == QEvent.MouseMove:
                        if self.ruler_start is not None and event.buttons() & Qt.LeftButton:
                            pos = self.get_image_coords(event.position())
                            if pos:
                                self.ruler_end = pos
                                self.update_ruler_stats()
                                self.refresh_video_label()
                    elif event.type() == QEvent.MouseButtonRelease:
                        if event.button() == Qt.LeftButton and self.ruler_start is not None:
                            pos = self.get_image_coords(event.position())
                            if pos:
                                self.ruler_end = pos
                                self.update_ruler_stats()
                                self.refresh_video_label()
        except RuntimeError:
            pass
        return super().eventFilter(watched, event)

    def refresh_video_label(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            # If Ruler is active and we have points, draw them on a temporary pixmap
            # We draw on the original resolution pixmap before scaling
            
            # Optimization: If we are just dragging, maybe we shouldn't clone the pixmap every time 
            # if the image is huge (5MP+). But for UI responsiveness it's usually fine.
            display_pixmap = self.current_pixmap
            
            if self.ruler_active and (self.ruler_start is not None):
                # Create a copy to draw on
                display_pixmap = self.current_pixmap.copy()
                painter = QPainter(display_pixmap)
                pen = QPen(QColor(0, 255, 255), 2) # Cyan, 2px
                painter.setPen(pen)
                
                start_pt = self.ruler_start
                end_pt = self.ruler_end if self.ruler_end is not None else start_pt
                
                painter.drawLine(start_pt, end_pt)
                
                # Draw endpoints
                pen.setWidth(4)
                painter.setPen(pen)
                painter.drawPoint(start_pt)
                painter.drawPoint(end_pt)
                
                painter.end()

            scaled = display_pixmap.scaled(
                self.ui.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
            self.ui.video_label.setPixmap(scaled)

    @Slot(str)
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        if hasattr(self.ui, "log_text_edit"):
            self.ui.log_text_edit.appendPlainText(f"[{timestamp}] {message}")

        # Parse Rx messages
        if message.startswith("Rx: "):
            content = message[4:].strip()
            self.process_serial_line(content)

    def process_serial_line(self, line):
        # Startup detection
        if "LED>" in line and not self.has_initialized_settings:
            self.has_initialized_settings = True
            QTimer.singleShot(
                500, lambda: self.send_serial_cmd_signal.emit("printsettings")
            )
            return

        parts = line.split()
        if not parts:
            return

        cmd = parts[0]

        try:
            if cmd == "level" and len(parts) >= 3:
                pin = int(parts[1])
                val = int(parts[2])
                # self.update_pin_status(pin, f"Level: {val}") # Removed per user request

                # If the pin was in the status list (e.g. PWM/Repeat), remove it as it's now just a simple level
                if pin in self.status_items_map:
                    item = self.status_items_map.pop(pin)
                    row = self.ui.list_status.row(item)
                    self.ui.list_status.takeItem(row)

                # Update Pin Button State
                if hasattr(self, "pin_buttons") and pin in self.pin_buttons:
                    btn = self.pin_buttons[pin]
                    if val == 1:
                        btn.setStyleSheet("background-color: red; color: white;")
                    else:
                        btn.setStyleSheet("background-color: none;")

                # Update old UI setters if they exist (legacy/fallback)
                if hasattr(self.ui, "spin_level_pin"):
                    self.ui.spin_level_pin.setValue(pin)
                if hasattr(self.ui, "combo_level_val"):
                    self.ui.combo_level_val.setCurrentIndex(val + 1)

            elif cmd == "pwm" and len(parts) >= 4:
                pin = int(parts[1])
                freq = int(parts[2])
                duty = int(parts[3])
                self.update_pin_status(pin, f"PWM: {freq}Hz, {duty}%")
                # Update UI setters
                self.ui.spin_pwm_pin.setValue(pin)
                self.ui.spin_pwm_freq.setValue(freq)
                self.ui.spin_pwm_duty.setValue(duty)

            elif cmd == "repeat" and len(parts) >= 4:
                pin = int(parts[1])
                freq = int(parts[2])
                dur = int(parts[3])
                self.update_pin_status(pin, f"Repeat: {freq}Hz, {dur}us")
                # Update UI setters
                self.ui.spin_repeat_pin.setValue(pin)
                self.ui.spin_repeat_freq.setValue(freq)
                self.ui.spin_repeat_dur.setValue(dur)

            elif cmd == "throb" and len(parts) >= 5:
                period = int(parts[1])
                p1 = int(parts[2])
                p2 = int(parts[3])
                p3 = int(parts[4])
                self.update_pin_status(p1, "Throb")
                self.update_pin_status(p2, "Throb")
                self.update_pin_status(p3, "Throb")
                # Update UI setters
                self.ui.spin_throb_period.setValue(period)
                self.ui.spin_throb_p1.setValue(p1)
                self.ui.spin_throb_p2.setValue(p2)
                self.ui.spin_throb_p3.setValue(p3)

            elif cmd == "interrupt" and len(parts) >= 5:
                pin = int(parts[1])
                edge = parts[2]
                tgt = int(parts[3])
                width = int(parts[4])
                self.update_interrupt_status(pin, f"{edge} -> Pulse {tgt} ({width}us)")
                # Update UI setters
                self.ui.spin_int_pin.setValue(pin)
                idx = self.ui.combo_int_edge.findText(edge)
                if idx >= 0:
                    self.ui.combo_int_edge.setCurrentIndex(idx)
                self.ui.spin_int_target.setValue(tgt)
                self.ui.spin_int_width.setValue(width)

        except ValueError:
            pass

    def update_pin_status(self, pin, status_text):
        text = f"Pin {pin}: {status_text}"
        if pin in self.status_items_map:
            self.status_items_map[pin].setText(text)
        else:
            item = QListWidgetItem(text)
            self.ui.list_status.addItem(item)
            self.status_items_map[pin] = item

    def update_interrupt_status(self, pin, status_text):
        text = f"Pin {pin}: {status_text}"
        if pin in self.interrupt_items_map:
            self.interrupt_items_map[pin].setText(text)
        else:
            item = QListWidgetItem(text)
            self.ui.list_interrupts.addItem(item)
            self.interrupt_items_map[pin] = item

    def update_detector(self):
        # SSIM Priority
        if hasattr(self.ui, 'chk_ssim_enable') and self.ui.chk_ssim_enable.isChecked():
            params = {'algo': 'SSIM'}
            self.update_worker_params_signal.emit(params)
            return

        # QR Code Priority
        if (
            hasattr(self.ui, "chk_qrcode_enable")
            and self.ui.chk_qrcode_enable.isChecked()
        ):
            params = {"algo": "QRCODE"}
            self.update_worker_params_signal.emit(params)
            return

        # ArUco Priority
        if (
            hasattr(self.ui, "chk_aruco_enable")
            and self.ui.chk_aruco_enable.isChecked()
        ):
            params = {"algo": "ARUCO"}
            if hasattr(self.ui, "aruco_dict"):
                params["dict"] = self.ui.aruco_dict.currentText()
            if hasattr(self.ui, "chk_aruco_show_ids"):
                params["show_ids"] = self.ui.chk_aruco_show_ids.isChecked()
            if hasattr(self.ui, "chk_aruco_show_rejected"):
                params["show_rejected"] = self.ui.chk_aruco_show_rejected.isChecked()
            if hasattr(self.ui, "spin_aruco_border_bits"):
                params["markerBorderBits"] = self.ui.spin_aruco_border_bits.value()
            self.update_worker_params_signal.emit(params)
            return

        tab_index = self.ui.tabs_matching.currentIndex()
        params = {}

        if tab_index == 0:  # ORB
            params["algo"] = "ORB"
            params["nfeatures"] = self.ui.orb_nfeatures.value()
            params["scaleFactor"] = self.ui.orb_scaleFactor.value()
            params["nlevels"] = self.ui.orb_nlevels.value()
            params["edgeThreshold"] = self.ui.orb_edgeThreshold.value()
            params["firstLevel"] = self.ui.orb_firstLevel.value()
            params["WTA_K"] = self.ui.orb_wta_k.value()
            params["scoreType"] = self.ui.orb_scoreType.currentIndex()
            params["patchSize"] = self.ui.orb_patchSize.value()
            params["fastThreshold"] = self.ui.orb_fastThreshold.value()

        elif tab_index == 1:  # SIFT
            params["algo"] = "SIFT"
            params["nfeatures"] = self.ui.sift_nfeatures.value()
            params["nOctaveLayers"] = self.ui.sift_nOctaveLayers.value()
            params["contrastThreshold"] = self.ui.sift_contrastThreshold.value()
            params["edgeThreshold"] = self.ui.sift_edgeThreshold.value()
            params["sigma"] = self.ui.sift_sigma.value()

        elif tab_index == 2:  # AKAZE
            params["algo"] = "AKAZE"
            combo_idx = self.ui.akaze_descriptor_type.currentIndex()
            mapping = [2, 3, 4, 5]
            params["descriptor_type"] = mapping[combo_idx]
            params["threshold"] = self.ui.akaze_threshold.value()
            params["nOctaves"] = self.ui.akaze_nOctaves.value()
            params["nOctaveLayers"] = self.ui.akaze_nOctaveLayers.value()

        self.update_worker_params_signal.emit(params)

    def on_detector_params_changed(self):
        self.update_detector()

    def frame_callback(self, width, height, bytes_per_line, fmt, data):
        # 1. Recording (High Priority)
        if self.video_thread.isRunning():
            try:
                self.video_thread.addFrameBytes(
                    width, height, bytes_per_line, fmt, data
                )
            except Exception as e:
                self.log(f"Recording error in callback: {e}")

        # 2. UI Update (Throttled)
        current_time = time.time()
        # Cap sending to UI/Worker at ~30 FPS to avoid overwhelming event loop
        if current_time - self.last_ui_update_time > 0.033:
            self.last_ui_update_time = current_time
            try:
                # Flow Control: Only send to worker if it's not busy
                if not self.worker_busy:
                    self.worker_busy = True
                    # Copy data for worker thread
                    data_copy = bytes(data)
                    self.process_frame_signal.emit(
                        width, height, bytes_per_line, fmt, data_copy
                    )
                # Else: Drop frame for UI display to prevent backlog/latency
            except Exception as e:
                self.worker_busy = False  # Reset on error
                self.log(f"Error in frame callback (UI): {e}")

    def fps_callback(self, fps):
        self.update_fps_signal.emit(fps)

    def error_callback(self, msg):
        self.error_signal.emit(msg)

    @Slot(QImage)
    def update_frame(self, image):
        self.worker_busy = False
        if not image.isNull():
            # Recording Start Trigger
            if self.recording_requested:
                self.recording_requested = False
                record_fps = self.current_fps if self.current_fps > 0.1 else 30.0
                self.video_thread.startRecording(
                    image.width(), image.height(), record_fps, "output.mkv"
                )
                self.ui.action_record.setText("Stop Recording")
                self.log("Recording started: output.mkv")

            # Display
            self.current_pixmap = QPixmap.fromImage(image)
            self.refresh_video_label()

    @Slot(float)
    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def handle_error(self, message):
        self.log(f"Camera Error: {message}")
        self.ui.video_label.setText(f"Error: {message}")
        self.ui.action_start_camera.setEnabled(True)
        self.ui.action_stop_camera.setEnabled(False)
        self.ui.controls_group.setEnabled(False)
        self.ui.trigger_group.setEnabled(False)
        self.ui.trigger_params_group.setEnabled(False)
        self.ui.ext_trigger_group.setEnabled(False)
        self.ui.strobe_group.setEnabled(False)

    def on_start_clicked(self):
        if self.camera.open():
            if self.camera.start():
                self.ui.action_start_camera.setEnabled(False)
                self.ui.action_stop_camera.setEnabled(True)
                self.ui.action_record.setEnabled(True)
                self.ui.action_snapshot.setEnabled(True)
                self.ui.controls_group.setEnabled(True)
                self.ui.trigger_group.setEnabled(True)
                self.ui.strobe_group.setEnabled(True)

                self.ui.video_label.setText("Starting stream...")
                self.sync_ui()
                self.apply_camera_settings()
                self.log("Camera started.")

    def on_stop_clicked(self):
        if self.video_thread.isRunning():
            self.on_record_clicked()

        self.camera.stop()
        self.camera.close()
        self.ui.action_start_camera.setEnabled(True)
        self.ui.action_stop_camera.setEnabled(False)
        self.ui.action_record.setEnabled(False)
        self.ui.action_snapshot.setEnabled(False)

        self.ui.controls_group.setEnabled(False)
        self.ui.trigger_group.setEnabled(False)
        self.ui.trigger_params_group.setEnabled(False)
        self.ui.ext_trigger_group.setEnabled(False)
        self.ui.strobe_group.setEnabled(False)

        self.ui.video_label.clear()
        self.ui.video_label.setText("Camera Stopped")
        self.fps_label.setText("FPS: 0.0")
        self.log("Camera stopped.")

    def on_record_clicked(self):
        if not self.video_thread.isRunning():
            self.recording_requested = True
            self.log("Recording requested...")
        else:
            self.video_thread.stopRecording()
            self.ui.action_record.setText("Start Recording")
            self.log("Recording stopped.")

    def on_snapshot_clicked(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            filename = f"snapshot_{int(time.time())}.png"
            self.current_pixmap.save(filename)
            self.log(f"Snapshot saved: {filename}")

    def on_aruco_enable_toggled(self, checked):
        if checked:
            # Mutual exclusion: disable template matching if active
            if (
                hasattr(self.ui, "chk_match_enable")
                and self.ui.chk_match_enable.isChecked()
            ):
                self.ui.chk_match_enable.blockSignals(True)
                self.ui.chk_match_enable.setChecked(False)
                self.ui.chk_match_enable.blockSignals(False)
                self.is_matching_ui_active = False

            # Mutual exclusion: disable QR Code if active
            if (
                hasattr(self.ui, "chk_qrcode_enable")
                and self.ui.chk_qrcode_enable.isChecked()
            ):
                self.ui.chk_qrcode_enable.blockSignals(True)
                self.ui.chk_qrcode_enable.setChecked(False)
                self.ui.chk_qrcode_enable.blockSignals(False)

            # Mutual exclusion: disable SSIM if active
            if (
                hasattr(self.ui, "chk_ssim_enable")
                and self.ui.chk_ssim_enable.isChecked()
            ):
                self.ui.chk_ssim_enable.blockSignals(True)
                self.ui.chk_ssim_enable.setChecked(False)
                self.ui.chk_ssim_enable.blockSignals(False)
            
            # Enable ArUco (update_detector picks up the checked state)
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            # Disable ArUco
            # Only stop worker if not switching to others (which shouldn't happen here directly)
            if not self.is_matching_ui_active:
                self.toggle_worker_matching_signal.emit(False)

    def on_load_template_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Template Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.set_worker_template_signal.emit(file_path)
            self.template_loaded = True

            # Update Label
            filename = os.path.basename(file_path)
            if hasattr(self.ui, "lbl_template_name"):
                self.ui.lbl_template_name.setText(filename)
                self.ui.lbl_template_name.setStyleSheet("color: black;")

            # If enabled, update worker
            if (
                hasattr(self.ui, "chk_match_enable")
                and self.ui.chk_match_enable.isChecked()
            ):
                # self.update_detector() # Redundant, worker should already be up to date
                self.toggle_worker_matching_signal.emit(True)

    def on_match_enable_toggled(self, checked):
        if checked:
            if not self.template_loaded:
                # If no template, try to load one
                self.on_load_template_clicked()
                # If still not loaded (user cancelled), uncheck
                if not self.template_loaded:
                    self.ui.chk_match_enable.setChecked(False)
                    return

            # Mutual Exclusion: Disable ArUco
            if (
                hasattr(self.ui, "chk_aruco_enable")
                and self.ui.chk_aruco_enable.isChecked()
            ):
                self.ui.chk_aruco_enable.blockSignals(True)
                self.ui.chk_aruco_enable.setChecked(False)
                self.ui.chk_aruco_enable.blockSignals(False)

            # Mutual Exclusion: Disable QR Code
            if (
                hasattr(self.ui, "chk_qrcode_enable")
                and self.ui.chk_qrcode_enable.isChecked()
            ):
                self.ui.chk_qrcode_enable.blockSignals(True)
                self.ui.chk_qrcode_enable.setChecked(False)
                self.ui.chk_qrcode_enable.blockSignals(False)

            # Mutual Exclusion: Disable SSIM
            if (
                hasattr(self.ui, "chk_ssim_enable")
                and self.ui.chk_ssim_enable.isChecked()
            ):
                self.ui.chk_ssim_enable.blockSignals(True)
                self.ui.chk_ssim_enable.setChecked(False)
                self.ui.chk_ssim_enable.blockSignals(False)

            self.is_matching_ui_active = True
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            self.is_matching_ui_active = False
            self.toggle_worker_matching_signal.emit(False)

    def on_qrcode_enable_toggled(self, checked):
        if checked:
            # Mutual exclusion: disable template matching if active
            if (
                hasattr(self.ui, "chk_match_enable")
                and self.ui.chk_match_enable.isChecked()
            ):
                self.ui.chk_match_enable.blockSignals(True)
                self.ui.chk_match_enable.setChecked(False)
                self.ui.chk_match_enable.blockSignals(False)
                self.is_matching_ui_active = False

            # Mutual exclusion: disable ArUco
            if (
                hasattr(self.ui, "chk_aruco_enable")
                and self.ui.chk_aruco_enable.isChecked()
            ):
                self.ui.chk_aruco_enable.blockSignals(True)
                self.ui.chk_aruco_enable.setChecked(False)
                self.ui.chk_aruco_enable.blockSignals(False)

            # Mutual exclusion: disable SSIM
            if (
                hasattr(self.ui, "chk_ssim_enable")
                and self.ui.chk_ssim_enable.isChecked()
            ):
                self.ui.chk_ssim_enable.blockSignals(True)
                self.ui.chk_ssim_enable.setChecked(False)
                self.ui.chk_ssim_enable.blockSignals(False)
            
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            if not self.is_matching_ui_active:
                self.toggle_worker_matching_signal.emit(False)

    @Slot(str)
    def handle_qr_found(self, text):
        if hasattr(self.ui, "lbl_qrcode_data"):
            self.ui.lbl_qrcode_data.setText(f"Decoded Data: {text}")

    @Slot(float)
    def handle_ssim_score(self, score):
        if hasattr(self.ui, "lbl_ssim_score"):
            self.ui.lbl_ssim_score.setText(f"Score: {score:.4f}")

    def on_load_ssim_ref_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Reference Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.set_worker_ssim_ref_signal.emit(file_path)
            self.ssim_ref_loaded = True
            
            filename = os.path.basename(file_path)
            if hasattr(self.ui, "lbl_ssim_ref_name"):
                self.ui.lbl_ssim_ref_name.setText(filename)
                self.ui.lbl_ssim_ref_name.setStyleSheet("color: black;")
            
            if hasattr(self.ui, "chk_ssim_enable") and self.ui.chk_ssim_enable.isChecked():
                self.update_detector()
                self.toggle_worker_matching_signal.emit(True)

    def on_ssim_enable_toggled(self, checked):
        if checked:
            if not self.ssim_ref_loaded:
                self.on_load_ssim_ref_clicked()
                if not self.ssim_ref_loaded:
                    self.ui.chk_ssim_enable.setChecked(False)
                    return

            # Mutual Exclusion
            for chk_name in ["chk_match_enable", "chk_aruco_enable", "chk_qrcode_enable"]:
                if hasattr(self.ui, chk_name):
                    chk = getattr(self.ui, chk_name)
                    if chk.isChecked():
                        chk.blockSignals(True)
                        chk.setChecked(False)
                        chk.blockSignals(False)
            
            self.is_matching_ui_active = True
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            self.is_matching_ui_active = False
            self.toggle_worker_matching_signal.emit(False)

    def on_toggle_contours_toggled(self, checked):
        if checked:
            self.btn_toggle_contours.setText("Disable Contours")
        else:
            self.btn_toggle_contours.setText("Enable Contours")

        self.on_contour_params_changed()
        self.toggle_worker_contours_signal.emit(checked)

    def on_contour_params_changed(self):
        # Update visibility of controls based on mode
        mode = self.combo_contour_mode.currentText()
        is_canny = mode == "Canny"

        self.slider_canny.setEnabled(is_canny)
        self.slider_threshold.setEnabled(not is_canny)

        # Update labels
        self.lbl_threshold_val.setText(str(self.slider_threshold.value()))
        c_min, c_max = self.slider_canny.getValues()
        self.lbl_canny_val.setText(f"{c_min} - {c_max}")

        params = {
            "mode": mode,
            "threshold": self.slider_threshold.value(),
            "thresh_min": c_min,
            "thresh_max": c_max,
            "min_area": self.spin_min_area.value(),
            "max_area": self.spin_max_area.value(),
            "fill": self.chk_fill_contours.isChecked(),
            "box": self.chk_show_box.isChecked(),
        }
        self.update_worker_contour_params_signal.emit(params)

    def sync_ui(self):
        # Ranges
        min_exp, max_exp = self.camera.getExposureTimeRange()
        step_exp = self.camera.getExposureTimeStep()

        self.ui.spin_exposure_time.setRange(min_exp, max_exp)
        if step_exp > 0:
            self.ui.spin_exposure_time.setSingleStep(step_exp)
            self.ui.slider_exposure.setRange(0, int((max_exp - min_exp) / step_exp))
        else:
            self.ui.slider_exposure.setRange(0, 10000)

        min_gain, max_gain = self.camera.getAnalogGainRange()
        self.ui.spin_gain.setRange(min_gain, max_gain)
        self.ui.slider_gain.setRange(min_gain, max_gain)

        # Values
        is_auto = self.camera.getAutoExposure()
        self.ui.chk_auto_exposure.setChecked(is_auto)
        self.ui.spin_exposure_time.setEnabled(not is_auto)
        self.ui.slider_exposure.setEnabled(not is_auto)
        self.ui.spin_ae_target.setEnabled(is_auto)
        self.ui.slider_ae_target.setEnabled(is_auto)

        # AE Target
        if hasattr(self.camera, "getAutoExposureTarget"):
            try:
                if hasattr(self.camera, "getAutoExposureTargetRange"):
                    min_ae, max_ae = self.camera.getAutoExposureTargetRange()
                    self.ui.spin_ae_target.setRange(min_ae, max_ae)
                    self.ui.slider_ae_target.setRange(min_ae, max_ae)

                current_ae = self.camera.getAutoExposureTarget()
                self.ui.spin_ae_target.blockSignals(True)
                self.ui.spin_ae_target.setValue(current_ae)
                self.ui.spin_ae_target.blockSignals(False)

                self.ui.slider_ae_target.blockSignals(True)
                self.ui.slider_ae_target.setValue(current_ae)
                self.ui.slider_ae_target.blockSignals(False)
            except Exception as e:
                self.log(f"Error syncing AE target: {e}")

        current_exp = self.camera.getExposureTime()
        self.ui.spin_exposure_time.blockSignals(True)
        self.ui.spin_exposure_time.setValue(current_exp)
        self.ui.spin_exposure_time.blockSignals(False)

        self.update_slider_from_time(current_exp, min_exp, max_exp)

        self.ui.spin_gain.blockSignals(True)
        self.ui.spin_gain.setValue(self.camera.getAnalogGain())
        self.ui.spin_gain.blockSignals(False)

        self.ui.slider_gain.blockSignals(True)
        self.ui.slider_gain.setValue(self.camera.getAnalogGain())
        self.ui.slider_gain.blockSignals(False)

    def update_slider_from_time(self, current, min_val, max_val):
        self.ui.slider_exposure.blockSignals(True)
        step_exp = self.camera.getExposureTimeStep()
        if step_exp > 0:
            val = int(round((current - min_val) / step_exp))
            self.ui.slider_exposure.setValue(val)
        else:
            rng = max_val - min_val
            if rng > 0:
                val = int((current - min_val) / rng * 10000)
                self.ui.slider_exposure.setValue(val)
        self.ui.slider_exposure.blockSignals(False)

    def on_auto_exposure_toggled(self, checked):
        if self.camera.setAutoExposure(checked):
            self.ui.spin_exposure_time.setEnabled(not checked)
            self.ui.slider_exposure.setEnabled(not checked)
            self.ui.spin_ae_target.setEnabled(checked)
            self.ui.slider_ae_target.setEnabled(checked)
            if not checked:
                # Update manual values
                current_exp = self.camera.getExposureTime()
                self.ui.spin_exposure_time.setValue(current_exp)
                min_exp = self.ui.spin_exposure_time.minimum()
                max_exp = self.ui.spin_exposure_time.maximum()
                self.update_slider_from_time(current_exp, min_exp, max_exp)
        else:
            self.ui.chk_auto_exposure.setChecked(not checked)

    def on_roi_toggled(self, checked):
        if not self.camera.setRoi(checked):
            self.ui.chk_roi.setChecked(not checked)

    def on_exposure_time_changed(self, value):
        self.camera.setExposureTime(value)
        actual = self.camera.getExposureTime()
        self.ui.spin_exposure_time.blockSignals(True)
        self.ui.spin_exposure_time.setValue(actual)
        self.ui.spin_exposure_time.blockSignals(False)

        min_exp = self.ui.spin_exposure_time.minimum()
        max_exp = self.ui.spin_exposure_time.maximum()
        self.update_slider_from_time(actual, min_exp, max_exp)

    def on_exposure_slider_changed(self, value):
        min_exp = self.ui.spin_exposure_time.minimum()
        step_exp = self.camera.getExposureTimeStep()
        if step_exp > 0:
            new_time = min_exp + (value * step_exp)
        else:
            max_exp = self.ui.spin_exposure_time.maximum()
            rng = max_exp - min_exp
            new_time = min_exp + (value / 10000.0) * rng
        self.ui.spin_exposure_time.setValue(new_time)

    def on_gain_changed(self, value):
        self.camera.setAnalogGain(value)
        self.ui.slider_gain.blockSignals(True)
        self.ui.slider_gain.setValue(value)
        self.ui.slider_gain.blockSignals(False)

    def on_gain_slider_changed(self, value):
        self.ui.spin_gain.setValue(value)

    def on_ae_target_changed(self, value):
        if hasattr(self.camera, "setAutoExposureTarget"):
            self.camera.setAutoExposureTarget(value)
            self.ui.slider_ae_target.blockSignals(True)
            self.ui.slider_ae_target.setValue(value)
            self.ui.slider_ae_target.blockSignals(False)

    def on_ae_slider_changed(self, value):
        self.ui.spin_ae_target.setValue(value)

    def on_trigger_mode_changed(self, id, checked):
        if checked:
            # 0=Continuous, 1=Software, 2=Hardware
            if self.camera.setTriggerMode(id):
                self.ui.btn_soft_trigger.setEnabled(id == 1)

                # Logic for Trigger Params Group
                # Enabled if Software (1) or Hardware (2)
                self.ui.trigger_params_group.setEnabled(id in [1, 2])

                # Logic for External Trigger Params Group
                # Enabled if Hardware (2)
                self.ui.ext_trigger_group.setEnabled(id == 2)

            else:
                self.log(f"Failed to set trigger mode {id}")

    def on_soft_trigger_clicked(self):
        self.camera.triggerSoftware()

    # Trigger Parameter Slots
    def on_trigger_count_changed(self, value):
        if hasattr(self.camera, "setTriggerCount"):
            self.camera.setTriggerCount(value)

    def on_trigger_delay_changed(self, value):
        if hasattr(self.camera, "setTriggerDelay"):
            self.camera.setTriggerDelay(value)

    def on_trigger_interval_changed(self, value):
        if hasattr(self.camera, "setTriggerInterval"):
            self.camera.setTriggerInterval(value)

    # External Trigger Slots
    def on_ext_mode_changed(self, index):
        if hasattr(self.camera, "setExternalTriggerSignalType"):
            self.camera.setExternalTriggerSignalType(index)

    def on_ext_jitter_changed(self, value):
        if hasattr(self.camera, "setExternalTriggerJitterTime"):
            self.camera.setExternalTriggerJitterTime(value)

    def on_ext_shutter_changed(self, index):
        if hasattr(self.camera, "setExternalTriggerShutterMode"):
            self.camera.setExternalTriggerShutterMode(index)

    # Strobe Slots
    def on_strobe_mode_changed(self, index):
        # 0 = Auto, 1 = Manual/Semi-Auto
        if hasattr(self.camera, "setStrobeMode"):
            self.camera.setStrobeMode(index)
            # Enable manual controls if index == 1
            is_manual = index == 1
            self.ui.combo_strobe_polarity.setEnabled(is_manual)
            self.ui.spin_strobe_delay.setEnabled(is_manual)
            self.ui.spin_strobe_width.setEnabled(is_manual)

    def on_strobe_polarity_changed(self, index):
        if hasattr(self.camera, "setStrobePolarity"):
            self.camera.setStrobePolarity(index)

    def on_strobe_delay_changed(self, value):
        if hasattr(self.camera, "setStrobeDelayTime"):
            self.camera.setStrobeDelayTime(value)

    def on_strobe_width_changed(self, value):
        if hasattr(self.camera, "setStrobePulseWidth"):
            self.camera.setStrobePulseWidth(value)

    # --- Serial Control Methods ---
    def refresh_serial_ports(self):
        self.ui.combo_serial_port.clear()
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.ui.combo_serial_port.addItem(f"{port.device}")
        else:
            self.ui.combo_serial_port.addItem("No pyserial")
            self.ui.btn_serial_connect.setEnabled(False)

    def on_btn_serial_connect_clicked(self, checked):
        if checked:
            port = self.ui.combo_serial_port.currentText().split()[
                0
            ]  # Handle "COM3 - Desc" if needed
            if port:
                self.connect_serial_signal.emit(port, 115200)
                self.ui.btn_serial_connect.setText("Connecting...")
            else:
                self.ui.btn_serial_connect.setChecked(False)
        else:
            self.disconnect_serial_signal.emit()

    @Slot(bool)
    def on_serial_status_changed(self, connected):
        self.ui.btn_serial_connect.setChecked(connected)
        self.ui.btn_serial_connect.setText("Disconnect" if connected else "Connect")
        self.ui.tabs_serial_cmds.setEnabled(connected)
        self.ui.combo_serial_port.setEnabled(not connected)
        self.ui.btn_serial_refresh.setEnabled(not connected)

        if not connected:
            self.has_initialized_settings = False
            self.ui.list_status.clear()
            self.status_items_map.clear()
            self.ui.list_interrupts.clear()
            self.interrupt_items_map.clear()

    def on_cmd_pulse(self):
        pin = self.ui.spin_pulse_pin.value()
        val = self.ui.spin_pulse_val.value()
        dur = self.ui.spin_pulse_dur.value()
        self.send_serial_cmd_signal.emit(f"pulse {pin} {val} {dur}")

    def toggle_pin_level(self, pin):
        btn = self.pin_buttons.get(pin)
        if not btn:
            return

        # Check current visual state (High=Red)
        is_high = "red" in btn.styleSheet()
        # Toggle: If high, set to 0. If low, set to 1.
        new_val = 0 if is_high else 1

        self.send_serial_cmd_signal.emit(f"level {pin} {new_val}")

    def on_cmd_pwm(self):
        pin = self.ui.spin_pwm_pin.value()
        freq = self.ui.spin_pwm_freq.value()
        duty = self.ui.spin_pwm_duty.value()
        self.send_serial_cmd_signal.emit(f"pwm {pin} {freq} {duty}")

    def on_cmd_stoppwm(self):
        pin = self.ui.spin_pwm_pin.value()
        self.send_serial_cmd_signal.emit(f"stoppwm {pin}")

        # Remove from Modified Pins list
        if pin in self.status_items_map:
            item = self.status_items_map.pop(pin)
            row = self.ui.list_status.row(item)
            self.ui.list_status.takeItem(row)

        # Reset button state
        if hasattr(self, "pin_buttons") and pin in self.pin_buttons:
            self.pin_buttons[pin].setStyleSheet("background-color: none;")

    def on_cmd_repeat(self):
        pin = self.ui.spin_repeat_pin.value()
        freq = self.ui.spin_repeat_freq.value()
        dur = self.ui.spin_repeat_dur.value()
        self.send_serial_cmd_signal.emit(f"repeat {pin} {freq} {dur}")

    def on_cmd_stoprepeat(self):
        pin = self.ui.spin_repeat_pin.value()
        self.send_serial_cmd_signal.emit(f"stoprepeat {pin}")

        # Remove from Modified Pins list
        if pin in self.status_items_map:
            item = self.status_items_map.pop(pin)
            row = self.ui.list_status.row(item)
            self.ui.list_status.takeItem(row)

        # Reset button state
        if hasattr(self, "pin_buttons") and pin in self.pin_buttons:
            self.pin_buttons[pin].setStyleSheet("background-color: none;")

    def on_cmd_interrupt(self):
        pin = self.ui.spin_int_pin.value()
        edge = self.ui.combo_int_edge.currentText()
        tgt = self.ui.spin_int_target.value()
        width = self.ui.spin_int_width.value()
        self.send_serial_cmd_signal.emit(f"interrupt {pin} {edge} {tgt} {width}")

    def on_cmd_stopinterrupt(self):
        pin = self.ui.spin_int_pin.value()
        self.send_serial_cmd_signal.emit(f"stopinterrupt {pin}")

        # Remove from Interrupts list
        if pin in self.interrupt_items_map:
            item = self.interrupt_items_map.pop(pin)
            row = self.ui.list_interrupts.row(item)
            self.ui.list_interrupts.takeItem(row)

    def on_cmd_throb(self):
        period = self.ui.spin_throb_period.value()
        p1 = self.ui.spin_throb_p1.value()
        p2 = self.ui.spin_throb_p2.value()
        p3 = self.ui.spin_throb_p3.value()
        self.send_serial_cmd_signal.emit(f"throb {period} {p1} {p2} {p3}")

    def on_cmd_stopthrob(self):
        self.send_serial_cmd_signal.emit("stopthrob")

    def on_cmd_mem(self):
        addr = self.ui.edit_mem_addr.text().strip()
        if addr:
            self.send_serial_cmd_signal.emit(f"mem {addr}")

    def on_ruler_toggled(self, checked):
        self.ruler_active = checked
        self.group_ruler.setVisible(checked)
        if not checked:
            self.ruler_start = None
            self.ruler_end = None
            self.refresh_video_label()
            self.intensity_chart.hide()
            self.chk_show_profile.setChecked(False)

    def on_show_profile_toggled(self, checked):
        if checked:
            self.intensity_chart.show()
            self.update_ruler_stats() # Initial update
        else:
            self.intensity_chart.hide()

    def get_image_coords(self, mouse_pos):
        if not self.current_pixmap:
            return None
        
        # Helper to map mouse position on label to image coordinates
        lbl_w = self.ui.video_label.width()
        lbl_h = self.ui.video_label.height()
        
        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()
        
        if img_w == 0 or img_h == 0:
            return None

        # Calculate scale and offsets (KeepAspectRatio)
        ratio_w = lbl_w / img_w
        ratio_h = lbl_h / img_h
        scale = min(ratio_w, ratio_h)
        
        disp_w = img_w * scale
        disp_h = img_h * scale
        
        off_x = (lbl_w - disp_w) / 2
        off_y = (lbl_h - disp_h) / 2
        
        # Mouse relative to image rect
        rx = mouse_pos.x() - off_x
        ry = mouse_pos.y() - off_y
        
        # Check bounds (optional, or clamp)
        # rx = max(0, min(rx, disp_w))
        # ry = max(0, min(ry, disp_h))
        
        ox = rx / scale
        oy = ry / scale
        
        return QPointF(ox, oy)

    def update_ruler_stats(self):
        if self.ruler_start is None or self.ruler_end is None:
            return
            
        line = QLineF(self.ruler_start, self.ruler_end)
        dist_px = line.length()
        
        self.lbl_ruler_px.setText(f"{dist_px:.2f} px")
        
        if self.ruler_calibration:
            dist_mm = dist_px / self.ruler_calibration
            self.lbl_ruler_meas.setText(f"{dist_mm:.2f} mm")
        else:
            self.lbl_ruler_meas.setText("N/A")

        # Update Intensity Chart if visible
        if self.intensity_chart.isVisible() and self.current_pixmap:
            # Get Image Data
            # Note: current_pixmap is for display. We should ideally use the source image data if available.
            # But converting QPixmap -> QImage is easiest here since we don't store the raw raw frame in MainWindow permanently (only temp).
            # Actually we store self.current_pixmap. Let's convert to QImage.
            
            qimg = self.current_pixmap.toImage()
            
            # Sampling
            num_samples = int(dist_px)
            if num_samples < 2:
                self.intensity_chart.set_data([])
                return

            x0, y0 = self.ruler_start.x(), self.ruler_start.y()
            x1, y1 = self.ruler_end.x(), self.ruler_end.y()
            
            # Clamp to image bounds
            w, h = qimg.width(), qimg.height()
            
            xs = np.linspace(x0, x1, num_samples)
            ys = np.linspace(y0, y1, num_samples)
            
            data = []
            
            # Accessing pixels from QImage in Python loop is slow. 
            # Better: convert QImage to numpy array first?
            # Or assume we can just grab pixel color.
            # QImage.pixelColor(x, y).value() gives grayscale for 8-bit or max component?
            # Let's use simple QImage.pixel() for now, optimized later if needed.
            # Actually, converting QImage to numpy is better.
            
            ptr = qimg.bits()
            # If default format is RGB888
            # Warning: This depends on format. Our worker emits Format_RGB888.
            
            # Fallback to slow pixel access for safety/robustness across formats
            for x, y in zip(xs, ys):
                ix, iy = int(x), int(y)
                if 0 <= ix < w and 0 <= iy < h:
                    c = qimg.pixelColor(ix, iy)
                    # Use Lightness/Value or Gray
                    val = (c.red() + c.green() + c.blue()) / 3.0
                    data.append(val)
                else:
                    data.append(0)
            
            self.intensity_chart.set_data(data)

    def calibrate_ruler(self):
        if self.ruler_start is None or self.ruler_end is None:
            self.log("Ruler: Draw a line first to calibrate.")
            return
            
        line = QLineF(self.ruler_start, self.ruler_end)
        dist_px = line.length()
        
        if dist_px < 1.0:
            self.log("Ruler: Line too short for calibration.")
            return
            
        known_len = self.spin_ruler_len.value()
        if known_len <= 0:
            self.log("Ruler: Known length must be > 0.")
            return
            
        self.ruler_calibration = dist_px / known_len # px per mm
        
        self.lbl_ruler_calib.setText(f"{self.ruler_calibration:.2f} px/mm")
        self.lbl_ruler_meas.setText(f"{known_len:.2f} mm") # Should match known length
        self.log(f"Ruler calibrated: {self.ruler_calibration:.2f} px/mm")

    def save_settings(self):
        settings = {}

        # --- Camera Settings ---
        settings["auto_exposure"] = self.ui.chk_auto_exposure.isChecked()
        settings["exposure_time"] = self.ui.spin_exposure_time.value()
        settings["gain"] = self.ui.spin_gain.value()
        settings["ae_target"] = self.ui.spin_ae_target.value()
        settings["roi"] = self.ui.chk_roi.isChecked()
        settings["trigger_mode"] = self.trigger_bg.checkedId()
        
        # Trigger Params
        settings["trigger_count"] = self.ui.spin_trigger_count.value()
        settings["trigger_delay"] = self.ui.spin_trigger_delay.value()
        settings["trigger_interval"] = self.ui.spin_trigger_interval.value()
        
        # External Trigger
        settings["ext_mode"] = self.ui.combo_ext_mode.currentIndex()
        settings["ext_jitter"] = self.ui.spin_ext_jitter.value()
        settings["ext_shutter"] = self.ui.combo_ext_shutter.currentIndex()
        
        # Strobe
        settings["strobe_mode"] = self.ui.combo_strobe_mode.currentIndex()
        settings["strobe_polarity"] = self.ui.combo_strobe_polarity.currentIndex()
        settings["strobe_delay"] = self.ui.spin_strobe_delay.value()
        settings["strobe_width"] = self.ui.spin_strobe_width.value()

        # --- Matching Settings ---
        settings["match_enable"] = getattr(self.ui, "chk_match_enable", QCheckBox()).isChecked()
        settings["aruco_enable"] = getattr(self.ui, "chk_aruco_enable", QCheckBox()).isChecked()
        settings["qrcode_enable"] = getattr(self.ui, "chk_qrcode_enable", QCheckBox()).isChecked()
        settings["ssim_enable"] = getattr(self.ui, "chk_ssim_enable", QCheckBox()).isChecked()
        settings["match_tab_index"] = self.ui.tabs_matching.currentIndex()
        
        # ORB
        settings["orb_nfeatures"] = self.ui.orb_nfeatures.value()
        settings["orb_scaleFactor"] = self.ui.orb_scaleFactor.value()
        settings["orb_nlevels"] = self.ui.orb_nlevels.value()
        settings["orb_edgeThreshold"] = self.ui.orb_edgeThreshold.value()
        settings["orb_firstLevel"] = self.ui.orb_firstLevel.value()
        settings["orb_wta_k"] = self.ui.orb_wta_k.value()
        settings["orb_scoreType"] = self.ui.orb_scoreType.currentIndex()
        settings["orb_patchSize"] = self.ui.orb_patchSize.value()
        settings["orb_fastThreshold"] = self.ui.orb_fastThreshold.value()
        
        # SIFT
        settings["sift_nfeatures"] = self.ui.sift_nfeatures.value()
        settings["sift_nOctaveLayers"] = self.ui.sift_nOctaveLayers.value()
        settings["sift_contrastThreshold"] = self.ui.sift_contrastThreshold.value()
        settings["sift_edgeThreshold"] = self.ui.sift_edgeThreshold.value()
        settings["sift_sigma"] = self.ui.sift_sigma.value()
        
        # AKAZE
        settings["akaze_descriptor_type"] = self.ui.akaze_descriptor_type.currentIndex()
        settings["akaze_threshold"] = self.ui.akaze_threshold.value()
        settings["akaze_nOctaves"] = self.ui.akaze_nOctaves.value()
        settings["akaze_nOctaveLayers"] = self.ui.akaze_nOctaveLayers.value()
        
        # ArUco
        if hasattr(self.ui, "aruco_dict"):
            settings["aruco_dict"] = self.ui.aruco_dict.currentIndex()
        if hasattr(self.ui, "spin_aruco_border_bits"):
            settings["aruco_border_bits"] = self.ui.spin_aruco_border_bits.value()
        if hasattr(self.ui, "chk_aruco_show_ids"):
            settings["aruco_show_ids"] = self.ui.chk_aruco_show_ids.isChecked()
        if hasattr(self.ui, "chk_aruco_show_rejected"):
            settings["aruco_show_rejected"] = self.ui.chk_aruco_show_rejected.isChecked()

        # --- Contours ---
        settings["contours_enabled"] = self.btn_toggle_contours.isChecked()
        settings["contour_mode"] = self.combo_contour_mode.currentIndex()
        settings["threshold"] = self.slider_threshold.value()
        settings["canny_values"] = self.slider_canny.getValues()
        settings["min_area"] = self.spin_min_area.value()
        settings["max_area"] = self.spin_max_area.value()
        settings["fill_contours"] = self.chk_fill_contours.isChecked()
        settings["show_box"] = self.chk_show_box.isChecked()

        # --- Ruler ---
        settings["ruler_active"] = self.action_ruler.isChecked()
        settings["ruler_len"] = self.spin_ruler_len.value()
        settings["ruler_calibration"] = self.ruler_calibration

        # --- Paths ---
        # Note: We can't easily save the loaded template/SSIM images, but we could save paths if we tracked them.
        # The worker sets them, but MainWindow doesn't explicitly store the path unless we add it.
        # For now, let's assume user re-loads them or we add path tracking.
        # I'll rely on the user to reload images, but saving other params is key.

        try:
            with open(os.path.join(script_dir, "settings.json"), "w") as f:
                json.dump(settings, f, indent=4)
                self.log("Settings saved.")
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def load_settings(self):
        try:
            path = os.path.join(script_dir, "settings.json")
            if not os.path.exists(path):
                return
            
            with open(path, "r") as f:
                settings = json.load(f)

            # --- Restore Software Settings (UI Only) ---
            # Block signals where necessary to avoid triggering heavy logic prematurely
            
            # Matching
            if "match_tab_index" in settings:
                self.ui.tabs_matching.setCurrentIndex(settings["match_tab_index"])
            
            # ORB
            self.ui.orb_nfeatures.setValue(settings.get("orb_nfeatures", 500))
            self.ui.orb_scaleFactor.setValue(settings.get("orb_scaleFactor", 1.2))
            self.ui.orb_nlevels.setValue(settings.get("orb_nlevels", 8))
            self.ui.orb_edgeThreshold.setValue(settings.get("orb_edgeThreshold", 31))
            self.ui.orb_firstLevel.setValue(settings.get("orb_firstLevel", 0))
            self.ui.orb_wta_k.setValue(settings.get("orb_wta_k", 2))
            if "orb_scoreType" in settings: self.ui.orb_scoreType.setCurrentIndex(settings["orb_scoreType"])
            self.ui.orb_patchSize.setValue(settings.get("orb_patchSize", 31))
            self.ui.orb_fastThreshold.setValue(settings.get("orb_fastThreshold", 20))
            
            # SIFT
            self.ui.sift_nfeatures.setValue(settings.get("sift_nfeatures", 0))
            self.ui.sift_nOctaveLayers.setValue(settings.get("sift_nOctaveLayers", 3))
            self.ui.sift_contrastThreshold.setValue(settings.get("sift_contrastThreshold", 0.04))
            self.ui.sift_edgeThreshold.setValue(settings.get("sift_edgeThreshold", 10))
            self.ui.sift_sigma.setValue(settings.get("sift_sigma", 1.6))
            
            # AKAZE
            if "akaze_descriptor_type" in settings: self.ui.akaze_descriptor_type.setCurrentIndex(settings["akaze_descriptor_type"])
            self.ui.akaze_threshold.setValue(settings.get("akaze_threshold", 0.0012))
            self.ui.akaze_nOctaves.setValue(settings.get("akaze_nOctaves", 4))
            self.ui.akaze_nOctaveLayers.setValue(settings.get("akaze_nOctaveLayers", 4))
            
            # ArUco
            if hasattr(self.ui, "aruco_dict") and "aruco_dict" in settings:
                self.ui.aruco_dict.setCurrentIndex(settings["aruco_dict"])
            if hasattr(self.ui, "spin_aruco_border_bits") and "aruco_border_bits" in settings:
                self.ui.spin_aruco_border_bits.setValue(settings["aruco_border_bits"])
            if hasattr(self.ui, "chk_aruco_show_ids") and "aruco_show_ids" in settings:
                self.ui.chk_aruco_show_ids.setChecked(settings["aruco_show_ids"])
            if hasattr(self.ui, "chk_aruco_show_rejected") and "aruco_show_rejected" in settings:
                self.ui.chk_aruco_show_rejected.setChecked(settings["aruco_show_rejected"])

            # Contours
            if "contours_enabled" in settings: self.btn_toggle_contours.setChecked(settings["contours_enabled"])
            if "contour_mode" in settings: self.combo_contour_mode.setCurrentIndex(settings["contour_mode"])
            if "threshold" in settings: self.slider_threshold.setValue(settings["threshold"])
            if "canny_values" in settings: self.slider_canny.setValues(*settings["canny_values"])
            if "min_area" in settings: self.spin_min_area.setValue(settings["min_area"])
            if "max_area" in settings: self.spin_max_area.setValue(settings["max_area"])
            if "fill_contours" in settings: self.chk_fill_contours.setChecked(settings["fill_contours"])
            if "show_box" in settings: self.chk_show_box.setChecked(settings["show_box"])

            # Ruler
            if "ruler_active" in settings: self.action_ruler.setChecked(settings["ruler_active"])
            if "ruler_len" in settings: self.spin_ruler_len.setValue(settings["ruler_len"])
            if "ruler_calibration" in settings: 
                self.ruler_calibration = settings["ruler_calibration"]
                if self.ruler_calibration:
                    self.lbl_ruler_calib.setText(f"{self.ruler_calibration:.2f} px/mm")

            # Enable Checkboxes (Last, to trigger updates if needed, though without camera open updates are partial)
            # Note: Toggling these might auto-uncheck others due to mutual exclusion logic in handlers.
            # We should set them in a specific order or just let the last one win.
            if settings.get("match_enable"): 
                if hasattr(self.ui, "chk_match_enable"): self.ui.chk_match_enable.setChecked(True)
            elif settings.get("aruco_enable"):
                if hasattr(self.ui, "chk_aruco_enable"): self.ui.chk_aruco_enable.setChecked(True)
            elif settings.get("qrcode_enable"):
                if hasattr(self.ui, "chk_qrcode_enable"): self.ui.chk_qrcode_enable.setChecked(True)
            elif settings.get("ssim_enable"):
                if hasattr(self.ui, "chk_ssim_enable"): self.ui.chk_ssim_enable.setChecked(True)

            # --- Restore Camera UI (Do not trigger camera calls yet) ---
            # We store these settings to apply them AFTER camera starts
            self.saved_camera_settings = settings
            
            # Update UI to reflect saved settings immediately, but block signals for camera controls 
            # so we don't try to set parameters on a closed camera (which might error or be ignored).
            
            self.ui.chk_auto_exposure.blockSignals(True)
            self.ui.chk_auto_exposure.setChecked(settings.get("auto_exposure", True))
            self.ui.chk_auto_exposure.blockSignals(False)

            self.ui.spin_exposure_time.blockSignals(True)
            self.ui.spin_exposure_time.setValue(settings.get("exposure_time", 2000))
            self.ui.spin_exposure_time.blockSignals(False)

            self.ui.spin_gain.blockSignals(True)
            self.ui.spin_gain.setValue(settings.get("gain", 1))
            self.ui.spin_gain.blockSignals(False)
            
            self.ui.spin_ae_target.blockSignals(True)
            self.ui.spin_ae_target.setValue(settings.get("ae_target", 120))
            self.ui.spin_ae_target.blockSignals(False)

            self.ui.chk_roi.blockSignals(True)
            self.ui.chk_roi.setChecked(settings.get("roi", False))
            self.ui.chk_roi.blockSignals(False)

            # Trigger Mode
            tm = settings.get("trigger_mode", 0)
            btn = self.trigger_bg.button(tm)
            if btn:
                self.trigger_bg.blockSignals(True)
                btn.setChecked(True)
                self.trigger_bg.blockSignals(False)
                # Manually update UI enable states
                self.on_trigger_mode_changed(tm, True)

            # Trigger Params
            self.ui.spin_trigger_count.blockSignals(True)
            self.ui.spin_trigger_count.setValue(settings.get("trigger_count", 1))
            self.ui.spin_trigger_count.blockSignals(False)
            
            self.ui.spin_trigger_delay.blockSignals(True)
            self.ui.spin_trigger_delay.setValue(settings.get("trigger_delay", 0))
            self.ui.spin_trigger_delay.blockSignals(False)

            self.ui.spin_trigger_interval.blockSignals(True)
            self.ui.spin_trigger_interval.setValue(settings.get("trigger_interval", 1000))
            self.ui.spin_trigger_interval.blockSignals(False)

            # Ext Trigger
            self.ui.combo_ext_mode.blockSignals(True)
            self.ui.combo_ext_mode.setCurrentIndex(settings.get("ext_mode", 0))
            self.ui.combo_ext_mode.blockSignals(False)
            
            self.ui.spin_ext_jitter.blockSignals(True)
            self.ui.spin_ext_jitter.setValue(settings.get("ext_jitter", 0))
            self.ui.spin_ext_jitter.blockSignals(False)
            
            self.ui.combo_ext_shutter.blockSignals(True)
            self.ui.combo_ext_shutter.setCurrentIndex(settings.get("ext_shutter", 0))
            self.ui.combo_ext_shutter.blockSignals(False)

            # Strobe
            self.ui.combo_strobe_mode.blockSignals(True)
            self.ui.combo_strobe_mode.setCurrentIndex(settings.get("strobe_mode", 0))
            self.ui.combo_strobe_mode.blockSignals(False)
            # Update UI state
            self.on_strobe_mode_changed(settings.get("strobe_mode", 0))

            self.ui.combo_strobe_polarity.blockSignals(True)
            self.ui.combo_strobe_polarity.setCurrentIndex(settings.get("strobe_polarity", 0))
            self.ui.combo_strobe_polarity.blockSignals(False)
            
            self.ui.spin_strobe_delay.blockSignals(True)
            self.ui.spin_strobe_delay.setValue(settings.get("strobe_delay", 0))
            self.ui.spin_strobe_delay.blockSignals(False)
            
            self.ui.spin_strobe_width.blockSignals(True)
            self.ui.spin_strobe_width.setValue(settings.get("strobe_width", 0))
            self.ui.spin_strobe_width.blockSignals(False)

            self.log("Settings loaded.")
        except Exception as e:
            self.log(f"Error loading settings: {e}")

    def apply_camera_settings(self):
        # Applies settings to the camera hardware.
        # This should be called after camera.start() and sync_ui()
        if not hasattr(self, "saved_camera_settings") or not self.saved_camera_settings:
            return

        s = self.saved_camera_settings
        
        # Apply Trigger Mode first (might reset others?)
        if "trigger_mode" in s:
            self.camera.setTriggerMode(s["trigger_mode"])
            # Update UI again to be sure
            self.trigger_bg.button(s["trigger_mode"]).setChecked(True)

        if "trigger_count" in s: self.camera.setTriggerCount(s["trigger_count"])
        if "trigger_delay" in s: self.camera.setTriggerDelay(s["trigger_delay"])
        if "trigger_interval" in s: self.camera.setTriggerInterval(s["trigger_interval"])
        
        if "ext_mode" in s: self.camera.setExternalTriggerSignalType(s["ext_mode"])
        if "ext_jitter" in s: self.camera.setExternalTriggerJitterTime(s["ext_jitter"])
        if "ext_shutter" in s: self.camera.setExternalTriggerShutterMode(s["ext_shutter"])

        if "strobe_mode" in s: self.camera.setStrobeMode(s["strobe_mode"])
        if "strobe_polarity" in s: self.camera.setStrobePolarity(s["strobe_polarity"])
        if "strobe_delay" in s: self.camera.setStrobeDelayTime(s["strobe_delay"])
        if "strobe_width" in s: self.camera.setStrobePulseWidth(s["strobe_width"])

        # Exposure / Gain
        # Note: Auto exposure setting might override manual time/gain
        if "auto_exposure" in s:
            self.camera.setAutoExposure(s["auto_exposure"])
            self.ui.chk_auto_exposure.setChecked(s["auto_exposure"])
            
            if s["auto_exposure"]:
                if "ae_target" in s: 
                    self.camera.setAutoExposureTarget(s["ae_target"])
                    self.ui.spin_ae_target.setValue(s["ae_target"])
            else:
                if "exposure_time" in s:
                    self.camera.setExposureTime(s["exposure_time"])
                    self.ui.spin_exposure_time.setValue(s["exposure_time"])
                if "gain" in s:
                    self.camera.setAnalogGain(s["gain"])
                    self.ui.spin_gain.setValue(s["gain"])

        if "roi" in s:
            self.camera.setRoi(s["roi"])
            self.ui.chk_roi.setChecked(s["roi"])

        self.log("Saved camera settings applied.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, lambda sig, frame: window.close())

    # Timer to let the Python interpreter handle signals periodically
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec())
