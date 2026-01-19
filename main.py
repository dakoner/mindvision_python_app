import sys
import os
import platform
import time
import signal

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

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QButtonGroup, QFileDialog, QListWidget, QListWidgetItem, QVBoxLayout)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QFile, QObject, QEvent, QThread, QMutex
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader

try:
    from _mindvision_qobject_py import MindVisionCamera, VideoThread
except ImportError as e:
    print(f"Failed to import _mindvision_qobject_py: {e}")
    sys.exit(1)

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
                self.serial_port.write(full_cmd.encode('utf-8'))
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
                            line = self.serial_port.readline().decode('utf-8', errors='replace').strip()
                            if line:
                                self.log_signal.emit(f"Rx: {line}")
                        else:
                            break
                except Exception as e:
                    self.log_signal.emit(f"Read error: {e}")

class MatchingWorker(QObject):
    result_ready = Signal(QImage)
    log_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.bf = None
        self.template_img = None
        self.template_kp = None
        self.template_des = None
        self.is_matching_enabled = False
        self.mutex = QMutex()

    @Slot(dict)
    def update_params(self, params):
        # Update detector based on params
        with QMutexLocker(self.mutex):
            try:
                algo = params.get('algo', 'ORB')
                if algo == 'ORB':
                    self.detector = cv2.ORB_create(
                        nfeatures=params.get('nfeatures', 500),
                        scaleFactor=params.get('scaleFactor', 1.2),
                        nlevels=params.get('nlevels', 8),
                        edgeThreshold=params.get('edgeThreshold', 31),
                        firstLevel=params.get('firstLevel', 0),
                        WTA_K=params.get('WTA_K', 2),
                        scoreType=params.get('scoreType', 0),
                        patchSize=params.get('patchSize', 31),
                        fastThreshold=params.get('fastThreshold', 20)
                    )
                    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                elif algo == 'SIFT':
                    self.detector = cv2.SIFT_create(
                        nfeatures=params.get('nfeatures', 0),
                        nOctaveLayers=params.get('nOctaveLayers', 3),
                        contrastThreshold=params.get('contrastThreshold', 0.04),
                        edgeThreshold=params.get('edgeThreshold', 10),
                        sigma=params.get('sigma', 1.6)
                    )
                    self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                elif algo == 'AKAZE':
                    self.detector = cv2.AKAZE_create(
                        descriptor_type=params.get('descriptor_type', 5),
                        threshold=params.get('threshold', 0.0012),
                        nOctaves=params.get('nOctaves', 4),
                        nOctaveLayers=params.get('nOctaveLayers', 4)
                    )
                    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                # Recompute template if exists
                if self.template_img is not None:
                    self.template_kp, self.template_des = self.detector.detectAndCompute(self.template_img, None)
            except Exception as e:
                self.log_signal.emit(f"Worker update error: {e}")

    @Slot(str)
    def set_template(self, file_path):
        with QMutexLocker(self.mutex):
            if not file_path:
                self.is_matching_enabled = False
                self.template_img = None
                return

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and self.detector is not None:
                self.template_img = img
                self.template_kp, self.template_des = self.detector.detectAndCompute(img, None)
                if self.template_des is not None:
                    self.is_matching_enabled = True
                    self.log_signal.emit(f"Worker: Template loaded {file_path}")
                else:
                    self.log_signal.emit("Worker: No features in template")
            else:
                self.log_signal.emit("Worker: Failed to load template")

    @Slot(bool)
    def toggle_matching(self, enabled):
        with QMutexLocker(self.mutex):
            self.is_matching_enabled = enabled

    @Slot(int, int, int, int, bytes)
    def process_frame(self, width, height, bytes_per_line, fmt, data_bytes):
        # This runs in the worker thread
        # 'data_bytes' is a copy of the frame data passed from the main thread
        
        with QMutexLocker(self.mutex):
            matching_active = self.is_matching_enabled and self.template_des is not None and self.detector is not None
            # We need local references to avoid race conditions if params change mid-processing
            local_detector = self.detector
            local_bf = self.bf
            local_template_des = self.template_des
            local_template_kp = self.template_kp
            local_template_img = self.template_img
        
        try:
            channels = bytes_per_line // width
            img_np = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height, width, channels))
            
            # If we are matching
            if matching_active:
                if channels == 3:
                    gray_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_frame = img_np
                
                kp_frame, des_frame = local_detector.detectAndCompute(gray_frame, None)
                
                if des_frame is not None:
                    matches = local_bf.match(local_template_des, des_frame)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = matches[:20]
                    
                    res_img = cv2.drawMatches(local_template_img, local_template_kp, 
                                            img_np if channels == 1 else cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                                            kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
                    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    h, w, c = res_img_rgb.shape
                    # Must copy to decouple from numpy array
                    qimg = QImage(res_img_rgb.data, w, h, w * c, QImage.Format_RGB888).copy()
                    self.result_ready.emit(qimg)
                    return

            # Pass-through if not matching or failed
            # Create QImage from the numpy array (which is a view of data_bytes)
            # data_bytes is local to this call (sent via signal), so it persists.
            # But QImage needs to survive emission. .copy() is safest.
            qimg = QImage(img_np.data, width, height, bytes_per_line, QImage.Format(fmt)).copy()
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
    # Signal to toggle matching
    toggle_worker_matching_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        
        # Load UI from file
        loader = QUiLoader()
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
        self.toggle_worker_matching_signal.connect(self.worker.toggle_matching)
        
        self.worker.result_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log)
        
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
        self.status_items_map = {} # Map pin -> QListWidgetItem
        self.interrupt_items_map = {} # Map pin -> QListWidgetItem

        # Access serial layout
        serial_layout = self.ui.group_serial.layout()
        
        # Status List
        self.lbl_status = QLabel("Modified Pins:")
        self.list_status = QListWidget()
        self.list_status.setMaximumHeight(100)
        serial_layout.addWidget(self.lbl_status)
        serial_layout.addWidget(self.list_status)

        # Interrupt List
        self.lbl_interrupts = QLabel("Interrupts:")
        self.list_interrupts = QListWidget()
        self.list_interrupts.setMaximumHeight(80)
        serial_layout.addWidget(self.lbl_interrupts)
        serial_layout.addWidget(self.list_interrupts)
        
        # Serial UI Init
        self.refresh_serial_ports()
        self.ui.btn_serial_refresh.clicked.connect(self.refresh_serial_ports)
        self.ui.btn_serial_connect.clicked.connect(self.on_btn_serial_connect_clicked)
        self.ui.btn_cmd_pulse.clicked.connect(self.on_cmd_pulse)
        self.ui.btn_cmd_level.clicked.connect(self.on_cmd_level)
        self.ui.btn_cmd_pwm.clicked.connect(self.on_cmd_pwm)
        self.ui.btn_cmd_stoppwm.clicked.connect(self.on_cmd_stoppwm)
        self.ui.btn_cmd_repeat.clicked.connect(self.on_cmd_repeat)
        self.ui.btn_cmd_stoprepeat.clicked.connect(self.on_cmd_stoprepeat)
        self.ui.btn_cmd_interrupt.clicked.connect(self.on_cmd_interrupt)
        self.ui.btn_cmd_stopinterrupt.clicked.connect(self.on_cmd_stopinterrupt)
        self.ui.btn_cmd_throb.clicked.connect(self.on_cmd_throb)
        self.ui.btn_cmd_stopthrob.clicked.connect(self.on_cmd_stopthrob)
        self.ui.btn_cmd_info.clicked.connect(lambda: self.send_serial_cmd_signal.emit("info"))
        self.ui.btn_cmd_wifi.clicked.connect(lambda: self.send_serial_cmd_signal.emit("wifi"))
        self.ui.btn_cmd_mem.clicked.connect(self.on_cmd_mem)
        
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
        self.ui.spin_trigger_interval.valueChanged.connect(self.on_trigger_interval_changed)
        
        # New Connections for External Trigger Params
        self.ui.combo_ext_mode.currentIndexChanged.connect(self.on_ext_mode_changed)
        self.ui.spin_ext_jitter.valueChanged.connect(self.on_ext_jitter_changed)
        self.ui.combo_ext_shutter.currentIndexChanged.connect(self.on_ext_shutter_changed)
        
        # New Connections for Strobe Params
        self.ui.combo_strobe_mode.currentIndexChanged.connect(self.on_strobe_mode_changed)
        self.ui.combo_strobe_polarity.currentIndexChanged.connect(self.on_strobe_polarity_changed)
        self.ui.spin_strobe_delay.valueChanged.connect(self.on_strobe_delay_changed)
        self.ui.spin_strobe_width.valueChanged.connect(self.on_strobe_width_changed)
        
        self.ui.start_btn.clicked.connect(self.on_start_clicked)
        self.ui.stop_btn.clicked.connect(self.on_stop_clicked)
        self.ui.record_btn.clicked.connect(self.on_record_clicked)
        self.ui.snapshot_btn.clicked.connect(self.on_snapshot_clicked)
        self.ui.btn_find_template.clicked.connect(self.on_find_template_clicked)
        
        # Matching Tabs Connection
        self.ui.tabs_matching.currentChanged.connect(self.on_detector_params_changed)

        # ORB Parameter Connections
        self.ui.orb_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_scaleFactor.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_nlevels.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_firstLevel.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_wta_k.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_scoreType.currentIndexChanged.connect(self.on_detector_params_changed)
        self.ui.orb_patchSize.valueChanged.connect(self.on_detector_params_changed)
        self.ui.orb_fastThreshold.valueChanged.connect(self.on_detector_params_changed)

        # SIFT Parameter Connections
        self.ui.sift_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_nOctaveLayers.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_contrastThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.sift_sigma.valueChanged.connect(self.on_detector_params_changed)

        # AKAZE Parameter Connections
        self.ui.akaze_descriptor_type.currentIndexChanged.connect(self.on_detector_params_changed)
        self.ui.akaze_threshold.valueChanged.connect(self.on_detector_params_changed)
        self.ui.akaze_nOctaves.valueChanged.connect(self.on_detector_params_changed)
        self.ui.akaze_nOctaveLayers.valueChanged.connect(self.on_detector_params_changed)

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

    def show(self):
        self.ui.showMaximized()

    def close(self):
        # Triggers closeEvent via the widget
        self.ui.close()

    def eventFilter(self, watched, event):
        try:
            if watched is self.ui and event.type() == QEvent.Close:
                self.on_stop_clicked()
                self.matching_thread.quit()
                self.matching_thread.wait()
                self.ui.removeEventFilter(self)
                try:
                    self.ui.video_label.removeEventFilter(self)
                except RuntimeError:
                    pass
            elif watched == self.ui.video_label and event.type() == QEvent.Resize:
                self.refresh_video_label()
        except RuntimeError:
            pass
        return super().eventFilter(watched, event)

    def refresh_video_label(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(self.ui.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ui.video_label.setPixmap(scaled)

    @Slot(str)
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        if hasattr(self.ui, 'log_text_edit'):
            self.ui.log_text_edit.appendPlainText(f"[{timestamp}] {message}")
        
        # Parse Rx messages
        if message.startswith("Rx: "):
            content = message[4:].strip()
            self.process_serial_line(content)

    def process_serial_line(self, line):
        # Startup detection
        if "LED>" in line and not self.has_initialized_settings:
            self.has_initialized_settings = True
            QTimer.singleShot(500, lambda: self.send_serial_cmd_signal.emit("printsettings"))
            return

        parts = line.split()
        if not parts:
            return
        
        cmd = parts[0]
        
        try:
            if cmd == "level" and len(parts) >= 3:
                pin = int(parts[1])
                val = int(parts[2])
                self.update_pin_status(pin, f"Level: {val}")
                # Update UI setters
                self.ui.spin_level_pin.setValue(pin)
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
                if idx >= 0: self.ui.combo_int_edge.setCurrentIndex(idx)
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
            self.list_status.addItem(item)
            self.status_items_map[pin] = item

    def update_interrupt_status(self, pin, status_text):
        text = f"Pin {pin}: {status_text}"
        if pin in self.interrupt_items_map:
            self.interrupt_items_map[pin].setText(text)
        else:
            item = QListWidgetItem(text)
            self.list_interrupts.addItem(item)
            self.interrupt_items_map[pin] = item

    def update_detector(self):
        tab_index = self.ui.tabs_matching.currentIndex()
        params = {}
        
        if tab_index == 0: # ORB
            params['algo'] = 'ORB'
            params['nfeatures'] = self.ui.orb_nfeatures.value()
            params['scaleFactor'] = self.ui.orb_scaleFactor.value()
            params['nlevels'] = self.ui.orb_nlevels.value()
            params['edgeThreshold'] = self.ui.orb_edgeThreshold.value()
            params['firstLevel'] = self.ui.orb_firstLevel.value()
            params['WTA_K'] = self.ui.orb_wta_k.value()
            params['scoreType'] = self.ui.orb_scoreType.currentIndex()
            params['patchSize'] = self.ui.orb_patchSize.value()
            params['fastThreshold'] = self.ui.orb_fastThreshold.value()

        elif tab_index == 1: # SIFT
            params['algo'] = 'SIFT'
            params['nfeatures'] = self.ui.sift_nfeatures.value()
            params['nOctaveLayers'] = self.ui.sift_nOctaveLayers.value()
            params['contrastThreshold'] = self.ui.sift_contrastThreshold.value()
            params['edgeThreshold'] = self.ui.sift_edgeThreshold.value()
            params['sigma'] = self.ui.sift_sigma.value()

        elif tab_index == 2: # AKAZE
            params['algo'] = 'AKAZE'
            combo_idx = self.ui.akaze_descriptor_type.currentIndex()
            mapping = [2, 3, 4, 5]
            params['descriptor_type'] = mapping[combo_idx]
            params['threshold'] = self.ui.akaze_threshold.value()
            params['nOctaves'] = self.ui.akaze_nOctaves.value()
            params['nOctaveLayers'] = self.ui.akaze_nOctaveLayers.value()

        self.update_worker_params_signal.emit(params)

    def on_detector_params_changed(self):
        self.update_detector()

    def frame_callback(self, width, height, bytes_per_line, fmt, data):
        # 1. Recording (High Priority)
        if self.video_thread.isRunning():
            try:
                self.video_thread.addFrameBytes(width, height, bytes_per_line, fmt, data)
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
                    self.process_frame_signal.emit(width, height, bytes_per_line, fmt, data_copy)
                # Else: Drop frame for UI display to prevent backlog/latency
            except Exception as e:
                self.worker_busy = False # Reset on error
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
                self.video_thread.startRecording(image.width(), image.height(), record_fps, "output.mkv")
                self.ui.record_btn.setText("Stop Recording")
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
        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)
        self.ui.controls_group.setEnabled(False)
        self.ui.trigger_group.setEnabled(False)
        self.ui.trigger_params_group.setEnabled(False)
        self.ui.ext_trigger_group.setEnabled(False)
        self.ui.strobe_group.setEnabled(False)
        self.ui.btn_find_template.setEnabled(False)

    def on_start_clicked(self):
        if self.camera.open():
            if self.camera.start():
                self.ui.start_btn.setEnabled(False)
                self.ui.stop_btn.setEnabled(True)
                self.ui.record_btn.setEnabled(True)
                self.ui.snapshot_btn.setEnabled(True)
                self.ui.btn_find_template.setEnabled(True)
                self.ui.controls_group.setEnabled(True)
                self.ui.trigger_group.setEnabled(True)
                self.ui.strobe_group.setEnabled(True)
                
                self.ui.video_label.setText("Starting stream...")
                self.sync_ui()
                self.log("Camera started.")

    def on_stop_clicked(self):
        if self.video_thread.isRunning():
            self.on_record_clicked()
        
        self.camera.stop()
        self.camera.close()
        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)
        self.ui.record_btn.setEnabled(False)
        self.ui.snapshot_btn.setEnabled(False)
        self.ui.btn_find_template.setEnabled(False)
        
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
            self.ui.record_btn.setText("Start Recording")
            self.log("Recording stopped.")

    def on_snapshot_clicked(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            filename = f"snapshot_{int(time.time())}.png"
            self.current_pixmap.save(filename)
            self.log(f"Snapshot saved: {filename}")

    def on_find_template_clicked(self):
        if self.is_matching_ui_active:
            self.is_matching_ui_active = False
            self.toggle_worker_matching_signal.emit(False)
            self.ui.btn_find_template.setText("Find template in image")
        else:
            file_path, _ = QFileDialog.getOpenFileName(self.ui, "Select Template Image", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                self.set_worker_template_signal.emit(file_path)
                self.is_matching_ui_active = True
                self.toggle_worker_matching_signal.emit(True)
                self.ui.btn_find_template.setText("Stop Matching")

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
        if hasattr(self.camera, 'getAutoExposureTarget'):
            try:
                if hasattr(self.camera, 'getAutoExposureTargetRange'):
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
        if hasattr(self.camera, 'setAutoExposureTarget'):
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
        if hasattr(self.camera, 'setTriggerCount'):
            self.camera.setTriggerCount(value)

    def on_trigger_delay_changed(self, value):
        if hasattr(self.camera, 'setTriggerDelay'):
            self.camera.setTriggerDelay(value)

    def on_trigger_interval_changed(self, value):
        if hasattr(self.camera, 'setTriggerInterval'):
            self.camera.setTriggerInterval(value)

    # External Trigger Slots
    def on_ext_mode_changed(self, index):
        if hasattr(self.camera, 'setExternalTriggerSignalType'):
            self.camera.setExternalTriggerSignalType(index)

    def on_ext_jitter_changed(self, value):
        if hasattr(self.camera, 'setExternalTriggerJitterTime'):
            self.camera.setExternalTriggerJitterTime(value)
    
    def on_ext_shutter_changed(self, index):
        if hasattr(self.camera, 'setExternalTriggerShutterMode'):
            self.camera.setExternalTriggerShutterMode(index)

    # Strobe Slots
    def on_strobe_mode_changed(self, index):
        # 0 = Auto, 1 = Manual/Semi-Auto
        if hasattr(self.camera, 'setStrobeMode'):
            self.camera.setStrobeMode(index)
            # Enable manual controls if index == 1
            is_manual = (index == 1)
            self.ui.combo_strobe_polarity.setEnabled(is_manual)
            self.ui.spin_strobe_delay.setEnabled(is_manual)
            self.ui.spin_strobe_width.setEnabled(is_manual)

    def on_strobe_polarity_changed(self, index):
        if hasattr(self.camera, 'setStrobePolarity'):
            self.camera.setStrobePolarity(index)

    def on_strobe_delay_changed(self, value):
        if hasattr(self.camera, 'setStrobeDelayTime'):
            self.camera.setStrobeDelayTime(value)

    def on_strobe_width_changed(self, value):
        if hasattr(self.camera, 'setStrobePulseWidth'):
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
            port = self.ui.combo_serial_port.currentText().split()[0] # Handle "COM3 - Desc" if needed
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
            self.list_status.clear()
            self.status_items_map.clear()
            self.list_interrupts.clear()
            self.interrupt_items_map.clear()

    def on_cmd_pulse(self):
        pin = self.ui.spin_pulse_pin.value()
        val = self.ui.spin_pulse_val.value()
        dur = self.ui.spin_pulse_dur.value()
        self.send_serial_cmd_signal.emit(f"pulse {pin} {val} {dur}")

    def on_cmd_level(self):
        pin = self.ui.spin_level_pin.value()
        idx = self.ui.combo_level_val.currentIndex()
        if idx == 0: # Read
            self.send_serial_cmd_signal.emit(f"level {pin}")
        else: # Set 0 or 1. Index 1=Low(0), 2=High(1)
            val = idx - 1
            self.send_serial_cmd_signal.emit(f"level {pin} {val}")

    def on_cmd_pwm(self):
        pin = self.ui.spin_pwm_pin.value()
        freq = self.ui.spin_pwm_freq.value()
        duty = self.ui.spin_pwm_duty.value()
        self.send_serial_cmd_signal.emit(f"pwm {pin} {freq} {duty}")

    def on_cmd_stoppwm(self):
        pin = self.ui.spin_pwm_pin.value()
        self.send_serial_cmd_signal.emit(f"stoppwm {pin}")

    def on_cmd_repeat(self):
        pin = self.ui.spin_repeat_pin.value()
        freq = self.ui.spin_repeat_freq.value()
        dur = self.ui.spin_repeat_dur.value()
        self.send_serial_cmd_signal.emit(f"repeat {pin} {freq} {dur}")

    def on_cmd_stoprepeat(self):
        pin = self.ui.spin_repeat_pin.value()
        self.send_serial_cmd_signal.emit(f"stoprepeat {pin}")

    def on_cmd_interrupt(self):
        pin = self.ui.spin_int_pin.value()
        edge = self.ui.combo_int_edge.currentText()
        tgt = self.ui.spin_int_target.value()
        width = self.ui.spin_int_width.value()
        self.send_serial_cmd_signal.emit(f"interrupt {pin} {edge} {tgt} {width}")

    def on_cmd_stopinterrupt(self):
        pin = self.ui.spin_int_pin.value()
        self.send_serial_cmd_signal.emit(f"stopinterrupt {pin}")

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
