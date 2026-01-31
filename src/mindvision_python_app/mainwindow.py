import os
import sys
import time
import json
import PySide6.QtWidgets
import serial
import serial.tools.list_ports
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QButtonGroup,
    QFileDialog,
    QListWidgetItem,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QPushButton,
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
    QPointF,
    QLineF,
)
from PySide6.QtGui import QPixmap, QAction, QPainter, QPen, QColor, QIcon, QImage
from PySide6.QtUiTools import QUiLoader

from _mindvision_qobject_py import MindVisionCamera, VideoThread
from range_slider import RangeSlider
from intensity_chart import IntensityChart
from color_picker_widget import ColorPickerWidget
from matching_worker import MatchingWorker
from serial_worker import SerialWorker, HAS_SERIAL
from cnc_control_panel import CNCControlPanel

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

        # Add the directory containing the generated module to sys.path
        # Note: In the refactored structure, we rely on main.py to set sys.path for _mindvision_qobject_py,
        # but for loading the UI file we need the directory of this file.
        self.script_dir = os.path.dirname(__file__)

        # Load UI from file
        loader = QUiLoader()
        loader.registerCustomWidget(RangeSlider)
        loader.registerCustomWidget(IntensityChart)
        loader.registerCustomWidget(ColorPickerWidget)
        ui_file_path = os.path.join(self.script_dir, "mainwindow.ui")
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

        # Parameter Poll Timer (for Auto Exposure updates)
        self.param_poll_timer = QTimer()
        self.param_poll_timer.timeout.connect(self.poll_camera_params)

        # --- CNC Control Panel Setup ---
        self.cnc_control_panel = CNCControlPanel()
        # Find the scroll_layout and the group_serial widget (now LED Controller)
        scroll_layout = self.ui.scroll_layout
        group_serial = self.ui.group_serial
        # Get the index of the group_serial widget and insert the cnc_control_panel after it
        index = scroll_layout.indexOf(group_serial)
        scroll_layout.insertWidget(index + 1, self.cnc_control_panel)
        self.cnc_control_panel.log_signal.connect(self.log)

        # Initial Detector config
        
        if hasattr(self.ui, 'chk_qrcode_enable'):
            self.ui.chk_qrcode_enable.toggled.connect(self.on_qrcode_enable_toggled)

        # SSIM UI
        if hasattr(self.ui, 'btn_load_ssim_ref'):
            self.ui.btn_load_ssim_ref.clicked.connect(self.on_load_ssim_ref_clicked)
        if hasattr(self.ui, 'chk_ssim_enable'):
            self.ui.chk_ssim_enable.toggled.connect(self.on_ssim_enable_toggled)
            
        self.ssim_ref_loaded = False

        self.intensity_chart = self.ui.intensity_chart
        self.action_color_picker = self.ui.action_color_picker
        self.tab_color_picker = self.ui.tab_color_picker
        
        # Restore Ruler UI Elements
        self.action_ruler = self.ui.action_ruler
        self.spin_ruler_len = self.ui.spin_ruler_len
        self.lbl_ruler_px = self.ui.lbl_ruler_px
        self.lbl_ruler_calib = self.ui.lbl_ruler_calib
        self.lbl_ruler_meas = self.ui.lbl_ruler_meas
        self.btn_ruler_calibrate = self.ui.btn_ruler_calibrate
        self.chk_show_profile = self.ui.chk_show_profile

        # Ruler / Measurement Tool Init
        self.ruler_active = False
        self.ruler_start = None
        self.ruler_end = None
        self.ruler_calibration = None 
        
        # Color Picker Init
        self.color_picker_active = False

        # Connect Signals
        self.action_ruler.toggled.connect(self.on_ruler_toggled)
        self.action_color_picker.toggled.connect(self.on_color_picker_toggled)
        self.btn_ruler_calibrate.clicked.connect(self.calibrate_ruler)
        self.chk_show_profile.toggled.connect(self.on_show_profile_toggled)

        # Measurement Tab Setup (Index 0)
        self.ruler_tab_index = 0
        if hasattr(self.ui, 'right_tab_widget'):
            self.ui.right_tab_widget.setTabVisible(self.ruler_tab_index, False)
            self.color_picker_tab_index = self.ui.right_tab_widget.indexOf(self.tab_color_picker)
            self.ui.right_tab_widget.setTabVisible(self.color_picker_tab_index, False)

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
        # Sync initial enabled state with checkbox
        if hasattr(self.ui, 'chk_match_enable'):
             self.ui.tabs_matching.setEnabled(self.ui.chk_match_enable.isChecked())

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

        # Hough Circle Parameter Connections
        self.ui.hough_dp.valueChanged.connect(self.on_detector_params_changed)
        self.ui.hough_minDist.valueChanged.connect(self.on_detector_params_changed)
        self.ui.hough_param1.valueChanged.connect(self.on_detector_params_changed)
        self.ui.hough_param2.valueChanged.connect(self.on_detector_params_changed)
        self.ui.hough_minRadius.valueChanged.connect(self.on_detector_params_changed)
        self.ui.hough_maxRadius.valueChanged.connect(self.on_detector_params_changed)

        # ArUco Parameter Connections
        self.ui.aruco_dict.currentTextChanged.connect(self.on_detector_params_changed)
        self.ui.chk_aruco_show_ids.toggled.connect(self.on_detector_params_changed)
        self.ui.chk_aruco_show_rejected.toggled.connect(self.on_detector_params_changed)
        self.ui.spin_aruco_border_bits.valueChanged.connect(
            self.on_detector_params_changed
        )

        # New ArUco Enable Checkbox
        self.ui.chk_aruco_enable.toggled.connect(self.on_aruco_enable_toggled)
        
        # Hough Enable Checkbox
        self.ui.chk_hough_enable.toggled.connect(self.on_hough_enable_toggled)

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
        self.is_camera_running = False

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
                elif self.color_picker_active:
                    if event.type() == QEvent.MouseMove or event.type() == QEvent.MouseButtonPress:
                        pos = self.get_image_coords(event.position())
                        if pos:
                            self.update_color_picker(pos)
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

        # Hough Circle Priority
        if (
            hasattr(self.ui, "chk_hough_enable")
            and self.ui.chk_hough_enable.isChecked()
        ):
            params = {
                "algo": "HOUGH_CIRCLE",
                "dp": self.ui.hough_dp.value(),
                "minDist": self.ui.hough_minDist.value(),
                "param1": self.ui.hough_param1.value(),
                "param2": self.ui.hough_param2.value(),
                "minRadius": self.ui.hough_minRadius.value(),
                "maxRadius": self.ui.hough_maxRadius.value()
            }
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

    def update_frame(self, image):
        self.worker_busy = False
        if not image.isNull():
            # Recording Start Trigger
            if self.recording_requested:
                self.recording_requested = False
                record_fps = self.current_fps if self.current_fps > 0.1 else 30.0
                
                # Create videos dir
                video_dir = os.path.join(self.script_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                filename = os.path.join(video_dir, f"recording_{int(time.time())}.mkv")
                
                self.video_thread.startRecording(
                    image.width(), image.height(), record_fps, filename
                )
                self.ui.action_record.setText("Stop Recording")
                self.log(f"Recording started: {os.path.basename(filename)}")

            # Display
            self.current_pixmap = QPixmap.fromImage(image)
            self.refresh_video_label()
            
            # Real-time Intensity Profile Update
            if self.ruler_active and self.chk_show_profile.isChecked() and self.ruler_start:
                end_pt = self.ruler_end if self.ruler_end is not None else self.ruler_start
                self.update_intensity_profile(self.ruler_start, end_pt, image)

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
                self.is_camera_running = True
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
                self.param_poll_timer.start(200) # Poll every 200ms
                self.log("Camera started.")

    def on_stop_clicked(self):
        self.param_poll_timer.stop()
        self.is_camera_running = False
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
    
    def poll_camera_params(self):
        if not self.is_camera_running:
            return

        # 1. Exposure Time
        # Always read from camera
        try:
            current_exp = self.camera.getExposureTime()
            # Update UI only if not interacting (or if disabled/auto)
            if not self.ui.slider_exposure.isSliderDown() and not self.ui.spin_exposure_time.hasFocus():
                if abs(self.ui.spin_exposure_time.value() - current_exp) > 1.0: # Threshold
                    self.ui.spin_exposure_time.blockSignals(True)
                    self.ui.spin_exposure_time.setValue(current_exp)
                    self.ui.spin_exposure_time.blockSignals(False)
                    
                    min_exp = self.ui.spin_exposure_time.minimum()
                    max_exp = self.ui.spin_exposure_time.maximum()
                    self.update_slider_from_time(current_exp, min_exp, max_exp)
        except Exception:
            pass

        # 2. Gain
        try:
            current_gain = self.camera.getAnalogGain()
            if not self.ui.slider_gain.isSliderDown() and not self.ui.spin_gain.hasFocus():
                if abs(self.ui.spin_gain.value() - current_gain) > 0.1:
                    self.ui.spin_gain.blockSignals(True)
                    self.ui.spin_gain.setValue(current_gain)
                    self.ui.spin_gain.blockSignals(False)

                    self.ui.slider_gain.blockSignals(True)
                    self.ui.slider_gain.setValue(current_gain)
                    self.ui.slider_gain.blockSignals(False)
        except Exception:
            pass

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
            snap_dir = os.path.join(self.script_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            
            filename = os.path.join(snap_dir, f"snapshot_{int(time.time())}.png")
            self.current_pixmap.save(filename)
            self.log(f"Snapshot saved: {os.path.basename(filename)}")

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

            # Mutual exclusion: disable Hough if active
            if (
                hasattr(self.ui, "chk_hough_enable")
                and self.ui.chk_hough_enable.isChecked()
            ):
                self.ui.chk_hough_enable.blockSignals(True)
                self.ui.chk_hough_enable.setChecked(False)
                self.ui.chk_hough_enable.blockSignals(False)
            
            # Enable ArUco (update_detector picks up the checked state)
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            # Disable ArUco
            # Only stop worker if not switching to others (which shouldn't happen here directly)
            if not self.is_matching_ui_active:
                self.toggle_worker_matching_signal.emit(False)

    def on_hough_enable_toggled(self, checked):
        if checked:
            # Mutual exclusion
            for chk_name in ["chk_match_enable", "chk_aruco_enable", "chk_qrcode_enable", "chk_ssim_enable"]:
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

            # Mutual Exclusion: Disable Hough
            if (
                hasattr(self.ui, "chk_hough_enable")
                and self.ui.chk_hough_enable.isChecked()
            ):
                self.ui.chk_hough_enable.blockSignals(True)
                self.ui.chk_hough_enable.setChecked(False)
                self.ui.chk_hough_enable.blockSignals(False)

            self.is_matching_ui_active = True
            self.ui.tabs_matching.setEnabled(True)
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            self.is_matching_ui_active = False
            self.ui.tabs_matching.setEnabled(False)
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

            # Mutual Exclusion: Disable Hough
            if (
                hasattr(self.ui, "chk_hough_enable")
                and self.ui.chk_hough_enable.isChecked()
            ):
                self.ui.chk_hough_enable.blockSignals(True)
                self.ui.chk_hough_enable.setChecked(False)
                self.ui.chk_hough_enable.blockSignals(False)
            
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
            for chk_name in ["chk_match_enable", "chk_aruco_enable", "chk_qrcode_enable", "chk_hough_enable"]:
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
        if hasattr(self.camera, "getAeTarget"):
            # Enable controls
            self.ui.spin_ae_target.setEnabled(is_auto)
            self.ui.slider_ae_target.setEnabled(is_auto)
            self.ui.spin_ae_target.setToolTip("")
            self.ui.slider_ae_target.setToolTip("")
            
            try:
                # Set range if available or default 0-255 (typical byte range)
                # MindVision AE target is often around 120 default.
                self.ui.spin_ae_target.setRange(0, 255)
                self.ui.slider_ae_target.setRange(0, 255)

                current_ae = self.camera.getAeTarget()
                self.ui.spin_ae_target.blockSignals(True)
                self.ui.spin_ae_target.setValue(current_ae)
                self.ui.spin_ae_target.blockSignals(False)

                self.ui.slider_ae_target.blockSignals(True)
                self.ui.slider_ae_target.setValue(current_ae)
                self.ui.slider_ae_target.blockSignals(False)
            except Exception as e:
                self.log(f"Error syncing AE target: {e}")
        else:
            # Feature not supported by current MindVisionCamera wrapper
            self.ui.spin_ae_target.setEnabled(False)
            self.ui.slider_ae_target.setEnabled(False)
            self.ui.spin_ae_target.setToolTip("Not supported by current camera wrapper")
            self.ui.slider_ae_target.setToolTip("Not supported by current camera wrapper")

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
        if hasattr(self.camera, "setAeTarget"):
            self.camera.setAeTarget(value)
            self.ui.slider_ae_target.blockSignals(True)
            self.ui.slider_ae_target.setValue(value)
            self.ui.slider_ae_target.blockSignals(False)
        else:
            self.log("Error: Camera does not support setAeTarget")

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

    def on_color_picker_toggled(self, checked):
        self.color_picker_active = checked
        
        if hasattr(self.ui, 'right_tab_widget'):
            self.ui.right_tab_widget.setTabVisible(self.color_picker_tab_index, checked)
            if checked:
                self.ui.right_tab_widget.setCurrentIndex(self.color_picker_tab_index)
                
                # Disable Ruler if active
                if self.ruler_active:
                    self.action_ruler.setChecked(False)

    def update_color_picker(self, pos):
        if not self.current_pixmap:
            return
            
        x = int(pos.x())
        y = int(pos.y())
        
        if 0 <= x < self.current_pixmap.width() and 0 <= y < self.current_pixmap.height():
            img = self.current_pixmap.toImage()
            color = QColor(img.pixel(x, y))
            self.tab_color_picker.update_color(x, y, color.red(), color.green(), color.blue())

    def on_ruler_toggled(self, checked):
        self.ruler_active = checked
        if hasattr(self.ui, 'right_tab_widget'):
            self.ui.right_tab_widget.setTabVisible(self.ruler_tab_index, checked)
            if checked:
                self.ui.right_tab_widget.setCurrentIndex(self.ruler_tab_index)
                
                # Disable Color Picker if active
                if self.color_picker_active:
                    self.action_color_picker.setChecked(False)
             
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
        # The QPixmap.scaled uses Qt.KeepAspectRatio, so we need to replicate that logic to find the rect where image is drawn.
        
        scale_w = lbl_w / img_w
        scale_h = lbl_h / img_h
        scale = min(scale_w, scale_h)
        
        drawn_w = img_w * scale
        drawn_h = img_h * scale
        
        off_x = (lbl_w - drawn_w) / 2
        off_y = (lbl_h - drawn_h) / 2
        
        mx = mouse_pos.x()
        my = mouse_pos.y()
        
        # Check bounds
        if mx < off_x or mx > (off_x + drawn_w) or my < off_y or my > (off_y + drawn_h):
            return None
            
        # Map to image coords
        img_x = (mx - off_x) / scale
        img_y = (my - off_y) / scale
        
        return QPointF(img_x, img_y)

    def calibrate_ruler(self):
        if self.ruler_start is None or self.ruler_end is None:
            return
            
        line = QLineF(self.ruler_start, self.ruler_end)
        px_dist = line.length()
        
        if px_dist < 1.0:
            self.log("Ruler: Line too short to calibrate.")
            return
            
        known_mm = self.spin_ruler_len.value()
        if known_mm <= 0:
            return
            
        self.ruler_calibration = px_dist / known_mm # pixels per mm
        self.lbl_ruler_calib.setText(f"{self.ruler_calibration:.2f} px/mm")
        self.log(f"Ruler Calibrated: {self.ruler_calibration:.2f} px/mm")
        self.update_ruler_stats()

    def update_ruler_stats(self):
        if self.ruler_start is None:
            return
            
        end_pt = self.ruler_end if self.ruler_end is not None else self.ruler_start
        line = QLineF(self.ruler_start, end_pt)
        px_dist = line.length()
        
        self.lbl_ruler_px.setText(f"{px_dist:.1f} px")
        
        if self.ruler_calibration:
            mm_dist = px_dist / self.ruler_calibration
            self.lbl_ruler_meas.setText(f"{mm_dist:.2f} mm")
        else:
            self.lbl_ruler_meas.setText("0.00 mm")
            
        # Update Profile
        if self.chk_show_profile.isChecked() and self.current_pixmap:
            self.update_intensity_profile(self.ruler_start, end_pt)

    def update_intensity_profile(self, p1, p2, image=None):
        # We need the underlying image data (grayscale preferably)
        if image:
             qimg = image
        elif self.current_pixmap:
             qimg = self.current_pixmap.toImage()
        else:
             return
        
        # Sample points along the line
        line = QLineF(p1, p2)
        length = int(line.length())
        if length < 2: 
            return
            
        data = []
        for i in range(length):
            pt = line.pointAt(i / length)
            # Round to nearest pixel
            x = int(pt.x())
            y = int(pt.y())
            
            if 0 <= x < qimg.width() and 0 <= y < qimg.height():
                # Get grayscale value
                col = QColor(qimg.pixel(x, y))
                # Simple average or lightness
                val = int((col.red() + col.green() + col.blue()) / 3)
                data.append(val)
                
        self.intensity_chart.set_data(data)

    def load_settings(self):
        settings_file = os.path.join(self.script_dir, "camera_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                    
                    # Apply camera settings
                    self.ui.spin_exposure_time.setValue(settings.get("exposure_time", 2000))
                    self.ui.spin_gain.setValue(settings.get("gain", 1))
                    
                    # Apply detector params
                    if "detector" in settings:
                        self.load_detector_settings(settings["detector"])
                        
                    if "ruler_calibration" in settings:
                        self.ruler_calibration = settings["ruler_calibration"]
                        self.lbl_ruler_calib.setText(f"{self.ruler_calibration:.2f} px/mm")

                    if "led_controller_port" in settings:
                        led_port = settings["led_controller_port"]
                        if led_port:
                            index = self.ui.combo_serial_port.findText(led_port)
                            if index != -1:
                                self.ui.combo_serial_port.setCurrentIndex(index)
                    
                    if "cnc_controller_port" in settings and hasattr(self, 'cnc_control_panel'):
                        cnc_port = settings["cnc_controller_port"]
                        if cnc_port:
                            index = self.cnc_control_panel.serial_port_combo.findText(cnc_port)
                            if index != -1:
                                self.cnc_control_panel.serial_port_combo.setCurrentIndex(index)
                        
                    self.log("Settings loaded.")
            except Exception as e:
                self.log(f"Error loading settings: {e}")

    def save_settings(self):
        settings = {
            "exposure_time": self.ui.spin_exposure_time.value(),
            "gain": self.ui.spin_gain.value(),
            "detector": self.get_detector_settings(),
            "ruler_calibration": self.ruler_calibration
        }
        
        if hasattr(self.ui, 'combo_serial_port'):
            settings["led_controller_port"] = self.ui.combo_serial_port.currentText()
        if hasattr(self, 'cnc_control_panel'):
            settings["cnc_controller_port"] = self.cnc_control_panel.serial_port_combo.currentText()
        
        settings_file = os.path.join(self.script_dir, "camera_settings.json")
        try:
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=4)
                self.log("Settings saved.")
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def get_detector_settings(self):
        return {
            "orb_nfeatures": self.ui.orb_nfeatures.value(),
            # ... capture other UI values ...
            # For brevity, saving just a few
        }

    def load_detector_settings(self, data):
        if "orb_nfeatures" in data:
            self.ui.orb_nfeatures.setValue(data["orb_nfeatures"])
        # ... load others

    def apply_camera_settings(self):
        # Force apply current UI values to camera
        self.camera.setExposureTime(self.ui.spin_exposure_time.value())
        self.camera.setAnalogGain(self.ui.spin_gain.value())
