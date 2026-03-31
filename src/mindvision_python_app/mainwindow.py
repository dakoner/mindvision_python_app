import os
import sys
import time
import json
import threading
from bisect import bisect_left
from collections import deque
import cv2
import numpy as np
import PySide6.QtWidgets
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
    QScrollArea,
    QPushButton,
    QDockWidget,
    QVBoxLayout,
    QDoubleSpinBox,
    QRadioButton,
    QComboBox,
    QTabWidget,
    QSplitter,
    QSizePolicy,
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
    QRect,
)

from PySide6.QtGui import QPixmap, QAction, QPainter, QPen, QColor, QIcon, QImage
from PySide6.QtUiTools import QUiLoader

from .bindings import MindVisionCamera, VideoThread
from .range_slider import RangeSlider
from .intensity_chart import IntensityChart
from .color_picker_widget import ColorPickerWidget
from .mosaic_window import MosaicPanel
from .matching_worker import MatchingWorker
from .cnc_control_panel import CNCControlPanel
from .led_controller import LEDController
from .scan_config_panel import ScanConfigPanel
from .scan_status_dialog import ScanStatusDialog

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class MainWindow(QObject):
    update_fps_signal = Signal(float)
    error_signal = Signal(str)

    # Signal to send frame to worker
    process_frame_signal = Signal(int, int, int, int, bytes, object)
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
    
    # Signals for scan progress
    scan_status_signal = Signal(str)
    scan_progress_signal = Signal(int, int)


    def __init__(self):
        super().__init__()

        self.script_dir = os.path.dirname(__file__)

        # Load UI from file
        loader = QUiLoader()
        loader.registerCustomWidget(RangeSlider)
        loader.registerCustomWidget(IntensityChart)
        loader.registerCustomWidget(ColorPickerWidget)
        loader.registerCustomWidget(CNCControlPanel)

        # Load main window
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

        # Since mainwindow.ui is now loaded as self.ui, find these widgets directly
        # They are already part of the mainwindow.ui structure
        self.camera_settings_tabs = self.ui.findChild(QTabWidget, "camera_settings_tabs")
        self.hardware_tabs = self.ui.findChild(QTabWidget, "hardware_tabs")
        self.right_tab_widget = self.ui.findChild(QTabWidget, "right_tab_widget")
        
        # Ensure the main_h_splitter is also found
        self.main_h_splitter = self.ui.findChild(QSplitter, "main_h_splitter")
        self.center_v_splitter = self.ui.findChild(QSplitter, "center_v_splitter")
        
        # Center tab widget for main view and mosaic view
        self.center_tab_widget = self.ui.findChild(QTabWidget, "center_tab_widget")
        self.mosaic_container = self.ui.findChild(QWidget, "mosaic_container")

        # --- Find all widgets ---
        self.video_label = self.ui.findChild(QLabel, "video_label")
        self.chk_qrcode_enable = self.right_tab_widget.findChild(QCheckBox, "chk_qrcode_enable")
        self.btn_load_ssim_ref = self.right_tab_widget.findChild(QPushButton, "btn_load_ssim_ref")
        self.chk_ssim_enable = self.right_tab_widget.findChild(QCheckBox, "chk_ssim_enable")
        self.lbl_ssim_ref_name = self.right_tab_widget.findChild(QLabel, "lbl_ssim_ref_name")
        self.lbl_ssim_score = self.right_tab_widget.findChild(QLabel, "lbl_ssim_score")
        self.lbl_qrcode_data = self.right_tab_widget.findChild(QLabel, "lbl_qrcode_data")
        self.lbl_template_name = self.right_tab_widget.findChild(QLabel, "lbl_template_name")
        self.intensity_chart = self.right_tab_widget.findChild(IntensityChart, "intensity_chart")
        self.action_color_picker = self.ui.findChild(QAction, "action_color_picker")
        self.tab_color_picker = self.right_tab_widget.findChild(ColorPickerWidget, "tab_color_picker")
        self.action_ruler = self.ui.findChild(QAction, "action_ruler")
        self.spin_ruler_len = self.right_tab_widget.findChild(QDoubleSpinBox, "spin_ruler_len")
        self.lbl_ruler_px = self.right_tab_widget.findChild(QLabel, "lbl_ruler_px")
        self.lbl_ruler_calib = self.right_tab_widget.findChild(QLabel, "lbl_ruler_calib")
        self.lbl_ruler_meas = self.right_tab_widget.findChild(QLabel, "lbl_ruler_meas")
        self.btn_ruler_calibrate = self.right_tab_widget.findChild(QPushButton, "btn_ruler_calibrate")
        self.chk_show_profile = self.right_tab_widget.findChild(QCheckBox, "chk_show_profile")
        self.action_show_mosaic = self.ui.findChild(QAction, "action_show_mosaic")
        self.log_text_edit = self.ui.findChild(PySide6.QtWidgets.QPlainTextEdit, "log_text_edit")
        self.chk_auto_exposure = self.camera_settings_tabs.findChild(QCheckBox, "chk_auto_exposure")
        self.chk_roi = self.camera_settings_tabs.findChild(QCheckBox, "chk_roi")
        self.spin_exposure_time = self.camera_settings_tabs.findChild(QDoubleSpinBox, "spin_exposure_time")
        self.slider_exposure = self.camera_settings_tabs.findChild(PySide6.QtWidgets.QSlider, "slider_exposure")
        self.spin_gain = self.camera_settings_tabs.findChild(QSpinBox, "spin_gain")
        self.slider_gain = self.camera_settings_tabs.findChild(PySide6.QtWidgets.QSlider, "slider_gain")
        self.spin_ae_target = self.camera_settings_tabs.findChild(QSpinBox, "spin_ae_target")
        self.slider_ae_target = self.camera_settings_tabs.findChild(PySide6.QtWidgets.QSlider, "slider_ae_target")
        self.rb_continuous = self.camera_settings_tabs.findChild(QRadioButton, "rb_continuous")
        self.rb_software = self.camera_settings_tabs.findChild(QRadioButton, "rb_software")
        self.rb_hardware = self.camera_settings_tabs.findChild(QRadioButton, "rb_hardware")
        self.btn_soft_trigger = self.camera_settings_tabs.findChild(QPushButton, "btn_soft_trigger")
        self.spin_trigger_count = self.camera_settings_tabs.findChild(QSpinBox, "spin_trigger_count")
        self.spin_trigger_delay = self.camera_settings_tabs.findChild(QSpinBox, "spin_trigger_delay")
        self.spin_trigger_interval = self.camera_settings_tabs.findChild(QSpinBox, "spin_trigger_interval")
        self.combo_ext_mode = self.camera_settings_tabs.findChild(QComboBox, "combo_ext_mode")
        self.spin_ext_jitter = self.camera_settings_tabs.findChild(QSpinBox, "spin_ext_jitter")
        self.combo_ext_shutter = self.camera_settings_tabs.findChild(QComboBox, "combo_ext_shutter")
        self.combo_strobe_mode = self.camera_settings_tabs.findChild(QComboBox, "combo_strobe_mode")
        self.combo_strobe_polarity = self.camera_settings_tabs.findChild(QComboBox, "combo_strobe_polarity")
        self.spin_strobe_delay = self.camera_settings_tabs.findChild(QSpinBox, "spin_strobe_delay")
        self.spin_strobe_width = self.camera_settings_tabs.findChild(QSpinBox, "spin_strobe_width")
        self.action_start_camera = self.ui.findChild(QAction, "action_start_camera")
        self.action_stop_camera = self.ui.findChild(QAction, "action_stop_camera")
        self.action_record = self.ui.findChild(QAction, "action_record")
        self.action_snapshot = self.ui.findChild(QAction, "action_snapshot")
        self.action_predict = self.ui.findChild(QAction, "action_predict")
        self.action_home_and_run = self.ui.findChild(QAction, "action_home_and_run")
        self.btn_load_template = self.right_tab_widget.findChild(QPushButton, "btn_load_template")
        self.chk_match_enable = self.right_tab_widget.findChild(QCheckBox, "chk_match_enable")
        self.btn_toggle_contours = self.right_tab_widget.findChild(QPushButton, "btn_toggle_contours")
        self.combo_contour_mode = self.right_tab_widget.findChild(QComboBox, "combo_contour_mode")
        self.slider_threshold = self.right_tab_widget.findChild(PySide6.QtWidgets.QSlider, "slider_threshold")
        self.lbl_threshold_val = self.right_tab_widget.findChild(QLabel, "lbl_threshold_val")
        self.slider_canny = self.right_tab_widget.findChild(RangeSlider, "slider_canny")
        self.lbl_canny_val = self.right_tab_widget.findChild(QLabel, "lbl_canny_val")
        self.spin_min_area = self.right_tab_widget.findChild(QSpinBox, "spin_min_area")
        self.spin_max_area = self.right_tab_widget.findChild(QSpinBox, "spin_max_area")
        self.chk_fill_contours = self.right_tab_widget.findChild(QCheckBox, "chk_fill_contours")
        self.chk_show_box = self.right_tab_widget.findChild(QCheckBox, "chk_show_box")
        self.tabs_matching = self.right_tab_widget.findChild(QTabWidget, "tabs_matching")
        self.orb_nfeatures = self.right_tab_widget.findChild(QSpinBox, "orb_nfeatures")
        self.orb_scaleFactor = self.right_tab_widget.findChild(QDoubleSpinBox, "orb_scaleFactor")
        self.orb_nlevels = self.right_tab_widget.findChild(QSpinBox, "orb_nlevels")
        self.orb_edgeThreshold = self.right_tab_widget.findChild(QSpinBox, "orb_edgeThreshold")
        self.orb_firstLevel = self.right_tab_widget.findChild(QSpinBox, "orb_firstLevel")
        self.orb_wta_k = self.right_tab_widget.findChild(QSpinBox, "orb_wta_k")
        self.orb_scoreType = self.right_tab_widget.findChild(QComboBox, "orb_scoreType")
        self.orb_patchSize = self.right_tab_widget.findChild(QSpinBox, "orb_patchSize")
        self.orb_fastThreshold = self.right_tab_widget.findChild(QSpinBox, "orb_fastThreshold")
        self.sift_nfeatures = self.right_tab_widget.findChild(QSpinBox, "sift_nfeatures")
        self.sift_nOctaveLayers = self.right_tab_widget.findChild(QSpinBox, "sift_nOctaveLayers")
        self.sift_contrastThreshold = self.right_tab_widget.findChild(QDoubleSpinBox, "sift_contrastThreshold")
        self.sift_edgeThreshold = self.right_tab_widget.findChild(QDoubleSpinBox, "sift_edgeThreshold")
        self.sift_sigma = self.right_tab_widget.findChild(QDoubleSpinBox, "sift_sigma")
        self.akaze_descriptor_type = self.right_tab_widget.findChild(QComboBox, "akaze_descriptor_type")
        self.akaze_threshold = self.right_tab_widget.findChild(QDoubleSpinBox, "akaze_threshold")
        self.akaze_nOctaves = self.right_tab_widget.findChild(QSpinBox, "akaze_nOctaves")
        self.akaze_nOctaveLayers = self.right_tab_widget.findChild(QSpinBox, "akaze_nOctaveLayers")
        self.hough_dp = self.right_tab_widget.findChild(QDoubleSpinBox, "hough_dp")
        self.hough_minDist = self.right_tab_widget.findChild(QDoubleSpinBox, "hough_minDist")
        self.hough_param1 = self.right_tab_widget.findChild(QDoubleSpinBox, "hough_param1")
        self.hough_param2 = self.right_tab_widget.findChild(QDoubleSpinBox, "hough_param2")
        self.hough_minRadius = self.right_tab_widget.findChild(QSpinBox, "hough_minRadius")
        self.hough_maxRadius = self.right_tab_widget.findChild(QSpinBox, "hough_maxRadius")
        self.aruco_dict = self.right_tab_widget.findChild(QComboBox, "aruco_dict")
        self.chk_aruco_show_ids = self.right_tab_widget.findChild(QCheckBox, "chk_aruco_show_ids")
        self.chk_aruco_show_rejected = self.right_tab_widget.findChild(QCheckBox, "chk_aruco_show_rejected")
        self.spin_aruco_border_bits = self.right_tab_widget.findChild(QSpinBox, "spin_aruco_border_bits")
        self.chk_aruco_enable = self.right_tab_widget.findChild(QCheckBox, "chk_aruco_enable")
        self.chk_hough_enable = self.right_tab_widget.findChild(QCheckBox, "chk_hough_enable")
        self.main_h_splitter = self.ui.findChild(QSplitter, "main_h_splitter")
        # The following group boxes are children of camera_settings_tabs, which is now found directly in self.ui
        self.controls_group = self.ui.findChild(QGroupBox, "controls_group")
        self.trigger_group = self.ui.findChild(QGroupBox, "trigger_group")
        self.trigger_params_group = self.ui.findChild(QGroupBox, "trigger_params_group")
        self.ext_trigger_group = self.ui.findChild(QGroupBox, "ext_trigger_group")
        self.strobe_group = self.ui.findChild(QGroupBox, "strobe_group")

        # --- End of widget finding ---

        # Install event filter to handle close event
        self.ui.installEventFilter(self)

        self.video_label.installEventFilter(self)
        self.video_label.setFocusPolicy(Qt.StrongFocus)

        self.current_pixmap = None
        self.current_frame_image = QImage()
        self.prediction_model = None
        self.prediction_threshold = 0.1
        self.prediction_boxes = []
        self.prediction_enabled = False
        self.prediction_model_path = os.path.realpath(
            os.path.join(self.script_dir, "..", "..", "model", "weights", "best.pt")
        )
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

        # Parameter Poll Timer (for Auto Exposure updates)
        self.param_poll_timer = QTimer()
        self.param_poll_timer.timeout.connect(self.poll_camera_params)

        # --- CNC Control Panel Setup ---        
        self.cnc_control_panel = self.hardware_tabs.findChild(QGroupBox, "CNCControlPanel")
        if self.cnc_control_panel:
            self.cnc_control_panel.setupUi()
            self.cnc_control_panel.log_signal.connect(self.log)
        else:
            self.cnc_control_panel = None

        # Initial Detector config
        
        if hasattr(self, 'chk_qrcode_enable') and self.chk_qrcode_enable:
            self.chk_qrcode_enable.toggled.connect(self.on_qrcode_enable_toggled)

        # SSIM UI
        if hasattr(self, 'btn_load_ssim_ref') and self.btn_load_ssim_ref:
            self.btn_load_ssim_ref.clicked.connect(self.on_load_ssim_ref_clicked)
        if hasattr(self, 'chk_ssim_enable') and self.chk_ssim_enable:
            self.chk_ssim_enable.toggled.connect(self.on_ssim_enable_toggled)
            
        self.ssim_ref_loaded = False

        # Ruler / Measurement Tool Init
        self.ruler_active = False
        self.ruler_start = None
        self.ruler_end = None
        self.ruler_calibration = None 

        # Mosaic panel state - no longer a dock widget, but embedded in tab
        self.mosaic_panel = None
        self.scan_panel = None
        self.scan_status_dialog = None
        self.mosaic_panel_initialized = False
        
        self.current_cnc_x_mm = 0.0
        self.current_cnc_y_mm = 0.0
        self.current_cnc_z_mm = 0.0
        self.cnc_state = "Idle"
        self.mosaic_position_samples = deque(
            [(time.time_ns(), self.current_cnc_x_mm, self.current_cnc_y_mm, self.current_cnc_z_mm)]
        )
        self.pending_mosaic_frames = deque()
        self.mosaic_position_retention_ns = 10_000_000_000
        self.mosaic_interpolation_delay_ns = 75_000_000
        
        self.is_scanning = False
        self.scan_panel = None # Will be created with the mosaic panel
        self.scan_current_row = 0
        self.scan_current_area_total_rows = 0 # Total rows for the current scan area
        self.scan_total_rows = 0
        
        # Scan State for row-by-row processing
        self.scan_x_min = 0
        self.scan_y_min = 0
        self.scan_x_max = 0
        self.scan_y_max = 0
        self.scan_home_x = False
        self.scan_home_y = False
        self.scan_step_y = 0
        self.scan_current_y = 0
        self.scan_is_first_strip = True
        self.scan_fov_x_mm = 0
        self.scan_fov_y_mm = 0
        self.scan_started_recording = False
        
        # Stage Settings
        self.stage_settings = {}
        self.lbl_stage_width_mm = self.ui.findChild(QLabel, "lbl_stage_width_mm")
        self.lbl_stage_height_mm = self.ui.findChild(QLabel, "lbl_stage_height_mm")
        self.mosaic_circle_overlays_mm = [(51.6, 48.5, 8.0)]
        
        # Color Picker Init
        self.color_picker_active = False

        # Connect Signals
        if self.action_ruler:
            self.action_ruler.toggled.connect(self.on_ruler_toggled)
        if self.action_color_picker:
            self.action_color_picker.toggled.connect(self.on_color_picker_toggled)
        if self.btn_ruler_calibrate:
            self.btn_ruler_calibrate.clicked.connect(self.calibrate_ruler)
        if self.chk_show_profile:
            self.chk_show_profile.toggled.connect(self.on_show_profile_toggled)
        
        # Connect CNC position updates for mosaic
        if self.cnc_control_panel:
            self.cnc_control_panel.position_updated_signal.connect(self.on_cnc_position_updated)
            self.cnc_control_panel.state_updated_signal.connect(self.on_cnc_state_updated)
            self.cnc_control_panel.scan_start_ready_signal.connect(self.on_scan_start_ready)
            self.cnc_control_panel.scan_finished_signal.connect(self.on_scan_finished)

        # Measurement Tab Setup (Index 0)
        self.ruler_tab_index = 0
        if hasattr(self.ui, 'right_tab_widget'):
            self.ui.right_tab_widget.setTabVisible(self.ruler_tab_index, True)
            self.color_picker_tab_index = self.ui.right_tab_widget.indexOf(self.tab_color_picker)
            self.ui.right_tab_widget.setTabVisible(self.color_picker_tab_index, False)

        # Mosaic Window Action
        self.ui.action_show_mosaic.triggered.connect(self.on_show_mosaic_triggered)

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

        # --- LED Controller Setup ---
        self.led_controller = LEDController(self.hardware_tabs)
        self.led_controller.log_signal.connect(self.log)

        # Connections
        if self.chk_auto_exposure:
            self.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        if self.chk_roi:
            self.chk_roi.toggled.connect(self.on_roi_toggled)

        self.spin_exposure_time.valueChanged.connect(self.on_exposure_time_changed)
        self.slider_exposure.valueChanged.connect(self.on_exposure_slider_changed)

        self.spin_gain.valueChanged.connect(self.on_gain_changed)
        self.slider_gain.valueChanged.connect(self.on_gain_slider_changed)

        self.spin_ae_target.valueChanged.connect(self.on_ae_target_changed)
        self.slider_ae_target.valueChanged.connect(self.on_ae_slider_changed)

        # Recreate ButtonGroup for logic
        self.trigger_bg = QButtonGroup(self.ui)
        self.trigger_bg.addButton(self.rb_continuous, 0)
        self.trigger_bg.addButton(self.rb_software, 1)
        self.trigger_bg.addButton(self.rb_hardware, 2)
        self.trigger_bg.idToggled.connect(self.on_trigger_mode_changed)

        self.btn_soft_trigger.clicked.connect(self.on_soft_trigger_clicked)

        # New Connections for Trigger Params
        self.spin_trigger_count.valueChanged.connect(self.on_trigger_count_changed)
        self.spin_trigger_delay.valueChanged.connect(self.on_trigger_delay_changed)
        self.spin_trigger_interval.valueChanged.connect(
            self.on_trigger_interval_changed
        )

        # New Connections for External Trigger Params
        self.combo_ext_mode.currentIndexChanged.connect(self.on_ext_mode_changed)
        self.spin_ext_jitter.valueChanged.connect(self.on_ext_jitter_changed)
        self.combo_ext_shutter.currentIndexChanged.connect(
            self.on_ext_shutter_changed
        )

        # New Connections for Strobe Params
        self.combo_strobe_mode.currentIndexChanged.connect(self.on_strobe_mode_changed)
        self.combo_strobe_polarity.currentIndexChanged.connect(self.on_strobe_polarity_changed)
        self.spin_strobe_delay.valueChanged.connect(self.on_strobe_delay_changed)
        self.spin_strobe_width.valueChanged.connect(self.on_strobe_width_changed)
        
        # Retrieve Actions
        # self.ui.action_start_camera.triggered.connect(self.on_start_clicked)
        self.ui.action_stop_camera.triggered.connect(self.on_stop_clicked)
        self.ui.action_record.triggered.connect(self.on_record_clicked)
        self.ui.action_snapshot.triggered.connect(self.on_snapshot_clicked)
        if self.action_predict:
            self.action_predict.setCheckable(True)
            self.action_predict.toggled.connect(self.on_predict_toggled)
        self.ui.action_home_and_run.triggered.connect(self.on_home_and_run_clicked)

        # Template Matching UI
        if self.btn_load_template:
            self.btn_load_template.clicked.connect(self.on_load_template_clicked)
        if self.chk_match_enable:
            self.chk_match_enable.toggled.connect(self.on_match_enable_toggled)
            
        self.template_loaded = False
        
        # --- Contour Controls ---
        if self.slider_canny:
            self.slider_canny.setRange(0, 255)
            self.slider_canny.setValues(50, 150)

        # Connect Contour Signals
        if self.btn_toggle_contours:
            self.btn_toggle_contours.toggled.connect(self.on_toggle_contours_toggled)
        if self.combo_contour_mode:
            self.combo_contour_mode.currentTextChanged.connect(
                self.on_contour_params_changed
            )
        if self.slider_threshold:
            self.slider_threshold.valueChanged.connect(self.on_contour_params_changed)
        if self.slider_canny:
            self.slider_canny.valuesChanged.connect(self.on_contour_params_changed)
        if self.spin_min_area:
            self.spin_min_area.valueChanged.connect(self.on_contour_params_changed)
        if self.spin_max_area:
            self.spin_max_area.valueChanged.connect(self.on_contour_params_changed)
        if self.chk_fill_contours:
            self.chk_fill_contours.toggled.connect(self.on_contour_params_changed)
        if self.chk_show_box:
            self.chk_show_box.toggled.connect(self.on_contour_params_changed)
        # -----------------------------------------

        # Matching Tabs Connection
        if self.tabs_matching:
            self.tabs_matching.currentChanged.connect(self.on_detector_params_changed)
            # Sync initial enabled state with checkbox
            if self.chk_match_enable:
                self.tabs_matching.setEnabled(self.chk_match_enable.isChecked())

        # ORB Parameter Connections
        self.orb_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.orb_scaleFactor.valueChanged.connect(self.on_detector_params_changed)
        self.orb_nlevels.valueChanged.connect(self.on_detector_params_changed)
        self.orb_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.orb_firstLevel.valueChanged.connect(self.on_detector_params_changed)
        self.orb_wta_k.valueChanged.connect(self.on_detector_params_changed)
        self.orb_scoreType.currentIndexChanged.connect(
            self.on_detector_params_changed
        )
        self.orb_patchSize.valueChanged.connect(self.on_detector_params_changed)
        self.orb_fastThreshold.valueChanged.connect(self.on_detector_params_changed)

        # SIFT Parameter Connections
        self.sift_nfeatures.valueChanged.connect(self.on_detector_params_changed)
        self.sift_nOctaveLayers.valueChanged.connect(self.on_detector_params_changed)
        self.sift_contrastThreshold.valueChanged.connect(
            self.on_detector_params_changed
        )
        self.sift_edgeThreshold.valueChanged.connect(self.on_detector_params_changed)
        self.sift_sigma.valueChanged.connect(self.on_detector_params_changed)

        # AKAZE Parameter Connections
        self.akaze_descriptor_type.currentIndexChanged.connect(
            self.on_detector_params_changed
        )
        self.akaze_threshold.valueChanged.connect(self.on_detector_params_changed)
        self.akaze_nOctaves.valueChanged.connect(self.on_detector_params_changed)
        self.akaze_nOctaveLayers.valueChanged.connect(
            self.on_detector_params_changed
        )

        # Hough Circle Parameter Connections
        self.hough_dp.valueChanged.connect(self.on_detector_params_changed)
        self.hough_minDist.valueChanged.connect(self.on_detector_params_changed)
        self.hough_param1.valueChanged.connect(self.on_detector_params_changed)
        self.hough_param2.valueChanged.connect(self.on_detector_params_changed)
        self.hough_minRadius.valueChanged.connect(self.on_detector_params_changed)
        self.hough_maxRadius.valueChanged.connect(self.on_detector_params_changed)

        # ArUco Parameter Connections
        if self.aruco_dict:
            self.aruco_dict.currentTextChanged.connect(self.on_detector_params_changed)
        if self.chk_aruco_show_ids:
            self.chk_aruco_show_ids.toggled.connect(self.on_detector_params_changed)
        if self.chk_aruco_show_rejected:
            self.chk_aruco_show_rejected.toggled.connect(
                self.on_detector_params_changed
            )
        if self.spin_aruco_border_bits:
            self.spin_aruco_border_bits.valueChanged.connect(
                self.on_detector_params_changed
            )

        # New ArUco Enable Checkbox
        if self.chk_aruco_enable:
            self.chk_aruco_enable.toggled.connect(self.on_aruco_enable_toggled)

        # Hough Enable Checkbox
        if self.chk_hough_enable:
            self.chk_hough_enable.toggled.connect(self.on_hough_enable_toggled)

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
        self.metadata_filename = None
        self.recording_metadata_rows = []
        self.recording_metadata_lock = threading.Lock()
        self.recording_frame_index = 0
        self.recording_start_time_ns = 0
        self.recording_position_samples = deque()
        self.pending_recording_metadata = deque()
        self.recording_position_retention_ns = 10_000_000_000

        QTimer.singleShot(0, self.load_settings)
        QTimer.singleShot(100, self.on_start_clicked)

    def show(self):
        self.ui.showMaximized()
        # Set initial splitter sizes: [left, center, right]
        self.ui.main_h_splitter.setSizes([300, 1000, 350])
        if self.center_v_splitter:
            self.center_v_splitter.setChildrenCollapsible(False)
            self.center_v_splitter.setStretchFactor(0, 1)
            self.center_v_splitter.setStretchFactor(1, 0)
            self.center_v_splitter.setSizes([900, 120])
        if self.log_text_edit:
            self.log_text_edit.setMinimumHeight(80)
            self.log_text_edit.setMaximumHeight(120)

    def close(self):
        # Triggers closeEvent via the widget
        self.ui.close()

    def eventFilter(self, watched, event):
        try:
            if watched is self.ui and event.type() == QEvent.Close:
                self.save_settings()
                self.on_stop_clicked()

                if self.cnc_control_panel:
                    self.cnc_control_panel.stop()

                # Cleanup Matching Thread
                if self.matching_thread.isRunning():
                    self.matching_thread.quit()
                    self.matching_thread.wait()

                # Cleanup LED Controller
                self.led_controller.stop()

                self.ui.removeEventFilter(self)
                try:
                    self.ui.video_label.removeEventFilter(self)
                except RuntimeError:
                    pass
            elif watched is self.ui and event.type() == QEvent.KeyPress:
                if self.video_label and self.video_label.underMouse():
                    if self._handle_stage_arrow_key(event):
                        return True
            elif watched == self.ui.video_label:
                if event.type() == QEvent.Resize:
                    self.refresh_video_label()
                elif event.type() == QEvent.KeyPress:
                    if self._handle_stage_arrow_key(event):
                        return True
                
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.video_label.setFocus(Qt.MouseFocusReason)

                if self.ruler_active:
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

    def _handle_stage_arrow_key(self, event):
        if not self.cnc_control_panel:
            return False

        key_to_move = {
            Qt.Key_Left: self.cnc_control_panel.move_left,
            Qt.Key_Right: self.cnc_control_panel.move_right,
            Qt.Key_Up: self.cnc_control_panel.move_forward,
            Qt.Key_Down: self.cnc_control_panel.move_back,
            Qt.Key_PageUp: self.cnc_control_panel.move_up,
            Qt.Key_PageDown: self.cnc_control_panel.move_down,
        }

        move_action = key_to_move.get(event.key())
        if move_action is None:
            return False

        move_action()
        return True

    def refresh_video_label(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            display_pixmap = self.current_pixmap

            has_prediction_overlay = bool(self.prediction_boxes)
            has_ruler_overlay = self.ruler_active and (self.ruler_start is not None)

            if has_prediction_overlay or has_ruler_overlay:
                display_pixmap = self.current_pixmap.copy()
                painter = QPainter(display_pixmap)

                if has_prediction_overlay:
                    pred_pen = QPen(QColor(0, 0, 255), 2)
                    painter.setPen(pred_pen)
                    for prediction in self.prediction_boxes:
                        rect = prediction["rect"]
                        confidence = prediction["confidence"]
                        painter.drawRect(rect)
                        text_y = max(14, rect.y() - 4)
                        painter.drawText(rect.x(), text_y, f"{confidence:.2f}")

                if has_ruler_overlay:
                    ruler_pen = QPen(QColor(0, 255, 255), 2)
                    painter.setPen(ruler_pen)

                    start_pt = self.ruler_start
                    end_pt = self.ruler_end if self.ruler_end is not None else start_pt

                    painter.drawLine(start_pt, end_pt)

                    ruler_pen.setWidth(4)
                    painter.setPen(ruler_pen)
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

    def update_detector(self):
        # SSIM Priority
        if self.chk_ssim_enable and self.chk_ssim_enable.isChecked():
            params = {'algo': 'SSIM'}
            self.update_worker_params_signal.emit(params)
            return

        # QR Code Priority
        if self.chk_qrcode_enable and self.chk_qrcode_enable.isChecked():
            params = {"algo": "QRCODE"}
            self.update_worker_params_signal.emit(params)
            return

        # ArUco Priority
        if self.chk_aruco_enable and self.chk_aruco_enable.isChecked():
            params = {"algo": "ARUCO"}
            if self.aruco_dict:
                params["dict"] = self.aruco_dict.currentText()
            if self.chk_aruco_show_ids:
                params["show_ids"] = self.chk_aruco_show_ids.isChecked()
            if self.chk_aruco_show_rejected:
                params["show_rejected"] = self.chk_aruco_show_rejected.isChecked()
            if self.spin_aruco_border_bits:
                params["markerBorderBits"] = self.spin_aruco_border_bits.value()
            self.update_worker_params_signal.emit(params)
            return

        # Hough Circle Priority
        if self.chk_hough_enable and self.chk_hough_enable.isChecked():
            params = {
                "algo": "HOUGH_CIRCLE",
                "dp": self.hough_dp.value(),
                "minDist": self.hough_minDist.value(),
                "param1": self.hough_param1.value(),
                "param2": self.hough_param2.value(),
                "minRadius": self.hough_minRadius.value(),
                "maxRadius": self.hough_maxRadius.value()
            }
            self.update_worker_params_signal.emit(params)
            return

        tab_index = self.tabs_matching.currentIndex() if self.tabs_matching else 0
        params = {}

        if tab_index == 0:  # ORB
            params["algo"] = "ORB"
            params["nfeatures"] = self.orb_nfeatures.value()
            params["scaleFactor"] = self.orb_scaleFactor.value()
            params["nlevels"] = self.orb_nlevels.value()
            params["edgeThreshold"] = self.orb_edgeThreshold.value()
            params["firstLevel"] = self.orb_firstLevel.value()
            params["WTA_K"] = self.orb_wta_k.value()
            params["scoreType"] = self.orb_scoreType.currentIndex()
            params["patchSize"] = self.orb_patchSize.value()
            params["fastThreshold"] = self.orb_fastThreshold.value()

        elif tab_index == 1:  # SIFT
            params["algo"] = "SIFT"
            params["nfeatures"] = self.sift_nfeatures.value()
            params["nOctaveLayers"] = self.sift_nOctaveLayers.value()
            params["contrastThreshold"] = self.sift_contrastThreshold.value()
            params["edgeThreshold"] = self.sift_edgeThreshold.value()
            params["sigma"] = self.sift_sigma.value()

        elif tab_index == 2:  # AKAZE
            params["algo"] = "AKAZE"
            combo_idx = self.akaze_descriptor_type.currentIndex()
            mapping = [2, 3, 4, 5]
            params["descriptor_type"] = mapping[combo_idx]
            params["threshold"] = self.akaze_threshold.value()
            params["nOctaves"] = self.akaze_nOctaves.value()
            params["nOctaveLayers"] = self.akaze_nOctaveLayers.value()

        self.update_worker_params_signal.emit(params)

    def on_detector_params_changed(self):
        self.update_detector()

    def frame_callback(
        self, width, height, bytes_per_line, fmt, data, timestamp_ms=None
    ):
        frame_timestamp_ns = (
            int(timestamp_ms) * 1_000_000
            if timestamp_ms is not None
            else time.time_ns()
        )

        # 1. Recording (High Priority)
        if self.video_thread.isRunning():
            try:
                self.video_thread.addFrameBytes(
                    width, height, bytes_per_line, fmt, data
                )
                if self.metadata_filename:
                    current_ns = max(0, frame_timestamp_ns - self.recording_start_time_ns)
                    with self.recording_metadata_lock:
                        self.pending_recording_metadata.append(
                            (frame_timestamp_ns, current_ns, self.recording_frame_index)
                        )
                        self.recording_frame_index += 1
                        self._resolve_pending_recording_metadata_locked()
                        self._prune_recording_position_samples_locked()
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
                        width, height, bytes_per_line, fmt, data_copy, frame_timestamp_ns
                    )
                # Else: Drop frame for UI display to prevent backlog/latency
            except Exception as e:
                self.worker_busy = False  # Reset on error
                self.log(f"Error in frame callback (UI): {e}")

    def fps_callback(self, fps):
        self.update_fps_signal.emit(fps)

    def error_callback(self, msg):
        self.error_signal.emit(msg)

    def _interpolate_mosaic_position(self, frame_timestamp_ns, clamp_latest=False):
        samples = list(self.mosaic_position_samples)
        if not samples:
            return None

        timestamps = [sample[0] for sample in samples]
        sample_idx = bisect_left(timestamps, frame_timestamp_ns)

        if sample_idx <= 0:
            return samples[0][1:]

        if sample_idx >= len(samples):
            if not clamp_latest:
                return None
            return samples[-1][1:]

        before_ts, before_x, before_y, before_z = samples[sample_idx - 1]
        after_ts, after_x, after_y, after_z = samples[sample_idx]
        if after_ts <= before_ts:
            return (after_x, after_y, after_z)

        ratio = (frame_timestamp_ns - before_ts) / (after_ts - before_ts)
        ratio = max(0.0, min(1.0, ratio))
        x = before_x + ((after_x - before_x) * ratio)
        y = before_y + ((after_y - before_y) * ratio)
        z = before_z + ((after_z - before_z) * ratio)
        return (x, y, z)

    def _prune_mosaic_position_samples(self):
        if len(self.mosaic_position_samples) <= 2:
            return

        if self.pending_mosaic_frames:
            cutoff_ns = self.pending_mosaic_frames[0][0]
        else:
            cutoff_ns = self.mosaic_position_samples[-1][0] - self.mosaic_position_retention_ns

        while len(self.mosaic_position_samples) > 2:
            next_timestamp_ns = self.mosaic_position_samples[1][0]
            if next_timestamp_ns > cutoff_ns:
                break
            self.mosaic_position_samples.popleft()

    def _resolve_pending_mosaic_frames(self, force=False):
        if not self.pending_mosaic_frames or not self.mosaic_position_samples:
            return

        now_ns = time.time_ns()
        while self.pending_mosaic_frames:
            frame_timestamp_ns, image = self.pending_mosaic_frames[0]
            allow_clamp = force or ((now_ns - frame_timestamp_ns) >= self.mosaic_interpolation_delay_ns)
            position = self._interpolate_mosaic_position(
                frame_timestamp_ns,
                clamp_latest=allow_clamp,
            )
            if position is None:
                break

            self.pending_mosaic_frames.popleft()
            if self.mosaic_panel_initialized and self.mosaic_panel and self.center_tab_widget.currentIndex() == 1:
                if self.cnc_state != "Home":
                    self.mosaic_panel.update_mosaic(image, position[0], position[1])

        self._prune_mosaic_position_samples()

    def update_frame(self, image, frame_timestamp_ns=None):
        self.worker_busy = False
        if not image.isNull():
            self.current_frame_image = image.copy()
            # Recording Start Trigger
            if self.recording_requested:
                self.recording_requested = False
                record_fps = self.current_fps if self.current_fps > 0.1 else 30.0
                
                # Create videos dir
                video_dir = os.path.realpath(
                    os.path.join(self.script_dir, "..", "..", "videos")
                )
                os.makedirs(video_dir, exist_ok=True)
                
                timestamp = int(time.time())
                filename = os.path.join(video_dir, f"recording_{timestamp}.rgb")
                
                meta_filename = os.path.join(video_dir, f"recording_{timestamp}_meta.csv")
                self.metadata_filename = meta_filename
                self.recording_start_time_ns = time.time_ns()
                self.recording_frame_index = 0
                with self.recording_metadata_lock:
                    self.recording_metadata_rows = []
                    self.pending_recording_metadata.clear()
                    self.recording_position_samples.clear()
                    self.recording_position_samples.append(
                        (
                            self.recording_start_time_ns,
                            getattr(self, 'current_cnc_x_mm', 0.0),
                            getattr(self, 'current_cnc_y_mm', 0.0),
                            getattr(self, 'current_cnc_z_mm', 0.0),
                        )
                    )

                self.video_thread.startRecording(
                    image.width(), image.height(), record_fps, filename
                )
                self.ui.action_record.setText("Stop Recording")
                self.log(f"Recording started: {os.path.basename(filename)}")

            # Display
            #image = image.mirrored(True, False) # Flip horizontally
            if self.prediction_enabled:
                self._run_prediction_on_image(image, log_result=False)
            self.current_pixmap = QPixmap.fromImage(image)
            self.refresh_video_label()
            
            # Real-time Intensity Profile Update
            if self.ruler_active and self.chk_show_profile.isChecked() and self.ruler_start:
                end_pt = self.ruler_end if self.ruler_end is not None else self.ruler_start
                self.update_intensity_profile(self.ruler_start, end_pt, image)

            # Update Mosaic Window
            # Check if mosaic panel is initialized and the mosaic tab is currently visible
            if self.mosaic_panel_initialized and self.mosaic_panel and \
               self.center_tab_widget.currentIndex() == 1 and \
               self.current_cnc_x_mm is not None and self.current_cnc_y_mm is not None:
                if self.cnc_state != "Home":
                    mosaic_timestamp_ns = frame_timestamp_ns if frame_timestamp_ns is not None else time.time_ns()
                    self.pending_mosaic_frames.append((mosaic_timestamp_ns, image.copy()))
                    self._resolve_pending_mosaic_frames()
            else:
                self.pending_mosaic_frames.clear()

    def load_prediction_model(self):
        if YOLO is None:
            self.log("Prediction unavailable: ultralytics is not installed.")
            return False

        if self.prediction_model is not None:
            return True

        if not os.path.exists(self.prediction_model_path):
            self.log(f"Prediction model not found: {self.prediction_model_path}")
            return False

        try:
            self.prediction_model = YOLO(self.prediction_model_path)
            self.log(f"Prediction model loaded: {os.path.basename(self.prediction_model_path)}")
            return True
        except Exception as e:
            self.log(f"Failed to load prediction model: {e}")
            self.prediction_model = None
            return False

    def _run_prediction_on_image(self, qimage, log_result=True):
        if not self.load_prediction_model():
            return False

        rgb_image = qimage.convertToFormat(QImage.Format_RGB888)
        width = rgb_image.width()
        height = rgb_image.height()
        frame_bytes = rgb_image.bits().tobytes()
        frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        try:
            results = self.prediction_model.predict(source=frame_bgr, verbose=False, save=False)
        except Exception as e:
            if log_result:
                self.log(f"Prediction failed: {e}")
            return False

        predicted_boxes = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0].cpu().numpy())
                if confidence < self.prediction_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x = int(x1)
                y = int(y1)
                box_width = int(x2 - x1)
                box_height = int(y2 - y1)
                if box_width <= 0 or box_height <= 0:
                    continue

                predicted_boxes.append({
                    "rect": QRect(x, y, box_width, box_height),
                    "confidence": confidence,
                })

        self.prediction_boxes = predicted_boxes
        if log_result:
            if predicted_boxes:
                self.log(f"Added {len(predicted_boxes)} prediction rectangle(s).")
            else:
                self.log("No predictions above threshold in current frame.")
        return True

    def predict_current_frame(self):
        if self.current_frame_image.isNull():
            self.log("No frame available for prediction.")
            return

        if self._run_prediction_on_image(self.current_frame_image, log_result=True):
            self.refresh_video_label()

    def on_predict_toggled(self, checked):
        if checked:
            if not self.load_prediction_model():
                if self.action_predict:
                    self.action_predict.blockSignals(True)
                    self.action_predict.setChecked(False)
                    self.action_predict.blockSignals(False)
                return

            self.prediction_enabled = True
            self.log("Prediction enabled (running on every frame).")
            if not self.current_frame_image.isNull():
                self._run_prediction_on_image(self.current_frame_image, log_result=False)
                self.refresh_video_label()
            return

        self.prediction_enabled = False
        self.prediction_boxes = []
        self.refresh_video_label()
        self.log("Prediction disabled.")

    @Slot(float)
    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def handle_error(self, message):
        self.log(f"Camera Error: {message}")
        self.ui.video_label.setText(f"Error: {message}")
        # self.ui.action_start_camera.setEnabled(True)
        self.ui.action_stop_camera.setEnabled(False)
        self.controls_group.setEnabled(False)
        self.trigger_group.setEnabled(False)
        self.trigger_params_group.setEnabled(False)
        self.ext_trigger_group.setEnabled(False)
        self.strobe_group.setEnabled(False)

    def on_start_clicked(self):
        if self.camera.open():
            if self.camera.start():
                self.is_camera_running = True
                # self.ui.action_start_camera.setEnabled(False)
                self.ui.action_stop_camera.setEnabled(True)
                self.ui.action_record.setEnabled(True)
                self.ui.action_snapshot.setEnabled(True)
                if self.action_predict:
                    self.action_predict.setEnabled(True)
                self.controls_group.setEnabled(True)
                self.trigger_group.setEnabled(True)
                self.strobe_group.setEnabled(True)

                self.ui.video_label.setText("Starting stream...")
                self.apply_camera_settings()
                self.sync_ui()
                self.param_poll_timer.start(200) # Poll every 200ms
                self.log("Camera started.")

    def on_stop_clicked(self):
        self.param_poll_timer.stop()
        self.is_camera_running = False
        self.prediction_enabled = False
        self.prediction_boxes = []
        if self.video_thread.isRunning():
            self.on_record_clicked()

        self.camera.stop()
        self.camera.close()
        # self.ui.action_start_camera.setEnabled(True)
        self.ui.action_stop_camera.setEnabled(False)
        self.ui.action_record.setEnabled(False)
        self.ui.action_snapshot.setEnabled(False)
        if self.action_predict:
            self.action_predict.blockSignals(True)
            self.action_predict.setChecked(False)
            self.action_predict.blockSignals(False)
            self.action_predict.setEnabled(False)

        self.controls_group.setEnabled(False)
        self.trigger_group.setEnabled(False)
        self.trigger_params_group.setEnabled(False)
        self.ext_trigger_group.setEnabled(False)
        self.strobe_group.setEnabled(False)

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
            if not self.slider_exposure.isSliderDown() and not self.spin_exposure_time.hasFocus():
                if abs(self.spin_exposure_time.value() - current_exp) > 1.0: # Threshold
                    self.spin_exposure_time.blockSignals(True)
                    self.spin_exposure_time.setValue(current_exp)
                    self.spin_exposure_time.blockSignals(False)
                    
                    min_exp = self.spin_exposure_time.minimum()
                    max_exp = self.spin_exposure_time.maximum()
                    self.update_slider_from_time(current_exp, min_exp, max_exp)
        except Exception:
            pass

        # 2. Gain
        try:
            current_gain = self.camera.getAnalogGain()
            if not self.slider_gain.isSliderDown() and not self.spin_gain.hasFocus():
                if abs(self.spin_gain.value() - current_gain) > 0.1:
                    self.spin_gain.blockSignals(True)
                    self.spin_gain.setValue(current_gain)
                    self.spin_gain.blockSignals(False)

                    self.slider_gain.blockSignals(True)
                    self.slider_gain.setValue(current_gain)
                    self.slider_gain.blockSignals(False)
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
            self._flush_recording_metadata_csv()

    def _flush_recording_metadata_csv(self):
        metadata_filename = self.metadata_filename
        if not metadata_filename:
            return

        with self.recording_metadata_lock:
            self._resolve_pending_recording_metadata_locked(force=True)
            rows = list(self.recording_metadata_rows)
            self.recording_metadata_rows = []
            self.pending_recording_metadata.clear()
            self.recording_position_samples.clear()

        rows.sort(key=lambda row: row[1])

        try:
            with open(metadata_filename, "w") as meta_file:
                meta_file.write("timestamp_ns,frame_index,x,y,z\n")
                for timestamp_ns, frame_index, x, y, z in rows:
                    meta_file.write(f"{timestamp_ns},{frame_index},{x:.3f},{y:.3f},{z:.3f}\n")
        except Exception as e:
            self.log(f"Failed to write recording metadata CSV: {e}")
        finally:
            self.metadata_filename = None

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
            if self.chk_match_enable and self.chk_match_enable.isChecked():
                self.chk_match_enable.blockSignals(True)
                self.chk_match_enable.setChecked(False)
                self.chk_match_enable.blockSignals(False)
                self.is_matching_ui_active = False

            # Mutual exclusion: disable QR Code if active
            if self.chk_qrcode_enable and self.chk_qrcode_enable.isChecked():
                self.chk_qrcode_enable.blockSignals(True)
                self.chk_qrcode_enable.setChecked(False)
                self.chk_qrcode_enable.blockSignals(False)

            # Mutual exclusion: disable SSIM if active
            if self.chk_ssim_enable and self.chk_ssim_enable.isChecked():
                self.chk_ssim_enable.blockSignals(True)
                self.chk_ssim_enable.setChecked(False)
                self.chk_ssim_enable.blockSignals(False)

            # Mutual exclusion: disable Hough if active
            if self.chk_hough_enable and self.chk_hough_enable.isChecked():
                self.chk_hough_enable.blockSignals(True)
                self.chk_hough_enable.setChecked(False)
                self.chk_hough_enable.blockSignals(False)
            
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
                chk = getattr(self, chk_name, None)
                if chk and chk.isChecked():
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
            if self.lbl_template_name:
                self.lbl_template_name.setText(filename)
                self.lbl_template_name.setStyleSheet("color: black;")

            # If enabled, update worker
            if self.chk_match_enable and self.chk_match_enable.isChecked():
                # self.update_detector() # Redundant, worker should already be up to date
                self.toggle_worker_matching_signal.emit(True)

    def on_match_enable_toggled(self, checked):
        if checked:
            if not self.template_loaded:
                # If no template, try to load one
                self.on_load_template_clicked()
                # If still not loaded (user cancelled), uncheck
                if not self.template_loaded:
                    if self.chk_match_enable:
                        self.chk_match_enable.setChecked(False)
                    return

            # Mutual Exclusion: Disable ArUco
            if self.chk_aruco_enable and self.chk_aruco_enable.isChecked():
                self.chk_aruco_enable.blockSignals(True)
                self.chk_aruco_enable.setChecked(False)
                self.chk_aruco_enable.blockSignals(False)

            # Mutual Exclusion: Disable QR Code
            if self.chk_qrcode_enable and self.chk_qrcode_enable.isChecked():
                self.chk_qrcode_enable.blockSignals(True)
                self.chk_qrcode_enable.setChecked(False)
                self.chk_qrcode_enable.blockSignals(False)

            # Mutual Exclusion: Disable SSIM
            if self.chk_ssim_enable and self.chk_ssim_enable.isChecked():
                self.chk_ssim_enable.blockSignals(True)
                self.chk_ssim_enable.setChecked(False)
                self.chk_ssim_enable.blockSignals(False)

            # Mutual Exclusion: Disable Hough
            if self.chk_hough_enable and self.chk_hough_enable.isChecked():
                self.chk_hough_enable.blockSignals(True)
                self.chk_hough_enable.setChecked(False)
                self.chk_hough_enable.blockSignals(False)

            self.is_matching_ui_active = True
            if self.tabs_matching:
                self.tabs_matching.setEnabled(True)
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            self.is_matching_ui_active = False
            if self.tabs_matching:
                self.tabs_matching.setEnabled(False)
            self.toggle_worker_matching_signal.emit(False)

    def on_qrcode_enable_toggled(self, checked):
        if checked:
            # Mutual exclusion: disable template matching if active
            if self.chk_match_enable and self.chk_match_enable.isChecked():
                self.chk_match_enable.blockSignals(True)
                self.chk_match_enable.setChecked(False)
                self.chk_match_enable.blockSignals(False)
                self.is_matching_ui_active = False

            # Mutual exclusion: disable ArUco
            if self.chk_aruco_enable and self.chk_aruco_enable.isChecked():
                self.chk_aruco_enable.blockSignals(True)
                self.chk_aruco_enable.setChecked(False)
                self.chk_aruco_enable.blockSignals(False)

            # Mutual exclusion: disable SSIM
            if self.chk_ssim_enable and self.chk_ssim_enable.isChecked():
                self.chk_ssim_enable.blockSignals(True)
                self.chk_ssim_enable.setChecked(False)
                self.chk_ssim_enable.blockSignals(False)

            # Mutual Exclusion: Disable Hough
            if self.chk_hough_enable and self.chk_hough_enable.isChecked():
                self.chk_hough_enable.blockSignals(True)
                self.chk_hough_enable.setChecked(False)
                self.chk_hough_enable.blockSignals(False)
            
            self.update_detector()
            self.toggle_worker_matching_signal.emit(True)
        else:
            if not self.is_matching_ui_active:
                self.toggle_worker_matching_signal.emit(False)

    @Slot(str)
    def handle_qr_found(self, text):
        if self.lbl_qrcode_data:
            self.lbl_qrcode_data.setText(f"Decoded Data: {text}")

    @Slot(float)
    def handle_ssim_score(self, score):
        if self.lbl_ssim_score:
            self.lbl_ssim_score.setText(f"Score: {score:.4f}")

    def on_load_ssim_ref_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Reference Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.set_worker_ssim_ref_signal.emit(file_path)
            self.ssim_ref_loaded = True
            
            filename = os.path.basename(file_path)
            if self.lbl_ssim_ref_name:
                self.lbl_ssim_ref_name.setText(filename)
                self.lbl_ssim_ref_name.setStyleSheet("color: black;")
            
            if self.chk_ssim_enable and self.chk_ssim_enable.isChecked():
                self.update_detector()
                self.toggle_worker_matching_signal.emit(True)

    def on_ssim_enable_toggled(self, checked):
        if checked:
            if not self.ssim_ref_loaded:
                self.on_load_ssim_ref_clicked()
                if not self.ssim_ref_loaded:
                    if self.chk_ssim_enable:
                        self.chk_ssim_enable.setChecked(False)
                    return

            # Mutual Exclusion
            for chk_name in ["chk_match_enable", "chk_aruco_enable", "chk_qrcode_enable", "chk_hough_enable"]:
                chk = getattr(self, chk_name, None)
                if chk and chk.isChecked():
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

        self.spin_exposure_time.setRange(min_exp, max_exp)
        if step_exp > 0:
            self.spin_exposure_time.setSingleStep(step_exp)
            self.slider_exposure.setRange(0, int((max_exp - min_exp) / step_exp))
        else:
            self.slider_exposure.setRange(0, 10000)

        min_gain, max_gain = self.camera.getAnalogGainRange()
        self.spin_gain.setRange(min_gain, max_gain)
        self.slider_gain.setRange(min_gain, max_gain)

        # Values
        is_auto = self.camera.getAutoExposure()
        self.chk_auto_exposure.setChecked(is_auto)
        self.spin_exposure_time.setEnabled(not is_auto)
        self.slider_exposure.setEnabled(not is_auto)
        self.spin_ae_target.setEnabled(is_auto)
        self.slider_ae_target.setEnabled(is_auto)

        # AE Target
        if hasattr(self.camera, "getAeTarget"):
            # Enable controls
            self.spin_ae_target.setEnabled(is_auto)
            self.slider_ae_target.setEnabled(is_auto)
            self.spin_ae_target.setToolTip("")
            self.slider_ae_target.setToolTip("")
            
            try:
                # Set range if available or default 0-255 (typical byte range)
                # MindVision AE target is often around 120 default.
                self.spin_ae_target.setRange(0, 255)
                self.slider_ae_target.setRange(0, 255)

                current_ae = self.camera.getAeTarget()
                self.spin_ae_target.blockSignals(True)
                self.spin_ae_target.setValue(current_ae)
                self.spin_ae_target.blockSignals(False)

                self.slider_ae_target.blockSignals(True)
                self.slider_ae_target.setValue(current_ae)
                self.slider_ae_target.blockSignals(False)
            except Exception as e:
                self.log(f"Error syncing AE target: {e}")
        else:
            # Feature not supported by current MindVisionCamera wrapper
            self.spin_ae_target.setEnabled(False)
            self.slider_ae_target.setEnabled(False)
            self.spin_ae_target.setToolTip("Not supported by current camera wrapper")
            self.slider_ae_target.setToolTip("Not supported by current camera wrapper")

        current_exp = self.camera.getExposureTime()
        self.spin_exposure_time.blockSignals(True)
        self.spin_exposure_time.setValue(current_exp)
        self.spin_exposure_time.blockSignals(False)

        self.update_slider_from_time(current_exp, min_exp, max_exp)

        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(self.camera.getAnalogGain())
        self.spin_gain.blockSignals(False)

        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(self.camera.getAnalogGain())
        self.slider_gain.blockSignals(False)

    def update_slider_from_time(self, current, min_val, max_val):
        self.slider_exposure.blockSignals(True)
        step_exp = self.camera.getExposureTimeStep()
        if step_exp > 0:
            val = int(round((current - min_val) / step_exp))
            self.slider_exposure.setValue(val)
        else:
            rng = max_val - min_val
            if rng > 0:
                val = int((current - min_val) / rng * 10000)
                self.slider_exposure.setValue(val)
        self.slider_exposure.blockSignals(False)

    def on_auto_exposure_toggled(self, checked):
        if not self.is_camera_running:
            return
        if self.camera.setAutoExposure(checked):
            self.spin_exposure_time.setEnabled(not checked)
            self.slider_exposure.setEnabled(not checked)
            self.spin_ae_target.setEnabled(checked)
            self.slider_ae_target.setEnabled(checked)
            if not checked:
                # Update manual values
                current_exp = self.camera.getExposureTime()
                self.spin_exposure_time.setValue(current_exp)
                min_exp = self.spin_exposure_time.minimum()
                max_exp = self.spin_exposure_time.maximum()
                self.update_slider_from_time(current_exp, min_exp, max_exp)
        else:
            self.chk_auto_exposure.setChecked(not checked)

    def on_roi_toggled(self, checked):
        if not self.camera.setRoi(checked):
            self.chk_roi.setChecked(not checked)

    def on_exposure_time_changed(self, value):
        if not self.is_camera_running:
            return
        self.camera.setExposureTime(value)
        actual = self.camera.getExposureTime()
        self.spin_exposure_time.blockSignals(True)
        self.spin_exposure_time.setValue(actual)
        self.spin_exposure_time.blockSignals(False)

        min_exp = self.spin_exposure_time.minimum()
        max_exp = self.spin_exposure_time.maximum()
        self.update_slider_from_time(actual, min_exp, max_exp)

    def on_exposure_slider_changed(self, value):
        min_exp = self.spin_exposure_time.minimum()
        step_exp = self.camera.getExposureTimeStep()
        if step_exp > 0:
            new_time = min_exp + (value * step_exp)
        else:
            max_exp = self.spin_exposure_time.maximum()
            rng = max_exp - min_exp
            new_time = min_exp + (value / 10000.0) * rng
        self.spin_exposure_time.setValue(new_time)

    def on_gain_changed(self, value):
        if not self.is_camera_running:
            return
        self.camera.setAnalogGain(value)
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(value)
        self.slider_gain.blockSignals(False)

    def on_gain_slider_changed(self, value):
        self.spin_gain.setValue(value)

    def on_ae_target_changed(self, value):
        if not self.is_camera_running:
            return
        if hasattr(self.camera, "setAeTarget"):
            self.camera.setAeTarget(value)
            self.slider_ae_target.blockSignals(True)
            self.slider_ae_target.setValue(value)
            self.slider_ae_target.blockSignals(False)
        else:
            self.log("Error: Camera does not support setAeTarget")

    def on_ae_slider_changed(self, value):
        self.spin_ae_target.setValue(value)

    def on_trigger_mode_changed(self, id, checked):
        if checked:
            # 0=Continuous, 1=Software, 2=Hardware
            if self.camera.setTriggerMode(id):
                self.btn_soft_trigger.setEnabled(id == 1)

                # Logic for Trigger Params Group
                # Enabled if Software (1) or Hardware (2)
                self.trigger_params_group.setEnabled(id in [1, 2])

                # Logic for External Trigger Params Group
                # Enabled if Hardware (2)
                self.ext_trigger_group.setEnabled(id == 2)

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
            self.combo_strobe_polarity.setEnabled(is_manual)
            self.spin_strobe_delay.setEnabled(is_manual)
            self.spin_strobe_width.setEnabled(is_manual)

    def on_strobe_polarity_changed(self, index):
        if hasattr(self.camera, "setStrobePolarity"):
            self.camera.setStrobePolarity(index)

    def on_strobe_delay_changed(self, value):
        if hasattr(self.camera, "setStrobeDelayTime"):
            self.camera.setStrobeDelayTime(value)

    def on_strobe_width_changed(self, value):
        if hasattr(self.camera, "setStrobePulseWidth"):
            self.camera.setStrobePulseWidth(value)

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
        """
        Converts mouse position on video_label to pixel coordinates on the original image.
        Handles aspect ratio scaling.
        """
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
        self.init_mosaic_panel(force_recreate=True)

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

    @Slot()
    def on_show_mosaic_triggered(self):
        """Handles the action to show the mosaic panel by switching to the mosaic tab."""
        if not self.mosaic_panel_initialized:
            self.init_mosaic_panel()
        
        if self.mosaic_panel_initialized:
            # Switch to the Stage Mosaic tab (index 1)
            self.center_tab_widget.setCurrentIndex(1)
            self.ui.action_show_mosaic.setChecked(True)
        else:
            self.ui.action_show_mosaic.setChecked(False)

    def init_mosaic_panel(self, force_recreate=False):
        # If the panel already exists, just switch to its tab
        if self.mosaic_panel_initialized and not force_recreate:
            self.center_tab_widget.setCurrentIndex(1)
            return

        # Prerequisite checks
        if not self.ruler_calibration or self.ruler_calibration <= 0:
            self.log("Cannot create Mosaic Panel: Ruler not calibrated.")
            return
        if not self.stage_settings or "stage_width_mm" not in self.stage_settings:
            self.log("Cannot create Mosaic Panel: Stage settings not loaded.")
            return

        # Re-creation logic - clean up old panel if force recreating
        if self.mosaic_panel:
            self.mosaic_panel.setParent(None)
            self.mosaic_panel.deleteLater()
            self.mosaic_panel = None
        
        if self.scan_panel:
            self.scan_panel.setParent(None)
            self.scan_panel.deleteLater()
            self.scan_panel = None

        # --- Create container for mosaic and scan panel ---
        # Clear the existing mosaic_container
        old_layout = self.mosaic_container.layout()
        if old_layout is not None:
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            old_layout.deleteLater()

        container_layout = QVBoxLayout(self.mosaic_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Create the mosaic panel
        stage_width_mm = self.stage_settings["stage_width_mm"]
        stage_height_mm = self.stage_settings["stage_height_mm"]
        self.mosaic_panel = MosaicPanel(stage_width_mm, stage_height_mm, self.ruler_calibration)
        self.mosaic_panel.request_move_signal.connect(self.on_mosaic_move_requested)
        self.mosaic_panel.selections_changed.connect(self._on_mosaic_selections_changed)
        self.mosaic_panel.set_stage_circles(self.mosaic_circle_overlays_mm)
        self.mosaic_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mosaic_panel.setMinimumHeight(0)
        
        container_layout.addWidget(self.mosaic_panel, 1)

        # Create the scan panel
        self.scan_panel = ScanConfigPanel()
        self.scan_panel.start_scan_signal.connect(self.start_scan)
        self.scan_panel.cancel_scan_signal.connect(self.cancel_current_scan_area)
        self.scan_status_signal.connect(self.scan_panel.update_status)
        self.scan_progress_signal.connect(self.scan_panel.update_progress)
        self.scan_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        container_layout.addWidget(self.scan_panel, 0)
        container_layout.setStretch(0, 1)
        container_layout.setStretch(1, 0)
        
        # Create the scan status dialog
        self.scan_status_dialog = ScanStatusDialog(self.ui)
        self.scan_status_dialog.cancel_requested.connect(self.cancel_current_scan_area)
        self.scan_progress_signal.connect(self.scan_status_dialog.update_progress)

        # Initialize scan panel with 0 selected areas
        self.scan_panel.update_scan_areas([])

        self.mosaic_panel_initialized = True

    @Slot(list)
    def _on_mosaic_selections_changed(self, mm_rects: list):
        if self.scan_panel:
            self.scan_panel.update_scan_areas(mm_rects)
            if mm_rects:
                if self.is_scanning or not self.ruler_calibration or not self.current_pixmap:
                    return
                self.log(f"{len(mm_rects)} scan area(s) selected.")

    @Slot(float, float, float)
    def on_cnc_position_updated(self, x_mm: float, y_mm: float, z_mm: float):
        """Receives updated CNC position from the CNCControlPanel."""
        self.current_cnc_x_mm = x_mm
        self.current_cnc_y_mm = y_mm
        self.current_cnc_z_mm = z_mm
        self.mosaic_position_samples.append((time.time_ns(), x_mm, y_mm, z_mm))
        self._resolve_pending_mosaic_frames()
        self._prune_mosaic_position_samples()

        with self.recording_metadata_lock:
            self.recording_position_samples.append((time.time_ns(), x_mm, y_mm, z_mm))
            self._resolve_pending_recording_metadata_locked()
            self._prune_recording_position_samples_locked()

    def _resolve_pending_recording_metadata_locked(self, force=False):
        if not self.pending_recording_metadata:
            return

        if not self.recording_position_samples:
            if not force:
                return

            fallback_position = (
                getattr(self, "current_cnc_x_mm", 0.0),
                getattr(self, "current_cnc_y_mm", 0.0),
                getattr(self, "current_cnc_z_mm", 0.0),
            )
            while self.pending_recording_metadata:
                _, relative_timestamp_ns, frame_index = self.pending_recording_metadata.popleft()
                self.recording_metadata_rows.append(
                    (relative_timestamp_ns, frame_index, *fallback_position)
                )
            return

        latest_position_timestamp_ns = self.recording_position_samples[-1][0]
        while self.pending_recording_metadata:
            frame_timestamp_ns, relative_timestamp_ns, frame_index = self.pending_recording_metadata[0]
            if not force and frame_timestamp_ns > latest_position_timestamp_ns:
                break

            position = self._interpolate_stage_position_locked(
                frame_timestamp_ns,
                clamp_latest=force,
            )
            if position is None:
                break

            self.pending_recording_metadata.popleft()
            self.recording_metadata_rows.append(
                (relative_timestamp_ns, frame_index, *position)
            )

    def _interpolate_stage_position_locked(self, frame_timestamp_ns, clamp_latest=False):
        samples = list(self.recording_position_samples)
        if not samples:
            return None

        timestamps = [sample[0] for sample in samples]
        sample_idx = bisect_left(timestamps, frame_timestamp_ns)

        if sample_idx <= 0:
            return samples[0][1:]

        if sample_idx >= len(samples):
            if not clamp_latest:
                return None
            return samples[-1][1:]

        before_ts, before_x, before_y, before_z = samples[sample_idx - 1]
        after_ts, after_x, after_y, after_z = samples[sample_idx]
        if after_ts <= before_ts:
            return (after_x, after_y, after_z)

        ratio = (frame_timestamp_ns - before_ts) / (after_ts - before_ts)
        ratio = max(0.0, min(1.0, ratio))
        x = before_x + ((after_x - before_x) * ratio)
        y = before_y + ((after_y - before_y) * ratio)
        z = before_z + ((after_z - before_z) * ratio)
        return (x, y, z)

    def _prune_recording_position_samples_locked(self):
        if len(self.recording_position_samples) <= 2:
            return

        if self.pending_recording_metadata:
            cutoff_ns = self.pending_recording_metadata[0][0]
        else:
            cutoff_ns = self.recording_position_samples[-1][0] - self.recording_position_retention_ns

        while len(self.recording_position_samples) > 2:
            next_timestamp_ns = self.recording_position_samples[1][0]
            if next_timestamp_ns > cutoff_ns:
                break
            self.recording_position_samples.popleft()

    @Slot(str)
    def on_cnc_state_updated(self, state):
        self.cnc_state = state

    @Slot(float, float)
    def on_mosaic_move_requested(self, x, y):
        if self.cnc_control_panel:
            # Send G-code to move to absolute position (G90) using Feed Move (10)
            feedrate = self.cnc_control_panel.feedrate
            cmd = f"G90 G1 X{x:.3f} Y{y:.3f} F{feedrate}"
            self.cnc_control_panel.send_serial_cmd_signal.emit(cmd)
            self.log(f"Mosaic Click: Moving to X={x:.3f}, Y={y:.3f}")

    @Slot(float, float, float, float)
    def on_mosaic_scan_requested(self, x_min, y_min, x_max, y_max):
        if self.is_scanning:
            self.log("Scan already in progress.")
            return

        if not self.cnc_control_panel:
            self.log("CNC panel not available.")
            return
            
        if not self.ruler_calibration or self.ruler_calibration <= 0:
            self.log("Scan Error: Ruler not calibrated.")
            return
            
        if not self.current_pixmap:
            self.log("Scan Error: No camera image.")
            return
            
        if self.scan_panel:
            self.scan_panel.update_scan_areas([(x_min, y_min, x_max, y_max)])
            self.log("Scan area selected. Configure and start scan in the 'Stage Control' panel.")


    @Slot(list, bool, bool, bool)
    def start_scan(self, areas, home_x, home_y, is_serpentine):
        if not self.cnc_control_panel:
            return

        if not areas:
            self.log("No scan areas defined.")
            return

        self.scan_areas = areas
        self.scan_home_x, self.scan_home_y = home_x, home_y
        self.scan_is_serpentine = is_serpentine
        self.scan_started_recording = False

        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()
        self.scan_fov_x_mm = img_w / self.ruler_calibration
        self.scan_fov_y_mm = img_h / self.ruler_calibration
        self.scan_step_y = self.scan_fov_y_mm * 0.75
        
        self.scan_total_rows = 0
        self.area_params = []

        stage_w = self.stage_settings.get("stage_width_mm", 100)
        stage_h = self.stage_settings.get("stage_height_mm", 100)
        
        for area in areas:
            x_min, y_min, x_max, y_max = area
            x_min = max(0.0, min(x_min, stage_w))
            x_max = max(0.0, min(x_max, stage_w))
            y_min = max(0.0, min(y_min, stage_h))
            y_max = max(0.0, min(y_max, stage_h))

            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            scan_height = y_max - y_min
            
            if scan_height <= self.scan_fov_y_mm:
                total_rows = 1
                start_y = (y_max + y_min) / 2
            else:
                total_rows = int((scan_height - self.scan_fov_y_mm) / self.scan_step_y) + 1
                if (total_rows - 1) * self.scan_step_y + self.scan_fov_y_mm < scan_height:
                    total_rows += 1
                start_y = y_max - (self.scan_fov_y_mm / 2)

            self.scan_total_rows += total_rows
            self.area_params.append({
                "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                "total_rows": total_rows, "start_y": start_y
            })

        self.scan_current_row = 0
        self.is_scanning = True
        
        if hasattr(self, 'scan_status_dialog'):
            self.scan_status_dialog.show()
            overall_x_min = min(a["x_min"] for a in self.area_params)
            overall_y_min = min(a["y_min"] for a in self.area_params)
            overall_x_max = max(a["x_max"] for a in self.area_params)
            overall_y_max = max(a["y_max"] for a in self.area_params)
            scan_img = self.mosaic_panel.get_region_image(overall_x_min, overall_y_min, overall_x_max, overall_y_max)
            self.scan_status_dialog.update_image(scan_img)
            self.scan_status_dialog.update_progress(0, self.scan_total_rows)

        first_y, first_start_x, _ = self._get_scan_row_targets(self.area_params[0], 0)

        feedrate = self.cnc_control_panel.feedrate
        self.cnc_control_panel.send_serial_cmd_signal.emit("G90")
        self.cnc_control_panel.send_serial_cmd_signal.emit(f"F{feedrate}")
        if self.scan_home_y:
            self.cnc_control_panel.send_serial_cmd_signal.emit("$HY")
        if self.scan_home_x:
            self.cnc_control_panel.send_serial_cmd_signal.emit("$HX")
        self.cnc_control_panel.send_serial_cmd_signal.emit(f"G0 X{first_start_x:.3f} Y{first_y:.3f}")
        self.cnc_control_panel.send_serial_cmd_signal.emit("G4 P0.1 ; SCAN_START")
        
        self.log(f"Starting Mosaic Scan: {len(areas)} areas, {self.scan_total_rows} total rows.")
        self.scan_status_signal.emit(f"Starting scan of {self.scan_total_rows} rows.")
        self.scan_progress_signal.emit(0, self.scan_total_rows)

        self.scan_all_rows()

    def _get_scan_row_targets(self, area, row_idx):
        scan_y_min = area["y_min"]
        scan_y_max = area["y_max"]
        scan_x_min = area["x_min"]
        scan_x_max = area["x_max"]

        y_target = area["start_y"] - (row_idx * self.scan_step_y)
        if scan_y_max - scan_y_min > self.scan_fov_y_mm:
            y_target = max(y_target, scan_y_min + self.scan_fov_y_mm / 2)

        x_left = scan_x_min + (self.scan_fov_x_mm / 2)
        x_right = scan_x_max - (self.scan_fov_x_mm / 2)

        if x_left > x_right:
            x_center = (scan_x_min + scan_x_max) / 2
            x_left = x_center
            x_right = x_center

        is_ltr = True
        if self.scan_is_serpentine:
            is_ltr = (row_idx % 2 == 0)

        start_x = x_left if is_ltr else x_right
        end_x = x_right if is_ltr else x_left
        return y_target, start_x, end_x

    def scan_all_rows(self):
        if not self.is_scanning:
            return

        for area_index, area in enumerate(self.area_params):
            area_rows = area["total_rows"]
            
            is_first_strip = True

            for row_idx in range(area_rows):
                y_target, start_x, end_x = self._get_scan_row_targets(area, row_idx)
                cmds = []
                is_initial_scan_row = area_index == 0 and row_idx == 0

                if is_initial_scan_row:
                    is_first_strip = False
                else:
                    if is_first_strip:
                        cmds.append(f"G1 X{start_x:.3f} Y{y_target:.3f}")
                        is_first_strip = False
                    else:
                        cmds.append(f"G1 Y{y_target:.3f}")

                    if self.scan_home_y:
                        cmds.append("$HY")
                    if self.scan_home_x:
                        cmds.append("$HX")
                        cmds.append(f"G0 X{start_x:.3f} Y{y_target:.3f}")
                    else:
                        cmds.append(f"G1 X{start_x:.3f} Y{y_target:.3f}")

                if start_x != end_x:
                    cmds.append(f"G1 X{end_x:.3f} Y{y_target:.3f}")
                
                for cmd in cmds:
                    self.cnc_control_panel.send_serial_cmd_signal.emit(cmd)

                self.scan_current_row += 1
                self.scan_progress_signal.emit(self.scan_current_row, self.scan_total_rows)

        self.cnc_control_panel.send_serial_cmd_signal.emit("G4 P0.1 ; SCAN_DONE")

    @Slot()
    def on_scan_start_ready(self):
        if not self.is_scanning or self.scan_started_recording:
            return

        if self.video_thread.isRunning():
            self.log("Initial scan move completed.")
            return

        self.on_record_clicked()
        self.scan_started_recording = True
        self.log("Initial scan move completed, recording starting.")


    def cancel_current_scan_area(self):
        if hasattr(self, 'scan_status_dialog'):
            self.scan_status_dialog.hide()

        if self.is_scanning:
            self.is_scanning = False
            if self.cnc_control_panel:
                self.cnc_control_panel.clear_queue()
                #self.cnc_control_panel.send_raw_serial_cmd_signal.emit("!") 
            
            if getattr(self, 'scan_started_recording', False) and self.video_thread.isRunning():
                self.on_record_clicked()
                self.scan_started_recording = False
            
            self.log("Scan cancelled by user.")
            if self.scan_panel:
                self.scan_panel.scan_finished(success=False)


    def on_scan_finished(self):
        if hasattr(self, 'scan_status_dialog'):
            self.scan_status_dialog.hide()

        # This is triggered by "[ECHO:scan_finished]" from the CNC
        if self.is_scanning:
            if getattr(self, 'scan_started_recording', False) and self.video_thread.isRunning():
                self.on_record_clicked() # Stop recording
                self.scan_started_recording = False
                self.log("Mosaic scan finished, recording stopped.")
            else:
                self.log("Mosaic scan finished.")
            self.is_scanning = False

            if self.scan_panel:
                self.scan_panel.scan_finished(success=True)

    def on_home_and_run_clicked(self):
        if self.cnc_control_panel:
            feedrate = self.cnc_control_panel.feedrate
            self.cnc_control_panel.send_serial_cmd_signal.emit("$H")
            self.cnc_control_panel.send_serial_cmd_signal.emit("G90")
            self.cnc_control_panel.send_serial_cmd_signal.emit(f"G1 X0 Y0 F{feedrate}")
            self.cnc_control_panel.send_serial_cmd_signal.emit(f"G1 X100 Y0 F{feedrate}")
            self.cnc_control_panel.send_serial_cmd_signal.emit("G4 P10")
            self.log("Executing Home and Run.")

    def load_settings(self):
        settings_file = "camera_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                    
                    # Apply camera settings
                    self.spin_exposure_time.setValue(settings.get("exposure_time", 2000))
                    self.spin_gain.setValue(settings.get("gain", 1))
                    self.chk_auto_exposure.setChecked(settings.get("auto_exposure", True))
                    
                    # Apply detector params
                    if "detector" in settings:
                        self.load_detector_settings(settings["detector"])
                        
                    if "ruler_calibration" in settings:
                        self.ruler_calibration = settings["ruler_calibration"]
                        self.lbl_ruler_calib.setText(f"{self.ruler_calibration:.2f} px/mm")

                    if "led_controller_port" in settings:
                        led_port = settings.get("led_controller_port")
                        if led_port and hasattr(self, 'led_controller'):
                            self.led_controller.set_port(led_port)
                            if settings.get("led_controller_connected"):
                                QTimer.singleShot(1000, self.led_controller.ui.btn_serial_connect.toggle)

                    if "cnc_controller_port" in settings and hasattr(self, 'cnc_control_panel'):
                        cnc_port = settings.get("cnc_controller_port")
                        if cnc_port:
                            index = self.cnc_control_panel.serial_port_combo.findText(cnc_port)
                            if index != -1:
                                self.cnc_control_panel.serial_port_combo.setCurrentIndex(index)
                                if settings.get("cnc_controller_connected"):
                                    QTimer.singleShot(500, self.cnc_control_panel.connect_button.toggle)
                        
                    self.log("Settings loaded.")
            except Exception as e:
                self.log(f"Error loading settings: {e}")
        
        self._load_stage_settings()
        self.init_mosaic_panel()

    def _load_stage_settings(self):
        stage_settings_file = os.path.join(os.getcwd(), "stage_settings.json")
        if os.path.exists(stage_settings_file):
            try:
                with open(stage_settings_file, "r") as f:
                    self.stage_settings = json.load(f)
                    if self.lbl_stage_width_mm:
                        self.lbl_stage_width_mm.setText(f"{self.stage_settings.get('stage_width_mm', 'N/A'):.1f} mm")
                    if self.lbl_stage_height_mm:
                        self.lbl_stage_height_mm.setText(f"{self.stage_settings.get('stage_height_mm', 'N/A'):.1f} mm")
                    self.log("Stage settings loaded.")
            except FileNotFoundError:
                self.log(f"Stage settings file not found: {stage_settings_file}")
            except json.JSONDecodeError as e:
                self.log(f"Error decoding stage settings JSON: {e}")
            except Exception as e:
                self.log(f"Error loading stage settings: {e}")

    def save_settings(self):
        settings = {
            "exposure_time": self.spin_exposure_time.value(),
            "gain": self.spin_gain.value(),
            "auto_exposure": self.chk_auto_exposure.isChecked(),
            "detector": self.get_detector_settings(),
            "ruler_calibration": self.ruler_calibration
        }
        
        if hasattr(self, 'led_controller'):
            settings["led_controller_port"] = self.led_controller.get_port()
            settings["led_controller_connected"] = self.led_controller.ui.btn_serial_connect.isChecked()
        if hasattr(self, 'cnc_control_panel'):
            settings["cnc_controller_port"] = self.cnc_control_panel.serial_port_combo.currentText()
            settings["cnc_controller_connected"] = self.cnc_control_panel.connect_button.isChecked()
        
        settings_file = "camera_settings.json"
        try:
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=4)
                self.log("Settings saved.")
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def get_detector_settings(self):
        return {
            "orb_nfeatures": self.orb_nfeatures.value(),
            # ... capture other UI values ...
            # For brevity, saving just a few
        }

    def load_detector_settings(self, data):
        if "orb_nfeatures" in data:
            self.orb_nfeatures.setValue(data["orb_nfeatures"])
        # ... load others

    def apply_camera_settings(self):
        # Force apply current UI values to camera
        self.camera.setAutoExposure(self.chk_auto_exposure.isChecked())
        self.camera.setExposureTime(self.spin_exposure_time.value())
        self.camera.setAnalogGain(self.spin_gain.value())
