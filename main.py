import sys
import os
import platform
import time
import signal

# Add the directory containing the generated module to sys.path
# Assuming the module is built into a 'python_module' directory relative to the project root
script_dir = os.path.dirname(__file__)
release_dir = os.path.join(r"z:\src\mindvision_qobject", "release")
sys.path.insert(0, release_dir)

os.add_dll_directory(release_dir)
# Add Qt bin directory
os.add_dll_directory(r"C:\Qt\6.10.1\msvc2022_64\bin")
# Add MindVision SDK directory
os.add_dll_directory(r"C:\Program Files (x86)\MindVision\SDK\X64")

import _mindvision_qobject_py
import PySide6.QtWidgets

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                               QCheckBox, QDoubleSpinBox, QSlider, QSpinBox, QFormLayout, QSizePolicy,
                               QRadioButton, QButtonGroup)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

try:
    from _mindvision_qobject_py import MindVisionCamera, VideoThread
except ImportError as e:
    print(f"Failed to import _mindvision_qobject_py: {e}")
    sys.exit(1)

class MainWindow(QMainWindow):
    update_frame_signal = Signal(QImage)
    update_fps_signal = Signal(float)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MindVision Camera Viewer (Python - PySide6)")
        self.resize(800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Video Label
        self.video_label = QLabel("Camera Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color : black; color : white; }")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Expanding
        self.main_layout.addWidget(self.video_label)

        # Controls Group
        self.controls_group = QGroupBox("Camera Settings")
        self.controls_layout = QFormLayout(self.controls_group)

        self.chk_auto_exposure = QCheckBox("Auto Exposure")
        self.chk_auto_exposure.setChecked(True)
        self.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)

        self.chk_roi = QCheckBox("High FPS ROI (640x480)")
        self.chk_roi.setChecked(False)
        self.chk_roi.toggled.connect(self.on_roi_toggled)

        self.spin_exposure_time = QDoubleSpinBox()
        self.spin_exposure_time.setSuffix(" ms")
        self.spin_exposure_time.setDecimals(4)
        self.spin_exposure_time.setRange(0.1, 1000.0)
        self.spin_exposure_time.setEnabled(False)
        self.spin_exposure_time.valueChanged.connect(self.on_exposure_time_changed)

        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setRange(0, 10000)
        self.slider_exposure.setEnabled(False)
        self.slider_exposure.valueChanged.connect(self.on_exposure_slider_changed)

        self.spin_gain = QSpinBox()
        self.spin_gain.setRange(0, 100)
        self.spin_gain.valueChanged.connect(self.on_gain_changed)

        self.controls_layout.addRow(self.chk_auto_exposure)
        self.controls_layout.addRow(self.chk_roi)
        self.controls_layout.addRow("Exposure Time:", self.spin_exposure_time)
        self.controls_layout.addRow("Exposure Slider:", self.slider_exposure)
        self.controls_layout.addRow("Analog Gain:", self.spin_gain)
        
        self.controls_group.setEnabled(False)
        self.main_layout.addWidget(self.controls_group)

        # Trigger Group
        self.trigger_group = QGroupBox("Trigger Settings")
        self.trigger_layout = QVBoxLayout(self.trigger_group)
        
        self.rb_continuous = QRadioButton("Continuous")
        self.rb_software = QRadioButton("Software Trigger")
        self.rb_hardware = QRadioButton("Hardware Trigger")
        self.rb_continuous.setChecked(True)
        
        self.btn_soft_trigger = QPushButton("Trigger")
        self.btn_soft_trigger.setEnabled(False)
        
        self.trigger_bg = QButtonGroup(self)
        self.trigger_bg.addButton(self.rb_continuous, 0)
        self.trigger_bg.addButton(self.rb_software, 1)
        self.trigger_bg.addButton(self.rb_hardware, 2)
        
        self.trigger_bg.idToggled.connect(self.on_trigger_mode_changed)
        self.btn_soft_trigger.clicked.connect(self.on_soft_trigger_clicked)
        
        self.trigger_layout.addWidget(self.rb_continuous)
        self.trigger_layout.addWidget(self.rb_software)
        self.trigger_layout.addWidget(self.rb_hardware)
        self.trigger_layout.addWidget(self.btn_soft_trigger)
        
        self.trigger_group.setEnabled(False)
        self.main_layout.addWidget(self.trigger_group)

        # Buttons
        self.btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.on_record_clicked)
        self.record_btn.setEnabled(False)

        self.btn_layout.addWidget(self.start_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.btn_layout.addWidget(self.record_btn)
        self.main_layout.addLayout(self.btn_layout)

        # Status Bar
        self.fps_label = QLabel("FPS: 0.0")
        self.statusBar().addPermanentWidget(self.fps_label)

        # Camera Setup
        self.camera = MindVisionCamera()
        
        # Register Callbacks
        self.camera.registerFrameCallback(self.frame_callback)
        self.camera.registerFpsCallback(self.fps_callback)
        self.camera.registerErrorCallback(self.error_callback)

        # Signals
        self.update_frame_signal.connect(self.update_frame)
        self.update_fps_signal.connect(self.update_fps)
        self.error_signal.connect(self.handle_error)

        self.video_thread = VideoThread()
        self.recording_requested = False
        self.current_fps = 30.0
        self.last_ui_update_time = 0

    def frame_callback(self, width, height, bytes_per_line, fmt, data):
        # 1. Recording (High Priority, no copies if possible)
        # Note: 'data' is valid only during this callback.
        # VideoThread.addFrameBytes must copy the data internally.
        if self.video_thread.isRunning():
            try:
                self.video_thread.addFrameBytes(width, height, bytes_per_line, fmt, data)
            except Exception as e:
                print(f"Recording error in callback: {e}")

        # 2. UI Update (Throttled)
        # We don't need to display 100+ FPS. 30 FPS is enough for preview.
        current_time = time.time()
        if current_time - self.last_ui_update_time > 0.033: # ~30 FPS
            self.last_ui_update_time = current_time
            try:
                # Create QImage from data (deep copy for GUI thread)
                img = QImage(data, width, height, bytes_per_line, QImage.Format(fmt))
                self.update_frame_signal.emit(img.copy())
            except Exception as e:
                print(f"Error in frame callback (UI): {e}")

    def fps_callback(self, fps):
        self.update_fps_signal.emit(fps)

    def error_callback(self, msg):
        self.error_signal.emit(msg)

    def closeEvent(self, event):
        self.on_stop_clicked()
        super().closeEvent(event)

    @Slot(QImage)
    def update_frame(self, image):
        if not image.isNull():
            # Recording Start Trigger
            if self.recording_requested:
                self.recording_requested = False
                record_fps = self.current_fps if self.current_fps > 0.1 else 30.0
                self.video_thread.startRecording(image.width(), image.height(), record_fps, "output.mkv")
                self.record_btn.setText("Stop Recording")
            
            # Display
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(float)
    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def handle_error(self, message):
        print(f"Camera Error: {message}")
        self.video_label.setText(f"Error: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.controls_group.setEnabled(False)
        self.trigger_group.setEnabled(False)

    def on_start_clicked(self):
        if self.camera.open():
            if self.camera.start():
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.record_btn.setEnabled(True)
                self.controls_group.setEnabled(True)
                self.trigger_group.setEnabled(True)
                self.video_label.setText("Starting stream...")
                self.sync_ui()

    def on_stop_clicked(self):
        if self.video_thread.isRunning():
            self.on_record_clicked()
        
        self.camera.stop()
        self.camera.close()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.controls_group.setEnabled(False)
        self.trigger_group.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("Camera Stopped")
        self.fps_label.setText("FPS: 0.0")

    def on_record_clicked(self):
        if not self.video_thread.isRunning():
            self.recording_requested = True
            print("Recording requested...")
        else:
            self.video_thread.stopRecording()
            self.record_btn.setText("Start Recording")
            print("Recording stopped.")

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
        
        # Values
        is_auto = self.camera.getAutoExposure()
        self.chk_auto_exposure.setChecked(is_auto)
        self.spin_exposure_time.setEnabled(not is_auto)
        self.slider_exposure.setEnabled(not is_auto)
        
        current_exp = self.camera.getExposureTime()
        self.spin_exposure_time.blockSignals(True)
        self.spin_exposure_time.setValue(current_exp)
        self.spin_exposure_time.blockSignals(False)
        
        self.update_slider_from_time(current_exp, min_exp, max_exp)
        
        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(self.camera.getAnalogGain())
        self.spin_gain.blockSignals(False)

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
        if self.camera.setAutoExposure(checked):
            self.spin_exposure_time.setEnabled(not checked)
            self.slider_exposure.setEnabled(not checked)
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
        self.camera.setAnalogGain(value)

    def on_trigger_mode_changed(self, id, checked):
        if checked:
            # 0=Continuous, 1=Software, 2=Hardware
            if self.camera.setTriggerMode(id):
                self.btn_soft_trigger.setEnabled(id == 1)
            else:
                print(f"Failed to set trigger mode {id}")
                # Revert logic could be added here

    def on_soft_trigger_clicked(self):
        self.camera.triggerSoftware()

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
