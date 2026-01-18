import sys
import os
import platform
import time
import signal

# Add the directory containing the generated module to sys.path
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

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QButtonGroup)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QFile, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader

try:
    from _mindvision_qobject_py import MindVisionCamera, VideoThread
except ImportError as e:
    print(f"Failed to import _mindvision_qobject_py: {e}")
    sys.exit(1)

class MainWindow(QObject):
    update_frame_signal = Signal(QImage)
    update_fps_signal = Signal(float)
    error_signal = Signal(str)

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

        # Access Widgets
        self.video_label = self.ui.video_label
        self.controls_group = self.ui.controls_group
        self.chk_auto_exposure = self.ui.chk_auto_exposure
        self.chk_roi = self.ui.chk_roi
        self.spin_exposure_time = self.ui.spin_exposure_time
        self.slider_exposure = self.ui.slider_exposure
        self.spin_gain = self.ui.spin_gain
        self.trigger_group = self.ui.trigger_group
        self.rb_continuous = self.ui.rb_continuous
        self.rb_software = self.ui.rb_software
        self.rb_hardware = self.ui.rb_hardware
        self.btn_soft_trigger = self.ui.btn_soft_trigger
        self.start_btn = self.ui.start_btn
        self.stop_btn = self.ui.stop_btn
        self.record_btn = self.ui.record_btn
        
        # Status Bar (add permanent widget manually)
        self.fps_label = QLabel("FPS: 0.0")
        self.ui.statusBar().addPermanentWidget(self.fps_label)

        # Connections
        self.chk_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        self.chk_roi.toggled.connect(self.on_roi_toggled)
        self.spin_exposure_time.valueChanged.connect(self.on_exposure_time_changed)
        self.slider_exposure.valueChanged.connect(self.on_exposure_slider_changed)
        self.spin_gain.valueChanged.connect(self.on_gain_changed)
        
        # Recreate ButtonGroup for logic
        self.trigger_bg = QButtonGroup(self.ui)
        self.trigger_bg.addButton(self.rb_continuous, 0)
        self.trigger_bg.addButton(self.rb_software, 1)
        self.trigger_bg.addButton(self.rb_hardware, 2)
        self.trigger_bg.idToggled.connect(self.on_trigger_mode_changed)
        
        self.btn_soft_trigger.clicked.connect(self.on_soft_trigger_clicked)
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.record_btn.clicked.connect(self.on_record_clicked)

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

    def show(self):
        self.ui.show()

    def close(self):
        self.on_stop_clicked()
        self.ui.close()

    def frame_callback(self, width, height, bytes_per_line, fmt, data):
        # 1. Recording (High Priority)
        if self.video_thread.isRunning():
            try:
                self.video_thread.addFrameBytes(width, height, bytes_per_line, fmt, data)
            except Exception as e:
                print(f"Recording error in callback: {e}")

        # 2. UI Update (Throttled)
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