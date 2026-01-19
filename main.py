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
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QFile, QObject, QEvent
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

        # Install event filter to handle close event
        self.ui.installEventFilter(self)
        self.ui.video_label.installEventFilter(self)

        self.current_pixmap = None

        # Status Bar (add permanent widget manually)
        self.fps_label = QLabel("FPS: 0.0")
        self.ui.statusBar().addPermanentWidget(self.fps_label)

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
        self.ui.showMaximized()

    def close(self):
        # Triggers closeEvent via the widget
        self.ui.close()

    def eventFilter(self, watched, event):
        try:
            if watched is self.ui and event.type() == QEvent.Close:
                self.on_stop_clicked()
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
                self.video_thread.startRecording(image.width(), image.height(), record_fps, "c:\\Users\\davidek\\Desktop\\output.mkv")
                self.ui.record_btn.setText("Stop Recording")
            
            # Display
            self.current_pixmap = QPixmap.fromImage(image)
            self.refresh_video_label()

    @Slot(float)
    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def handle_error(self, message):
        print(f"Camera Error: {message}")
        self.ui.video_label.setText(f"Error: {message}")
        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)
        self.ui.controls_group.setEnabled(False)
        self.ui.trigger_group.setEnabled(False)
        self.ui.trigger_params_group.setEnabled(False)
        self.ui.ext_trigger_group.setEnabled(False)
        self.ui.strobe_group.setEnabled(False)

    def on_start_clicked(self):
        if self.camera.open():
            if self.camera.start():
                self.ui.start_btn.setEnabled(False)
                self.ui.stop_btn.setEnabled(True)
                self.ui.record_btn.setEnabled(True)
                self.ui.controls_group.setEnabled(True)
                self.ui.trigger_group.setEnabled(True)
                self.ui.strobe_group.setEnabled(True)
                
                self.ui.video_label.setText("Starting stream...")
                self.sync_ui()

    def on_stop_clicked(self):
        if self.video_thread.isRunning():
            self.on_record_clicked()
        
        self.camera.stop()
        self.camera.close()
        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)
        self.ui.record_btn.setEnabled(False)
        
        self.ui.controls_group.setEnabled(False)
        self.ui.trigger_group.setEnabled(False)
        self.ui.trigger_params_group.setEnabled(False)
        self.ui.ext_trigger_group.setEnabled(False)
        self.ui.strobe_group.setEnabled(False)
        
        self.ui.video_label.clear()
        self.ui.video_label.setText("Camera Stopped")
        self.fps_label.setText("FPS: 0.0")

    def on_record_clicked(self):
        if not self.video_thread.isRunning():
            self.recording_requested = True
            print("Recording requested...")
        else:
            self.video_thread.stopRecording()
            self.ui.record_btn.setText("Start Recording")
            print("Recording stopped.")

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
                print(f"Error syncing AE target: {e}")
        
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
                print(f"Failed to set trigger mode {id}")

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
