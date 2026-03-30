import sys
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
    QWidget, QMessageBox, QFileDialog, QStatusBar
)

from .bindings import MindVisionCamera, VideoThread

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class CameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize camera and video thread
        self.camera = MindVisionCamera()
        self.video_thread = VideoThread()
        
        # Recording state
        self.is_recording = False
        self.current_fps = 0.0
        self.last_frame_size = None
        self.last_queue_size = 0
        self.last_dropped_frames = 0
        self.record_file_dialog = None
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.rendered_frame_seq = 0
        
        # Frame counter for status updates
        self.frame_count = 0
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("MindVision Camera - Video Recorder")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Camera display label
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("QLabel { background-color: black; color: white; }")
        layout.addWidget(self.camera_label)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.open_button = QPushButton("Open Camera")
        self.open_button.clicked.connect(self.open_camera)
        button_layout.addWidget(self.open_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)
        
        self.close_button = QPushButton("Close Camera")
        self.close_button.clicked.connect(self.close_camera)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Camera closed. Click 'Open Camera' to start.")
        
        # FPS update timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start(100)  # Update every 100ms

        # Render preview at a fixed rate to keep UI load stable.
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_latest_frame)
        self.render_timer.start(33)  # ~30 FPS UI rendering
        
    def open_camera(self):
        """Open the camera and start capturing frames."""
        if self.camera.open():
            self.camera.registerFrameViewCallback(self.on_frame)
            self.camera.registerFpsCallback(self.on_fps_change)
            self.camera.registerQueueStatsCallback(self.on_queue_stats)
            self.video_thread.setFrameSource(self.camera)
            
            if self.camera.start():
                self.open_button.setEnabled(False)
                self.close_button.setEnabled(True)
                self.record_button.setEnabled(True)
                self.status_bar.showMessage("Camera opened and capturing frames.")
            else:
                QMessageBox.critical(self, "Error", "Failed to start camera capture.")
                self.camera.close()
        else:
            QMessageBox.critical(self, "Error", "Failed to open camera. Check if camera is connected.")
            
    def close_camera(self):
        """Close the camera and stop capturing."""
        if self.is_recording:
            self.toggle_recording()
            
        self.camera.stop()
        self.camera.close()
        self.video_thread.clearFrameSource()
        
        self.camera_label.clear()
        self.camera_label.setText("Camera feed will appear here")
        self.camera_label.setStyleSheet("QLabel { background-color: black; color: white; }")
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.rendered_frame_seq = 0
        
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status_bar.showMessage("Camera closed.")
        
    def on_frame(self, width, height, bytes_per_line, format, data, timestamp_ms):
        """Callback for new camera frames."""
        try:
            image_format = QImage.Format(format)
        except ValueError:
            return

        self.last_frame_size = (width, height)
        if self.camera_label.width() != width or self.camera_label.height() != height:
            self.camera_label.setFixedSize(width, height)
        self.frame_count += 1

        # Keep only the latest frame; rendering is performed on a fixed UI timer.
        image = QImage(data, width, height, bytes_per_line, image_format).copy()
        if not image.isNull():
            self.latest_frame = image
            self.latest_frame_seq += 1

    def render_latest_frame(self):
        """Render the latest frame at a fixed cadence to avoid UI spikes."""
        if self.latest_frame is None:
            return

        if self.rendered_frame_seq == self.latest_frame_seq:
            return

        pixmap = QPixmap.fromImage(self.latest_frame)
        self.camera_label.setPixmap(pixmap)
        self.rendered_frame_seq = self.latest_frame_seq
            
    def on_fps_change(self, fps):
        """Callback for FPS updates."""
        self.current_fps = fps

    def on_queue_stats(self, queue_size, dropped_frames):
        """Callback for queue size and dropped frame stats."""
        self.last_queue_size = queue_size
        self.last_dropped_frames = dropped_frames
        
    def update_fps_display(self):
        """Update the FPS display in the status bar."""
        status_text = f"FPS: {self.current_fps:.1f} | Frames: {self.frame_count}"
        status_text += f" | Queue: {self.last_queue_size} | Dropped: {self.last_dropped_frames}"
        if self.is_recording:
            status_text += " | RECORDING"
        self.status_bar.showMessage(status_text)
        
    def toggle_recording(self):
        """Start or stop video recording."""
        if not self.is_recording:
            if self.last_frame_size is None:
                QMessageBox.warning(self, "No Frame", "Wait for the first camera frame before starting recording.")
                return

            if self.record_file_dialog is not None and self.record_file_dialog.isVisible():
                self.record_file_dialog.raise_()
                self.record_file_dialog.activateWindow()
                return

            self.record_file_dialog = QFileDialog(self)
            self.record_file_dialog.setFileMode(QFileDialog.AnyFile)
            self.record_file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            self.record_file_dialog.setNameFilter("Matroska video files (*.mkv);;All files (*)")

            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"recording_{timestamp}.mkv"
            self.record_file_dialog.selectFile(default_filename)

            self.record_file_dialog.fileSelected.connect(self.on_record_file_selected)
            self.record_file_dialog.rejected.connect(self.on_record_file_cancelled)
            self.record_file_dialog.open()
        else:
            # Stop recording
            self.video_thread.stopRecording()
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.status_bar.showMessage("Recording stopped.")

    def on_record_file_selected(self, filename):
        """Handle async recording filename selection without blocking UI updates."""
        if not filename.lower().endswith('.mkv'):
            filename += '.mkv'

        if self.last_frame_size is None:
            QMessageBox.warning(self, "No Frame", "Wait for the first camera frame before starting recording.")
            return

        width, height = self.last_frame_size
        fps = self.current_fps if self.current_fps > 0 else 30.0

        self.video_thread.startRecording(width, height, fps, filename)
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.status_bar.showMessage(f"Recording started: {filename}")

    def on_record_file_cancelled(self):
        """Handle async recording dialog cancellation."""
        self.status_bar.showMessage("Recording cancelled.")
            
    def closeEvent(self, event):
        """Handle window close event."""
        self.close_camera()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("MindVision Camera Recorder")
    app.setOrganizationName("MicroTools")
    
    # Create and show the main window
    window = CameraGUI()
    window.show()

    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
