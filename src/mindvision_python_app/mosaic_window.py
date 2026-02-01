import os
from PySide6.QtWidgets import QMainWindow, QLabel, QScrollArea, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, Signal, Slot, QSize

class MosaicWindow(QMainWindow):
    """
    A window to display a mosaic of the stage area, updated with camera frames
    based on CNC position.
    """
    closed_signal = Signal()

    def __init__(self, stage_width_mm: float, stage_height_mm: float, ruler_calibration_px_per_mm: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stage Mosaic View")
        self.setMinimumSize(600, 600)

        self.stage_width_mm = stage_width_mm
        self.stage_height_mm = stage_height_mm
        self.ruler_calibration_px_per_mm = ruler_calibration_px_per_mm

        if self.ruler_calibration_px_per_mm <= 0:
            print("Error: Invalid ruler calibration for mosaic window. Calibration must be > 0.")
            self.mosaic_image = QImage(100, 100, QImage.Format_RGB32)
            self.mosaic_image.fill(QColor("red"))
            self.label = QLabel("Error: Invalid Calibration", self)
            self.setCentralWidget(self.label)
            return

        self.mosaic_width_px = int(self.stage_width_mm * self.ruler_calibration_px_per_mm)
        self.mosaic_height_px = int(self.stage_height_mm * self.ruler_calibration_px_per_mm)

        if self.mosaic_width_px <= 0 or self.mosaic_height_px <= 0:
            print(f"Error: Calculated mosaic dimensions are invalid: {self.mosaic_width_px}x{self.mosaic_height_px}")
            self.mosaic_image = QImage(100, 100, QImage.Format_RGB32)
            self.mosaic_image.fill(QColor("red"))
            self.label = QLabel("Error: Invalid Mosaic Dimensions", self)
            self.setCentralWidget(self.label)
            return

        self.mosaic_image = QImage(self.mosaic_width_px, self.mosaic_height_px, QImage.Format_RGB32)
        self.mosaic_image.fill(QColor("darkgray")) # Initial background for the mosaic

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(self.mosaic_image))
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft) # Align top-left for mosaic

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.label)

        self.setCentralWidget(self.scroll_area)

        # Store camera frame dimensions to calculate offset
        self.camera_frame_width_px = 0
        self.camera_frame_height_px = 0

    def closeEvent(self, event):
        """Emits a signal when the window is closed."""
        self.closed_signal.emit()
        super().closeEvent(event)

    @Slot(QImage, float, float)
    def update_mosaic(self, camera_frame: QImage, cnc_x_mm: float, cnc_y_mm: float):
        """
        Updates a portion of the mosaic image with the new camera frame.
        Assumes CNC (0,0) is bottom-left, Y-up. Mosaic (0,0) is top-left, Y-down.
        """
        if self.ruler_calibration_px_per_mm <= 0 or camera_frame.isNull():
            return

        # Update camera frame dimensions if they change (e.g., ROI changes)
        if self.camera_frame_width_px != camera_frame.width() or \
           self.camera_frame_height_px != camera_frame.height():
            self.camera_frame_width_px = camera_frame.width()
            self.camera_frame_height_px = camera_frame.height()

        # Camera's physical width/height in mm
        camera_fov_width_mm = self.camera_frame_width_px / self.ruler_calibration_px_per_mm
        camera_fov_height_mm = self.camera_frame_height_px / self.ruler_calibration_px_per_mm

        # Calculate the top-left corner of the camera's view on the stage in CNC (mm) coordinates
        # Assuming cnc_x_mm, cnc_y_mm is the center of the camera's view
        camera_top_left_x_mm = cnc_x_mm - (camera_fov_width_mm / 2)
        camera_top_left_y_mm = cnc_y_mm + (camera_fov_height_mm / 2) # Y increases upwards in CNC, so top edge is higher Y

        # Convert top-left mm coordinates to mosaic pixel coordinates (mosaic Y is inverted)
        mosaic_draw_x_px = int(camera_top_left_x_mm * self.ruler_calibration_px_per_mm)
        mosaic_draw_y_px = int((self.stage_height_mm - camera_top_left_y_mm) * self.ruler_calibration_px_per_mm)

        painter = QPainter(self.mosaic_image)
        painter.drawImage(mosaic_draw_x_px, mosaic_draw_y_px, camera_frame.convertToFormat(QImage.Format_RGB32))
        painter.end()

        self.label.setPixmap(QPixmap.fromImage(self.mosaic_image))
        # Optional: Scroll to the updated region
        # self.scroll_area.ensureVisible(mosaic_draw_x_px, mosaic_draw_y_px, camera_frame.width(), camera_frame.height())