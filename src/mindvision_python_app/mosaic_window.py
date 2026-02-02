import os
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QTransform
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRect

class MosaicWidget(QWidget):
    """
    A custom widget to display the mosaic image scaled to fit the window.
    """
    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

    def set_image(self, image: QImage):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if not self.image or self.image.isNull():
            # Draw black background if no image
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("black"))
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor("black"))
        
        # Calculate aspect ratio scaling to fit
        widget_rect = self.rect()
        img_size = self.image.size()
        
        # Use QPainter's ability to draw image into a target rect, but we want to maintain aspect ratio
        # QImage.scaled() is one way, but calculating rect is faster for painting
        
        scale_w = widget_rect.width() / img_size.width()
        scale_h = widget_rect.height() / img_size.height()
        scale = min(scale_w, scale_h)
        
        drawn_w = int(img_size.width() * scale)
        drawn_h = int(img_size.height() * scale)
        
        x = (widget_rect.width() - drawn_w) // 2
        y = (widget_rect.height() - drawn_h) // 2
        
        target_rect = QRect(x, y, drawn_w, drawn_h)
        painter.drawImage(target_rect, self.image)

    def mousePressEvent(self, event):
        if not self.image or self.image.isNull():
            return
        
        if event.button() != Qt.LeftButton:
            return

        widget_rect = self.rect()
        img_size = self.image.size()
        
        if img_size.width() == 0 or img_size.height() == 0:
            return
        
        scale_w = widget_rect.width() / img_size.width()
        scale_h = widget_rect.height() / img_size.height()
        scale = min(scale_w, scale_h)
        
        drawn_w = int(img_size.width() * scale)
        drawn_h = int(img_size.height() * scale)
        
        x_off = (widget_rect.width() - drawn_w) // 2
        y_off = (widget_rect.height() - drawn_h) // 2
        
        pos = event.position()
        mx = pos.x()
        my = pos.y()
        
        if mx >= x_off and mx < x_off + drawn_w and my >= y_off and my < y_off + drawn_h:
            img_x = (mx - x_off) / scale
            img_y = (my - y_off) / scale
            self.clicked.emit(img_x, img_y)

class MosaicWindow(QMainWindow):
    """
    A window to display a mosaic of the stage area, updated with camera frames
    based on CNC position.
    """
    closed_signal = Signal()
    request_move_signal = Signal(float, float)

    def __init__(self, stage_width_mm: float, stage_height_mm: float, ruler_calibration_px_per_mm: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stage Mosaic View")
        # Set window to be maximized by default
        self.setWindowState(Qt.WindowMaximized)

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

        self.display_widget = MosaicWidget(self)
        self.display_widget.set_image(self.mosaic_image)
        self.display_widget.clicked.connect(self.on_mosaic_clicked)
        self.setCentralWidget(self.display_widget)

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

        # Rotate the image 90 degrees to correct for camera orientation
        transform = QTransform()
        transform.rotate(90)
        camera_frame = camera_frame.transformed(transform)
        camera_frame = camera_frame.mirrored(False, True)

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

        self.display_widget.set_image(self.mosaic_image)

    @Slot(float, float)
    def on_mosaic_clicked(self, img_x, img_y):
        if self.ruler_calibration_px_per_mm <= 0:
            return
        
        # Mosaic (0,0) is Top-Left. CNC (0,0) is Bottom-Left.
        cnc_x = img_x / self.ruler_calibration_px_per_mm
        cnc_y = self.stage_height_mm - (img_y / self.ruler_calibration_px_per_mm)
        
        # Clamp to stage bounds
        cnc_x = max(0.0, min(cnc_x, self.stage_width_mm))
        cnc_y = max(0.0, min(cnc_y, self.stage_height_mm))
        
        self.request_move_signal.emit(cnc_x, cnc_y)