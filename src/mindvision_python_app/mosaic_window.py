import os
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QTransform, QPen
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRect, QPoint

class MosaicWidget(QWidget):
    """
    A custom widget to display the mosaic image scaled to fit the window.
    """
    clicked = Signal(float, float)
    selection_made = Signal(float, float, float, float) # x, y, w, h in image pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.start_pos = None
        self.current_pos = None
        self.is_dragging = False

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
        painter.fillRect(self.rect(), Qt.white)
        
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
        
        x = 0
        y = (widget_rect.height() - drawn_h) // 2
        
        target_rect = QRect(x, y, drawn_w, drawn_h)
        painter.drawImage(target_rect, self.image)
        
        # Draw selection rectangle if dragging
        if self.is_dragging and self.start_pos and self.current_pos:
            pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            
            rect = QRect(self.start_pos.toPoint(), self.current_pos.toPoint()).normalized()
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if not self.image or self.image.isNull():
            return
        
        if event.button() == Qt.LeftButton:
            self.start_pos = event.position()
            self.current_pos = event.position()
            self.is_dragging = True

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.current_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            end_pos = event.position()
            
            # Calculate distance to distinguish click from drag
            dist = (end_pos - self.start_pos).manhattanLength()

            widget_rect = self.rect()
            img_size = self.image.size()
            
            if img_size.width() == 0 or img_size.height() == 0:
                return
            
            scale_w = widget_rect.width() / img_size.width()
            scale_h = widget_rect.height() / img_size.height()
            scale = min(scale_w, scale_h)
            
            drawn_w = int(img_size.width() * scale)
            drawn_h = int(img_size.height() * scale)
            
            x_off = 0
            y_off = (widget_rect.height() - drawn_h) // 2
            
            if dist < 5:
                # Treat as Click
                pos = end_pos
                mx = pos.x()
                my = pos.y()
                
                if mx >= x_off and mx < x_off + drawn_w and my >= y_off and my < y_off + drawn_h:
                    img_x = (mx - x_off) / scale
                    img_y = (my - y_off) / scale
                    self.clicked.emit(img_x, img_y)
            else:
                # Treat as Selection Drag
                # Convert start and end to image coords
                p1 = self.start_pos
                p2 = end_pos
                
                # Normalize rect in widget coords
                x1 = min(p1.x(), p2.x())
                y1 = min(p1.y(), p2.y())
                x2 = max(p1.x(), p2.x())
                y2 = max(p1.y(), p2.y())
                
                # Clip to drawn image area
                x1 = max(x1, x_off)
                y1 = max(y1, y_off)
                x2 = min(x2, x_off + drawn_w)
                y2 = min(y2, y_off + drawn_h)
                
                if x2 > x1 and y2 > y1:
                    img_x = (x1 - x_off) / scale
                    img_y = (y1 - y_off) / scale
                    img_w = (x2 - x1) / scale
                    img_h = (y2 - y1) / scale
                    self.selection_made.emit(img_x, img_y, img_w, img_h)
            
            self.start_pos = None
            self.current_pos = None
            self.update()

    # Removed old mousePressEvent to avoid conflict, logic moved to mouseReleaseEvent/mousePressEvent above

class MosaicPanel(QWidget):
    """
    A panel to display a mosaic of the stage area, updated with camera frames
    based on CNC position.
    """
    # closed_signal = Signal() # No longer needed for a panel
    request_move_signal = Signal(float, float)
    request_scan_signal = Signal(float, float, float, float) # x_min, y_min, x_max, y_max

    def __init__(self, stage_width_mm: float, stage_height_mm: float, ruler_calibration_px_per_mm: float, parent=None):
        super().__init__(parent)

        self.stage_width_mm = stage_width_mm
        self.stage_height_mm = stage_height_mm
        self.ruler_calibration_px_per_mm = ruler_calibration_px_per_mm

        if self.ruler_calibration_px_per_mm <= 0:
            print("Error: Invalid ruler calibration for mosaic window. Calibration must be > 0.")
            # Handle error gracefully in UI
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Error: Invalid Calibration"))
            return

        self.mosaic_width_px = int(self.stage_width_mm * self.ruler_calibration_px_per_mm)
        self.mosaic_height_px = int(self.stage_height_mm * self.ruler_calibration_px_per_mm)

        if self.mosaic_width_px <= 0 or self.mosaic_height_px <= 0:
            print(f"Error: Calculated mosaic dimensions are invalid: {self.mosaic_width_px}x{self.mosaic_height_px}")
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Error: Invalid Mosaic Dimensions"))
            return

        self.mosaic_image = QImage(self.mosaic_width_px, self.mosaic_height_px, QImage.Format_RGB32)
        self.mosaic_image.fill(Qt.white) # Initial background for the mosaic

        self.display_widget = MosaicWidget(self)
        self.display_widget.set_image(self.mosaic_image)
        self.display_widget.clicked.connect(self.on_mosaic_clicked)
        self.display_widget.selection_made.connect(self.on_mosaic_selection)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.display_widget)

        # Store camera frame dimensions to calculate offset
        self.camera_frame_width_px = 0
        self.camera_frame_height_px = 0

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

    @Slot(float, float, float, float)
    def on_mosaic_selection(self, x, y, w, h):
        if self.ruler_calibration_px_per_mm <= 0:
            return
            
        # Convert to CNC coords
        # Image (0,0) is Top-Left. CNC (0,0) is Bottom-Left.
        
        # Top-Left of selection in CNC
        cnc_x1 = x / self.ruler_calibration_px_per_mm
        cnc_y1 = self.stage_height_mm - (y / self.ruler_calibration_px_per_mm)
        
        # Bottom-Right of selection in CNC
        cnc_x2 = (x + w) / self.ruler_calibration_px_per_mm
        cnc_y2 = self.stage_height_mm - ((y + h) / self.ruler_calibration_px_per_mm)
        
        x_min = min(cnc_x1, cnc_x2)
        x_max = max(cnc_x1, cnc_x2)
        y_min = min(cnc_y1, cnc_y2)
        y_max = max(cnc_y1, cnc_y2)
        
        self.request_scan_signal.emit(x_min, y_min, x_max, y_max)