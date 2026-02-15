import os
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QTransform, QPen
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRect, QPoint, QPointF, QRectF


class MosaicWidget(QWidget):
    """
    A custom widget to display a pan-and-zoomable mosaic image.
    """
    clicked = Signal(float, float)
    selection_made = Signal(float, float, float, float) # x, y, w, h in image pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.start_pos = None
        self.current_pos = None
        
        self.grid_width = 0
        self.grid_height = 0

        # Pan and Zoom state
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.last_mouse_pos = QPoint()

        # Dragging states
        self.is_panning = False
        self.is_zooming = False
        self.is_selecting = False

        self.stage_width_mm = 0
        self.stage_height_mm = 0
        self.ruler_calibration_px_per_mm = 0

    def set_stage_size(self, width_mm: float, height_mm: float):
        self.stage_width_mm = width_mm
        self.stage_height_mm = height_mm

    def set_calibration(self, calibration: float):
        self.ruler_calibration_px_per_mm = calibration

    def set_grid_size(self, width: int, height: int):
        self.grid_width = width
        self.grid_height = height
        self.update()

    def set_image(self, image: QImage):
        if self.image is None and not image.isNull():
            # First image, fit to view
            self.fit_to_window()
        self.image = image
        self.update()

    def widget_to_image_coords(self, widget_pos: QPoint) -> QPointF:
        """Converts widget coordinates to image coordinates."""
        if self.image is None:
            return QPointF()
        
        # Reverse the transformation: (pos - pan) / zoom
        return (QPointF(widget_pos) - self.pan_offset) / self.zoom_factor

    def image_to_widget_coords(self, image_pos: QPointF) -> QPointF:
        """Converts image coordinates to widget coordinates."""
        if self.image is None:
            return QPointF()
        
        # Apply the transformation: (pos * zoom) + pan
        return (image_pos * self.zoom_factor) + self.pan_offset

    def fit_to_window(self):
        if not self.image or self.image.isNull():
            return
        
        widget_rect = self.rect()
        img_size = self.image.size()

        if img_size.width() == 0 or img_size.height() == 0:
            return

        scale_w = widget_rect.width() / img_size.width()
        scale_h = widget_rect.height() / img_size.height()
        self.zoom_factor = min(scale_w, scale_h)

        # Center the image
        drawn_w = img_size.width() * self.zoom_factor
        drawn_h = img_size.height() * self.zoom_factor
        
        self.pan_offset = QPointF(
            (widget_rect.width() - drawn_w) / 2,
            (widget_rect.height() - drawn_h) / 2
        )
        self.update()

    def wheelEvent(self, event):
        if not self.image or self.image.isNull():
            return
            
        # Zoom Factor
        delta = event.angleDelta().y()
        zoom_direction = 1 if delta > 0 else -1
        zoom_increment = 0.1 * zoom_direction
        
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor + zoom_increment))

        # Zoom towards the mouse cursor
        mouse_pos = event.position()
        
        # Position of the mouse cursor in image coordinates before zoom
        img_pos_before = self.widget_to_image_coords(mouse_pos)
        
        # Expected position of the image point in widget coordinates after zoom
        widget_pos_after = img_pos_before * self.zoom_factor
        
        # The new pan offset is the difference between the mouse position and the new widget pos
        self.pan_offset = mouse_pos - widget_pos_after
        
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_window()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self.image and not self.image.isNull():
            painter.save()
            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)
            painter.drawImage(0, 0, self.image)

            # Draw grid
            if self.grid_width > 0 and self.grid_height > 0:
                pen = QPen(QColor(128, 128, 128, 100), 1 / self.zoom_factor)
                painter.setPen(pen)
                # Vertical lines
                for i in range(1, int(self.image.width() / self.grid_width) + 1):
                    x = i * self.grid_width
                    painter.drawLine(x, 0, x, self.image.height())
                # Horizontal lines
                for i in range(1, int(self.image.height() / self.grid_height) + 1):
                    y = i * self.grid_height
                    painter.drawLine(0, y, self.image.width(), y)
            painter.restore()

        self.draw_axes(painter)

        # Draw selection rectangle if dragging
        if self.is_selecting and self.start_pos and self.current_pos:
            pen = QPen(QColor(0, 255, 255, 200), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 100, 100, 50))
            rect = QRect(self.start_pos.toPoint(), self.current_pos.toPoint()).normalized()
            painter.drawRect(rect)

    def draw_axes(self, painter):
        if not hasattr(self, 'stage_width_mm') or self.image is None:
            return

        margin = 40
        rect = self.rect()
        axis_rect = rect.adjusted(margin, margin, -margin, -margin)

        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.setFont(self.font())

        # Visible image area in image coordinates
        tl_img = self.widget_to_image_coords(axis_rect.topLeft())
        br_img = self.widget_to_image_coords(axis_rect.bottomRight())
        
        # Min/Max labels
        painter.drawText(QRectF(0, axis_rect.top() - 20, margin, 20), Qt.AlignRight | Qt.AlignVCenter, f"{self.stage_height_mm:.1f}")
        painter.drawText(QRectF(0, axis_rect.bottom() - 10, margin, 20), Qt.AlignRight | Qt.AlignVCenter, "0.0")
        painter.drawText(QRectF(axis_rect.left(), axis_rect.bottom(), 40, 20), Qt.AlignLeft | Qt.AlignVCenter, "0.0")
        painter.drawText(QRectF(axis_rect.right() - 40, axis_rect.bottom(), 40, 20), Qt.AlignRight | Qt.AlignVCenter, f"{self.stage_width_mm:.1f}")


        # --- X AXIS (Bottom) ---
        num_ticks = 100
        for i in range(num_ticks + 1):
            img_x = (i / num_ticks) * self.image.width()
            widget_x = self.image_to_widget_coords(QPointF(img_x, 0)).x()
            
            if axis_rect.left() <= widget_x <= axis_rect.right():
                painter.drawLine(QPointF(widget_x, axis_rect.bottom()), QPointF(widget_x, axis_rect.bottom() + 5))

        num_labels = 11
        for i in range(num_labels + 1):
            val_mm = (i / num_labels) * self.stage_width_mm
            img_x = val_mm * self.ruler_calibration_px_per_mm
            widget_x = self.image_to_widget_coords(QPointF(img_x, 0)).x()
            
            if axis_rect.left() - 20 <= widget_x <= axis_rect.right() + 20:
                painter.drawText(QRectF(widget_x - 20, axis_rect.bottom() + 5, 40, 20), Qt.AlignCenter, f"{val_mm:.1f}")

        # --- Y AXIS (Left) ---
        for i in range(num_ticks + 1):
            img_y = (i / num_ticks) * self.image.height()
            widget_y = self.image_to_widget_coords(QPointF(0, img_y)).y()
            
            if axis_rect.top() <= widget_y <= axis_rect.bottom():
                painter.drawLine(QPointF(axis_rect.left(), widget_y), QPointF(axis_rect.left() - 5, widget_y))

        for i in range(num_labels + 1):
            val_mm = (i / num_labels) * self.stage_height_mm
            # Y is inverted
            img_y = self.image.height() - (val_mm * self.ruler_calibration_px_per_mm)
            widget_y = self.image_to_widget_coords(QPointF(0, img_y)).y()

            if axis_rect.top() - 10 <= widget_y <= axis_rect.bottom() + 10:
                painter.drawText(QRectF(axis_rect.left() - margin, widget_y - 10, margin - 5, 20), Qt.AlignRight | Qt.AlignVCenter, f"{val_mm:.1f}")

    def mousePressEvent(self, event):
        if not self.image or self.image.isNull():
            return
        
        self.last_mouse_pos = event.position()
        
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
            self.is_selecting = True
            self.start_pos = event.position()
            self.current_pos = event.position()
        elif event.button() == Qt.LeftButton:
            self.is_panning = True
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            self.is_zooming = True
            self.setCursor(Qt.SizeVerCursor)


    def mouseMoveEvent(self, event):
        if not self.image or self.image.isNull():
            return
            
        delta = event.position() - self.last_mouse_pos
        
        if self.is_panning:
            self.pan_offset += delta
            self.update()
            
        elif self.is_zooming:
            zoom_increment = -delta.y() * 0.005 # Negative so moving up zooms in
            
            old_zoom = self.zoom_factor
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor + zoom_increment))

            # Zoom towards the mouse cursor
            mouse_pos = event.position()
            img_pos_before = (mouse_pos - delta - self.pan_offset) / old_zoom
            new_widget_pos = img_pos_before * self.zoom_factor
            self.pan_offset = mouse_pos - delta - new_widget_pos
            
            self.update()

        elif self.is_selecting:
            self.current_pos = event.position()
            self.update()
            
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        if not self.image or self.image.isNull():
            return

        if self.is_selecting:
            dist = (event.position() - self.start_pos).manhattanLength()
            
            if dist < 5: # Treat as a click
                img_coords = self.widget_to_image_coords(event.position())
                if not img_coords.isNull():
                    self.clicked.emit(img_coords.x(), img_coords.y())
            else: # Treat as a selection
                start_img = self.widget_to_image_coords(self.start_pos)
                end_img = self.widget_to_image_coords(event.position())
                
                if not start_img.isNull() and not end_img.isNull():
                    selection_rect = QRectF(start_img, end_img).normalized()
                    self.selection_made.emit(
                        selection_rect.x(), 
                        selection_rect.y(), 
                        selection_rect.width(), 
                        selection_rect.height()
                    )
            self.start_pos = None
            self.current_pos = None
            self.update()

        # Reset all states
        self.is_panning = False
        self.is_zooming = False
        self.is_selecting = False
        self.setCursor(Qt.ArrowCursor)


class MosaicPanel(QWidget):
    """
    A panel to display a mosaic of the stage area, updated with camera frames
    based on CNC position.
    """
    request_move_signal = Signal(float, float)
    request_scan_signal = Signal(float, float, float, float) # x_min, y_min, x_max, y_max

    def __init__(self, stage_width_mm: float, stage_height_mm: float, ruler_calibration_px_per_mm: float, parent=None):
        super().__init__(parent)

        self.stage_width_mm = stage_width_mm
        self.stage_height_mm = stage_height_mm
        self.ruler_calibration_px_per_mm = ruler_calibration_px_per_mm

        if self.ruler_calibration_px_per_mm <= 0:
            print("Error: Invalid ruler calibration for mosaic window. Calibration must be > 0.")
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
        self.display_widget.set_stage_size(self.stage_width_mm, self.stage_height_mm)
        self.display_widget.set_calibration(self.ruler_calibration_px_per_mm)
        self.display_widget.set_image(self.mosaic_image)
        self.display_widget.clicked.connect(self.on_mosaic_clicked)
        self.display_widget.selection_made.connect(self.on_mosaic_selection)
        
        self.position_label = QLabel("X: 0 px, Y: 0 px")
        self.position_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.display_widget)
        layout.addWidget(self.position_label)

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

        # Update camera frame dimensions if they change (e.g., ROI changes)
        if self.camera_frame_width_px != camera_frame.width() or \
           self.camera_frame_height_px != camera_frame.height():
            self.camera_frame_width_px = camera_frame.width()
            self.camera_frame_height_px = camera_frame.height()
            self.display_widget.set_grid_size(self.camera_frame_width_px, self.camera_frame_height_px)

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

        if not self.mosaic_image.isNull():
            painter = QPainter(self.mosaic_image)
            painter.drawImage(mosaic_draw_x_px, mosaic_draw_y_px, camera_frame.convertToFormat(QImage.Format_RGB32))
            painter.end()
        else:
            print(f"Error: Mosaic image is null (Dimensions: {self.mosaic_width_px}x{self.mosaic_height_px}). Cannot update mosaic.")

        self.display_widget.set_image(self.mosaic_image)

        # Update position label
        if self.ruler_calibration_px_per_mm > 0:
            pixel_x = cnc_x_mm * self.ruler_calibration_px_per_mm
            pixel_y = cnc_y_mm * self.ruler_calibration_px_per_mm
            self.position_label.setText(f"X: {pixel_x:.0f} px, Y: {pixel_y:.0f} px")

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
