import os
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout, QScrollBar
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QTransform, QPen
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRect, QPoint, QPointF, QRectF, QFile
from PySide6.QtUiTools import QUiLoader


class MosaicWidget(QWidget):
    """
    A custom widget to display a pan-and-zoomable mosaic image.
    """
    clicked = Signal(float, float) # Emits image_x, image_y
    selections_changed = Signal(list) # Emits list of QRectF in image pixels
    mouse_moved = Signal(float, float)
    selection_made = Signal(float, float, float, float) # Emits x, y, width, height

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.tiles = {}
        self.tile_size = 32768
        self.total_width = 0
        self.total_height = 0
        
        self.start_pos = None
        self.current_pos = None
        
        self.grid_width = 0
        self.grid_height = 0

        # Pan and Zoom state
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.last_mouse_pos = QPoint()

        # Dragging states and selections
        self.is_panning = False 
        self.is_selecting = False

        self.stage_width_mm = 0
        self.stage_height_mm = 0
        self.ruler_calibration_px_per_mm = 0
        
        # Overlay Rectangles
        self.current_frame_rect = QRectF()
        self.selected_rects = [] # List of QRectF for selected scan areas
        self.overlay_circles = [] # List of (QPointF center, float radius_px)

    def setup_scrollbars(self, h_bar, v_bar):
        self.h_scrollbar = h_bar
        self.v_scrollbar = v_bar
        self.h_scrollbar.show()
        self.v_scrollbar.show()
        self.h_scrollbar.valueChanged.connect(self.on_h_scroll)
        self.v_scrollbar.valueChanged.connect(self.on_v_scroll)
        self._updating_scrollbars = False

    def on_h_scroll(self, value):
        if hasattr(self, '_updating_scrollbars') and self._updating_scrollbars:
            return
        self.pan_offset.setX(-value)
        self.update()

    def on_v_scroll(self, value):
        if hasattr(self, '_updating_scrollbars') and self._updating_scrollbars:
            return
        self.pan_offset.setY(-value)
        self.update()

    def clamp_pan_offset(self):
        view_w = self.rect().width()
        view_h = self.rect().height()
        drawn_w = self.total_width * self.zoom_factor
        drawn_h = self.total_height * self.zoom_factor
        
        if drawn_w > view_w:
            x = max(-(drawn_w - view_w), min(0, self.pan_offset.x()))
        else:
            x = (view_w - drawn_w) / 2
            
        if drawn_h > view_h:
            y = max(-(drawn_h - view_h), min(0, self.pan_offset.y()))
        else:
            y = (view_h - drawn_h) / 2
            
        self.pan_offset = QPointF(x, y)

    def update_scrollbars(self):
        if not hasattr(self, 'h_scrollbar') or not hasattr(self, 'v_scrollbar'):
            return
            
        view_w = self.rect().width()
        view_h = self.rect().height()
        drawn_w = self.total_width * self.zoom_factor
        drawn_h = self.total_height * self.zoom_factor
        
        self._updating_scrollbars = True
        
        if drawn_w > view_w:
            self.h_scrollbar.setRange(0, int(drawn_w - view_w))
            self.h_scrollbar.setPageStep(view_w)
            self.h_scrollbar.setValue(int(-self.pan_offset.x()))
            self.h_scrollbar.setEnabled(True)
        else:
            self.h_scrollbar.setRange(0, 0)
            self.h_scrollbar.setPageStep(max(1, view_w))
            self.h_scrollbar.setValue(0)
            self.h_scrollbar.setEnabled(False)
            
        if drawn_h > view_h:
            self.v_scrollbar.setRange(0, int(drawn_h - view_h))
            self.v_scrollbar.setPageStep(view_h)
            self.v_scrollbar.setValue(int(-self.pan_offset.y()))
            self.v_scrollbar.setEnabled(True)
        else:
            self.v_scrollbar.setRange(0, 0)
            self.v_scrollbar.setPageStep(max(1, view_h))
            self.v_scrollbar.setValue(0)
            self.v_scrollbar.setEnabled(False)
            
        self._updating_scrollbars = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.clamp_pan_offset()
        self.update_scrollbars()

    def set_stage_size(self, width_mm: float, height_mm: float):
        self.stage_width_mm = width_mm
        self.stage_height_mm = height_mm

    def set_calibration(self, calibration: float):
        self.ruler_calibration_px_per_mm = calibration

    def set_grid_size(self, width: int, height: int):
        self.grid_width = width
        self.grid_height = height
        self.update()

    def set_current_frame_rect(self, rect: QRectF):
        self.current_frame_rect = rect # This is for the current camera FOV
        self.update()

    def set_overlay_circles(self, circles: list[tuple[QPointF, float]]):
        self.overlay_circles = circles
        self.update()

    def reset_mosaic(self, width: int, height: int, tile_size: int):
        self.total_width = width
        self.total_height = height
        self.tile_size = tile_size
        self.tiles = {}
        
        if width > 0 and height > 0:
             self.fit_to_window()
        self.update()

    def update_tile(self, row: int, col: int, image: QImage):
        self.tiles[(row, col)] = image
        self.update()

    def widget_to_image_coords(self, widget_pos: QPoint) -> QPointF:
        """Converts widget coordinates to image coordinates."""
        if self.total_width == 0:
            return QPointF()
        
        # Reverse the transformation: (pos - pan) / zoom
        return (QPointF(widget_pos) - self.pan_offset) / self.zoom_factor

    def image_to_widget_coords(self, image_pos: QPointF) -> QPointF:
        """Converts image coordinates to widget coordinates."""
        if self.total_width == 0:
            return QPointF()
        
        # Apply the transformation: (pos * zoom) + pan
        return (image_pos * self.zoom_factor) + self.pan_offset

    def fit_to_window(self):
        if self.total_width == 0 or self.total_height == 0:
            return
        
        widget_rect = self.rect()
        
        scale_w = widget_rect.width() / self.total_width
        scale_h = widget_rect.height() / self.total_height
        self.zoom_factor = min(scale_w, scale_h)

        # Center the image
        drawn_w = self.total_width * self.zoom_factor
        drawn_h = self.total_height * self.zoom_factor
        
        self.pan_offset = QPointF(
            (widget_rect.width() - drawn_w) / 2,
            (widget_rect.height() - drawn_h) / 2
        )
        self.clamp_pan_offset()
        self.update_scrollbars()
        self.update()

    def do_zoom(self, zoom_direction, center_pos=None):
        if self.total_width == 0:
            return

        zoom_increment = 0.1 * zoom_direction
        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor + zoom_increment))

        if center_pos is None:
            center_pos = QPointF(self.rect().width() / 2, self.rect().height() / 2)
            
        img_pos_before = (center_pos - self.pan_offset) / old_zoom
        widget_pos_after = img_pos_before * self.zoom_factor
        self.pan_offset = center_pos - widget_pos_after
        
        self.clamp_pan_offset()
        self.update_scrollbars()
        self.update()

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if self.total_width == 0:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_PageUp:
            self.do_zoom(1)
            event.accept()
        elif event.key() == Qt.Key_PageDown:
            self.do_zoom(-1)
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_window()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self.total_width > 0 and self.total_height > 0:
            painter.save()
            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)
            
            for (row, col), tile in self.tiles.items():
                x = col * self.tile_size
                y = row * self.tile_size
                painter.drawImage(x, y, tile)

            # Draw Scan Region (Pink)
            for rect in self.selected_rects:
                pen = QPen(QColor(255, 105, 180), 3 / self.zoom_factor)
                painter.setPen(pen)
                painter.setBrush(QColor(255, 105, 180, 50)) # Semi-transparent fill
                painter.drawRect(rect)
                
                # Draw a cross in the middle
                painter.drawLine(rect.center().x() - 5 / self.zoom_factor, rect.center().y(), rect.center().x() + 5 / self.zoom_factor, rect.center().y())
                painter.drawLine(rect.center().x(), rect.center().y() - 5 / self.zoom_factor, rect.center().x(), rect.center().y() + 5 / self.zoom_factor)

            
            # Draw Current Frame Rect (Green)
            if not self.current_frame_rect.isNull():
                 pen = QPen(Qt.green, 2 / self.zoom_factor)
                 painter.setPen(pen)
                 painter.setBrush(Qt.NoBrush)
                 painter.drawRect(self.current_frame_rect)

            for center, radius_px in self.overlay_circles:
                pen = QPen(QColor(255, 196, 0), 3 / self.zoom_factor)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, radius_px, radius_px)

            # Draw grid
            if self.grid_width > 0 and self.grid_height > 0:
                pen = QPen(QColor(128, 128, 128, 100), 1 / self.zoom_factor)
                painter.setPen(pen)
                # Vertical lines
                # Limit grid lines to avoid freezing on huge images
                max_grid_lines = 2000 
                
                num_v_lines = int(self.total_width / self.grid_width)
                if num_v_lines < max_grid_lines:
                    for i in range(1, num_v_lines + 1):
                        x = i * self.grid_width
                        painter.drawLine(x, 0, x, self.total_height)
                
                num_h_lines = int(self.total_height / self.grid_height)
                if num_h_lines < max_grid_lines:
                    for i in range(1, num_h_lines + 1):
                        y = i * self.grid_height
                        painter.drawLine(0, y, self.total_width, y)
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
        if not hasattr(self, 'stage_width_mm') or self.total_width == 0:
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
            img_x = (i / num_ticks) * self.total_width
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
            img_y = (i / num_ticks) * self.total_height
            widget_y = self.image_to_widget_coords(QPointF(0, img_y)).y()
            
            if axis_rect.top() <= widget_y <= axis_rect.bottom():
                painter.drawLine(QPointF(axis_rect.left(), widget_y), QPointF(axis_rect.left() - 5, widget_y))

        for i in range(num_labels + 1):
            val_mm = (i / num_labels) * self.stage_height_mm
            # Y is inverted
            img_y = self.total_height - (val_mm * self.ruler_calibration_px_per_mm)
            widget_y = self.image_to_widget_coords(QPointF(0, img_y)).y()

            if axis_rect.top() - 10 <= widget_y <= axis_rect.bottom() + 10:
                painter.drawText(QRectF(axis_rect.left() - margin, widget_y - 10, margin - 5, 20), Qt.AlignRight | Qt.AlignVCenter, f"{val_mm:.1f}")

    def mousePressEvent(self, event):
        if self.total_width == 0:
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
            self.selected_rects.clear()
            self.selections_changed.emit(self.selected_rects)
            self.update()
    def mouseMoveEvent(self, event):
        if self.total_width == 0:
            return
            
        # Emit mouse move signal
        img_coords = self.widget_to_image_coords(event.position())
        # (0,0) is a valid coordinate so don't check isNull()
        self.mouse_moved.emit(img_coords.x(), img_coords.y())

        delta = event.position() - self.last_mouse_pos
        
        if self.is_panning:
            self.pan_offset += delta
            self.clamp_pan_offset()
            self.update_scrollbars()
            self.update()

        elif self.is_selecting:
            self.current_pos = event.position()
            self.update()
            
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        if self.total_width == 0:
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
                    # Constrain to valid image boundaries
                    selection_rect = selection_rect.intersected(QRectF(0, 0, self.total_width, self.total_height))
                    if not selection_rect.isEmpty():
                        self.selected_rects.append(selection_rect)
                        self.selections_changed.emit(self.selected_rects)
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
        self.is_selecting = False
        self.setCursor(Qt.ArrowCursor)


class MosaicPanel(QWidget):
    """
    A panel to display a mosaic of the stage area, updated with camera frames
    based on CNC position.
    """
    request_move_signal = Signal(float, float)
    request_scan_signal = Signal(float, float, float, float) # x_min, y_min, x_max, y_max
    selections_changed = Signal(list) # Emits list of (x_min_mm, y_min_mm, x_max_mm, y_max_mm)
    
    TILE_SIZE = 32768
    SCALE_FACTOR = 0.1  # Downscale mosaic to 1/10 of true size

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

        self.mosaic_width_px = int(self.stage_width_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        self.mosaic_height_px = int(self.stage_height_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)

        if self.mosaic_width_px <= 0 or self.mosaic_height_px <= 0:
            print(f"Error: Calculated mosaic dimensions are invalid: {self.mosaic_width_px}x{self.mosaic_height_px}")
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Error: Invalid Mosaic Dimensions"))
            return
            
        print(f"Mosaic dimensions: {self.mosaic_width_px}x{self.mosaic_height_px}")
        
        # Tile initialization
        self.tiles = {} # (row, col) -> QImage
        self.cols = (self.mosaic_width_px + self.TILE_SIZE - 1) // self.TILE_SIZE
        self.rows = (self.mosaic_height_px + self.TILE_SIZE - 1) // self.TILE_SIZE
        
        print(f"Initializing {self.rows}x{self.cols} tiles (Tile Size: {self.TILE_SIZE})...")
        
        # Load UI
        loader = QUiLoader()
        loader.registerCustomWidget(MosaicWidget)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(script_dir, "MosaicPanel.ui")
        ui_file = QFile(ui_file_path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        if self.ui is None:
            print(f"Error loading UI: {loader.errorString()}")
            return
            
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ui)
        
        self.display_widget = self.ui.findChild(MosaicWidget, "display_widget")
        self.position_label = self.ui.findChild(QLabel, "position_label")
        self.cursor_label = self.ui.findChild(QLabel, "cursor_label")
        
        self.h_scrollbar = self.ui.findChild(QScrollBar, "h_scrollbar")
        self.v_scrollbar = self.ui.findChild(QScrollBar, "v_scrollbar")
        if self.h_scrollbar and self.v_scrollbar:
            self.display_widget.setup_scrollbars(self.h_scrollbar, self.v_scrollbar)

        self.display_widget.set_stage_size(self.stage_width_mm, self.stage_height_mm)
        self.display_widget.set_calibration(self.ruler_calibration_px_per_mm)
        self.display_widget.reset_mosaic(self.mosaic_width_px, self.mosaic_height_px, self.TILE_SIZE)
        
        for r in range(self.rows):
            for c in range(self.cols):
                w = min(self.TILE_SIZE, self.mosaic_width_px - c * self.TILE_SIZE)
                h = min(self.TILE_SIZE, self.mosaic_height_px - r * self.TILE_SIZE)
                img = QImage(w, h, QImage.Format_RGB32)
                img.fill(Qt.white)
                self.tiles[(r, c)] = img
                self.display_widget.update_tile(r, c, img)
        self.display_widget.clicked.connect(self.on_mosaic_clicked)
        self.display_widget.selections_changed.connect(self._on_selections_changed) # Connect to internal slot
        self.display_widget.mouse_moved.connect(self.on_mosaic_mouse_moved)

        # Store camera frame dimensions to calculate offset
        self.camera_frame_width_px = 0
        self.camera_frame_height_px = 0
        self.stage_circle_overlays_mm = []

    def set_stage_circles(self, circles_mm: list[tuple[float, float, float]]):
        self.stage_circle_overlays_mm = circles_mm

        circle_overlays_px = []
        for center_x_mm, center_y_mm, radius_mm in circles_mm:
            center_x_px = center_x_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR
            center_y_px = (self.stage_height_mm - center_y_mm) * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR
            radius_px = radius_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR
            circle_overlays_px.append((QPointF(center_x_px, center_y_px), radius_px))

        self.display_widget.set_overlay_circles(circle_overlays_px)

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
            scaled_grid_width = int(self.camera_frame_width_px * self.SCALE_FACTOR)
            scaled_grid_height = int(self.camera_frame_height_px * self.SCALE_FACTOR)
            self.display_widget.set_grid_size(scaled_grid_width, scaled_grid_height)

        # Camera's physical width/height in mm
        camera_fov_width_mm = self.camera_frame_width_px / self.ruler_calibration_px_per_mm
        camera_fov_height_mm = self.camera_frame_height_px / self.ruler_calibration_px_per_mm

        # Calculate the top-left corner of the camera's view on the stage in CNC (mm) coordinates
        # Assuming cnc_x_mm, cnc_y_mm is the center of the camera's view
        camera_top_left_x_mm = cnc_x_mm - (camera_fov_width_mm / 2)
        camera_top_left_y_mm = cnc_y_mm + (camera_fov_height_mm / 2) # Y increases upwards in CNC, so top edge is higher Y

        # Convert top-left mm coordinates to mosaic pixel coordinates (mosaic Y is inverted)
        mosaic_draw_x_px = int(camera_top_left_x_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        mosaic_draw_y_px = int((self.stage_height_mm - camera_top_left_y_mm) * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)

        # Rect of the camera frame in mosaic coordinates (scaled down)
        scaled_frame_width = int(self.camera_frame_width_px * self.SCALE_FACTOR)
        scaled_frame_height = int(self.camera_frame_height_px * self.SCALE_FACTOR)
        frame_rect = QRect(mosaic_draw_x_px, mosaic_draw_y_px, scaled_frame_width, scaled_frame_height)
        
        # Update current frame overlay
        self.display_widget.set_current_frame_rect(QRectF(frame_rect))
        
        # Determine intersecting tiles
        start_col = max(0, frame_rect.left() // self.TILE_SIZE) # Integer division
        end_col = min(self.cols - 1, frame_rect.right() // self.TILE_SIZE)
        start_row = max(0, frame_rect.top() // self.TILE_SIZE)
        end_row = min(self.rows - 1, frame_rect.bottom() // self.TILE_SIZE)
        
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                if (r, c) in self.tiles:
                    tile = self.tiles[(r, c)]
                    
                    tile_x = c * self.TILE_SIZE
                    tile_y = r * self.TILE_SIZE
                    tile_rect = QRect(tile_x, tile_y, tile.width(), tile.height())
                    
                    # Intersection of frame and tile
                    intersection = frame_rect.intersected(tile_rect)
                    
                    if not intersection.isEmpty():
                        # Destination on tile
                        dest_x = intersection.x() - tile_x
                        dest_y = intersection.y() - tile_y
                        
                        # Source from camera frame (need to scale back to original frame coordinates)
                        src_x = int((intersection.x() - frame_rect.x()) / self.SCALE_FACTOR)
                        src_y = int((intersection.y() - frame_rect.y()) / self.SCALE_FACTOR)
                        src_w = int(intersection.width() / self.SCALE_FACTOR)
                        src_h = int(intersection.height() / self.SCALE_FACTOR)
                        
                        painter = QPainter(tile)
                        # Scale down the camera frame region and flip vertically
                        scaled_frame = camera_frame.copy(src_x, src_y, src_w, src_h).scaled(
                            intersection.width(), intersection.height(), 
                            Qt.IgnoreAspectRatio, Qt.SmoothTransformation).mirrored(True, True)
                        painter.drawImage(dest_x, dest_y, scaled_frame.convertToFormat(QImage.Format_RGB32))
                        painter.end()
                        
                        self.display_widget.update_tile(r, c, tile)

        # Update position label
        if self.ruler_calibration_px_per_mm > 0:
            self.position_label.setText(f"CNC: {cnc_x_mm:.1f} mm, {cnc_y_mm:.1f} mm")

    @Slot(list)
    def _on_selections_changed(self, qrectf_list: list[QRectF]):
        """
        Converts the list of QRectF (image pixels) to a list of (x_min_mm, y_min_mm, x_max_mm, y_max_mm)
        and emits it.
        """
        mm_rects = []
        for qrectf in qrectf_list:
            # Convert to CNC coords
            # Image (0,0) is Top-Left. CNC (0,0) is Bottom-Left.
            # Account for scaling
            cnc_x1 = qrectf.x() / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
            cnc_y1 = self.stage_height_mm - (qrectf.y() / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR))
            cnc_x2 = qrectf.right() / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
            cnc_y2 = self.stage_height_mm - (qrectf.bottom() / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR))
            
            mm_rects.append((min(cnc_x1, cnc_x2), min(cnc_y1, cnc_y2), max(cnc_x1, cnc_x2), max(cnc_y1, cnc_y2)))
            
        self.selections_changed.emit(mm_rects)

    def get_region_image(self, x_min_mm: float, y_min_mm: float, x_max_mm: float, y_max_mm: float) -> QImage:
        if self.ruler_calibration_px_per_mm <= 0:
            return QImage()
        
        top_left_x_px = int(x_min_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        top_left_y_px = int((self.stage_height_mm - y_max_mm) * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        
        bottom_right_x_px = int(x_max_mm * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        bottom_right_y_px = int((self.stage_height_mm - y_min_mm) * self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        
        width = max(1, bottom_right_x_px - top_left_x_px)
        height = max(1, bottom_right_y_px - top_left_y_px)
        
        region_rect = QRect(top_left_x_px, top_left_y_px, width, height)
        
        result_image = QImage(width, height, QImage.Format_RGB32)
        result_image.fill(Qt.white)
        
        painter = QPainter(result_image)
        
        start_col = max(0, region_rect.left() // self.TILE_SIZE)
        end_col = min(self.cols - 1, region_rect.right() // self.TILE_SIZE)
        start_row = max(0, region_rect.top() // self.TILE_SIZE)
        end_row = min(self.rows - 1, region_rect.bottom() // self.TILE_SIZE)
        
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                if (r, c) in self.tiles:
                    tile = self.tiles[(r, c)]
                    tile_x = c * self.TILE_SIZE
                    tile_y = r * self.TILE_SIZE
                    
                    dest_x = tile_x - top_left_x_px
                    dest_y = tile_y - top_left_y_px
                    painter.drawImage(dest_x, dest_y, tile)
                    
        painter.end()
        return result_image

    @Slot(float, float)
    def on_mosaic_clicked(self, img_x, img_y):
        if self.ruler_calibration_px_per_mm <= 0:
            return
        
        # Mosaic (0,0) is Top-Left. CNC (0,0) is Bottom-Left.
        # Account for scaling: divide by (ruler_calibration * SCALE_FACTOR)
        cnc_x = img_x / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        cnc_y = self.stage_height_mm - (img_y / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR))
        
        # Clamp to stage bounds
        cnc_x = max(0.0, min(cnc_x, self.stage_width_mm))
        cnc_y = max(0.0, min(cnc_y, self.stage_height_mm))
        
        self.request_move_signal.emit(cnc_x, cnc_y)

    @Slot(float, float)
    def on_mosaic_mouse_moved(self, img_x, img_y):
        if self.ruler_calibration_px_per_mm <= 0:
            return
            
        # Mosaic (0,0) is Top-Left. CNC (0,0) is Bottom-Left.
        # Account for scaling: divide by (ruler_calibration * SCALE_FACTOR)
        cnc_x = img_x / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR)
        cnc_y = self.stage_height_mm - (img_y / (self.ruler_calibration_px_per_mm * self.SCALE_FACTOR))
        
        self.cursor_label.setText(f"Cursor: {cnc_x:.1f} mm, {cnc_y:.1f} mm ({img_x:.0f}, {img_y:.0f} px)")
