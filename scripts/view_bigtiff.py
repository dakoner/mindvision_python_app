#!/usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import tifffile
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QAction, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QWidget


def _normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array

    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return array

    raise ValueError(f"Unsupported TIFF array shape: {array.shape}")


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.uint8) * 255

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max == info.min:
            return np.zeros_like(array, dtype=np.uint8)
        scaled = (array.astype(np.float32) - info.min) * (255.0 / (info.max - info.min))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    if np.issubdtype(array.dtype, np.floating):
        finite_mask = np.isfinite(array)
        if not np.any(finite_mask):
            return np.zeros_like(array, dtype=np.uint8)
        finite_values = array[finite_mask]
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        if math.isclose(min_value, max_value):
            return np.zeros_like(array, dtype=np.uint8)
        scaled = (array.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))
        scaled[~finite_mask] = 0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported TIFF dtype: {array.dtype}")


def _to_qimage(array: np.ndarray) -> QImage:
    normalized = _normalize_array(array)
    as_uint8 = _to_uint8(normalized)

    if as_uint8.ndim == 2:
        contiguous = np.ascontiguousarray(as_uint8)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_Grayscale8,
        )
        return image.copy()

    if as_uint8.shape[-1] == 1:
        return _to_qimage(as_uint8[..., 0])

    if as_uint8.shape[-1] == 3:
        contiguous = np.ascontiguousarray(as_uint8)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format_RGB888,
        )
        return image.copy()

    contiguous = np.ascontiguousarray(as_uint8)
    image = QImage(
        contiguous.data,
        contiguous.shape[1],
        contiguous.shape[0],
        contiguous.strides[0],
        QImage.Format_RGBA8888,
    )
    return image.copy()


class BigTiffView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image_array = None
        self.image_width = 0
        self.image_height = 0
        self.zoom = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.drag_last_pos = QPoint()
        self.dragging = False
        self.status_label = None
        self._cached_pixmap = None
        self._cached_target = QRect()
        self._cached_key = None

    def set_status_label(self, label: QLabel):
        self.status_label = label
        self._update_status()

    def load_array(self, array: np.ndarray):
        self.image_array = _normalize_array(array)
        self.image_height = int(self.image_array.shape[0])
        self.image_width = int(self.image_array.shape[1])
        self.zoom = 1.0
        self.offset = QPointF(0.0, 0.0)
        self._invalidate_cache()
        self.fit_to_window()

    def has_image(self) -> bool:
        return self.image_array is not None

    def fit_to_window(self):
        if not self.has_image() or self.width() <= 0 or self.height() <= 0:
            return

        scale_x = self.width() / self.image_width
        scale_y = self.height() / self.image_height
        self.zoom = min(scale_x, scale_y)
        if self.zoom <= 0:
            self.zoom = 1.0

        draw_width = self.image_width * self.zoom
        draw_height = self.image_height * self.zoom
        self.offset = QPointF(
            (self.width() - draw_width) / 2.0,
            (self.height() - draw_height) / 2.0,
        )
        self._invalidate_cache()
        self._clamp_offset()
        self._update_status()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.has_image() and self._cached_pixmap is None:
            self.fit_to_window()
        else:
            self._clamp_offset()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self.has_image():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a BigTIFF file to view it")
            return

        target = self._target_rect()
        if target.isEmpty():
            return

        if self._cached_pixmap is None or self._cached_target != target:
            self._render_target(target)

        if self._cached_pixmap is not None:
            painter.drawPixmap(target, self._cached_pixmap)

    def wheelEvent(self, event):
        if not self.has_image():
            return

        angle = event.angleDelta().y()
        if angle == 0:
            return

        zoom_factor = 1.2 if angle > 0 else 1 / 1.2
        old_zoom = self.zoom
        new_zoom = max(0.02, min(32.0, self.zoom * zoom_factor))
        if math.isclose(old_zoom, new_zoom):
            return

        cursor_pos = event.position()
        image_x = (cursor_pos.x() - self.offset.x()) / old_zoom
        image_y = (cursor_pos.y() - self.offset.y()) / old_zoom

        self.zoom = new_zoom
        self.offset = QPointF(
            cursor_pos.x() - (image_x * self.zoom),
            cursor_pos.y() - (image_y * self.zoom),
        )
        self._clamp_offset()
        self._invalidate_cache()
        self._update_status(cursor_pos)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.has_image():
            self.dragging = True
            self.drag_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_last_pos
            self.offset += QPointF(delta.x(), delta.y())
            self.drag_last_pos = event.pos()
            self._clamp_offset()
            self._invalidate_cache()
            self.update()

        self._update_status(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)

    def _target_rect(self) -> QRect:
        if not self.has_image():
            return QRect()

        draw_x = int(round(self.offset.x()))
        draw_y = int(round(self.offset.y()))
        draw_width = max(1, int(round(self.image_width * self.zoom)))
        draw_height = max(1, int(round(self.image_height * self.zoom)))
        return QRect(draw_x, draw_y, draw_width, draw_height).intersected(self.rect())

    def _render_target(self, target: QRect):
        source_left = max(0.0, (target.left() - self.offset.x()) / self.zoom)
        source_top = max(0.0, (target.top() - self.offset.y()) / self.zoom)
        source_right = min(float(self.image_width), (target.right() - self.offset.x() + 1.0) / self.zoom)
        source_bottom = min(float(self.image_height), (target.bottom() - self.offset.y() + 1.0) / self.zoom)

        source_width = max(1, int(math.ceil(source_right - source_left)))
        source_height = max(1, int(math.ceil(source_bottom - source_top)))

        step_x = max(1, int(math.ceil(source_width / max(1, target.width()))))
        step_y = max(1, int(math.ceil(source_height / max(1, target.height()))))

        source_x0 = int(source_left)
        source_y0 = int(source_top)
        source_x1 = min(self.image_width, source_x0 + (source_width * step_x))
        source_y1 = min(self.image_height, source_y0 + (source_height * step_y))

        cache_key = (
            source_x0,
            source_y0,
            source_x1,
            source_y1,
            step_x,
            step_y,
            target.width(),
            target.height(),
        )
        if cache_key == self._cached_key and self._cached_pixmap is not None:
            self._cached_target = QRect(target)
            return

        sampled = self.image_array[source_y0:source_y1:step_y, source_x0:source_x1:step_x]
        preview = np.ascontiguousarray(sampled)
        image = _to_qimage(preview)
        pixmap = QPixmap.fromImage(image)
        if pixmap.width() != target.width() or pixmap.height() != target.height():
            pixmap = pixmap.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self._cached_pixmap = pixmap
        self._cached_target = QRect(target)
        self._cached_key = cache_key

    def _clamp_offset(self):
        if not self.has_image():
            return

        draw_width = self.image_width * self.zoom
        draw_height = self.image_height * self.zoom

        if draw_width <= self.width():
            offset_x = (self.width() - draw_width) / 2.0
        else:
            min_x = self.width() - draw_width
            max_x = 0.0
            offset_x = min(max(self.offset.x(), min_x), max_x)

        if draw_height <= self.height():
            offset_y = (self.height() - draw_height) / 2.0
        else:
            min_y = self.height() - draw_height
            max_y = 0.0
            offset_y = min(max(self.offset.y(), min_y), max_y)

        self.offset = QPointF(offset_x, offset_y)

    def _invalidate_cache(self):
        self._cached_pixmap = None
        self._cached_target = QRect()
        self._cached_key = None

    def _update_status(self, cursor_pos=None):
        if self.status_label is None:
            return

        if not self.has_image():
            self.status_label.setText("No image loaded")
            return

        message = f"{self.image_width} x {self.image_height} px | zoom {self.zoom:.3f}x"
        if cursor_pos is not None:
            image_x = int((cursor_pos.x() - self.offset.x()) / self.zoom)
            image_y = int((cursor_pos.y() - self.offset.y()) / self.zoom)
            if 0 <= image_x < self.image_width and 0 <= image_y < self.image_height:
                message += f" | cursor ({image_x}, {image_y})"

        self.status_label.setText(message)


class BigTiffViewer(QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("BigTIFF Viewer")
        self.resize(1400, 900)

        self.image_view = BigTiffView(self)
        self.setCentralWidget(self.image_view)

        self.status_label = QLabel(self)
        self.statusBar().addPermanentWidget(self.status_label, 1)
        self.image_view.set_status_label(self.status_label)

        self._tiff_file = None
        self._image_array = None

        self._create_actions()

        if initial_path:
            self.load_path(initial_path)

    def _create_actions(self):
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_dialog)
        file_menu.addAction(open_action)

        view_menu = self.menuBar().addMenu("View")
        fit_action = QAction("Fit to Window", self)
        fit_action.triggered.connect(self.image_view.fit_to_window)
        view_menu.addAction(fit_action)

    def open_dialog(self):
        start_dir = os.getcwd()
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open BigTIFF",
            start_dir,
            "TIFF Files (*.tif *.tiff)",
        )
        if selected_path:
            self.load_path(selected_path)

    def load_path(self, path: str):
        try:
            if self._tiff_file is not None:
                self._tiff_file.close()

            self._tiff_file = tifffile.TiffFile(path)
            if not self._tiff_file.series:
                raise ValueError("No image series found in TIFF file")

            series = self._tiff_file.series[0]
            self._image_array = series.asarray(out="memmap")
            self.image_view.load_array(self._image_array)
            self.setWindowTitle(f"BigTIFF Viewer - {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not open TIFF file:\n{exc}")

    def closeEvent(self, event):
        if self._tiff_file is not None:
            self._tiff_file.close()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a BigTIFF file in a simple pan/zoom viewer.")
    parser.add_argument("path", nargs="?", help="Path to a .tif or .tiff file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv)
    viewer = BigTiffViewer(initial_path=args.path)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())