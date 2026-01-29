from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QBrush, QColor, QPalette

class RangeSlider(QWidget):
    valuesChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_val = 0
        self.max_val = 255
        self.low = 50
        self.high = 150
        self.pressed_handle = None # 'low', 'high', or None
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(30)

    def setRange(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def setValues(self, low, high):
        self.low = max(self.min_val, min(low, self.max_val))
        self.high = max(self.min_val, min(high, self.max_val))
        if self.low > self.high:
            self.low, self.high = self.high, self.low
        self.update()
        self.valuesChanged.emit(self.low, self.high)

    def getValues(self):
        return self.low, self.high

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw track
        width = self.width()
        height = self.height()
        track_h = 4
        track_y = height // 2 - track_h // 2
        
        # Background track
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, track_y, width, track_h, 2, 2)
        
        # Active range
        x1 = self._value_to_pos(self.low)
        x2 = self._value_to_pos(self.high)
        
        # Ensure x1 is left of x2
        if x1 > x2: x1, x2 = x2, x1
        
        painter.setBrush(QBrush(self.palette().color(QPalette.Highlight)))
        painter.drawRoundedRect(x1, track_y, x2 - x1, track_h, 2, 2)
        
        # Handles
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QColor(100, 100, 100))
        
        self._draw_handle(painter, x1, track_y + track_h//2)
        self._draw_handle(painter, x2, track_y + track_h//2)

    def _draw_handle(self, painter, x, cy):
        size = 16
        painter.drawEllipse(int(x - size//2), int(cy - size//2), int(size), int(size))

    def _value_to_pos(self, value):
        width = self.width() - 20 # padding
        if self.max_val == self.min_val: return 10
        return 10 + (value - self.min_val) / (self.max_val - self.min_val) * width

    def _pos_to_value(self, x):
        width = self.width() - 20
        if width <= 0: return self.min_val
        val = self.min_val + (x - 10) / width * (self.max_val - self.min_val)
        return max(self.min_val, min(int(val), self.max_val))

    def mousePressEvent(self, event):
        pos = event.position().x()
        low_pos = self._value_to_pos(self.low)
        high_pos = self._value_to_pos(self.high)
        
        if abs(pos - low_pos) < 15:
            self.pressed_handle = 'low'
        elif abs(pos - high_pos) < 15:
            self.pressed_handle = 'high'
        else:
            # Click jump
            if abs(pos - low_pos) < abs(pos - high_pos):
                self.pressed_handle = 'low'
            else:
                self.pressed_handle = 'high'
            self._update_from_mouse(pos)

    def mouseMoveEvent(self, event):
        if self.pressed_handle:
            self._update_from_mouse(event.position().x())

    def mouseReleaseEvent(self, event):
        self.pressed_handle = None

    def _update_from_mouse(self, x):
        val = self._pos_to_value(x)
        if self.pressed_handle == 'low':
            if val > self.high:
                self.low = self.high
                self.high = val
                self.pressed_handle = 'high'
            else:
                self.low = val
        elif self.pressed_handle == 'high':
            if val < self.low:
                self.high = self.low
                self.low = val
                self.pressed_handle = 'low'
            else:
                self.high = val
        self.update()
        self.valuesChanged.emit(self.low, self.high)
    
    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.update()
