
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush
from PySide6.QtCore import Qt, QPointF

class IntensityChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intensity Profile")
        self.resize(400, 300)
        self.data = []
        self.setAttribute(Qt.WA_DeleteOnClose, False) # Don't destroy on close, just hide

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if not self.data:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Data")
            return

        w = self.width()
        h = self.height()
        margin = 20
        
        # Draw Axes
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawLine(margin, h - margin, w - margin, h - margin) # X Axis
        painter.drawLine(margin, margin, margin, h - margin) # Y Axis
        
        # Plot Data
        # Y axis is 0-255 intensity
        # X axis is index along line
        
        max_val = 255
        count = len(self.data)
        if count < 2:
            return

        step_x = (w - 2 * margin) / (count - 1)
        scale_y = (h - 2 * margin) / max_val
        
        # Draw Grid (optional)
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.DotLine))
        painter.drawLine(margin, h - margin - (128 * scale_y), w - margin, h - margin - (128 * scale_y))
        
        # Draw Line
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        
        points = []
        for i, val in enumerate(self.data):
            x = margin + i * step_x
            y = h - margin - val * scale_y
            points.append(QPointF(x, y))
            
        painter.drawPolyline(points)
        
        # Min/Max labels
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(5, h - margin + 15, "0")
        painter.drawText(5, margin + 10, "255")
        
        # Current Value (if hovering? for now just static)
