from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Signal

class ColorPickerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_coords = None
        self.lbl_rgb = None
        self.lbl_hex = None
        self.lbl_intensity = None
        self.lbl_preview = None

    def setup_ui(self):
        self.lbl_coords = self.findChild(QLabel, "lbl_cp_coords")
        self.lbl_rgb = self.findChild(QLabel, "lbl_cp_rgb")
        self.lbl_hex = self.findChild(QLabel, "lbl_cp_hex")
        self.lbl_intensity = self.findChild(QLabel, "lbl_cp_intensity")
        self.lbl_preview = self.findChild(QLabel, "lbl_cp_color_preview")

    def update_color(self, x, y, r, g, b):
        # Lazy initialization to ensure children are found after loading
        if self.lbl_coords is None:
            self.setup_ui()

        if self.lbl_coords:
            self.lbl_coords.setText(f"{x}, {y}")
        
        if self.lbl_rgb:
            self.lbl_rgb.setText(f"{r}, {g}, {b}")
            
        hex_val = f"#{r:02x}{g:02x}{b:02x}"
        if self.lbl_hex:
            self.lbl_hex.setText(hex_val.upper())
            
        intensity = int((r + g + b) / 3)
        if self.lbl_intensity:
            self.lbl_intensity.setText(str(intensity))
            
        if self.lbl_preview:
            self.lbl_preview.setStyleSheet(f"background-color: {hex_val}; border: 1px solid gray;")
