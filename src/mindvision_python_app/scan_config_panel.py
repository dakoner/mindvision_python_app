
import PySide6.QtWidgets
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QLabel,
    QProgressBar,
    QDialogButtonBox,
)
from PySide6.QtCore import Signal, Slot

class ScanConfigPanel(QWidget):
    """
    A dialog to configure and monitor a mosaic scan.
    """

    # Signal arguments: x_min, y_min, x_max, y_max, home_x, home_y, record_video
    start_scan_signal = Signal(float, float, float, float, bool, bool, bool)
    cancel_scan_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # No extra margins
        layout.setSpacing(0)

        # Scan Area Info
        self.scan_area_label = QLabel("Scan Area: (select area on mosaic)")
        layout.addWidget(self.scan_area_label)
        
        # Homing Options
        self.home_x_checkbox = QCheckBox("Home X before each row")
        self.home_y_checkbox = QCheckBox("Home Y before each row")
        self.home_x_checkbox.setChecked(True)
        layout.addWidget(self.home_x_checkbox)
        layout.addWidget(self.home_y_checkbox)

        # Video recording option
        self.record_video_checkbox = QCheckBox("Take videos of each strip")
        self.record_video_checkbox.setChecked(False)
        layout.addWidget(self.record_video_checkbox)

        # Status Display
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Scan")
        self.cancel_button = QPushButton("Cancel Scan")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(False) # Disabled until an area is selected

        # Connections
        self.start_button.clicked.connect(self.on_start_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        
    def update_scan_area(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.scan_area_label.setText(f"Scan Area: X({x_min:.2f} to {x_max:.2f}), Y({y_min:.2f} to {y_max:.2f})")
        self.start_button.setEnabled(True)
        self.scan_finished(success=False) # Reset state


    def on_start_clicked(self):
        self.start_button.setEnabled(False)
        self.home_x_checkbox.setEnabled(False)
        self.home_y_checkbox.setEnabled(False)
        self.record_video_checkbox.setEnabled(False)
        self.cancel_button.setEnabled(True)

        home_x = self.home_x_checkbox.isChecked()
        home_y = self.home_y_checkbox.isChecked()
        record_video = self.record_video_checkbox.isChecked()

        self.start_scan_signal.emit(self.x_min, self.y_min, self.x_max, self.y_max, home_x, home_y, record_video)
        self.update_status("Scan started...")

    def on_cancel_clicked(self):
        self.cancel_scan_signal.emit()
        self.update_status("Scan cancelled by user.")
        self.scan_finished(success=False)

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(int, int)
    def update_progress(self, current_row, total_rows):
        if total_rows > 0:
            progress = int((current_row / total_rows) * 100)
            self.progress_bar.setValue(progress)
            self.update_status(f"Scanning row {current_row} of {total_rows}")

    @Slot(bool)
    def scan_finished(self, success=True):
        self.start_button.setEnabled(True)
        self.home_x_checkbox.setEnabled(True)
        self.home_y_checkbox.setEnabled(True)
        self.record_video_checkbox.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        if success:
             self.update_status("Scan completed successfully!")
             self.progress_bar.setValue(100)
        else:
             self.update_status("Idle")
             self.progress_bar.setValue(0)

