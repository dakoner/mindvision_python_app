
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
    QRadioButton,
    QButtonGroup,
)
from PySide6.QtCore import Signal, Slot

class ScanConfigPanel(QWidget):
    """
    A dialog to configure and monitor a mosaic scan.
    """
    # Signal arguments: areas_list, home_x, home_y, is_serpentine
    start_scan_signal = Signal(list, bool, bool, bool)
    cancel_scan_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.num_selected_areas = 0
        self.scan_areas = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # No extra margins
        layout.setSpacing(0)

        # Scan Area Info
        self.scan_area_label = QLabel("Scan Area: (select area on mosaic)")
        layout.addWidget(self.scan_area_label) # type: ignore
        
        # Scan Style Options
        scan_style_layout = QHBoxLayout()
        self.radio_left_right = QRadioButton("Left-Right")
        self.radio_serpentine = QRadioButton("Serpentine")
        self.radio_serpentine.setChecked(True)
        self.scan_style_group = QButtonGroup()
        self.scan_style_group.addButton(self.radio_left_right)
        self.scan_style_group.addButton(self.radio_serpentine)
        scan_style_layout.addWidget(self.radio_left_right)
        scan_style_layout.addWidget(self.radio_serpentine)
        layout.addLayout(scan_style_layout) # type: ignore

        # Homing Options
        self.home_x_checkbox = QCheckBox("Home X before each row")
        self.home_y_checkbox = QCheckBox("Home Y before each row")
        self.home_x_checkbox.setChecked(True)
        layout.addWidget(self.home_x_checkbox) # type: ignore
        layout.addWidget(self.home_y_checkbox) # type: ignore

        # Status Display
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar) # type: ignore
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Scan")
        self.cancel_button = QPushButton("Cancel Scan")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button) # type: ignore
        layout.addLayout(button_layout) # type: ignore

        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(False) # Disabled until an area is selected

        # Connections
        self.start_button.clicked.connect(self.on_start_clicked)
        self.cancel_button.clicked.connect(self.cancel_scan_signal.emit)

    def update_scan_areas(self, areas):
        self.scan_areas = areas
        self.num_selected_areas = len(areas)
        self.scan_area_label.setText(f"Scan Area: {self.num_selected_areas} area(s) selected")
        self.start_button.setEnabled(self.num_selected_areas > 0)
        self.scan_finished(success=False) # Reset UI state

    def on_start_clicked(self):
        self.start_button.setEnabled(False)
        self.home_x_checkbox.setEnabled(False)
        self.home_y_checkbox.setEnabled(False)
        self.radio_left_right.setEnabled(False)
        self.radio_serpentine.setEnabled(False)
        self.cancel_button.setEnabled(True)

        home_x = self.home_x_checkbox.isChecked()
        home_y = self.home_y_checkbox.isChecked()
        is_serpentine = self.radio_serpentine.isChecked()

        self.start_scan_signal.emit(
            self.scan_areas,
            home_x, home_y, is_serpentine
        )
        self.update_status("Scan started...")

    def set_scanning_active(self, active: bool):
        self.start_button.setEnabled(not active and self.num_selected_areas > 0)
        self.cancel_button.setEnabled(active)

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
        self.radio_left_right.setEnabled(True)
        self.radio_serpentine.setEnabled(True)
        self.set_scanning_active(False) # Update buttons based on active state
        
        if success:
             self.update_status("Scan completed successfully!")
             self.progress_bar.setValue(100)
        else:
             self.update_status("Idle")
             self.progress_bar.setValue(0)
