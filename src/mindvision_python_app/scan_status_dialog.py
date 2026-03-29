import os
from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QProgressBar, QLabel
from PySide6.QtCore import QFile, Slot, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader

class ScanStatusDialog(QDialog):
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Load UI
        loader = QUiLoader()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(script_dir, "ScanStatusDialog.ui")
        ui_file = QFile(ui_file_path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        if self.ui is None:
            print(f"Error loading UI: {loader.errorString()}")
            return
            
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ui)
        
        self.btn_cancel = self.ui.findChild(QPushButton, "btn_cancel")
        self.progressBar = self.ui.findChild(QProgressBar, "progressBar")
        self.label_status = self.ui.findChild(QLabel, "label_status")
        self.label_image = self.ui.findChild(QLabel, "label_image")
        
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        
    @Slot(int, int)
    def update_progress(self, current, total):
        if total > 0:
            self.progressBar.setMaximum(total)
            self.progressBar.setValue(current)
            percent = int((current / total) * 100)
            self.label_status.setText(f"Scanning... {percent}% ({current}/{total} rows)")

    @Slot(QImage)
    def update_image(self, image):
        from PySide6.QtCore import Qt
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            # Scale pixmap to fit label while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.label_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_image.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update image scaling when dialog resizes if there's a pixmap
        if self.label_image.pixmap() and not self.label_image.pixmap().isNull():
            # Actually, since we only scale it when set, we might need to store the original image to rescale it properly.
            pass
