from PySide6 import QtWidgets, QtCore, QtGui

class MosaicWindow(QtWidgets.QWidget):
    def __init__(self, width_mm, height_mm, calibration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stitching Mosaic")

        if not calibration or calibration <= 0:
            raise ValueError("Invalid calibration value provided.")

        self.calibration = calibration  # px/mm
        self.width_px = int(width_mm * self.calibration)
        self.height_px = int(height_mm * self.calibration)

        # Create the main canvas/image
        self.mosaic_image = QtGui.QImage(
            self.width_px, self.height_px, QtGui.QImage.Format.Format_RGB32
        )
        self.mosaic_image.fill(QtCore.Qt.GlobalColor.lightGray)

        # UI Elements
        self.image_label = QtWidgets.QLabel()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(self.mosaic_image))
        self.image_label.setMinimumSize(self.width_px, self.height_px)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        self.resize(800, 600)  # Set a reasonable initial size for the window

    @QtCore.Slot(QtGui.QImage, float, float)
    def paste_frame(self, frame_image, x_mm, y_mm):
        """
        Pastes a given QImage at the specified millimeter coordinates.

        Args:
            frame_image (QtGui.QImage): The image to paste.
            x_mm (float): The X coordinate in millimeters.
            y_mm (float): The Y coordinate in millimeters.
        """
        if frame_image.isNull():
            return

        # Calculate top-left pixel position for pasting
        # We assume (0,0) mm is the top-left corner of the mosaic
        px_x = int(x_mm * self.calibration)
        px_y = int(y_mm * self.calibration)

        # Draw the new frame onto our main mosaic image
        painter = QtGui.QPainter(self.mosaic_image)
        # For now, let's anchor the paste by the center of the frame
        paste_x = px_x - frame_image.width() // 2
        paste_y = px_y - frame_image.height() // 2
        
        painter.drawImage(paste_x, paste_y, frame_image)
        painter.end()

        # Update the display
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(self.mosaic_image))

    def closeEvent(self, event):
        # Prevent this window from being destroyed on close, just hide it.
        # The owner (MainWindow) will manage its lifecycle.
        self.hide()
        event.ignore()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # Example usage: 140x120 mm with a calibration of 100 px/mm
    calib = 100.0
    main_win = MosaicWindow(width_mm=140, height_mm=120, calibration=calib)
    
    # Example paste
    test_frame = QtGui.QImage(100, 80, QtGui.QImage.Format.Format_RGB32)
    test_frame.fill(QtCore.Qt.GlobalColor.blue)
    main_win.paste_frame(test_frame, x_mm=70, y_mm=60) # Paste in the center
    
    main_win.show()
    sys.exit(app.exec())
