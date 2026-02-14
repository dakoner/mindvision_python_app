import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGroupBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Slot

from cnc_control_panel import CNCControlPanel


class CncTestWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CNC Control Panel Test")

        # Load the mainwindow.ui to get the CNCControlPanel widget
        loader = QUiLoader()
        script_dir = os.path.dirname(__file__)
        ui_file_path = os.path.join(script_dir, "mainwindow.ui")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Cannot open {ui_file_path}: {ui_file.errorString()}")

        # The loader's load method returns the top-level widget from the UI file,
        # but doesn't set it as the central widget. We just need to find our groupbox.
        main_ui_widget = loader.load(ui_file)
        ui_file.close()

        if not main_ui_widget:
            raise RuntimeError(loader.errorString())

        cnc_group_box_widget = main_ui_widget.findChild(QGroupBox, "CNCControlPanel")
        if not cnc_group_box_widget:
            raise RuntimeError("Could not find QGroupBox 'CNCControlPanel' in UI file.")
        
        # The CNCControlPanel's parent is the widget from the UI file, which we don't show.
        # To make it part of our new window, we need to reparent it.
        cnc_group_box_widget.setParent(self)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(cnc_group_box_widget)
        layout.addWidget(self.log_text_edit)
        self.setCentralWidget(central_widget)

        self.cnc_control_panel = CNCControlPanel(cnc_group_box_widget, self)
        self.cnc_control_panel.log_signal.connect(self.log)
        
        self.resize(400, 600)

    @Slot(str)
    def log(self, message):
        self.log_text_edit.append(message)

    def closeEvent(self, event):
        self.cnc_control_panel.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = CncTestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
