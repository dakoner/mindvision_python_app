import sys
import os
import signal
from PySide6.QtWidgets import QApplication

# Setup paths before importing MainWindow
script_dir = os.path.dirname(__file__)
release_dir = os.path.join(r"/home/dek/src/microtools/mindvision_qobject", "release")
sys.path.insert(0, release_dir)

from mainwindow import MainWindow

def main():
    # Enable Ctrl+C termination
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
