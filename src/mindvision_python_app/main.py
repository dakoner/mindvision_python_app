import signal
import sys
from PySide6.QtWidgets import QApplication

from .mainwindow import MainWindow

def main():
    # Enable Ctrl+C termination
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
