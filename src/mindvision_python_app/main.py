import sys
import os
import signal
from PySide6.QtWidgets import QApplication

# Setup paths before importing MainWindow
script_dir = os.path.dirname(__file__)
release_dir = os.path.realpath(os.path.join(script_dir, "..", "..", "..", "mindvision_qobject", "release"))
print(release_dir)
sys.path.insert(0, release_dir)



from mainwindow import MainWindow

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
