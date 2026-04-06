import sys
import os

# Add the native release folder to sys.path so Python can find _serial_qobject_py.so / .pyd
current_dir = os.path.dirname(__file__)
native_dir = os.path.realpath(os.path.join(current_dir, "..", "..", "native", "serial_qobject", "release"))
if native_dir not in sys.path:
    sys.path.append(native_dir)

from PySide6.QtCore import QCoreApplication, QTimer
# Pre-load the PySide6 bundled QtSerialPort library to prevent conflicts with the system Qt libraries
from PySide6.QtSerialPort import QSerialPort

try:
    import _serial_qobject_py
except ImportError as e:
    print(f"Failed to import _serial_qobject_py: {e}")
    print(f"Make sure you have built the native module in {native_dir}")
    sys.exit(1)

def main():
    # A Qt application instance is strictly required to pump the QSerialPort events
    app = QCoreApplication(sys.argv)
    
    worker = _serial_qobject_py.SerialWorker()
    
    def log_cb(msg):
        print(f"[Worker Log] {msg}")
        
    def status_cb(connected):
        print(f"[Worker Status] Connected: {connected}")
        
    worker.register_log_callback(log_cb)
    worker.register_status_callback(status_cb)
    
    # Allow overriding via command line, default to common COM/tty paths
    port = sys.argv[1] if len(sys.argv) > 1 else ("COM3" if os.name == 'nt' else "/dev/ttyUSB1")
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    print(f"Connecting to {port} at {baud} baud...")
    worker.connect_serial(port, baud)
    
    def send_test_cmd():
        print("Sending test command '?\\n' ...")
        worker.send_command("?")
        
    # Delay sending the test command slightly to let connection initialize completely
    QTimer.singleShot(1500, send_test_cmd)
    
    # Automatically quit the application after 5 seconds to conclude the test
    QTimer.singleShot(5000, app.quit)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
