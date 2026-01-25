import serial
import serial.tools.list_ports
from PySide6.QtCore import QObject, Signal, Slot, QMutex
from utils import QMutexLocker

try:
    import serial
    import serial.tools.list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("Warning: pyserial not installed. Serial features disabled.")

class SerialWorker(QObject):
    log_signal = Signal(str)
    connection_status = Signal(bool)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.mutex = QMutex()

    @Slot(str, int)
    def connect_serial(self, port_name, baud_rate=115200):
        if not HAS_SERIAL:
            self.log_signal.emit("Error: pyserial not installed.")
            return

        with QMutexLocker(self.mutex):
            if self.is_connected:
                self.disconnect_serial()

            try:
                self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
                self.is_connected = True
                self.connection_status.emit(True)
                self.log_signal.emit(f"Connected to {port_name} at {baud_rate}")
            except Exception as e:
                self.log_signal.emit(f"Failed to connect to {port_name}: {e}")
                self.connection_status.emit(False)

    @Slot()
    def disconnect_serial(self):
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.close()
                except Exception as e:
                    self.log_signal.emit(f"Error closing port: {e}")
            self.serial_port = None
            self.is_connected = False
            self.connection_status.emit(False)
            self.log_signal.emit("Disconnected.")

    @Slot(str)
    def send_command(self, cmd):
        with QMutexLocker(self.mutex):
            if not self.is_connected or not self.serial_port:
                self.log_signal.emit("Error: Not connected to serial port.")
                return

            try:
                full_cmd = cmd.strip() + "\n"
                self.serial_port.write(full_cmd.encode("utf-8"))
                self.log_signal.emit(f"Tx: {cmd}")
            except Exception as e:
                self.log_signal.emit(f"Send error: {e}")
                self.disconnect_serial()

    @Slot()
    def read_loop(self):
        # This is intended to be called periodically or run in a loop if we had a dedicated thread loop
        # For simplicity with QThread/moveToThread, we can use a QTimer in the worker or just rely on manual reads
        # But a robust way is a loop. However, since we are using standard QThread, let's just expose a read method
        # and let the main thread trigger it or use a timer in this worker.
        # BETTER: Use a timer inside this worker to poll.
        pass

    @Slot()
    def poll_serial(self):
        # Called by a timer in the worker thread
        with QMutexLocker(self.mutex):
            if self.is_connected and self.serial_port and self.serial_port.in_waiting:
                try:
                    # Read up to 10 lines to catch up but not freeze UI
                    for _ in range(10):
                        if self.serial_port.in_waiting:
                            line = (
                                self.serial_port.readline()
                                .decode("utf-8", errors="replace")
                                .strip()
                            )
                            if line:
                                self.log_signal.emit(f"Rx: {line}")
                        else:
                            break
                except Exception as e:
                    self.log_signal.emit(f"Read error: {e}")
