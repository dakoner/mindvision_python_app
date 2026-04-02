import serial
import serial.tools.list_ports
from PySide6.QtCore import QObject, Signal, Slot, QMutex
from .utils import QMutexLocker

try:
    import serial
    import serial.tools.list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("Warning: pyserial not installed. Serial features disabled.")


def _format_hex_bytes(data):
    return " ".join(f"{byte:02x}" for byte in data)


def _pop_complete_serial_lines(buffer):
    lines = []
    while True:
        newline_positions = [index for index in (buffer.find(b"\n"), buffer.find(b"\r")) if index != -1]
        if not newline_positions:
            break

        split_index = min(newline_positions)
        lines.append(bytes(buffer[:split_index]))
        del buffer[: split_index + 1]

        while buffer[:1] in (b"\n", b"\r"):
            del buffer[:1]

    return lines


def _sanitize_serial_line(raw_line):
    if not raw_line:
        return "", b""

    cleaned = bytearray()
    dropped = bytearray()

    for byte in raw_line:
        if byte == 0x09 or 0x20 <= byte <= 0x7E:
            cleaned.append(byte)
        else:
            dropped.append(byte)

    return cleaned.decode("ascii", errors="ignore").strip(), bytes(dropped)

class SerialWorker(QObject):
    log_signal = Signal(str)
    connection_status = Signal(bool)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.mutex = QMutex()
        self._rx_buffer = bytearray()
        self._rx_buffer_limit = 8192

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
            self._rx_buffer.clear()
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

    @Slot(str)
    def send_raw_command(self, cmd):
        with QMutexLocker(self.mutex):
            if not self.is_connected or not self.serial_port:
                self.log_signal.emit("Error: Not connected to serial port.")
                return

            try:
                self.serial_port.write(cmd.encode("utf-8"))
                # Do not log raw commands to avoid spamming the log with '?'
                # self.log_signal.emit(f"Tx (raw): {cmd}")
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
                    raw = self.serial_port.read(self.serial_port.in_waiting)
                    if not raw:
                        return

                    self._rx_buffer.extend(raw)

                    if len(self._rx_buffer) > self._rx_buffer_limit:
                        dropped = bytes(self._rx_buffer[:-self._rx_buffer_limit])
                        del self._rx_buffer[:-self._rx_buffer_limit]
                        self.log_signal.emit(
                            f"Rx dropped oversized buffered bytes: {_format_hex_bytes(dropped)}"
                        )

                    for raw_line in _pop_complete_serial_lines(self._rx_buffer):
                        line, dropped = _sanitize_serial_line(raw_line)
                        if dropped:
                            self.log_signal.emit(
                                f"Rx filtered bytes: {_format_hex_bytes(dropped)}"
                            )
                        if line:
                            self.log_signal.emit(f"Rx: {line}")
                except Exception as e:
                    self.log_signal.emit(f"Read error: {e}")
