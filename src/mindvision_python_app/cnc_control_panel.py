from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QThread, Signal, Slot, QObject
import serial.tools.list_ports
import re

from serial_worker import SerialWorker, HAS_SERIAL


class CNCControlPanel(QObject): # Changed base class from QGroupBox to QObject
    log_signal = Signal(str)
    connect_serial_signal = Signal(str, int)
    disconnect_serial_signal = Signal()
    send_serial_cmd_signal = Signal(str)
    send_raw_serial_cmd_signal = Signal(str)
    poll_serial_signal = Signal()

    def __init__(self, cnc_group_box_widget: QtWidgets.QGroupBox, parent=None):
        super().__init__(parent)

        # The UI for the CNCControlPanel QGroupBox is now loaded as part of mainwindow.ui.
        # We are passed the QGroupBox widget instance from mainwindow.py.
        self._cnc_group_box_widget = cnc_group_box_widget

        # Now, find the children widgets by their names within the passed QGroupBox
        self.serial_port_combo = self._cnc_group_box_widget.findChild(QtWidgets.QComboBox, "serial_port_combo")
        self.refresh_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "refresh_button")
        self.connect_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "connect_button")
        self.forward_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "forward_button")
        self.back_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "back_button")
        self.left_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "left_button")
        self.right_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "right_button")
        self.up_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "up_button")
        self.down_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "down_button")
        self.home_button = self._cnc_group_box_widget.findChild(QtWidgets.QPushButton, "home_button")
        self.step_input = self._cnc_group_box_widget.findChild(QtWidgets.QDoubleSpinBox, "step_input")
        self.z_step_input = self._cnc_group_box_widget.findChild(QtWidgets.QDoubleSpinBox, "z_step_input")
        self.status_label = self._cnc_group_box_widget.findChild(QtWidgets.QLabel, "status_label")
        self.wpos_x_label = self._cnc_group_box_widget.findChild(QtWidgets.QLabel, "wpos_x_label")
        self.wpos_y_label = self._cnc_group_box_widget.findChild(QtWidgets.QLabel, "wpos_y_label")
        self.wpos_z_label = self._cnc_group_box_widget.findChild(QtWidgets.QLabel, "wpos_z_label")

        self.step_size = 0.1
        self.z_step_size = 0.01
        self.step_input.setValue(self.step_size)
        self.z_step_input.setValue(self.z_step_size)

        # --- Serial Worker Setup ---
        self.serial_thread = QThread()
        self.serial_worker = SerialWorker()
        self.serial_worker.moveToThread(self.serial_thread)

        # Connect signals
        self.connect_serial_signal.connect(self.serial_worker.connect_serial)
        self.disconnect_serial_signal.connect(self.serial_worker.disconnect_serial)
        self.send_serial_cmd_signal.connect(self.serial_worker.send_command)
        self.send_raw_serial_cmd_signal.connect(self.serial_worker.send_raw_command)
        self.poll_serial_signal.connect(self.serial_worker.poll_serial)

        self.serial_worker.log_signal.connect(self.on_log_message)
        self.serial_worker.connection_status.connect(self.on_serial_status_changed)

        self.serial_thread.start()

        # Timer for polling serial read
        self.serial_poll_timer = QtCore.QTimer()
        self.serial_poll_timer.timeout.connect(self.poll_serial_signal.emit)
        self.serial_poll_timer.start(50)  # Poll every 50ms

        # Timer for polling CNC status
        self.status_poll_timer = QtCore.QTimer(self)
        self.status_poll_timer.timeout.connect(self.poll_status)

        # Connect signals from UI widgets
        self.refresh_button.clicked.connect(self.refresh_serial_ports)
        self.connect_button.toggled.connect(self.on_connect_toggled)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)
        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.forward_button.clicked.connect(self.move_forward)
        self.back_button.clicked.connect(self.move_back)
        self.home_button.clicked.connect(self.home)
        self.step_input.valueChanged.connect(self.on_step_size_changed)
        self.z_step_input.valueChanged.connect(self.on_z_step_size_changed)

        self.refresh_serial_ports()
        self.on_serial_status_changed(False)  # Initial state

    def refresh_serial_ports(self):
        self.serial_port_combo.clear()
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.serial_port_combo.addItem(f"{port.device}")
        else:
            self.serial_port_combo.addItem("No pyserial")
            self.connect_button.setEnabled(False)

    def on_connect_toggled(self, checked):
        if checked:
            port = self.serial_port_combo.currentText()
            if port and port != "No pyserial":
                self.connect_serial_signal.emit(port, 115200)
            else:
                self.connect_button.setChecked(False)
        else:
            self.disconnect_serial_signal.emit()

    @Slot(bool)
    def on_serial_status_changed(self, connected):
        self.connect_button.setChecked(connected)
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.serial_port_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)

        # Enable/disable movement buttons
        for button in [
            self.up_button,
            self.down_button,
            self.left_button,
            self.right_button,
            self.forward_button,
            self.back_button,
            self.home_button,
        ]:
            button.setEnabled(connected)
            
        if connected:
            self.status_poll_timer.start(250) # Poll every 250ms
        else:
            self.status_poll_timer.stop()
            # Reset status on disconnect
            self.status_label.setText("N/A")
            self.wpos_x_label.setText("0.000")
            self.wpos_y_label.setText("0.000")
            self.wpos_z_label.setText("0.000")


    @Slot(str)
    def on_log_message(self, msg):
        # Intercept status messages for internal handling
        if msg.startswith("Rx: <") and msg.endswith(">"):
            self._parse_status(msg[4:-1])
        else:
            # Pass all other messages through to the main window log
            self.log_signal.emit(msg)

    def _parse_status(self, status_str):
        # State is always the first part before a pipe or the end
        parts = status_str.split("|")
        if parts:
            self.status_label.setText(parts[0])

        # Use regex for a more robust search for WPos or MPos
        match = re.search(r"(?:WPos|MPos):(-?[\d\.]+),(-?[\d\.]+),(-?[\d\.]+)", status_str)
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                self.wpos_x_label.setText(f"{x:.3f}")
                self.wpos_y_label.setText(f"{y:.3f}")
                self.wpos_z_label.setText(f"{z:.3f}")
            except (ValueError, IndexError):
                # This might happen if the regex matches something that isn't a valid float
                pass
        # If no match, the labels are simply not updated.

    def poll_status(self):
        self.send_raw_serial_cmd_signal.emit("?")

    def on_step_size_changed(self, value):
        self.step_size = value

    def on_z_step_size_changed(self, value):
        self.z_step_size = value

    def move_up(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Z{self.z_step_size} F4000")

    def move_down(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Z-{self.z_step_size} F4000")

    def move_left(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 X-{self.step_size} F4000")

    def move_right(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 X{self.step_size} F4000")

    def move_forward(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Y{self.step_size} F4000")

    def move_back(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Y-{self.step_size} F4000")

    def home(self):
        self.send_serial_cmd_signal.emit(f"$H")

    def stop(self):
        self.status_poll_timer.stop()
        if self.serial_thread.isRunning():
            self.serial_thread.quit()
            self.serial_thread.wait()