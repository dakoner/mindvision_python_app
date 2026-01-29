from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QThread, Signal, Slot
import serial.tools.list_ports

from serial_worker import SerialWorker, HAS_SERIAL

class CNCControlPanel(QtWidgets.QDockWidget):
    log_signal = Signal(str)
    connect_serial_signal = Signal(str, int)
    disconnect_serial_signal = Signal()
    send_serial_cmd_signal = Signal(str)
    poll_serial_signal = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__("CNC Control", *args, **kwargs)
        
        self.step_size = 0.1

        # --- Serial Worker Setup ---
        self.serial_thread = QThread()
        self.serial_worker = SerialWorker()
        self.serial_worker.moveToThread(self.serial_thread)

        # Connect signals
        self.connect_serial_signal.connect(self.serial_worker.connect_serial)
        self.disconnect_serial_signal.connect(self.serial_worker.disconnect_serial)
        self.send_serial_cmd_signal.connect(self.serial_worker.send_command)
        self.poll_serial_signal.connect(self.serial_worker.poll_serial)
        
        self.serial_worker.log_signal.connect(self.on_log_message)
        self.serial_worker.connection_status.connect(self.on_serial_status_changed)

        self.serial_thread.start()

        # Timer for polling serial read
        self.serial_poll_timer = QtCore.QTimer()
        self.serial_poll_timer.timeout.connect(self.poll_serial_signal.emit)
        self.serial_poll_timer.start(50) # Poll every 50ms

        # Create a widget to hold the layout
        self.control_widget = QtWidgets.QWidget()
        self.setWidget(self.control_widget)
        
        # Create a grid layout
        self.grid_layout = QtWidgets.QGridLayout()
        self.control_widget.setLayout(self.grid_layout)
        self.grid_layout.setSpacing(5)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)

        # Serial connection UI
        self.serial_port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setCheckable(True)

        self.grid_layout.addWidget(self.serial_port_combo, 0, 0, 1, 2)
        self.grid_layout.addWidget(self.refresh_button, 0, 2, 1, 1)
        self.grid_layout.addWidget(self.connect_button, 0, 3, 1, 1)

        # Create buttons
        self.forward_button = QtWidgets.QPushButton("Y+")
        self.back_button = QtWidgets.QPushButton("Y-")
        self.left_button = QtWidgets.QPushButton("X-")
        self.right_button = QtWidgets.QPushButton("X+")
        self.up_button = QtWidgets.QPushButton("Z+")
        self.down_button = QtWidgets.QPushButton("Z-")
        self.home_button = QtWidgets.QPushButton("Home")

        # Add buttons to the layout
        self.grid_layout.addWidget(self.forward_button, 1, 1) # Y+
        self.grid_layout.addWidget(self.left_button, 2, 0) # X-
        self.grid_layout.addWidget(self.right_button, 2, 2) # X+
        self.grid_layout.addWidget(self.back_button, 3, 1) # Y-
        
        self.grid_layout.addWidget(self.up_button, 1, 3) # Z+
        self.grid_layout.addWidget(self.down_button, 2, 3) # Z-
        
        self.grid_layout.addWidget(self.home_button, 3, 3)

        # Step size controls
        self.step_label = QtWidgets.QLabel("Step (mm):")
        self.step_input = QtWidgets.QDoubleSpinBox()
        self.step_input.setDecimals(3)
        self.step_input.setSingleStep(0.1)
        self.step_input.setValue(self.step_size)
        
        self.grid_layout.addWidget(self.step_label, 4, 0, 1, 1)
        self.grid_layout.addWidget(self.step_input, 4, 1, 1, 1)

        # Set size policies for tighter layout
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)

        # Connect signals
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

        self.refresh_serial_ports()
        self.on_serial_status_changed(False) # Initial state

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
        for button in [self.up_button, self.down_button, self.left_button, self.right_button, self.forward_button, self.back_button, self.home_button]:
            button.setEnabled(connected)

    @Slot(str)
    def on_log_message(self, msg):
        self.log_signal.emit(msg)

    def on_step_size_changed(self, value):
        self.step_size = value
    
    def move_up(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Z{self.step_size} F4000")

    def move_down(self):
        self.send_serial_cmd_signal.emit(f"$J=G91 Z-{self.step_size} F4000")

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

    def closeEvent(self, event):
        self.serial_thread.quit()
        self.serial_thread.wait()
        super().closeEvent(event)