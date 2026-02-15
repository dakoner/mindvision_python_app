import serial
import serial.tools.list_ports
from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer
from PySide6.QtWidgets import QListWidgetItem
from serial_worker import SerialWorker, HAS_SERIAL

class LEDController(QObject):
    log_signal = Signal(str)

    # Serial Signals
    connect_serial_signal = Signal(str, int)
    disconnect_serial_signal = Signal()
    send_serial_cmd_signal = Signal(str)
    poll_serial_signal = Signal()

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        
        self.has_initialized_settings = False
        self.status_items_map = {}  # Map pin -> QListWidgetItem
        self.interrupt_items_map = {}  # Map pin -> QListWidgetItem
        self.pin_buttons = {}

        # --- Serial Worker Setup ---
        self.serial_thread = QThread()
        self.serial_worker = SerialWorker()
        self.serial_worker.moveToThread(self.serial_thread)

        # Connect signals
        self.connect_serial_signal.connect(self.serial_worker.connect_serial)
        self.disconnect_serial_signal.connect(self.serial_worker.disconnect_serial)
        self.send_serial_cmd_signal.connect(self.serial_worker.send_command)
        self.poll_serial_signal.connect(self.serial_worker.poll_serial)

        self.serial_worker.log_signal.connect(self.on_serial_log)
        self.serial_worker.connection_status.connect(self.on_serial_status_changed)

        self.serial_thread.start()

        # Timer for polling serial read
        self.serial_poll_timer = QTimer()
        self.serial_poll_timer.timeout.connect(lambda: self.poll_serial_signal.emit())
        self.serial_poll_timer.start(50) # Poll every 50ms

        # Serial UI Init
        self.refresh_serial_ports()
        self.ui.btn_serial_refresh.clicked.connect(self.refresh_serial_ports)
        self.ui.btn_serial_connect.clicked.connect(self.on_btn_serial_connect_clicked)
        self.ui.btn_cmd_pulse.clicked.connect(self.on_cmd_pulse)
        # self.ui.btn_cmd_level connection removed
        self.ui.btn_cmd_pwm.clicked.connect(self.on_cmd_pwm)
        self.ui.btn_cmd_stoppwm.clicked.connect(self.on_cmd_stoppwm)
        self.ui.btn_cmd_repeat.clicked.connect(self.on_cmd_repeat)
        self.ui.btn_cmd_stoprepeat.clicked.connect(self.on_cmd_stoprepeat)
        self.ui.btn_cmd_interrupt.clicked.connect(self.on_cmd_interrupt)
        self.ui.btn_cmd_stopinterrupt.clicked.connect(self.on_cmd_stopinterrupt)
        self.ui.btn_cmd_throb.clicked.connect(self.on_cmd_throb)
        self.ui.btn_cmd_stopthrob.clicked.connect(self.on_cmd_stopthrob)
        self.ui.btn_cmd_info.clicked.connect(
            lambda: self.send_serial_cmd_signal.emit("info")
        )
        self.ui.btn_cmd_wifi.clicked.connect(
            lambda: self.send_serial_cmd_signal.emit("wifi")
        )
        self.ui.btn_cmd_mem.clicked.connect(self.on_cmd_mem)

        # --- Setup Pin Level Buttons ---
        valid_pins = [4, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33]
        
        for pin in valid_pins:
            btn_name = f"btn_pin_{pin}"
            if hasattr(self.ui, btn_name):
                btn = getattr(self.ui, btn_name)
                btn.clicked.connect(lambda checked=False, p=pin: self.toggle_pin_level(p))
                self.pin_buttons[pin] = btn

        # Disable serial tabs initially
        self.ui.tabs_serial_cmds.setEnabled(False)

    def stop(self):
        if self.serial_thread.isRunning():
            self.serial_thread.quit()
            self.serial_thread.wait()

    @Slot(str)
    def on_serial_log(self, message):
        self.log_signal.emit(message)
        # Parse Rx messages
        if message.startswith("Rx: "):
            content = message[4:].strip()
            self.process_serial_line(content)

    def process_serial_line(self, line):
        # Startup detection
        if "LED>" in line and not self.has_initialized_settings:
            self.has_initialized_settings = True
            QTimer.singleShot(
                500, lambda: self.send_serial_cmd_signal.emit("printsettings")
            )
            return

        parts = line.split()
        if not parts:
            return

        cmd = parts[0]

        try:
            if cmd == "level" and len(parts) >= 3:
                pin = int(parts[1])
                val = int(parts[2])
                
                # If the pin was in the status list (e.g. PWM/Repeat), remove it as it's now just a simple level
                if pin in self.status_items_map:
                    item = self.status_items_map.pop(pin)
                    row = self.ui.list_status.row(item)
                    self.ui.list_status.takeItem(row)

                # Update Pin Button State
                if pin in self.pin_buttons:
                    btn = self.pin_buttons[pin]
                    if val == 1:
                        btn.setStyleSheet("background-color: red; color: white;")
                    else:
                        btn.setStyleSheet("background-color: none;")

                # Update old UI setters if they exist (legacy/fallback)
                if hasattr(self.ui, "spin_level_pin"):
                    self.ui.spin_level_pin.setValue(pin)
                if hasattr(self.ui, "combo_level_val"):
                    self.ui.combo_level_val.setCurrentIndex(val + 1)

            elif cmd == "pwm" and len(parts) >= 4:
                pin = int(parts[1])
                freq = int(parts[2])
                duty = int(parts[3])
                self.update_pin_status(pin, f"PWM: {freq}Hz, {duty}%")
                # Update UI setters
                self.ui.spin_pwm_pin.setValue(pin)
                self.ui.spin_pwm_freq.setValue(freq)
                self.ui.spin_pwm_duty.setValue(duty)

            elif cmd == "repeat" and len(parts) >= 4:
                pin = int(parts[1])
                freq = int(parts[2])
                dur = int(parts[3])
                self.update_pin_status(pin, f"Repeat: {freq}Hz, {dur}us")
                # Update UI setters
                self.ui.spin_repeat_pin.setValue(pin)
                self.ui.spin_repeat_freq.setValue(freq)
                self.ui.spin_repeat_dur.setValue(dur)

            elif cmd == "throb" and len(parts) >= 5:
                period = int(parts[1])
                p1 = int(parts[2])
                p2 = int(parts[3])
                p3 = int(parts[4])
                self.update_pin_status(p1, "Throb")
                self.update_pin_status(p2, "Throb")
                self.update_pin_status(p3, "Throb")
                # Update UI setters
                self.ui.spin_throb_period.setValue(period)
                self.ui.spin_throb_p1.setValue(p1)
                self.ui.spin_throb_p2.setValue(p2)
                self.ui.spin_throb_p3.setValue(p3)

            elif cmd == "interrupt" and len(parts) >= 5:
                pin = int(parts[1])
                edge = parts[2]
                tgt = int(parts[3])
                width = int(parts[4])
                self.update_interrupt_status(pin, f"{edge} -> Pulse {tgt} ({width}us)")
                # Update UI setters
                self.ui.spin_int_pin.setValue(pin)
                idx = self.ui.combo_int_edge.findText(edge)
                if idx >= 0:
                    self.ui.combo_int_edge.setCurrentIndex(idx)
                self.ui.spin_int_target.setValue(tgt)
                self.ui.spin_int_width.setValue(width)

        except ValueError:
            pass

    def update_pin_status(self, pin, status_text):
        text = f"Pin {pin}: {status_text}"
        if pin in self.status_items_map:
            self.status_items_map[pin].setText(text)
        else:
            item = QListWidgetItem(text)
            self.ui.list_status.addItem(item)
            self.status_items_map[pin] = item

    def update_interrupt_status(self, pin, status_text):
        text = f"Pin {pin}: {status_text}"
        if pin in self.interrupt_items_map:
            self.interrupt_items_map[pin].setText(text)
        else:
            item = QListWidgetItem(text)
            self.ui.list_interrupts.addItem(item)
            self.interrupt_items_map[pin] = item

    def refresh_serial_ports(self):
        self.ui.combo_serial_port.clear()
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.ui.combo_serial_port.addItem(f"{port.device}")
        else:
            self.ui.combo_serial_port.addItem("No pyserial")
            self.ui.btn_serial_connect.setEnabled(False)

    def on_btn_serial_connect_clicked(self, checked):
        if checked:
            port = self.ui.combo_serial_port.currentText().split()[0]
            if port:
                self.connect_serial_signal.emit(port, 115200)
                self.ui.btn_serial_connect.setText("Connecting...")
            else:
                self.ui.btn_serial_connect.setChecked(False)
        else:
            self.disconnect_serial_signal.emit()

    @Slot(bool)
    def on_serial_status_changed(self, connected):
        self.ui.btn_serial_connect.setChecked(connected)
        self.ui.btn_serial_connect.setText("Disconnect" if connected else "Connect")
        self.ui.tabs_serial_cmds.setEnabled(connected)
        self.ui.combo_serial_port.setEnabled(not connected)
        self.ui.btn_serial_refresh.setEnabled(not connected)

        if not connected:
            self.has_initialized_settings = False
            self.ui.list_status.clear()
            self.status_items_map.clear()
            self.ui.list_interrupts.clear()
            self.interrupt_items_map.clear()

    def on_cmd_pulse(self):
        pin = self.ui.spin_pulse_pin.value()
        val = self.ui.spin_pulse_val.value()
        dur = self.ui.spin_pulse_dur.value()
        self.send_serial_cmd_signal.emit(f"pulse {pin} {val} {dur}")

    def toggle_pin_level(self, pin):
        btn = self.pin_buttons.get(pin)
        if not btn:
            return

        # Check current visual state (High=Red)
        is_high = "red" in btn.styleSheet()
        # Toggle: If high, set to 0. If low, set to 1.
        new_val = 0 if is_high else 1

        self.send_serial_cmd_signal.emit(f"level {pin} {new_val}")

    def on_cmd_pwm(self):
        pin = self.ui.spin_pwm_pin.value()
        freq = self.ui.spin_pwm_freq.value()
        duty = self.ui.spin_pwm_duty.value()
        self.send_serial_cmd_signal.emit(f"pwm {pin} {freq} {duty}")

    def on_cmd_stoppwm(self):
        pin = self.ui.spin_pwm_pin.value()
        self.send_serial_cmd_signal.emit(f"stoppwm {pin}")

        # Remove from Modified Pins list
        if pin in self.status_items_map:
            item = self.status_items_map.pop(pin)
            row = self.ui.list_status.row(item)
            self.ui.list_status.takeItem(row)

        # Reset button state
        if pin in self.pin_buttons:
            self.pin_buttons[pin].setStyleSheet("background-color: none;")

    def on_cmd_repeat(self):
        pin = self.ui.spin_repeat_pin.value()
        freq = self.ui.spin_repeat_freq.value()
        dur = self.ui.spin_repeat_dur.value()
        self.send_serial_cmd_signal.emit(f"repeat {pin} {freq} {dur}")

    def on_cmd_stoprepeat(self):
        pin = self.ui.spin_repeat_pin.value()
        self.send_serial_cmd_signal.emit(f"stoprepeat {pin}")

        # Remove from Modified Pins list
        if pin in self.status_items_map:
            item = self.status_items_map.pop(pin)
            row = self.ui.list_status.row(item)
            self.ui.list_status.takeItem(row)

        # Reset button state
        if pin in self.pin_buttons:
            self.pin_buttons[pin].setStyleSheet("background-color: none;")

    def on_cmd_interrupt(self):
        pin = self.ui.spin_int_pin.value()
        edge = self.ui.combo_int_edge.currentText()
        tgt = self.ui.spin_int_target.value()
        width = self.ui.spin_int_width.value()
        self.send_serial_cmd_signal.emit(f"interrupt {pin} {edge} {tgt} {width}")

    def on_cmd_stopinterrupt(self):
        pin = self.ui.spin_int_pin.value()
        self.send_serial_cmd_signal.emit(f"stopinterrupt {pin}")

        # Remove from Interrupts list
        if pin in self.interrupt_items_map:
            item = self.interrupt_items_map.pop(pin)
            row = self.ui.list_interrupts.row(item)
            self.ui.list_interrupts.takeItem(row)

    def on_cmd_throb(self):
        period = self.ui.spin_throb_period.value()
        p1 = self.ui.spin_throb_p1.value()
        p2 = self.ui.spin_throb_p2.value()
        p3 = self.ui.spin_throb_p3.value()
        self.send_serial_cmd_signal.emit(f"throb {period} {p1} {p2} {p3}")

    def on_cmd_stopthrob(self):
        self.send_serial_cmd_signal.emit("stopthrob")

    def on_cmd_mem(self):
        addr = self.ui.edit_mem_addr.text().strip()
        if addr:
            self.send_serial_cmd_signal.emit(f"mem {addr}")

    def set_port(self, port_name):
        index = self.ui.combo_serial_port.findText(port_name)
        if index != -1:
            self.ui.combo_serial_port.setCurrentIndex(index)

    def get_port(self):
        return self.ui.combo_serial_port.currentText()

