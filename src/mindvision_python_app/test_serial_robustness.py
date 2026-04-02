import unittest

from PySide6.QtWidgets import QApplication

from .cnc_control_panel import CNCControlPanel, _classify_rx_content
from .serial_worker import SerialWorker, _pop_complete_serial_lines, _sanitize_serial_line


app = QApplication.instance() or QApplication([])


class FakeSerialPort:
    def __init__(self, chunks):
        self._chunks = [bytes(chunk) for chunk in chunks]

    @property
    def in_waiting(self):
        if not self._chunks:
            return 0
        return len(self._chunks[0])

    def read(self, size):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class SerialWorkerRobustnessTests(unittest.TestCase):
    def test_pop_complete_serial_lines_preserves_partial_tail(self):
        buffer = bytearray(b"first\r\nsecond\npartial")

        lines = _pop_complete_serial_lines(buffer)

        self.assertEqual(lines, [b"first", b"second"])
        self.assertEqual(buffer, bytearray(b"partial"))

    def test_sanitize_serial_line_drops_non_ascii_bytes(self):
        line, dropped = _sanitize_serial_line(b"\xef\xbf\xbdok\x00")

        self.assertEqual(line, "ok")
        self.assertEqual(dropped, b"\xef\xbf\xbd\x00")

    def test_poll_serial_logs_filtered_bytes_and_clean_line(self):
        worker = SerialWorker()
        worker.is_connected = True
        worker.serial_port = FakeSerialPort([b"\xef\xbf\xbd", b"ok\n"])
        messages = []
        worker.log_signal.connect(messages.append)

        worker.poll_serial()
        worker.poll_serial()

        self.assertEqual(
            messages,
            [
                "Rx filtered bytes: ef bf bd",
                "Rx: ok",
            ],
        )


class CNCControlPanelRobustnessTests(unittest.TestCase):
    def test_classify_rx_content_treats_wrapped_ok_as_ack(self):
        kind, normalized = _classify_rx_content("\ufffd[ok]\ufffd")

        self.assertEqual(kind, "ok")
        self.assertEqual(normalized, "[ok]")

    def test_on_log_message_unblocks_queue_on_noisy_ok(self):
        panel = CNCControlPanel()
        process_calls = []
        logged_messages = []
        panel.waiting_for_ok = True
        panel.last_sent_command = "G1 X1"
        panel.process_queue = lambda: process_calls.append("called")
        panel.log_signal.connect(logged_messages.append)

        panel.on_log_message("Rx: \ufffdok")

        self.assertFalse(panel.waiting_for_ok)
        self.assertEqual(process_calls, ["called"])
        self.assertEqual(logged_messages, ["Rx: \ufffdok"])


if __name__ == "__main__":
    unittest.main()