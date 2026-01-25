
class QMutexLocker:
    def __init__(self, mutex):
        self.mutex = mutex

    def __enter__(self):
        self.mutex.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.mutex.unlock()
