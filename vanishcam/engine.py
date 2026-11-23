from pathlib import Path
import threading
import time

import cv2

from .capture import CameraCapture
from .gesture import ShushDetector
from .image_processing import composite
from .states import Visibility, VisibilityController
from .virtual_camera import VirtualCameraOutput


class VideoEngine:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.capture = CameraCapture()
        self.controller = VisibilityController()
        self.detector = None
        self.background = None
        self.running = False
        self.started = False
        self.virtual_enabled = False
        self.virtual = None
        self.error = None
        self._lock = threading.Lock()
        self._latest = None
        self._latest_raw = None
        self._thread = None

    def start_worker(self):
        self.capture.open()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="VanishCam video", daemon=True)
        self._thread.start()

    def capture_background(self) -> bool:
        with self._lock:
            if self._latest_raw is None or self.started:
                return False
            self.background = self._latest_raw.copy()
            return True

    def start_processing(self, virtual_enabled: bool):
        if self.background is None:
            raise RuntimeError("Captura primero el fondo vacío.")
        self.detector = ShushDetector(self.model_dir)
        self.virtual_enabled = virtual_enabled
        self.started = True

    def reappear(self):
        self.controller.reappear()

    def latest(self):
        with self._lock:
            return None if self._latest is None else (self._latest.copy(), self.controller.state)

    def _run(self):
        try:
            while self.running:
                frame = self.capture.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    self._latest_raw = frame.copy()
                output = frame
                if self.started:
                    if self.controller.state is Visibility.INVISIBLE:
                        self.detector.reset_hold()
                    elif self.detector.detect(frame):
                        self.controller.shush_detected()
                    output = composite(frame, self.background, self.controller.alpha())
                    if self.virtual_enabled:
                        if self.virtual is None:
                            self.virtual = VirtualCameraOutput(frame.shape[1], frame.shape[0])
                        self.virtual.send(output)
                with self._lock:
                    self._latest = output.copy()
        except Exception as exc:
            self.error = str(exc)
            self.running = False
        finally:
            if self.detector is not None:
                self.detector.close()
            if self.virtual is not None:
                self.virtual.close()
            self.capture.close()

    def close(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
