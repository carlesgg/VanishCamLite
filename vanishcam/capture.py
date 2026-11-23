import cv2


class CameraCapture:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.camera = None

    def open(self) -> None:
        self.camera = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        if not self.camera.isOpened():
            self.close()
            raise RuntimeError("No se pudo abrir la webcam (índice 0).")

    def read(self):
        if self.camera is None:
            return None
        ok, frame = self.camera.read()
        return frame if ok else None

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
