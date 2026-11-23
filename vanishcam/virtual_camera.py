class VirtualCameraOutput:
    def __init__(self, width: int, height: int, fps: int = 30):
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError("Instala las dependencias fijadas para activar la cámara virtual.") from exc
        self.camera = pyvirtualcam.Camera(
            width=width, height=height, fps=fps, device="OBS Virtual Camera", print_fps=False
        )

    def send(self, bgr_frame) -> None:
        import cv2
        self.camera.send(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        self.camera.sleep_until_next_frame()

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
