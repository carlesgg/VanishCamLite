from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


class ShushDetector:
    """Detecta índice vertical sobre los labios de una cara visible."""

    def __init__(self, model_dir: Path):
        hand_path = model_dir / "hand_landmarker.task"
        face_path = model_dir / "face_landmarker.task"
        missing = [p.name for p in (hand_path, face_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan modelos locales en models/: " + ", ".join(missing)
            )
        vision = mp.tasks.vision
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.5,
            )
        )
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.started_at = None
        self.last_timestamp_ms = 0

    def detect(self, frame) -> bool:
        now = time.monotonic()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        hand_result = self.hand.detect_for_video(image, timestamp_ms)
        face_result = self.face.detect_for_video(image, timestamp_ms)
        matched = self._matches(hand_result, face_result, frame.shape[1], frame.shape[0])
        if matched:
            if self.started_at is None:
                self.started_at = now
            return now - self.started_at >= 0.6
        self.started_at = None
        return False

    @staticmethod
    def _matches(hand_result, face_result, width: int, height: int) -> bool:
        if not hand_result.hand_landmarks or not face_result.face_landmarks:
            return False
        hand = hand_result.hand_landmarks[0]
        face = face_result.face_landmarks[0]

        # Face Mesh: puntos interiores del labio superior/inferior.
        mouth = np.array([(face[i].x, face[i].y) for i in (13, 14)])
        mouth_center = mouth.mean(axis=0)
        face_width = max(face[454].x - face[234].x, 0.08)

        wrist, mcp, pip, tip = (np.array([hand[i].x, hand[i].y]) for i in (0, 5, 6, 8))
        index_vector = tip - pip
        finger_length = np.linalg.norm(tip - mcp)
        if finger_length < 0.035 or index_vector[1] >= -0.015:
            return False
        # La punta queda cerca del centro de los labios; el dedo apunta hacia arriba.
        if abs(tip[0] - mouth_center[0]) > face_width * 0.18:
            return False
        if abs(tip[1] - mouth_center[1]) > face_width * 0.22:
            return False
        if abs(index_vector[0]) > max(abs(index_vector[1]) * 0.65, 0.015):
            return False
        # El índice debe extenderse desde la palma; los otros tres dedos se recogen.
        if np.linalg.norm(tip - wrist) < np.linalg.norm(pip - wrist) * 1.08:
            return False
        for tip_id, pip_id in ((12, 10), (16, 14), (20, 18)):
            if np.linalg.norm(np.array([hand[tip_id].x, hand[tip_id].y]) - wrist) > \
                    np.linalg.norm(np.array([hand[pip_id].x, hand[pip_id].y]) - wrist) * 1.15:
                return False
        return True

    def close(self) -> None:
        self.hand.close()
        self.face.close()

    def reset_hold(self) -> None:
        self.started_at = None
