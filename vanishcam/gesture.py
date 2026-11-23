from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


class GestureDetector:
    """Detecta Shush o brazos cruzados durante 600 ms."""

    def __init__(self, model_dir: Path):
        hand_path = model_dir / "hand_landmarker.task"
        face_path = model_dir / "face_landmarker.task"
        pose_path = model_dir / "pose_landmarker_lite.task"
        missing = [p.name for p in (hand_path, face_path, pose_path) if not p.is_file()]
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
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
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
        pose_result = self.pose.detect_for_video(image, timestamp_ms)
        matched = self._arms_crossed(pose_result)
        if not matched:
            hand_result = self.hand.detect_for_video(image, timestamp_ms)
            face_result = self.face.detect_for_video(image, timestamp_ms)
            matched = self._shush(hand_result, face_result)
        if matched:
            if self.started_at is None:
                self.started_at = now
            return now - self.started_at >= 0.6
        self.started_at = None
        return False

    @staticmethod
    def _arms_crossed(pose_result) -> bool:
        if not pose_result.pose_landmarks:
            return False
        pose = pose_result.pose_landmarks[0]
        left_shoulder, right_shoulder = pose[11], pose[12]
        left_elbow, right_elbow = pose[13], pose[14]
        left_wrist, right_wrist = pose[15], pose[16]
        if min(p.visibility for p in (left_shoulder, right_shoulder, left_elbow,
                                      right_elbow, left_wrist, right_wrist)) < 0.5:
            return False

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_span = abs(left_shoulder.x - right_shoulder.x)
        if shoulder_span < 0.08:
            return False
        # Each wrist must pass the torso center onto the opposite side of its shoulder.
        left_crossed = (left_shoulder.x - center_x) * (left_wrist.x - center_x) < 0
        right_crossed = (right_shoulder.x - center_x) * (right_wrist.x - center_x) < 0
        wrists_swapped = ((left_shoulder.x - right_shoulder.x) *
                          (left_wrist.x - right_wrist.x)) < 0
        margin = shoulder_span * 0.08
        clear_cross = (abs(left_wrist.x - center_x) > margin and
                       abs(right_wrist.x - center_x) > margin)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        chest_bottom = shoulder_y + shoulder_span * 1.6
        chest_level = all(shoulder_y - shoulder_span * 0.15 < wrist.y < chest_bottom
                          for wrist in (left_wrist, right_wrist))
        return left_crossed and right_crossed and wrists_swapped and clear_cross and chest_level

    @staticmethod
    def _shush(hand_result, face_result) -> bool:
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
        self.pose.close()

    def reset_hold(self) -> None:
        self.started_at = None
