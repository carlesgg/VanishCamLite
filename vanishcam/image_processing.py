import cv2


def composite(frame, background, alpha: float):
    if background is None:
        return frame
    if alpha >= 1:
        return frame
    if alpha <= 0:
        return background
    return cv2.addWeighted(frame, alpha, background, 1.0 - alpha, 0)
