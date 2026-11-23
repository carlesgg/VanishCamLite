# VanishCam Lite

A local Windows app that replaces the webcam image with a captured background when it detects the 🤫 gesture. The app has no telemetry or download code and needs no network connection while running.

## Setup

Requires 64-bit Python 3.11. From this folder, create an environment and install the pinned versions manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Manually download the official MediaPipe models and put them in `models\` with these names:

- [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) — Hand Landmarker model.
- [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) — Face Landmarker model.

The models are data files and are never downloaded or executed by the app. See the MediaPipe guides for [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).

Run `python main.py`. Keep the camera view empty and click **Capture empty background**; then enter the frame and click **Start**. Hold your index finger vertically in front of your lips for 600 ms. The output fades to the background over 500 ms. Landmarkers are skipped in the INVISIBLE state. Press F8 or click **Reappear** to return.

## Virtual camera and Teams

Manually install OBS Studio from [obsproject.com](https://obsproject.com/), launch OBS once to register **OBS Virtual Camera**, and close OBS if you do not need it. In VanishCam Lite, enable **Send to virtual camera (OBS)** before clicking **Start**. In Teams, open **Settings > Devices > Camera** and select **OBS Virtual Camera**. The virtual camera carries the processed image; choose your microphone separately in Teams.

OBS and the dependencies are installed manually. While running, VanishCam Lite only accesses the local webcam and virtual camera device.

## MVP limitations

- Uses Windows camera device index 0, at 640×480 when supported by the driver.
- Detection requires both a visible face and hand. The geometric gesture rule is intentionally simple and may need good lighting, distance and framing adjustments.
- The background is captured as-is; keep the camera still and the background clear of people.
- Windows virtual camera output requires the OBS device to be installed. The app does not install drivers.
