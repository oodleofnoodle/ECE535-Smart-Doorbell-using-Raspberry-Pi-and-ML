"""
Capture frames from the back camera, save them to disk, reload, and classify with faces.h5.
Takes multiple pictures (spaced by capture_interval_sec) and optionally sends SMS alerts
for suspicious detections.
"""

from pathlib import Path
from datetime import datetime
import sys
import queue
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from keras.models import load_model

from camera_capture import CameraCapture
from sms import send_sms


MODEL_PATH = "/home/aandoni/Desktop/ECE535-SmartDoorbell/ECE535-Smart-Doorbell-using-Raspberry-Pi-and-ML/faces.h5"

# Update the class names if you retrain the model with different labels.
CLASS_NAMES = ["den", "brad", "angie", "dman", "sussy"]
ALERT_CLASSES = {"dman", "sussy"}

# Input size used during training.
IMG_SIZE: Tuple[int, int] = (224, 224)


def load_classifier(model_path: Path):
    return load_model(model_path)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_person(model, frame: np.ndarray) -> str:
    input_tensor = preprocess_frame(frame)
    preds = model.predict(input_tensor, verbose=0)
    class_id = int(np.argmax(preds, axis=1)[0])
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return str(class_id)


def get_latest_frame(capture: CameraCapture, timeout: float) -> Optional[np.ndarray]:
    """
    Pull the most recent frame from the capture queue, dropping buffered frames.
    """
    try:
        frame_data = capture.frame_queue.get(timeout=timeout)
    except queue.Empty:
        return None

    try:
        while True:
            frame_data = capture.frame_queue.get_nowait()
    except queue.Empty:
        pass

    return frame_data.frame if frame_data is not None else None


def save_frame_to_dir(frame: np.ndarray, output_dir: Path) -> Path:
    """
    Save a frame to disk with a timestamped filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"back_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def maybe_send_sms(label: str, to_number: Optional[str]):
    """
    Send an SMS when the label matches alert classes.
    """
    if label not in ALERT_CLASSES:
        return

    if not to_number:
        print("SMS alert skipped: SMS recipient not configured.", file=sys.stderr)
        return

    message = f"Doorbell alert: {label} detected on the back camera."
    try:
        send_sms(to_number, message)
    except Exception as exc:
        print(f"SMS send failed: {exc}", file=sys.stderr)


def main():
    # Fixed configuration values (no CLI args).
    source = "0"  # back camera index
    use_picamera2 = True  # set True if you are on Raspberry Pi with PiCamera2
    resolution: Tuple[int, int] = (640, 480)
    target_fps = 5
    rotate_180 = False
    timeout = 3.0
    output_dir = Path(
        "/home/aandoni/Desktop/ECE535-SmartDoorbell/ECE535-Smart-Doorbell-using-Raspberry-Pi-and-ML/pictures/captured_back_frames/"
    )
    capture_interval_sec = 5.0  # seconds between captures
    num_pictures = 3  # how many pictures to take in this run
    sms_to_number: Optional[str] = None  # e.g., "+15555555555" to enable alerts
    model = load_classifier(Path(MODEL_PATH))

    capture = CameraCapture(
        camera_name="back_camera",
        source=source,
        target_fps=target_fps,
        resolution=resolution,
        use_picamera2=use_picamera2,
        max_queue_size=5,
    )

    capture.start()

    try:
        for shot_idx in range(num_pictures):
            frame = get_latest_frame(capture, timeout=timeout)
            if frame is None:
                raise RuntimeError("No frame received from back camera.")

            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            saved_path = save_frame_to_dir(frame, output_dir)

            reloaded = cv2.imread(str(saved_path))
            if reloaded is None:
                raise RuntimeError(f"Failed to reload saved image at {saved_path}")

            label = predict_person(model, reloaded)
            print(label)
            print(f"Saved frame to {saved_path}")

            maybe_send_sms(label, sms_to_number)

            if shot_idx < num_pictures - 1:
                time.sleep(capture_interval_sec)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
