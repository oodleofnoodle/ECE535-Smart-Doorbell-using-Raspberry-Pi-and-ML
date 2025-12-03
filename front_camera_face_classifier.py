"""
Classify a face from the front camera using the faces.h5 model.
Captures a single frame from camera index 0, runs the classifier, and prints the predicted class.
Sends an SMS alert for specific classes when configured.
"""

from pathlib import Path
import argparse
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from sms import send_sms



MODEL_PATH = "/home/aandoni/Desktop/ECE535-SmartDoorbell/ECE535-Smart-Doorbell-using-Raspberry-Pi-and-ML/faces.h5"

# Update the class names if you retrain the model with different labels.
CLASS_NAMES = ["den", "brad", "angie", "dman", "sussy"]

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


def open_camera(camera_index: int, resolution: Tuple[int, int]) -> cv2.VideoCapture:
    """
    Open a camera device with the requested resolution.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return cap


def capture_frame(
    cap: cv2.VideoCapture,
    timeout: float,
    rotate_180: bool = False,
) -> Optional[np.ndarray]:
    """
    Capture a single frame from an open VideoCapture within a timeout.
    """
    deadline = time.time() + timeout
    last_frame: Optional[np.ndarray] = None

    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            last_frame = frame
            break
        time.sleep(0.05)

    if last_frame is None:
        return None

    if rotate_180:
        last_frame = cv2.rotate(last_frame, cv2.ROTATE_180)

    return last_frame


def maybe_send_sms(label: str, to_number: Optional[str]):
    """
    Send an SMS when the label matches alert classes.
    """
    alert_classes = {"dman", "sussy"}
    if label not in alert_classes:
        return

    if not to_number:
        print("SMS alert skipped: no --sms-to number provided.", file=sys.stderr)
        return

    message = f"Doorbell alert: {label} detected at the door."
    try:
        send_sms(to_number, message)
    except Exception as exc:  # Catch-all so detection still returns a result
        print(f"SMS send failed: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Classify a face from camera index 0.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="Capture resolution (default: 640 480).",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate the captured frame by 180 degrees.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a frame before failing (default: 3.0).",
    )
    parser.add_argument(
        "--sms-to",
        help="Phone number to receive SMS alerts when 'dman' or 'sussy' are detected.",
    )
    args = parser.parse_args()

    model = load_classifier(MODEL_PATH)

    cap = open_camera(args.camera_index, tuple(args.resolution))

    try:
        frame = capture_frame(
            cap=cap,
            timeout=args.timeout,
            rotate_180=args.rotate_180,
        )
        if frame is None:
            raise RuntimeError("No frame received from the camera.")

        label = predict_person(model, frame)
        print(label)

        maybe_send_sms(label, args.sms_to)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
