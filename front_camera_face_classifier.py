"""
Classify a face from the front camera using the faces.h5 model.
Captures a single frame, runs the classifier, and prints the predicted class.
"""

from pathlib import Path
import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import queue
from tensorflow.keras.models import load_model

from camera_capture import CameraCapture


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "faces.h5"

# Update the class names if you retrain the model with different labels.
CLASS_NAMES = ["den", "brad", "angie", "dman", "sussy"]

# Input size used during training.
IMG_SIZE: Tuple[int, int] = (224, 224)


def load_classifier(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file at {model_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Classify a face from the front camera.")
    parser.add_argument(
        "--source",
        default="1",
        help="Camera index or path for the front camera (default: 1).",
    )
    parser.add_argument(
        "--use-picamera2",
        action="store_true",
        help="Use PiCamera2 for capture (set when running on Raspberry Pi).",
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
        default=3.0,
        help="Seconds to wait for a frame before failing (default: 3.0).",
    )
    args = parser.parse_args()

    model = load_classifier(MODEL_PATH)

    capture = CameraCapture(
        camera_name="front_camera",
        source=args.source,
        target_fps=5,
        resolution=tuple(args.resolution),
        use_picamera2=args.use_picamera2,
        max_queue_size=5,
    )

    capture.start()

    try:
        frame = get_latest_frame(capture, timeout=args.timeout)
        if frame is None:
            raise RuntimeError("No frame received from front camera.")

        if args.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        label = predict_person(model, frame)
        print(label)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
