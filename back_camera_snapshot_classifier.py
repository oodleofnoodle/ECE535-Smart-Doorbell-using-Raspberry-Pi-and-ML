"""
Capture one frame from the back camera, save it to disk, reload it, and classify with faces.h5.
Uses CameraCapture for grabbing frames and mirrors the prototype-style save->process flow.
"""

from pathlib import Path
from datetime import datetime
import sys
import queue
from typing import Optional, Tuple

import cv2
import numpy as np
from keras.models import load_model

from camera_capture import CameraCapture


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
    model = load_classifier(MODEL_PATH)

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
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
