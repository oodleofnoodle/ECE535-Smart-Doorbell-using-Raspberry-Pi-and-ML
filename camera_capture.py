
"""
Camera Capture Module
Handles video/camera feed capture with support for both file-based testing and live Pi camera feeds.
"""

import cv2
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple, Dict


# Optional PiCamera2 support for Raspberry Pi cameras
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    _PICAMERA2_AVAILABLE = False

import numpy as np


class FrameData:
    """Container for frame data and metadata"""
    def __init__(self, frame, timestamp: float, frame_number: int, camera_name: str):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.camera_name = camera_name
        self.metadata = {}

    def __repr__(self):
        return f"FrameData(camera={self.camera_name}, frame={self.frame_number}, timestamp={self.timestamp:.2f})"


class CameraCapture:
    """
    Captures frames from video file or camera feed at specified FPS.
    Designed to be Pi-camera compatible.
    """

    def __init__(
        self,
        camera_name: str,
        source: str,
        target_fps: int = 10,
        resolution: Tuple[int, int] = (320, 240),
        use_picamera2: bool = False,
        max_queue_size: int = 30
    ):
        """
        Initialize camera capture.

        Args:
            camera_name: Identifier for this camera (e.g., "back_camera", "front_camera")
            source: Video file path or camera index (0, 1, etc.)
            target_fps: Target frame extraction rate
            resolution: Desired frame resolution (width, height)
            use_picamera2: Whether to use PiCamera2 library (for Raspberry Pi)
            max_queue_size: Maximum frames to buffer in queue
        """
        self.camera_name = camera_name
        self.source = source
        self.target_fps = target_fps
        self.resolution = resolution
        self.use_picamera2 = use_picamera2
        self.max_queue_size = max_queue_size

        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.capture = None
        self.picamera = None
        self.is_running = False
        self.capture_thread = None
        self.frame_count = 0

        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize the video capture source"""
        if self.use_picamera2:
            self._initialize_picamera()
        else:
            self._initialize_opencv_capture()

    def _initialize_opencv_capture(self):
        """Initialize OpenCV VideoCapture (for video files or USB cameras)"""
        # Try to parse source as integer (camera index) or use as file path
        try:
            source = int(self.source)
        except ValueError:
            source = self.source

        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set resolution if using live camera
        if isinstance(source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Get actual resolution
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)

    def _initialize_picamera(self):
        """Initialize PiCamera2 (Raspberry Pi Camera Module 3)."""
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "PiCamera2 library is not available. Install it with "
                "'sudo apt install python3-picamera2' or set use_picamera2=False."
            )

        # Treat self.source as camera index (0, 1, ...) when using PiCamera2
        try:
            cam_index = int(self.source)
        except (TypeError, ValueError):
            cam_index = 0

        # Initialize Picamera2 instance for the selected camera
        self.picamera = Picamera2(camera_num=cam_index)

        # Configure video stream with optimized settings for performance
        config = self.picamera.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=4,  # Reduced from 4 for less memory usage
            queue=True,
        )
        self.picamera.configure(config)

        # Apply high-performance controls for 50fps
        controls = {
            "FrameRate": 50,       # 50 FPS camera capture
            # "ExposureTime": 10000, # Shorter exposure for high FPS (10000µs = 1/100s)
            # "AnalogueGain": 1.0,   # Minimal gain
            # "ColourGains": [1.0, 1.0]  # Neutral color balance
        }
        self.picamera.set_controls(controls)
        self.picamera.start()

    def start(self):
        """Start capturing frames in a separate thread"""
        if self.is_running:
            return

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        """Stop capturing frames"""
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.capture:
            self.capture.release()
            self.capture = None

        if self.picamera:
            try:
                # Gracefully stop and close the PiCamera2 instance
                self.picamera.stop()
                if hasattr(self.picamera, "close"):
                    self.picamera.close()
            except Exception:
                pass
            finally:
                self.picamera = None

    def _capture_loop(self):
        """High-performance capture loop optimized for 50fps"""
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        last_capture_time = 0
        frame_drop_count = 0

        # Pre-calculate resize check for performance
        needs_resize = self.resolution[0] != 640 or self.resolution[1] != 360  # Assuming 640x360 default

        while self.is_running:
            current_time = time.time()

            # Check if enough time has passed to capture next frame
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)  # Minimal sleep for 50fps
                continue

            # Aggressive frame dropping for high FPS stability
            queue_usage = self.frame_queue.qsize() / self.max_queue_size
            if queue_usage > 0.7:  # Drop frames earlier to maintain smoothness
                frame_drop_count += 1
                last_capture_time = current_time
                continue

            # Capture frame with minimal error handling for speed
            try:
                if self.use_picamera2:
                        frame = self.picamera.capture_array()
                        # Skip color conversion for speed - assume RGB is acceptable
                else:
                        ret, frame = self.capture.read()
                        if not ret:
                            if not self.use_picamera2:
                                break  # End of video file
                            continue
            except Exception:
                continue

            if frame is None:
                continue

            # Resize only if needed (optimized)
            # if needs_resize:
            #     frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_NEAREST)

            # Create FrameData object
            frame_data = FrameData(
                frame=frame,
                timestamp=current_time,
                frame_number=self.frame_count,
                camera_name=self.camera_name
            )

            # Add to queue (non-blocking)
            try:
                self.frame_queue.put(frame_data, block=False)
                self.frame_count += 1
                last_capture_time = current_time
                frame_drop_count = 0
            except queue.Full:
                frame_drop_count += 1

    def _capture_opencv_frame(self):
        """Capture a frame using OpenCV"""
        ret, frame = self.capture.read()
        return frame if ret else None

    def _capture_picamera_frame(self):
        """Capture a frame using PiCamera2"""
        if self.picamera is None:
            return None

        try:
            frame = self.picamera.capture_array()
        except Exception:
            return None

        if frame is None:
            return None

        # Picamera2 returns RGB; convert to BGR for OpenCV if needed
        # if frame.ndim == 3 and frame.shape[2] == 3:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Get next frame from queue.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            FrameData object or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        """Get current number of frames in queue"""
        return self.frame_queue.qsize()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class DualCameraCapture:
    """Manages two CameraCapture instances (e.g., front and back cameras)."""

    def __init__(self, back_camera_config: Dict, front_camera_config: Dict):
        """Initialize dual camera capture system."""
        

        self.back_camera = CameraCapture(
            camera_name=back_camera_config["camera_name"],
            source=back_camera_config["source"],
            target_fps=back_camera_config.get("target_fps", 10),
            resolution=back_camera_config.get("resolution", (320, 240)),
            use_picamera2=back_camera_config.get("use_picamera2", False),
            max_queue_size=back_camera_config.get("max_queue_size", 30)
        )
        
        self.front_camera = CameraCapture(
            camera_name=front_camera_config["camera_name"],
            source=front_camera_config["source"],
            target_fps=front_camera_config.get("target_fps", 10),
            resolution=front_camera_config.get("resolution", (320, 240)),
            use_picamera2=front_camera_config.get("use_picamera2", False),
            max_queue_size=front_camera_config.get("max_queue_size", 30)
        )

    def start(self):
        """Start both cameras"""
        self.back_camera.start()
        self.front_camera.start()

    def stop(self):
        """Stop both cameras"""
        self.back_camera.stop()
        self.front_camera.stop()

    def get_frames(self, timeout: float = 1.0) -> Tuple[Optional[FrameData], Optional[FrameData]]:
        """
        Get latest frames from both cameras.

        Returns:
            Tuple of (back_frame, front_frame)
        """
        back_frame = self.back_camera.get_frame(timeout=timeout)
        front_frame = self.front_camera.get_frame(timeout=timeout)
        return back_frame, front_frame


def create_dual_camera_capture(
    back_source: str = "0",
    front_source: str = "1",
    target_fps: int = 10,
    resolution: Tuple[int, int] = (320, 240),
    use_picamera2: bool = True,
    max_queue_size: int = 30
) -> DualCameraCapture:
    """
    Helper function to create a dual camera capture system.

    Args:
        back_source: Source for back camera (e.g., "0" for Pi camera 0)
        front_source: Source for front camera (e.g., "1" for Pi camera 1)
        target_fps: Target FPS for both cameras
        resolution: Resolution for both cameras
        use_picamera2: Whether to use PiCamera2 for both cameras
        max_queue_size: Max queue size for each camera

    Returns:
        DualCameraCapture instance
    """
    back_camera_config = {
        "camera_name": "back_camera",
        "source": back_source,
        "target_fps": target_fps,
        "resolution": resolution,
        "use_picamera2": use_picamera2,
        "max_queue_size": max_queue_size,
    }

    front_camera_config = {
        "camera_name": "front_camera",
        "source": front_source,
        "target_fps": target_fps,
        "resolution": resolution,
        "use_picamera2": use_picamera2,
        "max_queue_size": max_queue_size,
    }

    return DualCameraCapture(back_camera_config, front_camera_config)

if __name__ == "__main__":
    # Example usage / simple test
    # This will try to open two Pi cameras (index 0 and 1) using PiCamera2.
    # Press 'q' to quit.
    dual_capture = create_dual_camera_capture(
        back_source="0",
        front_source="1",
        target_fps=60,
        resolution=(640, 480),
        use_picamera2=True,
        max_queue_size=30,
    )

    try:
        dual_capture.start()
        print("Press Ctrl+C to stop...")

        while True:
            back_frame_data, front_frame_data = dual_capture.get_frames(timeout=0.25)

            if back_frame_data is not None:
                cv2.imshow("Back Camera", back_frame_data.frame)
            if front_frame_data is not None:
                cv2.imshow("Front Camera", front_frame_data.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dual_capture.stop()
        cv2.destroyAllWindows()