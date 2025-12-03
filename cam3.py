"""
Camera Capture Module

- CameraCapture: generic single-source capture (file / USB / Pi cam),
  used by your existing pipeline.
- LightSwitchedDualCapture: "front_camera"-style capture that
  switches between two Pi cameras (NoIR + Color) using a BH1750
  light sensor, like your working cam2.py.

This lets you keep your front/back camera naming and pipeline-style
FrameData / queue behavior, while using the light-based switching.
"""

import cv2
import time
import threading
import queue
from typing import Optional, Tuple, Dict
import logging

# ---------- Optional BH1750 light sensor ----------
try:
    import smbus  # from python3-smbus
except ImportError:
    smbus = None

# ---------- Optional PiCamera2 ----------
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    _PICAMERA2_AVAILABLE = False

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Common FrameData container
# =========================
class FrameData:
    """Container for frame data and metadata."""
    def __init__(self, frame, timestamp: float, frame_number: int, camera_name: str):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.camera_name = camera_name
        self.metadata: Dict = {}

    def __repr__(self):
        return (
            f"FrameData(camera={self.camera_name}, "
            f"frame={self.frame_number}, timestamp={self.timestamp:.2f})"
        )


# =========================
# Generic single-source capture
# =========================
class CameraCapture:
    """
    Captures frames from a single video source (file / USB / Pi camera)
    at a target FPS, putting FrameData into a queue.
    Designed to be compatible with your existing pipeline.py.
    """

    def __init__(
        self,
        camera_name: str,
        source: str,
        target_fps: int = 10,
        resolution: Tuple[int, int] = (320, 240),
        use_picamera2: bool = False,
        max_queue_size: int = 30,
    ):
        """
        Args:
            camera_name: Identifier for this camera (e.g. "back_camera", "front_camera").
            source: Video file path or camera index ("0", "1", etc.).
            target_fps: Target frame rate.
            resolution: (width, height).
            use_picamera2: Use PiCamera2 if True, otherwise OpenCV VideoCapture.
            max_queue_size: Max buffered frames.
        """
        self.camera_name = camera_name
        self.source = source
        self.target_fps = target_fps
        self.resolution = resolution
        self.use_picamera2 = use_picamera2
        self.max_queue_size = max_queue_size

        self.frame_queue: "queue.Queue[FrameData]" = queue.Queue(maxsize=max_queue_size)
        self.capture: Optional[cv2.VideoCapture] = None
        self.picamera: Optional[Picamera2] = None
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_count = 0

        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize the video capture source."""
        if self.use_picamera2:
            self._initialize_picamera()
        else:
            self._initialize_opencv_capture()

    def _initialize_opencv_capture(self):
        """Initialize OpenCV VideoCapture (for video files or USB cameras)."""
        try:
            source = int(self.source)
        except ValueError:
            source = self.source

        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set resolution if using a live camera index
        if isinstance(source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)

        logger.info(f"{self.camera_name}: Initialized OpenCV capture")
        logger.info(f"  Source: {self.source}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Source FPS: {source_fps:.2f}, Target FPS: {self.target_fps}")

    def _initialize_picamera(self):
        """Initialize PiCamera2 (Raspberry Pi Camera Module)."""
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "PiCamera2 library is not available. "
                "Install it with 'sudo apt install python3-picamera2' "
                "or set use_picamera2=False."
            )

        try:
            cam_index = int(self.source)
        except (TypeError, ValueError):
            cam_index = 0

        self.picamera = Picamera2(camera_num=cam_index)

        config = self.picamera.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=4,
            queue=True,
        )
        self.picamera.configure(config)

        controls = {
            "FrameRate": 50,  # good default
        }
        self.picamera.set_controls(controls)
        self.picamera.start()

        logger.info(f"{self.camera_name}: Initialized PiCamera2")
        logger.info(f"  Camera index: {cam_index}")
        logger.info(f"  Resolution: {self.resolution[0]}x{self.resolution[1]}")

    def start(self):
        """Start capturing frames in a background thread."""
        if self.is_running:
            logger.warning(f"{self.camera_name}: Already running")
            return

        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self.capture_thread.start()
        logger.info(f"{self.camera_name}: Started capture thread")

    def stop(self):
        """Stop capturing frames."""
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None

        if self.capture:
            self.capture.release()
            self.capture = None

        if self.picamera:
            try:
                self.picamera.stop()
                if hasattr(self.picamera, "close"):
                    self.picamera.close()
            except Exception as exc:
                logger.warning(f"{self.camera_name}: Error stopping PiCamera2: {exc}")
            finally:
                self.picamera = None

        logger.info(f"{self.camera_name}: Stopped capture")

    def _capture_loop(self):
        """Capture loop respecting target_fps."""
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        last_capture_time = 0.0

        while self.is_running:
            now = time.time()

            if frame_interval > 0 and (now - last_capture_time) < frame_interval:
                time.sleep(0.001)
                continue

            # Grab frame
            if self.use_picamera2:
                frame = self._capture_picamera_frame()
            else:
                frame = self._capture_opencv_frame()

            if frame is None:
                continue

            frame_data = FrameData(
                frame=frame,
                timestamp=now,
                frame_number=self.frame_count,
                camera_name=self.camera_name,
            )

            try:
                self.frame_queue.put(frame_data, block=False)
                self.frame_count += 1
                last_capture_time = now
            except queue.Full:
                # drop frame if consumer is slow
                pass

        logger.info(f"{self.camera_name}: Capture loop ended")

    def _capture_opencv_frame(self):
        if self.capture is None:
            return None
        ret, frame = self.capture.read()
        return frame if ret else None

    def _capture_picamera_frame(self):
        if self.picamera is None:
            return None
        try:
            frame = self.picamera.capture_array()
        except Exception as exc:
            logger.warning(f"{self.camera_name}: PiCamera2 capture failed: {exc}")
            return None
        return frame  # leave as RGB

    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get next frame from queue or None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        return self.frame_queue.qsize()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =========================
# Light-switched dual Pi camera (cam2 behavior)
# =========================
class LightSwitchedDualCapture:
    """
    Virtual camera that uses TWO Pi cameras (NoIR + Color),
    but only one runs at a time, selected via BH1750 light sensor.

    API is compatible with CameraCapture: start(), stop(), get_frame().

    Typical wiring for your case:
    - NoIR Module 3 on CAM/DISP 0  -> noir_index = 0
    - Color Module 3 on CAM/DISP 1 -> color_index = 1
    - BH1750 on I2C with ADDR -> GND (address 0x23)
    """

    BH1750_ADDR = 0x23
    CONTINUOUS_HIGH_RES_MODE = 0x10

    def __init__(
        self,
        camera_name: str = "front_camera",
        noir_index: int = 0,
        color_index: int = 1,
        target_fps: int = 75,
        resolution: Tuple[int, int] = (640, 480),
        max_queue_size: int = 30,
        day_threshold: float = 30.0,
        night_threshold: float = 10.0,
        brightness_poll_interval: float = 0.5,
    ):
        """
        Args:
            camera_name: Logical name for this virtual camera ("front_camera" for your ML).
            noir_index: Pi camera index for NoIR (slot 0).
            color_index: Pi camera index for Color (slot 1).
            target_fps: Display / queue FPS (sensor runs faster).
            resolution: (width, height).
            max_queue_size: Frame queue size.
            day_threshold: lux above which we choose the color camera.
            night_threshold: lux below which we choose the NoIR camera.
            brightness_poll_interval: seconds between lux checks.
        """
        if night_threshold >= day_threshold:
            raise ValueError("night_threshold must be < day_threshold for hysteresis")

        self.camera_name = camera_name
        self.noir_index = noir_index
        self.color_index = color_index
        self.target_fps = target_fps
        self.resolution = resolution
        self.max_queue_size = max_queue_size
        self.day_threshold = day_threshold
        self.night_threshold = night_threshold
        self.brightness_poll_interval = brightness_poll_interval

        # BH1750 setup
        self.bus = None
        if smbus is not None:
            try:
                self.bus = smbus.SMBus(1)
                # put sensor in continuous high-res mode
                self.bus.write_byte(self.BH1750_ADDR, self.CONTINUOUS_HIGH_RES_MODE)
                time.sleep(0.2)
                logger.info("BH1750 initialized in continuous high-res mode")
            except Exception as e:
                logger.warning(f"BH1750 init failed: {e}")
                self.bus = None
        else:
            logger.warning("smbus not available; light switching will use dummy lux")

        # PiCamera2 objects (created & configured once)
        self.noir_cam = self._create_configured_picamera(self.noir_index)
        self.color_cam = self._create_configured_picamera(self.color_index)

        self.active_cam: Optional[Picamera2] = None
        self.current_mode = "day"  # "day" -> color, "night" -> noir

        # Frame queue / thread
        self.frame_queue: "queue.Queue[FrameData]" = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_count = 0

        self._last_lux_check = 0.0

    def _create_configured_picamera(self, cam_index: int) -> Picamera2:
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "PiCamera2 is not available; install with "
                "'sudo apt install python3-picamera2'."
            )

        logger.info(f"Creating Picamera2 for camera index {cam_index}")
        picam = Picamera2(camera_num=cam_index)

        config = picam.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=4,
            queue=True,
        )
        picam.configure(config)

        controls = {
            "FrameRate": 75,  # drive sensor fast
            "AfMode": 2,      # continuous autofocus
            "AfRange": 1,     # macro/near range (tune 0/1/2 as needed)
            "ExposureTime": 10000,  # ~1/100 s
            "AnalogueGain": 2.0,
        }
        try:
            picam.set_controls(controls)
        except Exception as e:
            logger.warning(f"Failed to set controls on camera {cam_index}: {e}")

        return picam

    def _read_lux(self) -> Optional[float]:
        """Read lux from BH1750, or return dummy value if unavailable."""
        if self.bus is None:
            return 100.0  # default to "day"

        try:
            data = self.bus.read_i2c_block_data(
                self.BH1750_ADDR,
                self.CONTINUOUS_HIGH_RES_MODE,
                2,
            )
            raw = (data[0] << 8) | data[1]
            lux = raw / 1.2
            return lux
        except Exception as e:
            logger.warning(f"BH1750 read error: {e}")
            return None

    def _decide_initial_mode(self):
        lux = self._read_lux()
        if lux is not None:
            logger.info(f"Initial lux: {lux:.1f}")
            if lux < self.night_threshold:
                self.current_mode = "night"
            else:
                self.current_mode = "day"
        logger.info(f"Initial mode: {self.current_mode.upper()}")

    def start(self):
        """Start capture thread & appropriate camera based on light."""
        if self.is_running:
            return

        self._decide_initial_mode()

        if self.current_mode == "day":
            logger.info("Starting COLOR camera")
            self.color_cam.start()
            self.active_cam = self.color_cam
        else:
            logger.info("Starting NOIR camera")
            self.noir_cam.start()
            self.active_cam = self.noir_cam

        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self.capture_thread.start()
        logger.info(f"{self.camera_name}: LightSwitchedDualCapture started")

    def stop(self):
        """Stop thread and both cameras."""
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None

        # Stop both cams safely
        for cam, name in [(self.color_cam, "COLOR"), (self.noir_cam, "NOIR")]:
            try:
                logger.info(f"Stopping {name} camera")
                cam.stop()
            except Exception:
                pass
            if hasattr(cam, "close"):
                try:
                    cam.close()
                except Exception:
                    pass

        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None

        logger.info(f"{self.camera_name}: LightSwitchedDualCapture stopped")

    def _maybe_switch_mode(self, now: float):
        """Check lux every brightness_poll_interval and switch cameras if needed."""
        if now - self._last_lux_check < self.brightness_poll_interval:
            return

        self._last_lux_check = now
        lux = self._read_lux()

        if lux is None:
            return

        if self.current_mode == "day" and lux < self.night_threshold:
            # switch to night / NoIR
            logger.info(
                f"Lux {lux:.1f} < {self.night_threshold} → switching to NIGHT (NoIR)"
            )
            # stop color, start noir
            try:
                self.color_cam.stop()
            except Exception as e:
                logger.warning(f"Error stopping COLOR camera: {e}")
            time.sleep(0.1)
            self.noir_cam.start()
            self.active_cam = self.noir_cam
            self.current_mode = "night"

        elif self.current_mode == "night" and lux > self.day_threshold:
            # switch to day / Color
            logger.info(
                f"Lux {lux:.1f} > {self.day_threshold} → switching to DAY (Color)"
            )
            try:
                self.noir_cam.stop()
            except Exception as e:
                logger.warning(f"Error stopping NOIR camera: {e}")
            time.sleep(0.1)
            self.color_cam.start()
            self.active_cam = self.color_cam
            self.current_mode = "day"

    def _capture_loop(self):
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        last_frame_time = 0.0

        while self.is_running:
            now = time.time()

            # Check if we need to switch cameras based on lux
            self._maybe_switch_mode(now)

            # Enforce target FPS
            if frame_interval > 0 and (now - last_frame_time) < frame_interval:
                time.sleep(0.001)
                continue

            if self.active_cam is None:
                time.sleep(0.01)
                continue

            try:
                frame = self.active_cam.capture_array()
            except Exception as e:
                logger.warning(f"{self.camera_name}: capture error: {e}")
                continue

            if frame is None:
                continue

            frame_data = FrameData(
                frame=frame,
                timestamp=now,
                frame_number=self.frame_count,
                camera_name=self.camera_name,  # always "front_camera" etc.
            )

            try:
                self.frame_queue.put(frame_data, block=False)
                self.frame_count += 1
                last_frame_time = now
            except queue.Full:
                # drop if consumer is slow
                pass

        logger.info(f"{self.camera_name}: LightSwitchedDualCapture loop ended")

    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        return self.frame_queue.qsize()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =========================
# Simple demo / test
# =========================
if __name__ == "__main__":
    """
    Test script:

    - Assumes:
        NoIR camera on index 0 (slot 0)
        Color camera on index 1 (slot 1)
        BH1750 on I2C (ADDR->GND)

    - Shows a single window with frames from whichever camera
      is active ("DAY" or "NIGHT" logged in terminal).

    Use this instead of your old cam2.py.
    """

    capture = LightSwitchedDualCapture(
        camera_name="front_camera",
        noir_index=0,
        color_index=1,
        target_fps=75,
        resolution=(640, 480),
        max_queue_size=30,
        day_threshold=30.0,
        night_threshold=10.0,
        brightness_poll_interval=0.5,
    )

    try:
        with capture:
            print("Light-switched front_camera running. Press 'q' to quit.")
            while True:
                frame_data = capture.get_frame(timeout=1.0)
                if frame_data is None:
                    continue

                frame = frame_data.frame  # RGB from Picamera2

                label = f"{capture.current_mode.upper()} ({frame_data.camera_name})"
                cv2.putText(
                    frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("LightSwitched front_camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()
