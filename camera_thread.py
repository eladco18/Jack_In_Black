import cv2
import threading
import time
from collections import deque
import numpy as np
from typing import List
from PyQt5.QtCore import pyqtSignal, QThread
import queue

fps_limit = 15


class CameraThread(QThread):
    """
    Background thread dedicated to reading frames from the camera.
    Uses a Production-Grade Zero-Latency Producer-Consumer architecture.
    Features: Event-driven sync, UI throttling, Buffer Protection, and Telemetry.
    """
    frame_ready_signal = pyqtSignal(np.ndarray)

    def __init__(self, engine, camera_id: str = "0", buffer_size: int = 5, target_res: tuple = (1280, 720)) -> None:
        super().__init__()
        self.engine = engine
        self.camera_id = camera_id
        self.running: bool = True

        # Buffer for the logic engine. Will store tuples of (timestamp, frame_data)
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.lock: threading.Lock = threading.Lock()
        self.target_res = target_res

        # --- Variables for Zero-Latency & Synchronization ---
        self.latest_frame = None
        self.latest_frame_time: float = 0.0
        self.frame_id: int = 0  # Monotonic counter to track generation
        self.grab_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.cap = None
        self.grabber_thread = None
        self.render_thread = None

        # --- UI Throttling Parameters ---
        self.ui_fps_limit: int = fps_limit
        self.render_queue = queue.Queue(maxsize=1)
        self.min_ui_update_interval: float = 1.0 / self.ui_fps_limit
        self.last_ui_update_time: float = 0.0

        # --- Telemetry & Metrics ---
        self.stats_captured: int = 0
        self.stats_processed: int = 0
        self.stats_dropped: int = 0
        self.stats_skipped_duplicates: int = 0
        self.stats_rendered: int = 0

    def _grabber_loop(self):
        """
        [PRODUCER] A dedicated micro-thread that ONLY pulls frames.
        Updates the latest frame, increments telemetry, and signals the consumer.
        """
        consecutive_failures = 0
        MAX_FAILURES = 30

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                consecutive_failures = 0
                capture_time = time.time()

                with self.grab_lock:
                    self.latest_frame = frame
                    self.latest_frame_time = capture_time
                    self.frame_id += 1
                    self.stats_captured += 1

                # Signal the consumer thread
                self.new_frame_event.set()
            else:
                consecutive_failures += 1
                if consecutive_failures > MAX_FAILURES:
                    print(f"[CAMERA CRITICAL] Camera {self.camera_id} is dropping frames!")
                    time.sleep(0.05)

    def _print_telemetry(self):
        """Helper method to print system health metrics."""
        print(f"[TELEMETRY] Captured: {self.stats_captured} | "
              f"Logic Processed: {self.stats_processed} | "
              f"UI Rendered: {self.stats_rendered} | "
              f"Dropped: {self.stats_dropped} | "
              f"Skipped Duplicates: {self.stats_skipped_duplicates}")

    def get_recent_frames(self, num_frames: int = 5) -> List[np.ndarray]:
        """
        Returns clean, unannotated frames for the game logic.
        Strips the timestamp before returning to keep backward compatibility.
        """
        with self.lock:
            return [f[1] for f in list(self.frame_buffer)[-num_frames:]]

    def clear_buffer(self) -> None:
        with self.lock:
            self.frame_buffer.clear()
            print("[CAMERA] Frame buffer cleared.")

    def stop(self) -> None:
        self.running = False
        self.new_frame_event.set()
        self.wait()

    def _render_loop(self):
        """
        [RENDER THREAD] Dedicated lightweight thread for drawing and emitting UI frames.
        Runs at a strictly locked FPS, immune to upstream logic latency.
        """
        interval = 1.0 / self.ui_fps_limit
        next_frame_time = time.time()

        while self.running:
            next_frame_time += interval
            try:
                # Wait for a fresh frame (timeout allows clean thread shutdown)
                frame = self.render_queue.get(timeout=0.05)

                # Render the UI completely decoupled from the heavy logic
                annotated_frame = self.engine.render_ui(frame)

                # Technically emitting from a standard thread to PyQt, but safe via Qt's queued connections
                self.frame_ready_signal.emit(annotated_frame)
                self.stats_rendered += 1
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[RENDER ERROR] Exception during UI rendering: {e}")

            # Strict absolute sleeping to guarantee smooth Frame Pacing without drift
            sleep_for = next_frame_time - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def run(self) -> None:
        """
        [CONSUMER] Main processing loop. Protects logic frames, handles vision,
        and logs telemetry data.
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_res[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            self.running = False
            return

        # ==========================================================
        # START WORKER THREADS ONCE, BEFORE THE MAIN LOOP
        # ==========================================================
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.grabber_thread.start()

        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        last_processed_id = -1
        last_telemetry_print = time.time()

        # ==========================================================
        # SINGLE MAIN LOGIC LOOP
        # ==========================================================
        while self.running:
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()

            if not self.running:
                break

            with self.grab_lock:
                if self.latest_frame is None or self.frame_id == last_processed_id:
                    self.stats_skipped_duplicates += 1
                    continue

                current_frame = self.latest_frame.copy()
                current_frame_time = self.latest_frame_time
                current_frame_id = self.frame_id

                if last_processed_id != -1 and (current_frame_id - last_processed_id > 1):
                    self.stats_dropped += (current_frame_id - last_processed_id - 1)

                last_processed_id = current_frame_id

            current_frame = self.engine.normalize_frame(current_frame)

            # --- PROTECT THE LOGIC BUFFER ---
            logic_frame = current_frame.copy()
            with self.lock:
                self.frame_buffer.append((current_frame_time, logic_frame))

            # --- VISION LOGIC PROCESSING (Non-blocking to UI) ---
            try:
                self.engine.process_logic(current_frame)
                self.stats_processed += 1
            except Exception as e:
                print(f"[VISION ERROR] Exception during logic processing: {e}")

            # --- PUSH TO RENDER PIPELINE ---
            try:
                self.render_queue.put_nowait(current_frame)
            except queue.Full:
                pass

            # --- TELEMETRY REPORTING ---
            current_time = time.time()
            if current_time - last_telemetry_print > 5.0:
                self._print_telemetry()
                last_telemetry_print = current_time

        # ==========================================================
        # CLEAN SHUTDOWN (Executes only once)
        # ==========================================================
        self.cap.release()
        if self.grabber_thread and self.grabber_thread.is_alive():
            self.grabber_thread.join(timeout=1.0)
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
