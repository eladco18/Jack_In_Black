import sys
import threading
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QCheckBox
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import numpy as np
import math

# Import our custom modules
from vision_engine import VisionEngine
from game_logic import GameLogic
from game_manager import GameManager
from ui_manager import UIManager
from camera_thread import CameraThread

import os
import PyQt5

plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

# camera_id = "http://192.168.1.65:8080/video"      # wireless
# camera_id = "http://10.156.202.187:8080/video"    # wired


class IpConnectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Connection")
        self.setFixedSize(500, 250)
        self.setStyleSheet("""
            QDialog { background-color: #121212; border: 2px solid #333; }
            QLabel { color: #ffffff; font-size: 26px; font-weight: bold; }
            QLineEdit { 
                background-color: #2a2a2a; color: white; border: 1px solid #444; 
                padding: 10px; border-radius: 6px; font-size: 26px;
            }
            QPushButton {
                background-color: #00e5ff; color: #000; font-weight: bold;
                border-radius: 6px; padding: 12px; font-size: 26px;
            }
            QPushButton:hover { background-color: #00b8cc; }
            QCheckBox { 
                color: #ffffff; 
                font-size: 20px; 
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #00e5ff;
                border: 2px solid #00e5ff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        layout.addWidget(QLabel("Enter IP Webcam URL:"))

        self.ip_input = QLineEdit()
        self.ip_input.setText("http://192.168.1.65:8080/video")  # default IP
        layout.addWidget(self.ip_input)

        self.check_norm = QCheckBox("Board color normalization")
        self.check_norm.setChecked(False)  # Default is Off
        self.check_norm.setStyleSheet("color: white; font-size: 20px; margin-top: 10px;")
        layout.addWidget(self.check_norm)

        layout.addStretch()

        self.btn_connect = QPushButton("Connect Camera")
        self.btn_connect.clicked.connect(self.accept)
        layout.addWidget(self.btn_connect)

    def get_config(self):
        return self.ip_input.text().strip(), self.check_norm.isChecked()


def probe_camera_and_get_scale(cam_id: str) -> tuple:
    """
    Probes the camera at startup, calculates scaling dynamically,
    and ENFORCES a strict 16:9 aspect ratio.
    Returns: (actual_width, actual_height, system_scale)
    """
    print("[SYSTEM] Probing camera hardware...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"CRITICAL ERROR: Cannot connect to the camera at {cam_id}")

    # Read the actual physical output of the camera stream
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if actual_width == 0 or actual_height == 0:
        raise RuntimeError("CRITICAL ERROR: Camera returned a 0x0 frame.")

    # Calculate individual scales relative to our 720p baseline
    scale_x = actual_width / 1280.0
    scale_y = actual_height / 720.0

    # Strict Aspect Ratio Validation (Allowing for tiny floating-point rounding errors)
    if not math.isclose(scale_x, scale_y, rel_tol=0.01):
        raise ValueError(
            f"CRITICAL ERROR: Aspect Ratio Mismatch!\n"
            f"The camera is broadcasting at {int(actual_width)}x{int(actual_height)}.\n"
            f"This breaks the required 16:9 aspect ratio. The system cannot guarantee circular chip detection.\n"
            f"Please configure your IP Webcam to a 16:9 resolution (e.g., 1280x720, 1920x1080, 3840x2160)."
        )

    system_scale = scale_x

    print(f"[SYSTEM] Hardware successfully verified at {int(actual_width)}x{int(actual_height)}.")
    print(f"[SYSTEM] Global Scale Factor locked at: {system_scale:.2f}")

    return int(actual_width), int(actual_height), system_scale


class GameSignals(QObject):
    """
    The centralized Event Bus. Matches the UI Manager exactly!
    """
    # Video Feed
    video_update = pyqtSignal(np.ndarray)

    # Game Flow & Data
    update_timer = pyqtSignal(int)
    update_bet = pyqtSignal(str, int)
    update_hand = pyqtSignal(str, str)
    update_bankroll = pyqtSignal(str, int)
    update_phase = pyqtSignal(str)

    # Decisions & Strategy
    show_decision_prompt = pyqtSignal(bool)
    update_strategy = pyqtSignal(str, str, object)
    game_alert = pyqtSignal(str, str)

    # Decision Timer
    start_timer_signal = pyqtSignal(int, str)
    stop_timer_signal = pyqtSignal()

    # Phase Control
    calibration_complete = pyqtSignal(dict)
    game_start_request = pyqtSignal(dict)
    action_decision = pyqtSignal(str)

    # Shuffle Flow
    show_shuffle_button = pyqtSignal(bool)
    shuffle_complete = pyqtSignal()

    # End-of-Round Flow
    round_over_signal = pyqtSignal()
    new_round_clicked = pyqtSignal()
    new_game_clicked = pyqtSignal()

    financial_result = pyqtSignal(str, int, int)  # (player_name, payout_amount, original_bet)


def _on_calibration_complete(roi_dict: dict, vision_engine: object, game_manager: object) -> None:
    name_map = {
        "Felt Calibration": "felt_roi",
        "Dealer Cards": "dealer_cards",
        "Player 1 Cards": "player_1_cards",
        "Player 2 Cards": "player_2_cards",
        "Player 1 Chips": "player_1_chips",
        "Player 2 Chips": "player_2_chips",
        "P1 Decision": "player_1_decision",
        "P2 Decision": "player_2_decision",
    }

    for ui_name, roi_cords in roi_dict.items():
        internal_key = name_map.get(ui_name)
        if internal_key:
            x, y, w, h = roi_cords
            vision_engine.rois[internal_key] = (x, y, x + w, y + h)  # Convert (x,y,w,h) → (x1,y1,x2,y2)

    game_manager.current_state = game_manager.STATE_INIT_GAME
    print("[SYSTEM] Calibration complete. Transitioning to STATE_INIT_GAME.")


def get_camera_resolution_and_scale(cam_id: str) -> tuple:
    """
    Probes the camera to find its actual native resolution and calculates
    the scale factor relative to our baseline of 1280x720.
    """
    print("[SYSTEM] Probing camera for native resolution...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_FFMPEG)

    # Fallback to base resolution if camera fails to open temporarily
    if not cap.isOpened():
        print("[WARNING] Could not probe camera. Defaulting to 1280x720 baseline.")
        return 1280, 720, 1.0

    # Read the actual width and height directly from the camera hardware
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    # Safety check in case the stream returns 0
    if actual_width == 0 or actual_height == 0:
        return 1280, 720, 1.0

    # Calculate scale factor relative to 1280p base
    scale_factor = actual_width / 1280.0

    print(f"[SYSTEM] Camera locked at {int(actual_width)}x{int(actual_height)}.")
    print(f"[SYSTEM] System Scale Factor calculated as: {scale_factor:.2f}")

    return int(actual_width), int(actual_height), scale_factor


def main():
    app = QApplication(sys.argv)
    signals = GameSignals()

    # --- 1. IP Connection Flow ---
    camera_id = ""
    cam_width, cam_height, dynamic_scale = 0, 0, 1.0
    connected = False

    while not connected:
        ip_dialog = IpConnectDialog()
        if ip_dialog.exec_() == QDialog.Accepted:
            # Get the IP and the normalization flag
            camera_id, enable_norm = ip_dialog.get_config()
            try:
                # 1. Dynamically probe the camera before starting any engines
                cam_width, cam_height, dynamic_scale = probe_camera_and_get_scale(camera_id)
                connected = True
            except Exception as e:
                # If connection fails, show an error box and the loop will restart
                err_box = QMessageBox()
                err_box.setIcon(QMessageBox.Critical)
                err_box.setWindowTitle("Connection Error")
                err_box.setStyleSheet(""" 
                QLabel { color: black; font-size: 24px; min-height: 150px;} 
                QPushButton { font-size: 24px; padding: 10px; min-width: 100px;}""")
                err_box.setText(
                    f"<p>Failed to connect to the camera.<br>"
                    f"Make sure the IP is correct and the server is running.</p>"
                    f"<p><b>Details:</b>{str(e)}</p>"
                )
                err_box.exec_()
        else:
            # User closed the dialog window
            print("[SYSTEM] Setup cancelled by user. Exiting.")
            sys.exit(0)

    # --- 2. Initialize Game Systems (Only happens if connected successfully) ---
    # Inject the dynamic scale directly into the Vision Engine
    vision = VisionEngine(scale=dynamic_scale, enable_color_norm=enable_norm)
    logic = GameLogic()
    manager = GameManager()

    # UI Manager internalizes the signals and connects them itself
    ui = UIManager(signals, enable_color_norm=enable_norm)

    # 3. Tell the CameraThread exactly what resolution to use based on our probe
    camera_thread = CameraThread(engine=vision, camera_id=camera_id, buffer_size=20, target_res=(cam_width, cam_height))

    # Bridge the camera's local signal directly to the Event Bus!
    camera_thread.frame_ready_signal.connect(signals.video_update.emit)

    # Only wire signals that go BACK from the UI to the Core Logic
    signals.calibration_complete.connect(lambda roi_dict: _on_calibration_complete(roi_dict, vision, manager))
    signals.new_round_clicked.connect(manager.on_new_round_clicked)
    signals.new_game_clicked.connect(manager.on_new_game_clicked)
    signals.shuffle_complete.connect(manager.on_shuffle_complete_clicked)

    # Hook the UI game start request to initialize the players!
    signals.game_start_request.connect(manager.on_game_start)

    manager.inject_dependencies(
        vision_engine=vision,
        game_logic=logic,
        ui_manager=ui,
        camera_thread=camera_thread,
        signals=signals
    )

    camera_thread.start()
    manager_thread = threading.Thread(target=manager.run_game_loop, daemon=True)
    manager_thread.start()

    ui.showFullScreen()
    print("[SYSTEM] System initialized. Waiting for Calibration...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
