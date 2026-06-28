import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import os
import time
import threading

# Import set_system_scale properly from modules
from chips_classifier import chips_classifier, set_system_scale as set_chips_scale
from cards_classifier import RANK_W, RANK_H, SUIT_W, SUIT_H, CARD_W, CARD_H, CORNER_W, CORNER_H, \
    set_system_scale as set_cards_scale
from decision_chip_classifier import detect_decision_chip, set_system_scale as set_decision_scale
import cards_classifier
import decision_chip_classifier as ddc

# ==========================================
# Vision System Settings (Image Processing)
# ==========================================
SKIP_FRAMES = 5  # Number of frames to skip between detections to reduce CPU load
SENSITIVITY = 90  # White color mask sensitivity (lower means harder to detect in low light)
THRESHOLD_VALUE = 127  # Binary threshold value for template loading
MEMORY_TIMEOUT = 1  # Seconds to keep drawing a detected object after it's obscured/lost
DUPLICATE_DIST_THRESH = 30  # Base pixel distance (at 720p) to merge nearby detections (jitter filter)

# chip_trackers arguments (SYSTEM_SCALE multiplication is performed in ChipMemory)
max_dist = 15  # Max spatial distance allowed to link a new detection to an existing tracked object.
max_unseen = 5  # Max consecutive frames an object can be undetected before it is permanently removed from memory
min_hits = 2  # Min consecutive frames an object must be detected before it is confirmed as "real"
move_threshold = 8  # Min distance an object must shift to update its position


# Helper_function - takes a frame function to run over multiple frames for stable detection
def stabilize_detection(frames_list: List[np.ndarray], detection_func: Callable, *args, **kwargs) -> Any:
    """
    Processes a sequence of frames and returns the most stable detection result.
    Uses Majority Vote (Mode) for ALL detections (Bets, Cards, Decisions),
    assuming frames are captured when the table is clear of hands.
    """
    results = []

    for frame in frames_list:
        val = detection_func(frame, *args, **kwargs)

        if val is not None:
            if isinstance(val, list):
                val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
            results.append(val)

    if not results:
        return None

    # Safe Majority Vote using Counter (handles ties and edge cases gracefully)
    from collections import Counter
    counts = Counter(results)

    # most_common(1) returns a list with one tuple: [(winning_value, vote_count)]
    stable_val = counts.most_common(1)[0][0]

    # RESTORE ORIGINAL FORMAT: Revert tuples back to lists
    if isinstance(stable_val, tuple):
        stable_val = [list(x) if isinstance(x, tuple) else x for x in stable_val]

    return stable_val


class ChipMemoryTracker:
    def __init__(self, scale=1.0, max_dist=15, max_unseen=15, min_hits=2, move_threshold=7):
        self.chips = {}
        self.next_id = 0
        self.max_dist = max_dist * scale
        self.max_unseen = max_unseen
        self.min_hits = min_hits
        self.move_threshold = move_threshold * scale

    def update(self, detected_chips_list):
        new_memory = {}
        valid_chips_to_return = []

        for detected_chip in detected_chips_list:
            found_match = False
            curr_pos = detected_chip["pos"]

            for chip_id, chip_data in self.chips.items():
                # Calculate Euclidean distance
                dist = math.sqrt((curr_pos[0] - chip_data["pos"][0]) ** 2 + (curr_pos[1] - chip_data["pos"][1]) ** 2)

                if dist < self.max_dist:
                    votes = chip_data.get("votes", {})
                    color_votes = chip_data.get("color_votes", {})

                    det_color = detected_chip["color"]
                    det_val = detected_chip["value"]

                    if det_color != "UNKNOWN":
                        votes[det_val] = votes.get(det_val, 0) + 1
                        color_votes[det_color] = color_votes.get(det_color, 0) + 1

                    best_val = max(votes, key=votes.get) if votes else 0
                    best_color = max(color_votes, key=color_votes.get) if color_votes else "UNKNOWN"

                    # --- Exponential Moving Average (EMA) implementation ---
                    # Alpha controls the smoothing. Lower = smoother but slower reaction.
                    alpha = 0.35

                    old_cx, old_cy = chip_data["pos"]
                    new_cx, new_cy = curr_pos

                    # Smooth the position coordinates
                    smoothed_cx = int(old_cx * (1 - alpha) + new_cx * alpha)
                    smoothed_cy = int(old_cy * (1 - alpha) + new_cy * alpha)

                    # Smooth the radius to prevent UI flickering/breathing
                    old_r = chip_data["radius"]
                    new_r = detected_chip["radius"]
                    smoothed_radius = int(old_r * (1 - alpha) + new_r * alpha)

                    new_memory[chip_id] = {
                        "id": chip_id,
                        "color": best_color,
                        "value": best_val,
                        "pos": (smoothed_cx, smoothed_cy),
                        "radius": smoothed_radius,
                        "unseen": 0,
                        "hits": chip_data.get("hits", 0) + 1,
                        "votes": votes,
                        "color_votes": color_votes
                    }
                    found_match = True
                    break

            if not found_match and detected_chip["color"] != "UNKNOWN":
                # Register a completely new chip
                new_memory[self.next_id] = {
                    "id": self.next_id,  # CRITICAL: Assign ID on creation
                    "color": detected_chip["color"],
                    "value": detected_chip["value"],
                    "pos": curr_pos,
                    "radius": detected_chip["radius"],
                    "unseen": 0,
                    "hits": 1,
                    "votes": {detected_chip["value"]: 1},
                    "color_votes": {detected_chip["color"]: 1}
                }
                self.next_id += 1

        # Handle chips that were not seen in this frame
        for chip_id, chip_data in self.chips.items():
            if chip_id not in new_memory and chip_data["unseen"] < self.max_unseen:
                chip_data["unseen"] += 1
                new_memory[chip_id] = chip_data

        self.chips = new_memory

        # Filter out ghost detections (require minimum hits)
        for chip_id, chip_data in self.chips.items():
            if chip_data["hits"] >= self.min_hits:
                # Append a copy to prevent accidental reference mutations downstream
                valid_chips_to_return.append(chip_data.copy())

        return valid_chips_to_return


class VisionEngine:
    """
    Handles all computer vision tasks for the Blackjack game.
    Includes ROI calibration, chip detection (HSV & Hough Circles),
    and card detection (Template Matching).
    """

    RANK_TO_VALUE = {
        '2': 2, 'two': 2,
        '3': 3, 'three': 3,
        '4': 4, 'four': 4,
        '5': 5, 'five': 5,
        '6': 6, 'six': 6,
        '7': 7, 'seven': 7,
        '8': 8, 'eight': 8,
        '9': 9, 'nine': 9,
        '10': 10, 'ten': 10,
        'j': 10, 'jack': 10,
        'q': 10, 'queen': 10,
        'k': 10, 'king': 10,
        'a': 1, 'ace': 1
    }

    def __init__(self, scale: float = 1.0, enable_color_norm: bool = False) -> None:
        self.scale = scale
        self.DUPLICATE_DIST_THRESH = 35 * self.scale
        self.enable_color_norm = enable_color_norm

        # --- Phase A: Color Normalization Variables ---
        # Ideal Green (BGR) optimized for the existing chips HSV thresholds
        self.target_green_bgr = np.array([80.0, 94.0, 58.0], dtype=np.float32)
        self.color_gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.is_color_calibrated = False

        # Apply global scale to classifiers ONE TIME at startup
        set_chips_scale(self.scale)
        set_cards_scale(self.scale)
        set_decision_scale(self.scale)

        # Initialize ChipMemory tracking with NO scale parameter (it uses SYSTEM_SCALE internally now)
        self.chip_trackers = {
            "player_1_chips": ChipMemoryTracker(scale=self.scale, max_dist=15, max_unseen=5, min_hits=2),
            "player_2_chips": ChipMemoryTracker(scale=self.scale, max_dist=15, max_unseen=5, min_hits=2)
        }

        self.decision_trackers = {
            "player_1_decision": ChipMemoryTracker(scale=self.scale, max_dist=15, max_unseen=5, min_hits=2),
            "player_2_decision": ChipMemoryTracker(scale=self.scale, max_dist=15, max_unseen=5, min_hits=2)
        }

        # Dictionary to store the Regions of Interest (ROIs) coordinates.
        # Format for each ROI: (x1, y1, x2, y2)
        self.rois: Dict[str, Optional[Tuple[int, int, int, int]]] = {
            "player_1_cards": None,
            "player_1_chips": None,
            "player_1_decision": None,
            "player_2_cards": None,
            "player_2_chips": None,
            "player_2_decision": None,
            "dealer_cards": None
        }

        # Dictionary to hold the loaded card templates.
        # Key: Card name/value, Value: The template image matrix.
        self.rank_templates: Dict[str, np.ndarray] = {}
        self.suit_templates: Dict[str, np.ndarray] = {}
        self.load_templates()
        # Memory Cache for Visual Persistence (prevent flickering)
        self.drawing_memory = {}
        self.memory_lock = threading.Lock()

        self.logic_frame_counter = 0
        self.UI_SKIP_FRAMES = 5
        self.player_overlays = {1: None, 2: None}
        self.system_overlay: Optional[str] = None

    def set_player_overlay(self, player_num: int, text: str) -> None:
        """Sets the massive text overlay to be drawn over a player's cards."""
        self.player_overlays[player_num] = text

    def clear_player_overlays(self) -> None:
        """Clears all text overlays for the new round."""
        self.player_overlays = {1: None, 2: None}

    def load_folder(self, folder_path: str, width: int, height: int) -> Dict[str, np.ndarray]:
        """
        Helper function to read images from a folder, apply threshold,
        and resize them to the standard SYM_W x SYM_H dimensions.
        """
        templates = {}
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            return templates

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, filename)
                img = cv2.imread(full_path, 0)

                if img is not None:
                    _, b = cv2.threshold(img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

                    coords = cv2.findNonZero(b)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        b = b[y:y + h, x:x + w]

                    resized_img = cv2.resize(b, (width, height), interpolation=cv2.INTER_CUBIC)
                    card_name = os.path.splitext(filename)[0]
                    templates[card_name] = resized_img
        return templates

    def load_templates(self) -> None:
        """
        Loads card templates from local directories into separate dictionaries
        for ranks and suits.
        """
        self.rank_templates = self.load_folder('templates/ranks', RANK_W, RANK_H)
        self.suit_templates = self.load_folder('templates/suits', SUIT_W, SUIT_H)

    def analyze_player_bet(self, roi_frame: np.ndarray, player_name_vision=1) -> int:
        """
        Receives the cropped betting area.
        Returns the total bet amount using raw detection for the frame.
        (Stabilization is handled externally by GameManager evaluating multiple frames).
        """
        chips = chips_classifier(roi_frame, debug=False, player_name=player_name_vision)
        return sum(chip["value"] for chip in chips)

    def detect_decision_chip_in_vision(self, frame: np.ndarray, roi_coords, chosen_player=1) -> Optional[str]:
        """
        Acts as a clean Facade. Fetches the exact decision string (HIT/STAND/DOUBLE/SURRENDER)
        from the decision module and passes it directly to the GameManager.
        """
        return detect_decision_chip(frame, roi_coords, chosen_player, show_results=False)

    def detect_cards(self, roi_frame: np.ndarray) -> Tuple[Tuple[int, int, int], ...]:
        """
        Detects playing cards in the given ROI.
        Returns a sorted tuple of tuples in the format: ((value, center_x, center_y), ...)
        This spatial data format is critical for the GameManager's Centroid Tracking.
        """
        detected_items = []

        # 1. Process the ROI frame through the classification pipeline
        raw_cards_data = cards_classifier.analyze_frame(
            frame=roi_frame,
            rank_t=self.rank_templates,
            suit_t=self.suit_templates,
            sensitivity=SENSITIVITY
        )

        # 2. Extract numeric value and spatial center (cx, cy)
        for card in raw_cards_data:
            rank_str = card["rank"].lower()

            if rank_str in self.RANK_TO_VALUE:
                val = self.RANK_TO_VALUE[rank_str]
                cx, cy = card["center"]
                # Scale the spatial rounding grid (was 30.0)
                grid_size = 30.0 * self.scale
                stable_cx = int(round(cx / grid_size) * grid_size)
                stable_cy = int(round(cy / grid_size) * grid_size)
                detected_items.append((val, stable_cx, stable_cy))

        # 3. Sort by X coordinate (cx) to ensure the tuple order is consistent
        # for the stabilize_detection's Counter vote.
        detected_items.sort(key=lambda item: item[1])

        return tuple(detected_items)

    def process_logic(self, raw_frame: np.ndarray) -> None:
        """
        Runs the heavy computer vision logic (Hough, Templates) only on designated frames.
        Does NOT block the render thread during heavy computation.
        """
        self.logic_frame_counter += 1
        is_heavy_frame = (self.logic_frame_counter % self.UI_SKIP_FRAMES == 0)

        if not is_heavy_frame:
            return

        start_time_perf = time.perf_counter()

        # ====================================================================
        # HEAVY DETECTION PHASE: UNLOCKED! (Allows UI to keep rendering)
        # ====================================================================
        cards_to_draw = self._get_cards_draw_data(raw_frame)
        chips_to_draw = self._get_chips_draw_data(raw_frame)
        decisions_to_draw = self._get_decision_draw_data(raw_frame)
        all_new_detections = cards_to_draw + chips_to_draw + decisions_to_draw

        current_time = time.time()

        # ====================================================================
        # MEMORY UPDATE PHASE: SURGICAL LOCK (Very fast, ~0.1ms)
        # ====================================================================
        with self.memory_lock:
            for new_det in all_new_detections:
                new_cx, new_cy = new_det["center"]
                found_match = False

                for mem_id, mem_data in self.drawing_memory.items():
                    if mem_data["roi"] == new_det["roi"]:
                        old_cx, old_cy = mem_data["center"]
                        dist = math.hypot(new_cx - old_cx, new_cy - old_cy)

                        if dist < self.DUPLICATE_DIST_THRESH:
                            self.drawing_memory[mem_id]["time"] = current_time
                            self.drawing_memory[mem_id]["hits"] += 1
                            self.drawing_memory[mem_id]["label"] = new_det["label"]
                            self.drawing_memory[mem_id]["color"] = new_det["color"]

                            # Apply Jitter Filter
                            if dist > 3:
                                smooth_cx = int((old_cx + new_cx) / 2)
                                smooth_cy = int((old_cy + new_cy) / 2)
                                self.drawing_memory[mem_id]["center"] = (smooth_cx, smooth_cy)

                                if new_det["shape"] in ["chip", "decision"]:
                                    old_r = mem_data["shape_data"]["radius"]
                                    new_r = new_det["shape_data"]["radius"]
                                    self.drawing_memory[mem_id]["shape_data"]["radius"] = int((old_r + new_r) / 2)
                                else:
                                    self.drawing_memory[mem_id]["shape_data"] = new_det["shape_data"]

                            found_match = True
                            break

                if not found_match:
                    self.drawing_memory[new_det["id"]] = {
                        "center": new_det["center"],
                        "shape": new_det["shape"],
                        "shape_data": new_det["shape_data"],
                        "label": new_det["label"],
                        "color": new_det["color"],
                        "roi": new_det["roi"],
                        "time": current_time,
                        "hits": 1
                    }

        logic_time_ms = (time.perf_counter() - start_time_perf) * 1000
        print(f"[PROFILER] 🔴 HEAVY LOGIC: {logic_time_ms:.1f}ms")

    def render_ui(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Draws current memory state onto the frame. Runs blindly at 15 FPS.
        Protected by memory lock to ensure thread safety with the logic thread.
        """
        annotated_frame = raw_frame.copy()
        current_time = time.time()
        keys_to_delete = []

        # ====================================================================
        # DRAWING PHASE: PROTECTED READ (Extremely fast)
        # ====================================================================
        with self.memory_lock:
            for det_id, det_data in self.drawing_memory.items():
                if current_time - det_data["time"] < MEMORY_TIMEOUT:

                    if det_data["shape"] in ["chip", "decision"] and det_data["hits"] < 2:
                        continue

                    color = det_data["color"]
                    label = det_data["label"]
                    cx, cy = det_data["center"]
                    shape = det_data["shape"]
                    shape_data = det_data["shape_data"]

                    if shape == "card":
                        contour = shape_data["contour"]
                        pos_x, pos_y = shape_data["pos"]

                        base_thick = 4
                        scaled_thick = max(1, int(base_thick * self.scale))

                        cv2.drawContours(annotated_frame, [contour], -1, color, scaled_thick, cv2.LINE_AA)

                        font = cv2.FONT_HERSHEY_DUPLEX
                        scaled_font_scale = 0.55 * self.scale
                        scaled_font_thick = max(1, int(1 * self.scale))

                        (text_w, text_h), _ = cv2.getTextSize(label, font, scaled_font_scale, scaled_font_thick)

                        scaled_padding_x = int(6 * self.scale)
                        scaled_padding_y = int(4 * self.scale)

                        cv2.rectangle(annotated_frame,
                                      (pos_x, pos_y - text_h - scaled_padding_y * 2),
                                      (pos_x + text_w + scaled_padding_x * 2, pos_y),
                                      (30, 30, 30), -1)

                        cv2.rectangle(annotated_frame,
                                      (pos_x, pos_y - text_h - scaled_padding_y * 2),
                                      (pos_x + text_w + scaled_padding_x * 2, pos_y),
                                      color, max(1, int(1 * self.scale)), cv2.LINE_AA)

                        cv2.putText(annotated_frame, label,
                                    (pos_x + scaled_padding_x, pos_y - scaled_padding_y),
                                    font, scaled_font_scale, (255, 255, 255), scaled_font_thick, cv2.LINE_AA)

                    elif shape == "chip":
                        r = shape_data["radius"]
                        thick2 = max(1, int(2 * self.scale))
                        dot_radius = max(1, int(2 * self.scale))

                        cv2.circle(annotated_frame, (cx, cy), r, color, thick2, cv2.LINE_AA)
                        cv2.circle(annotated_frame, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)

                        font = cv2.FONT_HERSHEY_DUPLEX
                        scaled_font_scale = 0.65 * self.scale
                        display_text = f"{label}$"

                        text_x = cx - int(20 * self.scale)
                        text_y = cy - r - int(10 * self.scale)

                        shadow_thick = max(1, int(5 * self.scale))
                        text_thick = max(1, int(1.5 * self.scale))

                        cv2.putText(annotated_frame, display_text, (text_x, text_y),
                                    font, scaled_font_scale, (0, 0, 0), shadow_thick, cv2.LINE_AA)
                        cv2.putText(annotated_frame, display_text, (text_x, text_y),
                                    font, scaled_font_scale, color, text_thick, cv2.LINE_AA)

                    elif shape == "decision":
                        r = shape_data["radius"]
                        thick3 = max(1, int(3 * self.scale))
                        thick1 = max(1, int(1 * self.scale))
                        inner_offset = int(6 * self.scale)

                        cv2.circle(annotated_frame, (cx, cy), r, color, thick3, cv2.LINE_AA)
                        cv2.circle(annotated_frame, (cx, cy), r - inner_offset, color, thick1, cv2.LINE_AA)

                        font = cv2.FONT_HERSHEY_DUPLEX
                        scaled_font_scale = 0.55 * self.scale

                        (text_w, text_h), _ = cv2.getTextSize(label, font, scaled_font_scale, thick1)
                        text_x = cx - (text_w // 2)
                        text_y = cy - r - int(10 * self.scale)

                        shadow_thick = max(1, int(4 * self.scale))

                        cv2.putText(annotated_frame, label, (text_x, text_y),
                                    font, scaled_font_scale, (0, 0, 0), shadow_thick, cv2.LINE_AA)
                        cv2.putText(annotated_frame, label, (text_x, text_y),
                                    font, scaled_font_scale, color, thick1, cv2.LINE_AA)

                else:
                    keys_to_delete.append(det_id)

            # Cleanup expired memory
            for key in keys_to_delete:
                del self.drawing_memory[key]

        # =========================================================
        # OVERLAYS (No memory lock needed here)
        # =========================================================
        for p_num, text in self.player_overlays.items():
            if text:
                roi_key = f"player_{p_num}_cards"
                if self.rois.get(roi_key):
                    x1, y1, x2, y2 = self.rois[roi_key]
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2

                    color = (255, 229, 0) if p_num == 1 else (255, 0, 189)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    lines = text.split('\n')

                    if len(lines) > 1:
                        scaled_font_scale = 0.75 * self.scale
                        scaled_font_thick = max(1, int(2 * self.scale))
                        scaled_vertical_spacing = int(55 * self.scale)
                    else:
                        scaled_font_scale = 1.8 * self.scale
                        scaled_font_thick = max(1, int(4 * self.scale))
                        scaled_vertical_spacing = 0

                    total_block_h = len(lines) * scaled_vertical_spacing if len(lines) > 1 else 0
                    current_y_offset = cy - (total_block_h // 2)
                    shadow_offset = max(1, int(3 * self.scale))

                    for i, line in enumerate(lines):
                        (tw, th), _ = cv2.getTextSize(line, font, scaled_font_scale, scaled_font_thick)
                        tx = cx - (tw // 2)
                        ty = current_y_offset + (i * scaled_vertical_spacing) + th if len(lines) > 1 else cy + (th // 2)

                        cv2.putText(annotated_frame, line, (tx + shadow_offset, ty + shadow_offset), font,
                                    scaled_font_scale, (0, 0, 0), scaled_font_thick + max(1, int(2 * self.scale)),
                                    cv2.LINE_AA)
                        cv2.putText(annotated_frame, line, (tx, ty), font, scaled_font_scale, color,
                                    scaled_font_thick, cv2.LINE_AA)

        if getattr(self, 'system_overlay', None):
            lines = self.system_overlay.split('\n')
            font = cv2.FONT_HERSHEY_DUPLEX
            scaled_font_scale = 1.6 * self.scale
            scaled_font_thick = max(1, int(3 * self.scale))
            color = (0, 165, 255)
            frame_h, frame_w = annotated_frame.shape[:2]

            line_height = cv2.getTextSize("Q", font, scaled_font_scale, scaled_font_thick)[0][1] + int(20 * self.scale)
            total_h = len(lines) * line_height
            current_y = (frame_h - total_h) // 4
            shadow_offset = max(1, int(4 * self.scale))

            for line in lines:
                (tw, th), _ = cv2.getTextSize(line, font, scaled_font_scale, scaled_font_thick)
                tx = (frame_w - tw) // 2
                ty = current_y + th

                cv2.putText(annotated_frame, line, (tx + shadow_offset, ty + shadow_offset), font,
                            scaled_font_scale, (0, 0, 0), scaled_font_thick + max(1, int(2 * self.scale)), cv2.LINE_AA)
                cv2.putText(annotated_frame, line, (tx, ty), font, scaled_font_scale, color,
                            scaled_font_thick, cv2.LINE_AA)
                current_y += line_height

        return annotated_frame

    def _get_cards_draw_data(self, frame: np.ndarray) -> list:
        draw_data = []
        for roi_name, coords in self.rois.items():
            if coords is None or "cards" not in roi_name: continue
            x1, y1, x2, y2 = coords
            roi_frame = frame[y1:y2, x1:x2]

            detected_cards = cards_classifier.analyze_frame(
                frame=roi_frame, rank_t=self.rank_templates, suit_t=self.suit_templates)

            for i, card in enumerate(detected_cards):
                pts = np.array(card["contour"]) + [x1, y1]
                pos_x = int(card["pos"][0] + x1)
                pos_y = int(card["pos"][1] + y1)
                cx, cy = pos_x, pos_y

                loc_x, loc_y = (cx // 10) * 10, (cy // 10) * 10

                draw_data.append({
                    "id": f"card_{roi_name}_{card['rank']}_{card['suit']}_x{loc_x}_y{loc_y}",
                    "roi": roi_name,
                    "center": (cx, cy),
                    "shape": "card",
                    "shape_data": {"contour": pts, "pos": (pos_x, pos_y)},
                    "label": f"{card['rank']} {card['suit']}",
                    "color": card["color"]
                })
        return draw_data

    def _get_chips_draw_data(self, frame: np.ndarray) -> list:
        draw_data = []

        for roi_name, coords in self.rois.items():
            if coords is None or "chips" not in roi_name:
                continue

            x1, y1, x2, y2 = coords
            roi_frame = frame[y1:y2, x1:x2]
            player_num = 1 if "p1" in roi_name or "player_1" in roi_name else 2

            raw_local_chips = chips_classifier(roi_frame, debug=False, player_name=player_num)

            # Route through the stabilized ChipMemory
            if roi_name in self.chip_trackers:
                stable_chips = self.chip_trackers[roi_name].update(raw_local_chips)
            else:
                stable_chips = raw_local_chips

            for chip in stable_chips:
                abs_cx = chip["pos"][0] + x1
                abs_cy = chip["pos"][1] + y1
                r = chip["radius"]
                val = chip["value"]

                # Fetch the true ID from the memory tracker
                real_id = chip.get("id", f"{abs_cx}_{abs_cy}")

                # Map chip value to its specific Neon color
                if val == 5:
                    chip_color = (50, 50, 255)  # Neon Red
                elif val == 10:
                    chip_color = (50, 255, 50)  # Poison Green
                elif val == 25:
                    chip_color = (255, 100, 100)  # Sky Blue
                elif val == 50:
                    chip_color = (100, 100, 100)  # Dark Gray
                elif val == 1:
                    chip_color = (255, 255, 255)  # Pure White
                else:
                    chip_color = (0, 255, 255)  # Fallback Yellow

                draw_data.append({
                    # Use the stable tracked ID instead of generating one based on coordinates
                    "id": f"chip_{roi_name}_{real_id}",
                    "roi": roi_name,
                    "center": (abs_cx, abs_cy),
                    "shape": "chip",
                    "shape_data": {"radius": r},
                    "label": str(val),
                    "color": chip_color
                })

        return draw_data

    def _get_decision_draw_data(self, frame: np.ndarray) -> list:
        draw_data = []
        for roi_name, coords in self.rois.items():
            if coords is None or "decision" not in roi_name:
                continue

            player_num = 1 if "p1" in roi_name or "player_1" in roi_name else 2

            # 1. Fetch raw chip detections from the decision module
            raw_decision_chips = ddc.analyze_player_area(frame, coords, debug=False, player_name=f"Player {player_num}")

            # 2. Map the raw data to match the format expected by ChipMemory
            # We store the action string (e.g., "HIT") inside the "value" key
            mapped_chips = []
            for chip in raw_decision_chips:
                mapped_chips.append({
                    "color": "DECISION",  # Color is implicitly handled by the action label
                    "value": chip["action"].upper(),  # Store the string action as the value
                    "pos": chip["pos"],
                    "radius": chip["radius"]
                })

            # 3. Route through the stabilized memory tracker to apply EMA and preserve ID
            if roi_name in self.decision_trackers:
                stable_chips = self.decision_trackers[roi_name].update(mapped_chips)
            else:
                stable_chips = mapped_chips

            # 4. Generate drawing instructions using the smoothed data and stable ID
            for chip in stable_chips:
                cx, cy = chip["pos"]
                r = chip["radius"]
                action = chip["value"]  # Recover the action string

                # Fetch the true ID assigned by ChipMemory
                real_id = chip.get("id", f"{cx}_{cy}")

                # Map specific decisions to Neon colors
                if "HIT" in action:
                    color = (255, 100, 100)  # Neon Blue
                    label = "HIT"
                elif "STAND" in action:
                    color = (50, 50, 255)  # Bright Red
                    label = "STAND"
                elif "DOUBLE" in action:
                    color = (0, 215, 255)  # Gold/Yellow
                    label = "DOUBLE DOWN"
                elif "SURRENDER" in action:
                    color = (40, 100, 80)  # Brown
                    label = "SURRENDER"
                else:
                    color = (200, 200, 200)  # Fallback Gray
                    label = action

                draw_data.append({
                    # Use the stable ID to prevent UI flickering
                    "id": f"decision_{roi_name}_{real_id}",
                    "roi": roi_name,
                    "center": (cx, cy),
                    "shape": "decision",
                    "shape_data": {"radius": r},
                    "label": label,
                    "color": color
                })

        return draw_data

    def calibrate_rois(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            print("Error: The provided frame is empty or invalid.")
            return

        print("\n--- ROI Calibration Mode ---")
        print("Drag the mouse to draw a rectangle, then press ENTER.")
        print("To skip an area, just press ENTER without drawing.")

        areas_to_select = [
            ("player_1_cards", "Select Player 1 CARDS"),
            ("player_1_chips", "Select Player 1 CHIPS"),
            ("player_1_decision", "Select Player 1 DECISION"),
            ("player_2_cards", "Select Player 2 CARDS"),
            ("player_2_chips", "Select Player 2 CHIPS"),
            ("player_2_decision", "Select Player 2 DECISION"),
            ("dealer_cards", "Select Dealer CARDS")
        ]

        for roi_key, window_name in areas_to_select:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1000, 700)

            roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)

            x, y, w, h = roi
            if w > 0 and h > 0:
                self.rois[roi_key] = (x, y, x + w, y + h)
                print(f"[{roi_key}] Saved: {self.rois[roi_key]}")
            else:
                print(f"[{roi_key}] Skipped.")

        print("--- Calibration Complete ---\n")

    def set_system_overlay(self, text: Optional[str]) -> None:
        """Sets or clears a global massive text overlay in the center of the video."""
        self.system_overlay = text

    # In vision_engine.py
    def calibrate_lighting(self, empty_table_frame: np.ndarray) -> None:
        """
        Phase A: Calibration (Manual/Static Setup)
        Uses the manually selected ROI to sample the green felt and compute gains.
        """
        if not self.enable_color_norm or empty_table_frame is None:
            self.is_color_calibrated = False
            self.color_gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Neutral gains
            print("[VISION ENGINE] Color normalization skipped by user.")
            return

        h, w = empty_table_frame.shape[:2]

        # Check if a manual felt ROI exists, otherwise fallback to center crop
        if self.rois.get("felt_roi"):
            x1, y1, x2, y2 = self.rois["felt_roi"]
            # Ensure the ROI is within frame boundaries
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            center_patch = empty_table_frame[y1:y2, x1:x2]
        else:
            # Fallback to 20% central crop if selection was skipped
            center_patch = empty_table_frame[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]

        # If the patch is too small or empty, avoid processing
        if center_patch.size == 0:
            print("[VISION ERROR] Felt ROI is empty. Calibration failed.")
            return

        # Use Median to be robust against noise, shadows, or small debris
        c_b = np.median(center_patch[:, :, 0])
        c_g = np.median(center_patch[:, :, 1])
        c_r = np.median(center_patch[:, :, 2])

        # Prevent division by zero
        c_b = max(c_b, 1.0)
        c_g = max(c_g, 1.0)
        c_r = max(c_r, 1.0)

        # Compute Gains (Target / Current)
        k_b = self.target_green_bgr[0] / c_b
        k_g = self.target_green_bgr[1] / c_g
        k_r = self.target_green_bgr[2] / c_r

        self.color_gains = np.array([k_b, k_g, k_r], dtype=np.float32)
        self.is_color_calibrated = True
        print(
            f"[VISION ENGINE] Color Calibration Locked using Manual ROI -> Gains (B:{k_b:.2f}, G:{k_g:.2f}, R:{k_r:.2f})")

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Phase B: Real-Time Pre-Processing
        Applies the computed gains vector to the incoming frame efficiently using NumPy.
        """
        if not self.is_color_calibrated:
            return frame

        # Vectorized Normalization with fast clipping to prevent wrap-around artifacts
        normalized = frame.astype(np.float32) * self.color_gains
        return np.clip(normalized, 0, 255).astype(np.uint8)


# Final Debug Section with Interactive Calibration
if __name__ == "__main__":
    print("--- Starting VisionEngine Smart Debug ---")

    print("Please select camera resolution:")
    print("1: 720p (1280x720)")
    print("2: 1080p FHD (1920x1080)")
    print("3: 4K UHD (3840x2160)")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '2':
        cam_w, cam_h, system_scale = 1920, 1080, 1.5
    elif choice == '3':
        cam_w, cam_h, system_scale = 3840, 2160, 3.0
    else:
        # Default fallback (1)
        cam_w, cam_h, system_scale = 1280, 720, 1.0

    print(f"\n[SETUP] Resolution configured to {cam_w}x{cam_h} with Scale {system_scale}")

    engine = VisionEngine(scale=system_scale)

    cam_id = "http://192.168.1.65:8080/video"
    cap = cv2.VideoCapture(cam_id, cv2.CAP_FFMPEG)

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video from IP Webcam at {cam_id}")
        exit()

    engine.calibrate_rois(first_frame)

    print("Running smart annotation... Press 'q' to stop.")
    cv2.namedWindow("Smart UI - Calibrated", cv2.WINDOW_NORMAL)

    print("Running smart annotation... Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        engine.process_logic(frame)
        display_frame = engine.render_ui(frame)

        cv2.imshow("Smart UI - Calibrated", display_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
