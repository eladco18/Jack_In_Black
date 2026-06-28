import cv2
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional

# RANK_W, RANK_H = 39, 70
# SUIT_W, SUIT_H = 32, 70
CARD_W, CARD_H = 250, 350
CORNER_W, CORNER_H = 35, 90
RANK_W, RANK_H = 35, 39
SUIT_W, SUIT_H = 35, 32
CROP_TOP_PERCENT = 0.05
CROP_LEFT_PERCENT = 0.02
CROP_RIGHT_PERCENT = 0.02

SYSTEM_SCALE = 1.0
SCALED_MIN_AREA = 10000


def set_system_scale(scale_factor: float):
    global SYSTEM_SCALE, SCALED_MIN_AREA
    SYSTEM_SCALE = scale_factor
    SCALED_MIN_AREA = SCALED_MIN_AREA * (SYSTEM_SCALE ** 2)
    print(f"[CARDS CLASSIFIER] Spatial scale globally set to {SYSTEM_SCALE:.2f}")


'''
class CardMemory:
    """
    Manages the tracking of detected cards across multiple frames.
    Uses spatial proximity (distance between centers) to match cards
    and keeps cards in memory for a short period even if they are temporarily obscured.
    """

    def __init__(self, max_dist: int = 60, max_unseen: int = 12) -> None:
        """
        Initializes the CardMemory tracker.

        Args:
            max_dist (int): Maximum distance (in pixels) between card centers
                            to consider them the exact same card.
            max_unseen (int): Number of frames a card can be 'unseen' before
                              it is completely removed from memory.
        """
        self.cards: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 0
        self.max_dist: int = int(max_dist * SYSTEM_SCALE)
        self.max_unseen: int = max_unseen

    def update(self, detected_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the memory with newly detected cards from the current frame.

        Args:
            detected_list (List[Dict[str, Any]]): A list of dictionaries, where each dict
                                                  contains data for a newly detected card
                                                  (e.g., "rank", "suit", "center").

        Returns:
            List[Dict[str, Any]]: A list of all currently tracked cards, including
                                  those temporarily unseen (bridging the gaps).
        """
        new_memory: Dict[int, Dict[str, Any]] = {}

        for detected_card in detected_list:
            found_match = False
            curr_pos = detected_card["center"]

            # 1. Try to match the newly detected card with an existing one in memory
            for card_id, card_data in self.cards.items():

                # Calculate Euclidean distance between centers using math.hypot
                dist = math.hypot(curr_pos[0] - card_data["center"][0],
                                  curr_pos[1] - card_data["center"][1])

                if dist < self.max_dist:
                    # Found a match! Update data.
                    # If the new detection couldn't read the rank/suit (""), keep the old one from memory.
                    rank = detected_card["rank"] if detected_card["rank"] != "" else card_data["rank"]
                    suit = detected_card["suit"] if detected_card["suit"] != "" else card_data["suit"]

                    new_memory[card_id] = {
                        "rank": rank,
                        "suit": suit,
                        "pos": detected_card["pos"],
                        "center": curr_pos,
                        "contour": detected_card["contour"],
                        "unseen": 0,  # Reset unseen counter because the card was just seen
                        "color": (0, 255, 0)
                    }
                    found_match = True
                    break

            # 2. If no match was found, and we actually identified a rank, register as a new card
            if not found_match and detected_card["rank"] != "?":
                new_memory[self.next_id] = {
                    "rank": detected_card["rank"],
                    "suit": detected_card["suit"],
                    "pos": detected_card["pos"],
                    "center": curr_pos,
                    "contour": detected_card["contour"],
                    "unseen": 0,
                    "color": (0, 255, 0)
                }
                self.next_id += 1

        # 3. Retain cards that were not detected in the current frame but are still within the 'unseen' limit
        for card_id, card_data in self.cards.items():
            if card_id not in new_memory and card_data["unseen"] < self.max_unseen:
                card_data["unseen"] += 1
                new_memory[card_id] = card_data

        # 4. Update the official state
        self.cards = new_memory

        return list(self.cards.values())
'''


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates in a consistent, clockwise order:
    [top-left, top-right, bottom-right, bottom-left].
    This specific order is essential for correctly applying cv2.warpPerspective.

    Args:
        pts (np.ndarray): An array of 4 points (coordinates) detected from a contour.

    Returns:
        np.ndarray: A (4, 2) matrix of the ordered points, formatted as 'float32'
                    which is required by OpenCV's perspective transform functions.
    """
    # Ensure the array is structured as 4 rows (points) and 2 columns (x, y)
    pts = pts.reshape(4, 2)

    # Initialize an empty array of 4 points to hold the final ordered coordinates
    rect = np.zeros((4, 2), dtype="float32")

    # Calculate the sum of (x + y) for each point:
    # The top-left point will have the smallest sum (closest to origin 0,0)
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Calculate the difference between coordinates (y - x):
    # The top-right point will have the smallest difference
    # The bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    This enhances the local contrast, which is crucial for reliably detecting
    card ranks and suits under varying or uneven lighting conditions.

    Args:
        img (np.ndarray): The input image. Can be either a 3-channel BGR image
                          or a single-channel grayscale image.

    Returns:
        np.ndarray: A single-channel grayscale image with enhanced contrast.
    """
    # Convert to grayscale if the image has 3 channels (BGR color)
    # If it is already grayscale (length of shape is 2), use it as is
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    # Create a CLAHE object
    # clipLimit: Threshold for contrast limiting
    # tileGridSize: Divides the image into 8x8 blocks for local equalization
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Apply the CLAHE algorithm to the grayscale image and return
    return clahe_obj.apply(gray)


def is_card_face_up(card_img: np.ndarray) -> bool:
    """
    Determines whether a card is face-up or face-down by analyzing the edge density
    in the top-left corner.
    A face-down card (back) is full of lines and patterns (High Edge Density).
    A face-up card (front) is mostly smooth paper with a small number/suit (Low Edge Density).

    Args:
        card_img (np.ndarray): A cropped/warped BGR image of the card.

    Returns:
        bool: True if the card is face-up, False if it is face-down.
    """
    # 1. Crop the top-left corner (where the rank/suit or back pattern is located)
    h, w = card_img.shape[:2]  # Note: defined here for reference, though fixed slicing is used below
    corner_roi = card_img[0:80, 0:60]

    # 2. Convert to grayscale and apply slight blur (to remove standard camera noise)
    gray = cv2.cvtColor(corner_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge Detection
    # This turns any significant color/intensity change into a thin white line on a black background
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Count the edge pixels
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = corner_roi.shape[0] * corner_roi.shape[1]

    edge_ratio = edge_pixels / total_pixels

    # Debugging - uncomment to see the numbers during runtime
    # print(f"Edge Ratio: {edge_ratio:.3f}")

    # Decision:
    # On a normal face-up card (single number on a white background), the ratio is about 0.05 (5%)
    # On a face-down card (grid/pattern), the ratio is over 0.15 (15%)
    # We set the threshold at 0.12 to be safe
    return edge_ratio < 0.12


def identify_corner(
        corner_img: np.ndarray,
        rank_templates: Dict[str, np.ndarray],
        suit_templates: Dict[str, np.ndarray]
) -> Tuple[str, float, str, float, str]:
    """
    Identifies rank and suit using dynamic X-axis and Y-axis projection profiles.
    """
    img_h, img_w = corner_img.shape[:2]

    # 0. Minimal Safe Cropping (Only remove the physical outer edges of the card)
    crop_y = int(img_h * CROP_TOP_PERCENT)
    crop_x = int(img_w * CROP_LEFT_PERCENT)

    # Work on a slightly cleaner patch before binarization
    work_patch = corner_img[crop_y:, crop_x:]

    # 1. Smart Binarization
    gray = cv2.cvtColor(work_patch, cv2.COLOR_BGR2GRAY) if len(work_patch.shape) == 3 else work_patch
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Ultimate Color Detection using LAB Color Space
    lab_patch = cv2.cvtColor(work_patch, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab_patch, mask=thresh)
    l_val, a_val, b_val = mean_lab[0], mean_lab[1], mean_lab[2]

    is_red = a_val > 150
    color_name = "RED" if is_red else "BLACK"

    # print(f"     [DEBUG] Color: {color_name} (L:{l_val:.1f}, A:{a_val:.1f}, B:{b_val:.1f})")

    # 3. Dynamic X-Axis Width Scan (The User's Strategy)
    # Sum pixels vertically for each column to find where the symbol ends
    proj_x = np.sum(thresh, axis=0)

    # Find the first column with ink (Start X)
    non_zero_cols = np.where(proj_x > 0)[0]
    if len(non_zero_cols) == 0:
        return "?", 0.0, "?", 0.0, color_name

    start_x = non_zero_cols[0]

    # Scan right from start_x until we hit an empty column (End X)
    end_x = start_x
    for x in range(start_x, len(proj_x)):
        if proj_x[x] == 0:  # We hit the empty gap after the symbol!
            break
        end_x = x

    # Add a small 2-pixel buffer to the right, ensuring we don't exceed image width
    crop_r = min(end_x + 2, work_patch.shape[1])

    # Apply the dynamic width crop to both the color image and the threshold mask
    corner_img = work_patch[:, :crop_r]
    thresh = thresh[:, :crop_r]
    img_h, img_w = corner_img.shape[:2]

    # 4. Y-Axis Projection Profile (Histogram-based split)
    # Sum all white pixels in each row to find the empty "valley" between Rank and Suit
    proj_y = np.sum(thresh, axis=1)

    # Search for the splitting valley in the middle 40% of the image
    mid_start = int(img_h * 0.3)
    mid_end = int(img_h * 0.7)

    # Fallback in case of an empty image
    if mid_end <= mid_start:
        return "?", 0.0, "?", 0.0, color_name

    valley_y = mid_start + np.argmin(proj_y[mid_start:mid_end])

    # Split the image exactly at the valley
    rank_slice = thresh[:valley_y, :]
    suit_slice = thresh[valley_y:, :]

    # 5. Helper Function: Tight Crop and Direct Stretch
    def extract_and_stretch(binary_img: np.ndarray, target_w: int, target_h: int) -> Optional[np.ndarray]:
        """
        Crops the exact bounding box of the symbol and stretches it directly
        to match the exact dimensions of the template (ignoring aspect ratio).
        This matches how the original templates were created.
        """
        coords = cv2.findNonZero(binary_img)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w < 3 or h < 3:  # Filter out tiny noise
            return None

        # Crop tightly around the symbol's white pixels
        cropped = binary_img[y:y + h, x:x + w]

        # 2. Stretch directly to the target dimensions using High-Quality Up-scaling
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        return resized

    # Extract symbols using the new stretch logic
    r_roi = extract_and_stretch(rank_slice, RANK_W, RANK_H)
    s_roi = extract_and_stretch(suit_slice, SUIT_W, SUIT_H)

    if r_roi is None or s_roi is None:
        return "?", 0.0, "?", 0.0, color_name

    # 6. Template Matching
    best_r, best_rs = "?", 0.0
    for name, tmpl in rank_templates.items():
        score = np.max(cv2.matchTemplate(r_roi, tmpl, cv2.TM_CCOEFF_NORMED))
        if score > best_rs:
            best_rs, best_r = score, name

    best_s, best_ss = "?", 0.0
    for name, tmpl in suit_templates.items():
        # Filter suits by color to improve accuracy and speed
        if is_red and name not in ["hearts", "diamonds"]: continue
        if not is_red and name not in ["clubs", "spades"]: continue

        score = np.max(cv2.matchTemplate(s_roi, tmpl, cv2.TM_CCOEFF_NORMED))
        if score > best_ss:
            best_ss, best_s = score, name

    return best_r, best_rs, best_s, best_ss, color_name


def resolve_prediction(val1: str, conf1: float, val2: str, conf2: float) -> Tuple[str, float]:
    """
    Resolves conflicts between two corner predictions.
    If there is a disagreement or both scores are low, it simply returns
    the prediction with the highest confidence score.
    """
    # Scenario A: Mutual Agreement - both corners identified the exact same value
    if val1 == val2 and val1 != "?":
        return val1, max(conf1, conf2)

    # Scenario B: One corner completely failed (returned "?")
    if val1 == "?" and val2 != "?":
        return val2, conf2
    if val2 == "?" and val1 != "?":
        return val1, conf1
    if val1 == "?" and val2 == "?":
        return "?", 0.0

    # Scenario C: Shootout - Always trust the corner with the higher confidence
    if conf1 >= conf2:
        return val1, conf1
    else:
        return val2, conf2


def analyze_frame(
        frame: np.ndarray,
        rank_t: Dict[str, np.ndarray],
        suit_t: Dict[str, np.ndarray],
        sensitivity: int = 90
) -> List[Dict[str, Any]]:
    """
    Analyzes a given frame (ROI) to detect, warp, and classify playing cards.

    Args:
        frame (np.ndarray): The BGR image frame (or ROI) to analyze.
        rank_t (Dict[str, np.ndarray]): Loaded templates for card ranks.
        suit_t (Dict[str, np.ndarray]): Loaded templates for card suits.
        sensitivity (int): Tolerance for the white color mask.
        card_w (int): Target width for the warped card image.
        card_h (int): Target height for the warped card image.
        corner_w (int): Width of the corner crop for template matching.
        corner_h (int): Height of the corner crop.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries. Each dict contains the detected
                              card's 'rank', 'suit', 'contour', 'pos', and 'center'.
    """
    results: List[Dict[str, Any]] = []

    # 1. Color Masking to find white areas (potential cards)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 255 - sensitivity]),
        np.array([180, sensitivity, 255])
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > SCALED_MIN_AREA:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            if len(approx) == 4:
                try:
                    # 2. Perspective Transform (Flatten the card)
                    rect = order_points(approx)
                    M = cv2.getPerspectiveTransform(
                        rect,
                        np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
                    )
                    warped = cv2.warpPerspective(frame, M, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)

                    # 3. Filter out face-down cards
                    if not is_card_face_up(warped):
                        continue

                    # 4. Dual-Corner Extraction (Extract both immediately)
                    tl_patch = warped[0:CORNER_H, 0:CORNER_W]
                    br_patch = warped[CARD_H - CORNER_H: CARD_H, CARD_W - CORNER_W: CARD_W]
                    br_patch_rotated = cv2.rotate(br_patch, cv2.ROTATE_180)

                    # 5. Fast Track (Test TL Corner first)
                    r1, rs1, s1, ss1, c1 = identify_corner(tl_patch, rank_t, suit_t)

                    # If TL corner is absolutely perfect, skip the second corner to save CPU
                    if rs1 >= 0.85 and ss1 >= 0.85:
                        r, rs, s, ss, c = r1, rs1, s1, ss1, c1
                    else:
                        # 6. Backup Stage: Test BR Corner and run Decision Matrix
                        r2, rs2, s2, ss2, c2 = identify_corner(br_patch_rotated, rank_t, suit_t)

                        # Resolve Rank and Suit independently
                        r, rs = resolve_prediction(r1, rs1, r2, rs2)
                        s, ss = resolve_prediction(s1, ss1, s2, ss2)
                        c = c1 if ss1 >= ss2 else c2

                    # Custom logic fix for 'Clubs'
                    if rs > 0.70 and ss < 0.4:
                        s = "clubs"

                    # Stop processing if resolution failed (returned "?")
                    if r == "?" or s == "?":
                        continue

                    # Calculate center point
                    moments = cv2.moments(approx)
                    cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else int(rect[0][0])
                    cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else int(rect[0][1])

                    # 7. Final Confidence Filter
                    if rs > 0.53:
                        results.append({
                            "rank": r,
                            "suit": s,
                            "contour": approx,
                            "pos": (int(rect[0][0]), int(rect[0][1])),
                            "center": (cx, cy),
                            "color": (60, 40, 220) if c == "RED" else (80, 80, 30)
                        })

                except Exception:
                    continue

    return results


if __name__ == "__main__":
    import os


    # --- Helper function to load templates for the standalone test ---
    def load_templates_for_test(folder_path: str, width: int, height: int) -> Dict[str, np.ndarray]:
        templates = {}
        if not os.path.exists(folder_path):
            print(f"[WARNING] Template folder not found: {folder_path}")
            return templates
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, filename)
                img = cv2.imread(full_path, 0)
                if img is not None:
                    # Binarize the template
                    _, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                    coords = cv2.findNonZero(b)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        if w >= 3 and h >= 3:
                            b = b[y:y + h, x:x + w]

                    resized_img = cv2.resize(b, (width, height), interpolation=cv2.INTER_CUBIC)
                    card_name = os.path.splitext(filename)[0]
                    templates[card_name] = resized_img
        return templates


    # --- Helper function for visual debugging pauses ---
    def show_and_wait(window_name: str, image: np.ndarray):
        cv2.imshow(window_name, image)
        print(f">>> Displaying: {window_name}. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


    print("=== Starting Visual Card Detection Pipeline Test ===")

    # 1. Load templates using the updated accurate dimensions
    rank_templates = load_templates_for_test('templates/ranks', RANK_W, RANK_H)
    suit_templates = load_templates_for_test('templates/suits', SUIT_W, SUIT_H)

    # 2. Load the test image
    img_path = "captured_image.jpg"  # WIN_20260503_14_36_00_Pro
    test_image = cv2.imread(img_path)

    if test_image is None:
        print(f"[ERROR] Could not load image: {img_path}")
        exit()

    # Scale down the original image for display purposes (if it's too large)
    display_height = 800
    display_ratio = display_height / test_image.shape[0]
    display_width = int(test_image.shape[1] * display_ratio)

    display_img = cv2.resize(test_image, (display_width, display_height))
    show_and_wait("1. Original Test Image", display_img)

    # 3. Preprocessing (Color Masking based on analyze_frame logic)
    hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 255, 255]))

    display_mask = cv2.resize(mask, (display_width, display_height))
    show_and_wait("2. Preprocessing (White Color Mask)", display_mask)

    # 4. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = test_image.copy()

    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > SCALED_MIN_AREA:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                valid_contours.append(approx)
                # Draw valid card contours in green
                cv2.drawContours(img_with_contours, [approx], -1, (0, 255, 0), 3)
            else:
                # Draw large but invalid contours in red (e.g. noise, hands)
                cv2.drawContours(img_with_contours, [approx], -1, (0, 0, 255), 2)

    display_contours = cv2.resize(img_with_contours, (display_width, display_height))
    show_and_wait(f"3. Detected Contours (Found {len(valid_contours)} cards)", display_contours)

    print(f"\n[INFO] Found {len(valid_contours)} valid card contours. Starting detailed analysis...\n")


    # --- Helper function for creating dashboard panels ---
    def create_panel(img: np.ndarray, title: str, panel_w: int = 160, panel_h: int = 300) -> np.ndarray:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        resized = cv2.resize(img, (panel_w, panel_h - 40))
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[40:, 0:panel_w] = resized

        cv2.putText(panel, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return panel


    def get_debug_visuals(corner_img_raw):
        """ Recreates the extraction steps specifically to yield real visual images for the dashboard """
        img_h, img_w = corner_img_raw.shape[:2]
        crop_y, crop_x = int(img_h * CROP_TOP_PERCENT), int(img_w * CROP_LEFT_PERCENT)
        work_patch = corner_img_raw[crop_y:, crop_x:]

        gray = cv2.cvtColor(work_patch, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh_full = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Dynamic X-Scan for Debug Dashboard
        proj_x = np.sum(thresh_full, axis=0)
        non_zero_cols = np.where(proj_x > 0)[0]
        if len(non_zero_cols) == 0:
            crop_r = work_patch.shape[1]
        else:
            start_x = non_zero_cols[0]
            end_x = start_x
            for x in range(start_x, len(proj_x)):
                if proj_x[x] == 0: break
                end_x = x
            crop_r = min(end_x + 2, work_patch.shape[1])
        tl_patch = work_patch[:, :crop_r]
        filtered = filtered[:, :crop_r]
        thresh = thresh_full[:, :crop_r]
        patch_h, patch_w = tl_patch.shape[:2]

        proj_y = np.sum(thresh, axis=1)
        mid_start, mid_end = int(patch_h * 0.3), int(patch_h * 0.7)
        valley_y = mid_start + np.argmin(proj_y[mid_start:mid_end]) if mid_start < mid_end else patch_h // 2

        thresh_viz = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.line(thresh_viz, (0, valley_y), (patch_w, valley_y), (0, 0, 255), 1)

        def stretch_real_pixels(binary_img, target_w, target_h):
            coords = cv2.findNonZero(binary_img)
            if coords is None: return np.zeros((target_h, target_w), dtype=np.uint8)
            x, y, w, h = cv2.boundingRect(coords)
            if w < 3 or h < 3: return np.zeros((target_h, target_w), dtype=np.uint8)
            return cv2.resize(binary_img[y:y + h, x:x + w], (target_w, target_h))

        # This extracts the ACTUAL pixels from your camera frame!
        real_r_roi = stretch_real_pixels(thresh[:valley_y, :], RANK_W, RANK_H)
        real_s_roi = stretch_real_pixels(thresh[valley_y:, :], SUIT_W, SUIT_H)

        extracted_canvas = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        extracted_canvas[0:patch_h // 2, :] = cv2.cvtColor(cv2.resize(real_r_roi, (patch_w, patch_h // 2)),
                                                           cv2.COLOR_GRAY2BGR)
        extracted_canvas[patch_h // 2:, :] = cv2.cvtColor(cv2.resize(real_s_roi, (patch_w, patch_h - patch_h // 2)),
                                                          cv2.COLOR_GRAY2BGR)

        return tl_patch, filtered, thresh_viz, extracted_canvas


    for i, approx in enumerate(valid_contours):
        print(f"--- Analyzing Card #{i + 1} ---")

        rect = order_points(approx)
        M = cv2.getPerspectiveTransform(
            rect, np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
        )
        warped = cv2.warpPerspective(test_image, M, (CARD_W, CARD_H))

        if not is_card_face_up(warped):
            print(f"[RESULT] Card {i + 1} is Face Down. Skipping.\n")
            continue

        # Extract both corners
        tl_patch_raw = warped[0:CORNER_H, 0:CORNER_W]
        br_patch_raw = warped[CARD_H - CORNER_H: CARD_H, CARD_W - CORNER_W: CARD_W]
        br_patch_rotated = cv2.rotate(br_patch_raw, cv2.ROTATE_180)

        # Run system identification
        r1, rs1, s1, ss1, c1 = identify_corner(tl_patch_raw, rank_templates, suit_templates)
        r2, rs2, s2, ss2, c2 = identify_corner(br_patch_rotated, rank_templates, suit_templates)

        # The Tribunal - Resolve Predictions
        final_r, final_rs = resolve_prediction(r1, rs1, r2, rs2)
        final_s, final_ss = resolve_prediction(s1, ss1, s2, ss2)
        final_c = c1 if ss1 >= ss2 else c2

        # Generate Real Visuals
        tl_vis = get_debug_visuals(tl_patch_raw)
        br_vis = get_debug_visuals(br_patch_rotated)

        # Build Top Row (TL)
        r1_tl = create_panel(tl_vis[0], "TL Cropped")
        r2_tl = create_panel(tl_vis[1], "TL Filtered")
        r3_tl = create_panel(tl_vis[2], "TL Proj")
        r4_tl = create_panel(tl_vis[3], f"TL: {r1}{s1} ({rs1:.2f})")
        row_tl = cv2.hconcat([r1_tl, r2_tl, r3_tl, r4_tl])

        # Build Bottom Row (BR)
        r1_br = create_panel(br_vis[0], "BR Cropped")
        r2_br = create_panel(br_vis[1], "BR Filtered")
        r3_br = create_panel(br_vis[2], "BR Proj")
        r4_br = create_panel(br_vis[3], f"BR: {r2}{s2} ({rs2:.2f})")
        row_br = cv2.hconcat([r1_br, r2_br, r3_br, r4_br])

        # Combine Rows
        matrix = cv2.vconcat([row_tl, row_br])

        # Build Verdict Panel (Right Side)
        verdict_panel = np.zeros((600, 250, 3), dtype=np.uint8)  # 600 height matches the 2 rows of 300
        cv2.putText(verdict_panel, "THE VERDICT", (15, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

        status_color = (0, 255, 0) if final_r != "?" and final_s != "?" else (0, 0, 255)
        res_text = f"{final_r.upper()} of {final_s.upper()}" if final_r != "?" else "FAILED"
        cv2.putText(verdict_panel, res_text, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        color_display_rgb = (0, 0, 255) if final_c == "RED" else (200, 200, 200)
        cv2.putText(verdict_panel, f"Color: {final_c}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_display_rgb, 1)

        cv2.putText(verdict_panel, f"Rank Conf: {final_rs:.2f}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        cv2.putText(verdict_panel, f"Suit Conf: {final_ss:.2f}", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        # Display Final Dashboard
        dashboard = cv2.hconcat([matrix, verdict_panel])
        show_and_wait(f"Card {i + 1} - Dual-Corner Tribunal", dashboard)

        print(f"[RESULT] Card {i + 1} Final Verdict: {final_r} of {final_s} (R: {final_rs:.2f}, S: {final_ss:.2f})\n")

    print("=== Visual Card Detection Pipeline Test Completed ===")
    cv2.destroyAllWindows()
