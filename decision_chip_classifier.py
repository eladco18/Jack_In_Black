import cv2
import numpy as np
from typing import Optional, Tuple

SYSTEM_SCALE = 1.0

min_leved_HSV = [65, 30, 30]
max_leved_HSV = [100, 255, 255]

outer_radius_roi = 1
inner_radius_roi = 0.4

CHIP_THRESHOLDS = {
    "blue=hit": {"lower": np.array([70, 75, 100]), "upper": np.array([170, 255, 255])},
    "pink=stand": {
            "lower": np.array([160, 70, 100]), "upper": np.array([180, 150, 255]),
            "lower2": np.array([0, 70, 100]), "upper2": np.array([10, 150, 255])},
    "orange=double": {"lower": np.array([0, 160, 200]), "upper": np.array([15, 255, 255])},
    "brown=surrender": {"lower": np.array([0, 70, 80]), "upper": np.array([15, 198, 200])}
}

hough_dp = 1
hough_param1 = 100
hough_param2 = 18

hough_minDist = 30
hough_minRadius = 35
hough_maxRadius = 50

open_kernel_size = (3, 3)
close_kernel_size = (5, 5)
gauss_k_size = 9
max_pixel_count = 20


def set_system_scale(scale_factor: float):
    """Updates the global scale factor and PRE-CALCULATES all spatial thresholds once."""
    global SYSTEM_SCALE, hough_minDist, hough_minRadius, hough_maxRadius
    global open_kernel_size, close_kernel_size, gauss_k_size, max_pixel_count

    SYSTEM_SCALE = scale_factor
    hough_minDist = int(30 * SYSTEM_SCALE)
    hough_minRadius = int(35 * SYSTEM_SCALE)
    hough_maxRadius = int(50 * SYSTEM_SCALE)

    k_open = max(3, int(3 * SYSTEM_SCALE) | 1)
    k_close = max(3, int(5 * SYSTEM_SCALE) | 1)
    open_kernel_size = (k_open, k_open)
    close_kernel_size = (k_close, k_close)

    gauss_k_size = max(3, int(9 * SYSTEM_SCALE) | 1)

    # Area scales by the square of the linear scale
    max_pixel_count = int(20 * (SYSTEM_SCALE ** 2))
    print(f"[DECISION CLASSIFIER] Spatial scale globally set to {SYSTEM_SCALE:.2f}")


def gauss_blur(frame):
    return cv2.GaussianBlur(frame, (gauss_k_size, gauss_k_size), 2)


def identify_chip_color(hsv_roi):
    """
    Finds the most dominant color in the ROI based on pixel counts,
    supporting multiple HSV ranges per color.
    """
    max_count = 20
    detected_color = "UNKNOWN"

    for color_name, ranges in CHIP_THRESHOLDS.items():
        mask = cv2.inRange(hsv_roi, ranges["lower"], ranges["upper"])

        if "lower2" in ranges and "upper2" in ranges:
            mask2 = cv2.inRange(hsv_roi, ranges["lower2"], ranges["upper2"])
            mask = cv2.bitwise_or(mask, mask2)

        count = cv2.countNonZero(mask)
        if count > max_count:
            max_count = count
            detected_color = color_name

    return detected_color


def chips_classifier(frame, debug=False, player_name="Player"):
    """
    Receives a frame (or a cropped ROI), detects the chips,
    and returns a list of dictionaries containing the action (hit/stand) and relative position.
    """
    if frame is None or frame.size == 0:
        return []

    raw_chips = []
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array(min_leved_HSV)
    upper_green = np.array(max_leved_HSV)

    background_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    chips_mask_noisy = cv2.bitwise_not(background_mask)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)

    chips_mask_clean = cv2.morphologyEx(chips_mask_noisy, cv2.MORPH_OPEN, kernel_open)
    chips_mask_clean = cv2.morphologyEx(chips_mask_clean, cv2.MORPH_CLOSE, kernel_close)

    blurred_mask = gauss_blur(chips_mask_clean)

    if debug:
        cv2.imshow(f"{player_name} - 1. Original ROI", frame)
        cv2.imshow(f"{player_name} - 2. Noisy Mask", chips_mask_noisy)
        cv2.imshow(f"{player_name} - 3. Clean Mask", chips_mask_clean)
        cv2.imshow(f"{player_name} - 4. Blurred (Input to Hough)", blurred_mask)

    circles = cv2.HoughCircles(blurred_mask, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_minDist,
                               param1=hough_param1, param2=hough_param2, minRadius=hough_minRadius,
                               maxRadius=hough_maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            mask_annulus = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

            r_outer = int(radius * outer_radius_roi)
            r_inner = int(radius * inner_radius_roi)

            cv2.circle(mask_annulus, center, r_outer, 255, thickness=-1)
            cv2.circle(mask_annulus, center, r_inner, 0, thickness=-1)

            hsv_roi_pixels = hsv_frame[mask_annulus == 255]

            if hsv_roi_pixels.size == 0:
                continue

            hsv_roi_pseudo_img = hsv_roi_pixels.reshape(1, -1, 3)
            label_str = identify_chip_color(hsv_roi_pseudo_img)

            raw_chips.append({
                "action": label_str,
                "pos": (int(center[0]), int(center[1])),
                "radius": int(radius)
            })

    elif debug:
        print(f"DEBUG: No circles detected for {player_name}")

    return raw_chips


def analyze_player_area(full_frame, roi_coords, debug=False, player_name="Player"):
    x1, y1, x2, y2 = roi_coords

    roi_image = full_frame[y1:y2, x1:x2]
    local_chips = chips_classifier(roi_image, debug=debug, player_name=player_name)

    global_chips = []

    for chip in local_chips:
        local_x, local_y = chip["pos"]

        global_x = local_x + x1
        global_y = local_y + y1

        global_chips.append({
            "action": chip["action"],
            "pos": (global_x, global_y),
            "radius": chip["radius"]
        })

    return global_chips


def detect_decision_chip(frame: np.ndarray, roi_coords: Tuple[int, int, int, int], player: int = 1,
                         show_results: bool = False) -> Optional[str]:
    """
    Analyzes a specific ROI from a single sampled frame to detect decision chips

    Args:
        frame (np.ndarray): A single sampled frame
        roi_coords (Tuple[int, int, int, int]): The coordinates of the Region of Interest (x1, y1, x2, y2).
        player (int): The player number (e.g., 1 or 2). Default is 1.
        show_results (bool): If True, displays an OpenCV window showing the bounding boxes. Default is False.

    Returns:
        Optional[str]: Returns "hit", "stand", "double" or "surrender" if detected, otherwise None.
    """
    if frame is None:
        return None

    final_display = frame.copy() if show_results else None

    # --- Player Analysis (Using your exact original logic on the sampled frame) ---
    player_chips = analyze_player_area(frame, roi_coords, debug=show_results, player_name=f"Player {player}")
    actions = [chip["action"] for chip in player_chips]

    result_string = ", ".join(actions)

    if show_results:
        color = (255, 0, 0) if player == 1 else (0, 0, 255)

        for chip in player_chips:
            cv2.circle(final_display, chip["pos"], chip["radius"], color, 2)
            cv2.putText(final_display, chip["action"],
                        (chip["pos"][0] - 25, chip["pos"][1] - chip["radius"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(final_display, (roi_coords[0], roi_coords[1]),
                      (roi_coords[2], roi_coords[3]), color, 2)

        cv2.namedWindow("Final Poker Table Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Poker Table Detection", 1000, 700)
        cv2.imshow("Final Poker Table Detection", final_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Return result ---
    result_lower = result_string.lower()
    if "hit" in result_lower:
        return "HIT"
    elif "stand" in result_lower:
        return "STAND"
    elif "double" in result_lower:
        return "DOUBLE"
    elif "surrender" in result_lower:
        return "SURRENDER"
    return None


if __name__ == "__main__":
    test_image_path = "full_table_hit.jpg"  # stand, surrender, double, hit
    sample_frame = cv2.imread(test_image_path)

    if sample_frame is None:
        print(f"Error: Could not load {test_image_path}")
    else:

        ROI_PLAYER_1 = (770, 580, 1130, 700)
        ROI_PLAYER_2 = (0, 580, 380, 710)

        print("--- Testing Player 1 ---")
        result_p1 = detect_decision_chip(
            frame=sample_frame,
            roi_coords=ROI_PLAYER_1,
            player=1,
            show_results=False
        )
        print(f"Player 1 Final Result: {result_p1}")

        print("\n--- Testing Player 2 ---")
        result_p2 = detect_decision_chip(
            frame=sample_frame,
            roi_coords=ROI_PLAYER_2,
            player=2,
            show_results=False
        )
        print(f"Player 2 Final Result: {result_p2}")
