import cv2
import numpy as np
import math

# ------------- Test Settings --------------- #
SYSTEM_SCALE = 1.5

# Static roi (BASE 720p)
BASE_ROI_PLAYER_1 = (658, 250, 849, 720)
BASE_ROI_PLAYER_2 = (433, 265, 618, 707)

ROI_PLAYER_1 = BASE_ROI_PLAYER_1
ROI_PLAYER_2 = BASE_ROI_PLAYER_2
# ------------------------------------------- #

min_leved_HSV = [65, 60, 30]
max_leved_HSV = [90, 255, 255]

'''
min_leved_HSV = [40, 30, 70]
max_leved_HSV = [90, 180, 135]
'''

outer_radius_roi = 0.7
inner_radius_roi = 0.4

CHIP_THRESHOLDS = {
    "5": {"lower": np.array([0, 100, 140]), "upper": np.array([10, 255, 255]),  # RED
          "lower2": np.array([170, 100, 140]), "upper2": np.array([180, 255, 255])},
    "10": {"lower": np.array([30, 20, 100]), "upper": np.array([65, 80, 255])},  # GREEN
    "25": {"lower": np.array([70, 65, 100]), "upper": np.array([120, 150, 180])},  # BLUE
    "50": {"lower": np.array([130, 0, 40]), "upper": np.array([180, 105, 130]),  # BLACK
           "lower2": np.array([0, 0, 40]), "upper2": np.array([20, 105, 130])},
    "1": {"lower": np.array([0, 10, 200]), "upper": np.array([30, 100, 255])},  # WHITE
}

'''
#gimi's reccommandation. didnt check yet!

min_leved_HSV = [56, 60, 30]
max_leved_HSV = [70, 255, 255]

CHIP_THRESHOLDS = {
    "5": {"lower": np.array([0, 100, 90]), "upper": np.array([10, 255, 255]),  # RED
          "lower2": np.array([170, 100, 90]), "upper2": np.array([180, 255, 255])},
    "10": {"lower": np.array([35, 60, 85]), "upper": np.array([55, 255, 255])},  # GREEN
    "25": {"lower": np.array([72, 60, 80]), "upper": np.array([130, 255, 255])},  # BLUE
    "50": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 50, 85])},  # BLACK
    "1": {"lower": np.array([0, 0, 180]), "upper": np.array([180, 40, 255])},  # WHITE
}
'''

# Base thresholds (calibrated for 720p)
hough_dp = 1.2  # Inverse ratio of the accumulator resolution to the image resolution
hough_param1 = 60  # Upper threshold for the internal Canny edge detector
hough_param2 = 20  # Accumulator threshold for circle centers (lower means more false circles)

# Pre-calculated dynamic thresholds
hough_minDist = 35  # Minimum distance between the centers of detected circles
hough_minRadius = 25  # Minimum radius of the circles to be detected
hough_maxRadius = 40  # Maximum radius of the circles to be detected
open_kernel_size = (3, 3)  # Kernel size for morphological opening (removes small noise)
close_kernel_size = (5, 5)  # Kernel size for morphological closing (fills small holes)
gauss_k_size = 9  # Kernel size for Gaussian blur (smooths the image to reduce noise)
max_pixel_count = -1  # Custom boundary flag or limit for maximum pixel area processing


def set_system_scale(scale_factor: float):
    """Updates the global scale factor and PRE-CALCULATES all spatial thresholds once."""
    global SYSTEM_SCALE, hough_minDist, hough_minRadius, hough_maxRadius
    global open_kernel_size, close_kernel_size, gauss_k_size, max_pixel_count
    global ROI_PLAYER_1, ROI_PLAYER_2

    SYSTEM_SCALE = scale_factor
    hough_minDist = int(40 * SYSTEM_SCALE)
    hough_minRadius = int(20 * SYSTEM_SCALE)
    hough_maxRadius = int(40 * SYSTEM_SCALE)

    k_open = max(3, int(3 * SYSTEM_SCALE) | 1)
    k_close = max(3, int(5 * SYSTEM_SCALE) | 1)
    open_kernel_size = (k_open, k_open)
    close_kernel_size = (k_close, k_close)

    gauss_k_size = max(3, int(9 * SYSTEM_SCALE) | 1)

    # Scale ROIs and force them to be Integers!
    ROI_PLAYER_1 = (
        int(BASE_ROI_PLAYER_1[0] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_1[1] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_1[2] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_1[3] * SYSTEM_SCALE)
    )

    ROI_PLAYER_2 = (
        int(BASE_ROI_PLAYER_2[0] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_2[1] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_2[2] * SYSTEM_SCALE),
        int(BASE_ROI_PLAYER_2[3] * SYSTEM_SCALE)
    )

    print(f"[CHIPS CLASSIFIER] Spatial scale globally set to {SYSTEM_SCALE:.2f}")


def gauss_blur(frame):
    # Uses the pre-calculated global variable
    return cv2.GaussianBlur(frame, (gauss_k_size, gauss_k_size), 2)


def identify_chip_color(hsv_roi):
    max_count = -1
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


def chips_classifier(frame, debug=False, player_name=1):
    if frame is None or frame.size == 0:
        return []

    raw_chips = []
    display_img = frame.copy() if debug else frame
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
        cv2.imshow(f"P{player_name} - Step 1: Original ROI", frame)
        cv2.imshow(f"P{player_name} - Step 2: Noisy Mask (No Green)", chips_mask_noisy)
        cv2.imshow(f"P{player_name} - Step 3: Clean Mask", chips_mask_clean)
        cv2.imshow(f"P{player_name} - Step 4: Blurred (Input to Hough)", blurred_mask)

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
            value_str = identify_chip_color(hsv_roi_pseudo_img)
            chip_value = int(value_str) if value_str != "UNKNOWN" else 0

            raw_chips.append({
                "color": value_str,
                "value": chip_value,
                "pos": (int(center[0]), int(center[1])),
                "radius": int(radius)
            })

            if debug:
                cv2.circle(display_img, center, radius, (0, 255, 0), 2)
                cv2.circle(display_img, center, r_outer, (0, 255, 255), 1)
                cv2.circle(display_img, center, r_inner, (0, 255, 255), 1)
                cv2.putText(display_img, str(chip_value), (center[0] - 10, center[1] - radius - 5),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)

        if debug:
            cv2.imshow(f"P{player_name} - Step 5: Final Circles & Values", display_img)

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
            "color": chip["color"],
            "value": chip["value"],
            "pos": (global_x, global_y),
            "radius": chip["radius"]
        })

    return global_chips


# =======================================
# --- MAIN VIDEO LOOP WITH STABILIZATION
# =======================================
if __name__ == "__main__":

    set_system_scale(SYSTEM_SCALE)
    image_source = "captured_image.jpg"
    frame = cv2.imread(image_source)

    if frame is None:
        print(f"Error: Cannot load image source {image_source}")
        exit()

    p1_raw_chips = analyze_player_area(frame, ROI_PLAYER_1, debug=True, player_name="1")
    p2_raw_chips = analyze_player_area(frame, ROI_PLAYER_2, debug=True, player_name="2")

    p1_total = sum(chip["value"] for chip in p1_raw_chips)
    p2_total = sum(chip["value"] for chip in p2_raw_chips)

    cv2.rectangle(frame, (ROI_PLAYER_1[0], ROI_PLAYER_1[1]), (ROI_PLAYER_1[2], ROI_PLAYER_1[3]), (255, 0, 0), 2)
    cv2.putText(frame, f"Player 1 Pot: ${p1_total}", (ROI_PLAYER_1[0], ROI_PLAYER_1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for chip in p1_raw_chips:
        cx, cy = chip["pos"]
        r = chip["radius"]
        val = chip["value"]
        cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.putText(frame, str(val), (int(cx) - 10, int(cy) - int(r) - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                    (0, 255, 255), 2)

    cv2.rectangle(frame, (ROI_PLAYER_2[0], ROI_PLAYER_2[1]), (ROI_PLAYER_2[2], ROI_PLAYER_2[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Player 2 Pot: ${p2_total}", (ROI_PLAYER_2[0], ROI_PLAYER_2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for chip in p2_raw_chips:
        cx, cy = chip["pos"]
        r = chip["radius"]
        val = chip["value"]
        cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.putText(frame, str(val), (int(cx) - 10, int(cy) - int(r) - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                    (0, 255, 255), 2)

    cv2.namedWindow("Poker Chips Tracking - Raw Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poker Chips Tracking - Raw Detection", 1280, 720)
    cv2.imshow("Poker Chips Tracking - Raw Detection", frame)

    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
