import cv2
import numpy as np
import mediapipe as mp
from manual_keyboard_calibration import (
    load_calibration,
    generate_white_polygons,
    generate_black_polygons,
    draw_polygons,
)

KEYBOARD_WIDTH = 5200
KEYBOARD_HEIGHT = 1600 # needs to be sufficiently large

# Initialize hand detection

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,      # start with images, turn to false for video
    max_num_hands=2,
    min_detection_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

def warp_frame(frame, H):

    warped = cv2.warpPerspective(
        frame,
        H,
        (KEYBOARD_WIDTH, KEYBOARD_HEIGHT)
    )

    return warped

def polygon_mean(gray, polygon):

    mask = np.zeros(gray.shape, dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], 255)

    pixels = gray[mask == 255]

    if len(pixels) == 0:
        return 0

    return float(np.mean(pixels))

def compute_key_brightness(frame, keys):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = {}

    for key in keys:

        value = polygon_mean(
            gray,
            key["polygon"]
        )

        brightness[key["name"]] = value

    return brightness

def draw_brightness(image, keys, brightness):

    out = image.copy()

    for key in keys:

        x, y = map(int, key["center"])

        if key["type"] == "black":
            cv2.putText(
            out,
            f"{brightness[key['name']]:.0f}",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,255),
            1,
            )
        else:
            cv2.putText(
            out,
            f"{brightness[key['name']]:.0f}",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0 ,0),
            1,
            )

    return out

def detect_hands(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    return results

def draw_hands(image, results):

    out = image.copy()

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                out,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    return out

TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
def fingertip_positions(results, width, height):

    fingertips = []

    if not results.multi_hand_landmarks:
        return fingertips

    for hand in results.multi_hand_landmarks:

        for finger, idx in TIP_IDS.items():

            lm = hand.landmark[idx]

            x = int(lm.x * width)
            y = int(lm.y * height)

            fingertips.append({
                "finger": finger,
                "point": (x, y)
            })

    return fingertips

def draw_fingertips(image, fingertips):

    out = image.copy()

    for tip in fingertips:

        x, y = tip["point"]

        cv2.circle(
            out,
            (x, y),
            8,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            out,
            tip["finger"],
            (x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    return out



def key_under_fingertip(point, keys):

    # Black keys have higher priority, check black first

    black_keys = [key for key in keys if key["type"] == "black"]
    white_keys = [key for key in keys if key["type"] == "white"]

    for key in black_keys:

        inside = cv2.pointPolygonTest(
            key["polygon"].astype(np.float32),
            point,
            False
        )

        if inside >= 0:
            return key
        
    for key in white_keys:

        inside = cv2.pointPolygonTest(
            key["polygon"].astype(np.float32),
            point,
            False
        )

        if inside >= 0:
            return key

    return None


def main(image_path, calibration_path):

    image = cv2.imread(image_path)

    H, corners, white, black = load_calibration(
        calibration_path
    )

    warped = warp_frame(image, H)

    keys = (
    generate_white_polygons(corners, white) +
    generate_black_polygons(corners, black)
    )

    brightness = compute_key_brightness(
        warped,
        keys
    )

    hand_results = detect_hands(warped)

    polygons = draw_polygons(
        warped,
        keys
    )

    brightness_values = draw_brightness(
        polygons,
        keys,
        brightness
    )

    fingertips = fingertip_positions(
    hand_results,
    warped.shape[1],
    warped.shape[0]
    )

    preview = draw_fingertips(brightness_values, fingertips)

    cv2.imshow("Detection", preview)

    for tip in fingertips:

        key = key_under_fingertip(tip["point"], keys)

        if key is not None:
            # check here if key is pressed
            print(f"{tip['finger']} -> {key['name']}")

    cv2.waitKey(0)


if __name__ == "__main__":

    import sys

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "keyboard.jpg"
    )

    calibration_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "keyboard_calibration.npz"
    )

    main(image_path, calibration_path)