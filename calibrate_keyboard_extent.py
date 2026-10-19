"""
keyboard_calibration.py

Manual calibration for an 88-key piano.

Workflow
--------
1. Open an image.
2. Click:
       Top Left
       Top Right
       Bottom Left
       Bottom Right
   of the WHITE KEY BED.
3. The program computes a perspective transform.
4. The warped keyboard is produced.
5. The 52 white keys are generated automatically.
6. Calibration is saved for future videos.

Author: ChatGPT
"""

import cv2
import numpy as np
import os

# -------------------------------------------------------------------
# Adjustable parameters
# -------------------------------------------------------------------

N_WHITE_KEYS = 52

# Size of rectified keyboard image
KEYBOARD_WIDTH = 5200
KEYBOARD_HEIGHT = 400

# -------------------------------------------------------------------
# Mouse interaction
# -------------------------------------------------------------------

clicked_points = []


def draw(display):
    """
    Draw clicked points and connecting lines.
    """

    img = display.copy()

    for i, pt in enumerate(clicked_points):

        cv2.circle(img, pt, 6, (0, 0, 255), -1)

        cv2.putText(
            img,
            str(i + 1),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(img, clicked_points[i], clicked_points[i + 1],
                     (255, 0, 0), 2)

    cv2.imshow("Keyboard Calibration", img)


def mouse_callback(event, x, y, flags, display):

    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            draw(display)


# -------------------------------------------------------------------
# User calibration
# -------------------------------------------------------------------

def select_keyboard(image):

    clicked = []

    window = "Keyboard Calibration"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    display = image.copy()

    while True:

        temp = display.copy()

        # Draw clicked points
        for i, p in enumerate(clicked):
            cv2.circle(temp, p, 6, (0,0,255), -1)
            cv2.putText(
                temp,
                str(i+1),
                (p[0]+8,p[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        # Draw polygon
        if len(clicked) >= 2:
            for i in range(len(clicked)-1):
                cv2.line(temp, clicked[i], clicked[i+1], (255,0,0), 2)

        if len(clicked) == 4:
            cv2.line(temp, clicked[3], clicked[0], (255,0,0), 2)

        cv2.imshow(window, temp)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'): # press r to reset calibration
            print("clicked r key")
            clicked = []

        elif key == ord('d'): # press d to delete most recently placed dot
            print("clicked d key")
            if clicked != []: 
                clicked.pop()

        elif key == 13 or key == 32:      # Enter or Space to accept calibration
            if len(clicked) == 4:
                break

        elif key == 27:                   # ESC to cancel 
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled.")

        def mouse(event, x, y, flags, param):
            nonlocal clicked

            if event == cv2.EVENT_LBUTTONDOWN:

                if len(clicked) < 4:
                    clicked.append((x,y))

            elif event == cv2.EVENT_RBUTTONDOWN:

                if clicked:
                    clicked.pop()

        cv2.setMouseCallback(window, mouse)

    cv2.destroyWindow(window)

    return np.float32(clicked)


# -------------------------------------------------------------------
# Perspective warp
# -------------------------------------------------------------------

def warp_keyboard(image, src_points):

    dst_points = np.float32(
        [
            [0, 0],
            [KEYBOARD_WIDTH, 0],
            [0, KEYBOARD_HEIGHT],
            [KEYBOARD_WIDTH, KEYBOARD_HEIGHT],
        ]
    )

    H = cv2.getPerspectiveTransform(src_points, dst_points)

    warped = cv2.warpPerspective(
        image,
        H,
        (KEYBOARD_WIDTH, KEYBOARD_HEIGHT),
    )

    return warped, H


# -------------------------------------------------------------------
# Generate key polygons
# -------------------------------------------------------------------

def generate_white_keys(white_width):
    keys = []

    white_names = [
        "A0","B0",
        "C1","D1","E1","F1","G1","A1","B1",
        "C2","D2","E2","F2","G2","A2","B2",
        "C3","D3","E3","F3","G3","A3","B3",
        "C4","D4","E4","F4","G4","A4","B4",
        "C5","D5","E5","F5","G5","A5","B5",
        "C6","D6","E6","F6","G6","A6","B6",
        "C7","D7","E7","F7","G7","A7","B7",
        "C8"
    ]

    for i, name in enumerate(white_names):

        x1 = int(round(i * white_width))
        x2 = int(round((i + 1) * white_width))

        keys.append({
            "name": name,
            "type": "white",
            "bbox": (x1,0,x2,KEYBOARD_HEIGHT)
        })

    return keys



def generate_black_keys(white_width, black_height):
    keys = []

    cde = [
        ("C#",0.1857,0.4000),
        ("D#",0.6000,0.8143),
    ]

    fgab = [
        ("F#",0.1398,0.3011),
        ("G#",0.4301,0.5914),
        ("A#",0.7204,0.8817),
    ]

    # A#0

    block = 2*white_width

    keys.append({
        "name":"A#0",
        "type":"black",
        "bbox":(
            int(fgab[2][1]*0 + fgab[2][1]),0,0,0
        )
    })

    # We'll overwrite the bbox correctly below.

    # Full octaves

    for octave in range(1,8):

        start = (2 + (octave-1)*7) * white_width

        # CDE

        for note,left,right in cde:

            x1 = int(start + left*3*white_width)
            x2 = int(start + right*3*white_width)

            keys.append({
                "name":f"{note}{octave}",
                "type":"black",
                "bbox":(
                    x1,
                    0,
                    x2,
                    black_height
                )
            })

        # FGAB

        start = (5 + (octave-1)*7) * white_width

        for note,left,right in fgab:

            x1 = int(start + left*4*white_width)
            x2 = int(start + right*4*white_width)

            keys.append({
                "name":f"{note}{octave}",
                "type":"black",
                "bbox":(
                    x1,
                    0,
                    x2,
                    black_height
                )
            })

    # Correct A#0

    left = fgab[2][1]
    print(left)

    return keys



def generate_key_polygons():

    keys = []

    white_width = KEYBOARD_WIDTH / 52
    black_height = int(KEYBOARD_HEIGHT * 0.62)

    keys.extend(generate_white_keys(white_width))
    keys.extend(generate_black_keys(white_width, black_height))

    return keys

# -------------------------------------------------------------------
# Draw all white and black keys
# -------------------------------------------------------------------

def draw_keyboard(image, keys):

    out = image.copy()

    for key in keys:

        x1,y1,x2,y2 = key["bbox"]

        color = (0,255,0)

        if key["type"] == "black":
            color = (255,0,255)

        cv2.rectangle(
            out,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.putText(
            out,
            key["name"],
            (x1+3,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1
        )

    return out

# -------------------------------------------------------------------
# Save calibration
# -------------------------------------------------------------------

def save_calibration(filename, H):

    np.savez(
        filename,
        H=H,
        keyboard_width=KEYBOARD_WIDTH,
        keyboard_height=KEYBOARD_HEIGHT,
    )

    print(f"Calibration saved to {filename}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def calibrate(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    src_points = select_keyboard(image)

    warped, H = warp_keyboard(image, src_points)

    all_keys = generate_key_polygons()

    annotated = draw_keyboard(warped, all_keys)

    base = os.path.splitext(image_path)[0]

    warped_path = base + "_warped.png"
    annotated_path = base + "_keys.png"
    calibration_path = base + "_calibration.npz"

    cv2.imwrite(warped_path, warped)
    cv2.imwrite(annotated_path, annotated)

    save_calibration(calibration_path, H)

    print()
    print("Finished.")
    print("Warped image:      ", warped_path)
    print("Annotated image:   ", annotated_path)

    cv2.imshow("Warped Keyboard", annotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H, all_keys


# -------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "test_keyboard.jpg"

    calibrate(path)