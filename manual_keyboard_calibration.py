# All colors are BGR, so ex: (0, 255, 255) is yellow, not teal

import cv2
import numpy as np
import os

WINDOW_NAME = "Keyboard Calibration"
KEYBOARD_WIDTH = 5200
KEYBOARD_HEIGHT = 600

def rgb(r, g, b):
    return (b, g, r)

def click_points(image, numpoints, prefix, mode):
    """
    Generic point-clicking routine.

    Parameters
    ----------
    image : ndarray
        Image to display.

    numpoints : int
        Number of points required.

    prefix : str
        Label prefix ("C", "W", or "B").

    mode : str
        Text shown at the top of the image.

    Returns
    -------
    list[(x,y)]
    """

    points = []

    selected = None      # currently moving point
    CLICK_RADIUS = 10

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, param):

        nonlocal points, selected

        if event != cv2.EVENT_LBUTTONDOWN and event != cv2.EVENT_RBUTTONDOWN:
            return

        # ------------------------------------------------
        # RIGHT CLICK
        # Delete nearest point
        # ------------------------------------------------

        if event == cv2.EVENT_RBUTTONDOWN:

            nearest = None

            for i, p in enumerate(points):

                if np.hypot(x-p[0], y-p[1]) < CLICK_RADIUS:

                    nearest = i
                    break

            if nearest is not None:

                points.pop(nearest)

                if selected is not None:

                    selected = None

            return

        # ------------------------------------------------
        # LEFT CLICK
        # ------------------------------------------------

        nearest = None

        for i, p in enumerate(points):

            if np.hypot(x-p[0], y-p[1]) < CLICK_RADIUS:

                nearest = i
                break

        # ------------------------------------------------
        # Drop selected point
        # ------------------------------------------------

        if selected is not None:

            points[selected] = (x, y)

            selected = None

            return

        # ------------------------------------------------
        # Pick up an existing point
        # ------------------------------------------------

        if nearest is not None:

            selected = nearest

            return

        # ------------------------------------------------
        # Add new point
        # ------------------------------------------------

        if len(points) < numpoints:

            points.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, mouse)

    while True:

        display = image.copy()

        # Draw previously clicked points
        for i, p in enumerate(points):

            color = (0,0,255)

            if selected == i:
                color = (0,255,255)

            cv2.circle(display, p, 6, color, -1)

            cv2.putText(
                display,
                f"{prefix}{i+1}",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
            )

        controls = [
            "CONTROLS:", 
            "Left click: add point",
            "Left click on point: move point",
            "Right click on point: delete point",
            "D: delete last point",
            "R: reset all points",
            "Enter: continue",
            "Esc: quit",
        ]

        ''' 
        void putText(InputOutputArray img, const String& text, Point org, 
                        int fontFace, double fontScale, Scalar color, 
                        int thickness = 1, int lineType = LINE_8, 
                        bool bottomLeftOrigin = false); 
        '''

        for i, line in enumerate(controls):
            cv2.putText(
                display,
                line,
                (20, 60 + i * 60),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 0, 255),
                4,
            )

        # Corners/Whites/Black mode
        cv2.putText(
            display,
            "Mode: " + mode,
            (1250, 120),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 0, 255),
            4,
        )

        # Progress
        cv2.putText(
            display,
            f"Points placed: {len(points)}/{numpoints}",
            (2500, 120),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 0, 255),
            4,
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(20) & 0xFF

        # Delete last point
        if key == ord("d"):

            if points:
                points.pop()

        # Reset
        elif key == ord("r"):

            points = []

        # Finish
        elif key == 13:

            if len(points) == numpoints:
                break

        # Cancel
        elif key == 27:

            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled.")

    return points

def select_keyboard_corners(image):
    return click_points(image, 4, "C", "Corners")

def select_white_keys(image):
    return click_points(image, 52, "W", "White keys")

def select_black_keys(image):
    return click_points(image, 36, "B", "Black keys")

def compute_edge_parameters(centers):
    """
    Given ordered key centers, compute the left/right boundaries
    between neighboring keys.

    Returns
    -------
    edges : ndarray
        Length len(centers)+1
    """

    xs = np.array([p[0] for p in centers], dtype=float)

    edges = np.zeros(len(xs)+1)

    # Interior edges
    edges[1:-1] = (xs[:-1] + xs[1:]) / 2

    # Extrapolate the first/last
    edges[0] = xs[0] - (edges[1] - xs[0])
    edges[-1] = xs[-1] + (xs[-1] - edges[-2])

    return edges


def interpolate_boundary(corners, x_fraction):
    """
    Parameters
    ----------
    corners
        [TL, TR, BR, BL]

    x_fraction
        Between 0 and 1

    Returns
    -------
    top_point
    bottom_point
    """

    TL, TR, BL, BR = [np.asarray(p, float) for p in corners]

    top = TL + x_fraction * (TR - TL)

    bottom = BL + x_fraction * (BR - BL)

    return top, bottom

WHITE_NAMES = [
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

BLACK_NAMES = [
    "Bb0",
    "C#1","Eb1","F#1","Ab1","Bb1",
    "C#2","Eb2","F#2","Ab2","Bb2",
    "C#3","Eb3","F#3","Ab3","Bb3",
    "C#4","Eb4","F#4","Ab4","Bb4",
    "C#5","Eb5","F#5","Ab5","Bb5",
    "C#6","Eb6","F#6","Ab6","Bb6",
    "C#7","Eb7","F#7","Ab7","Bb7"
]


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



def generate_white_polygons(corners, white_centers):

    edges = compute_edge_parameters(white_centers)

    width = edges[-1] - edges[0]

    keys = []

    for i in range(52):

        left = (edges[i] - edges[0]) / width
        right = (edges[i+1] - edges[0]) / width

        tl, bl = interpolate_boundary(corners, left)
        tr, br = interpolate_boundary(corners, right)

        poly = np.array([
            tl,
            tr,
            br,
            bl
        ], dtype=np.int32)

        keys.append({
            "name": WHITE_NAMES[i],
            "type": "white",
            "center": white_centers[i],
            "polygon": poly
        })

    return keys




BLACK_HEIGHT_RATIO = 0.62
BLACK_WIDTH_RATIO = 4.5 / 7


def generate_black_polygons(corners, black_centers):
    width = KEYBOARD_WIDTH / 52 * BLACK_WIDTH_RATIO

    keys = []

    for i in range(36):
        x, y = black_centers[i]

        left = x - width/2
        right = x + width/2

        left_frac = left/KEYBOARD_WIDTH
        right_frac = right/KEYBOARD_WIDTH

        tl, bl = interpolate_boundary(corners,left_frac)
        tr, br = interpolate_boundary(corners,right_frac)

        bl = tl + BLACK_HEIGHT_RATIO*(bl-tl)
        br = tr + BLACK_HEIGHT_RATIO*(br-tr)

        poly = np.array([
            tl,
            tr,
            br,
            bl
        ],dtype=np.int32)

        keys.append({
            "name":BLACK_NAMES[i],
            "type":"black",
            "center":black_centers[i],
            "polygon":poly
        })

    return keys


# NEXT TIME: revise this function for black keys, maybe create separate function
def draw_polygons(image, keys):

    out = image.copy()

    for key in keys:

        color = (0,255,0)

        if key["type"] == "black":
            color = (255,0,255)

        cv2.polylines(
            out,
            [key["polygon"]],
            True,
            color,
            2
        )

        c = tuple(np.array(key["center"],dtype=int))

        cv2.circle(out,c,3,color,-1)

        cv2.putText(
            out,
            key["name"],
            (c[0]+5,c[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1
        )

    return out


def save_calibration(filename,
                     H,
                     corners,
                     white_centers,
                     black_centers):

    np.savez(
        filename,
        H=H,
        corners=np.array(corners),
        white_centers=np.array(white_centers),
        black_centers=np.array(black_centers)
    )

    print(f"Saved calibration to {filename}")


def load_calibration(filename):

    data = np.load(filename, allow_pickle=True)

    return (
        data["H"],
        data["corners"],
        data["white_centers"],
        data["black_centers"]
    )

def main(path):

    # Try loading as an image first
    image = cv2.imread(path)

    if image is None:

        # If it isn't an image, try opening it as a video
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open image or video: {path}"
            )

        ret, image = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(
                f"Could not read first frame of video: {path}"
            )

    # --------------------------------------------------
    # User calibration
    # --------------------------------------------------

    corners = select_keyboard_corners(image)

    white = select_white_keys(image)

    black = select_black_keys(image)

    # --------------------------------------------------
    # Warp the keyboard
    # --------------------------------------------------

    warped_image, H = warp_keyboard(image, np.float32(corners))

    # --------------------------------------------------
    # Warp all clicked points
    # --------------------------------------------------

    corner_pts = np.array(corners, dtype=np.float32).reshape(-1,1,2)
    warped_corner = cv2.perspectiveTransform(corner_pts, H).reshape(-1,2)

    white_pts = np.array(white, dtype=np.float32).reshape(-1,1,2)
    warped_white = cv2.perspectiveTransform(white_pts, H).reshape(-1,2)

    black_pts = np.array(black, dtype=np.float32).reshape(-1,1,2)
    warped_black = cv2.perspectiveTransform(black_pts, H).reshape(-1,2)

    # --------------------------------------------------
    # Generate polygons
    # --------------------------------------------------

    white_keys = generate_white_polygons(warped_corner, warped_white)
    black_keys = generate_black_polygons(warped_corner, warped_black)

    keys = white_keys + black_keys

    # --------------------------------------------------
    # Draw polygons
    # --------------------------------------------------

    preview = draw_polygons(warped_image, keys)

    cv2.imshow("Calibration", preview)

    # --------------------------------------------------
    # Save calibration
    # --------------------------------------------------

    base = os.path.splitext(path)[0]
    preview_path = base + "_allkeys.png"
    cv2.imwrite(preview_path, preview)


    save_calibration(
        base + "_keyboard_calibration.npz",
        H,
        warped_corner,
        warped_white,
        warped_black,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "keyboard.jpg"

    main(path)