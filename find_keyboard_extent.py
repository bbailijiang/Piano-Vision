"""
Given a top-down photo of an 88-key keyboard, find the pixel regions of the
lowest key (A0) and highest key (C8).

Assumptions for this first pass:
  - The photo is roughly top-down (perspective correction comes later).
  - The keyboard is the dominant bright horizontal band in the image
    (i.e. the photo is already reasonably cropped to the keyboard, or the
    keyboard contrasts against a darker background/surface).
  - All 88 keys are present and white keys are evenly spaced (true on a
    real piano/keyboard).

Algorithm:
  1. Grayscale the image.
  2. For each column, take the MAX brightness value in that column. Every
     column that contains any part of the key bed will have a bright value
     somewhere in it (even under a black key, the lower part of the white
     key beneath/behind it is usually visible) -- so this profile marks
     out the full horizontal extent of the keyboard.
  3. Threshold that profile to get the left and right edges of the keybed.
  4. There are always 52 white keys on an 88-key keyboard, evenly spaced,
     so divide the detected width by 52 to get white-key width, and take
     the first slice as the lowest key (A0) and the last slice as the
     highest key (C8).
  5. Do the same thing row-wise (within the detected column range) to get
     the vertical extent of the keybed.
"""
import cv2
import numpy as np

N_WHITE_KEYS = 52  # fixed for 88-key keyboards


def find_keyboard_extent(image_path, brightness_thresh=None, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Auto threshold via Otsu if none given
    if brightness_thresh is None:
        brightness_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # --- Horizontal extent: column-wise max brightness ---
    col_max = gray.max(axis=0)
    key_cols = np.where(col_max > brightness_thresh)[0]
    if len(key_cols) == 0:
        raise RuntimeError("No bright key-bed region found. Try lowering the threshold.")
    left_edge, right_edge = int(key_cols.min()), int(key_cols.max())

    # --- Vertical extent: row-wise max brightness, restricted to keybed columns ---
    row_max = gray[:, left_edge:right_edge + 1].max(axis=1)
    key_rows = np.where(row_max > brightness_thresh)[0]
    top_edge, bottom_edge = int(key_rows.min()), int(key_rows.max())

    total_width = right_edge - left_edge + 1
    white_key_width = total_width / N_WHITE_KEYS

    lowest_key_box = (
        left_edge,
        top_edge,
        int(left_edge + white_key_width),
        bottom_edge,
    )
    highest_key_box = (
        int(right_edge - white_key_width),
        top_edge,
        right_edge,
        bottom_edge,
    )

    result = {
        "keyboard_bbox": (left_edge, top_edge, right_edge, bottom_edge),
        "white_key_width_px": white_key_width,
        "lowest_key": {"name": "A0", "bbox": lowest_key_box},
        "highest_key": {"name": "C8", "bbox": highest_key_box},
        "brightness_threshold_used": float(brightness_thresh),
    }

    if debug:
        annotated = img.copy()
        cv2.rectangle(annotated, (left_edge, top_edge), (right_edge, bottom_edge), (0, 255, 0), 2)
        cv2.rectangle(annotated, lowest_key_box[:2], lowest_key_box[2:], (0, 0, 255), 2)
        cv2.putText(annotated, "A0", (lowest_key_box[0], max(top_edge - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(annotated, highest_key_box[:2], highest_key_box[2:], (255, 0, 0), 2)
        cv2.putText(annotated, "C8", (highest_key_box[0], max(top_edge - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        out_path = image_path.rsplit(".", 1)[0] + "_annotated.png"
        cv2.imwrite(out_path, annotated)
        result["annotated_image"] = out_path

    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/test_keyboard.png"
    result = find_keyboard_extent(path, debug=True)
    for k, v in result.items():
        print(f"{k}: {v}")
