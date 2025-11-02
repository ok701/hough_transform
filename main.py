import cv2
import numpy as np
from pathlib import Path
import json

# === Video file setup ===
VIDEO = Path(__file__).with_name("sample1.mp4")
if not VIDEO.exists():
    raise FileNotFoundError(f"Video file not found: {VIDEO}")

# Try normal backend first → fallback to FFMPEG
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {VIDEO}")

# === ROI params (persisted) ===
PARAMS_FILE = Path(__file__).with_name("roi_params.json")

# defaults
horizon_ratio      = 0.8  # top y ratio
top_width_ratio    = 0.15  # trapezoid top width ratio
bottom_width_ratio = 0.38  # trapezoid bottom width ratio
bottom_y_ratio     = 0.95  # bottom y ratio

def load_params():
    """Load ROI ratios from disk if present."""
    global horizon_ratio, top_width_ratio, bottom_width_ratio, bottom_y_ratio
    if PARAMS_FILE.exists():
        try:
            data = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
            horizon_ratio      = float(data.get("horizon_ratio", horizon_ratio))
            top_width_ratio    = float(data.get("top_width_ratio", top_width_ratio))
            bottom_width_ratio = float(data.get("bottom_width_ratio", bottom_width_ratio))
            bottom_y_ratio     = float(data.get("bottom_y_ratio", bottom_y_ratio))
        except Exception:
            pass  # ignore malformed file

def save_params():
    """Persist current ROI ratios to disk."""
    data = {
        "horizon_ratio":      round(horizon_ratio, 4),
        "top_width_ratio":    round(top_width_ratio, 4),
        "bottom_width_ratio": round(bottom_width_ratio, 4),
        "bottom_y_ratio":     round(bottom_y_ratio, 4),
    }
    PARAMS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

load_params()  # load on startup

def make_trapezoid_roi(width, height):
    """
    Create trapezoid ROI polygon and return (roi_poly, top_y, bottom_y).
    Lines will be clipped between these y-bounds so their length changes with ROI.
    """
    top_y    = int(height * horizon_ratio)
    bottom_y = int(height * bottom_y_ratio)
    top_w    = int(width  * top_width_ratio)
    bottom_w = int(width  * bottom_width_ratio)

    top_x1 = (width - top_w) // 2
    top_x2 = top_x1 + top_w
    bot_x1 = (width - bottom_w) // 2
    bot_x2 = bot_x1 + bottom_w

    roi_poly = np.array([[
        (bot_x1, bottom_y),
        (top_x1, top_y),
        (top_x2, top_y),
        (bot_x2, bottom_y),
    ]], dtype=np.int32)
    return roi_poly, top_y, bottom_y

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# === Main loop ===
running = True
while running:
    ret, img = cap.read()
    if not ret:
        break

    height, width = img.shape[:2]

    # --- preprocessing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equalized_gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 140)

    # --- ROI & mask ---
    roi_array, top_y, bottom_y = make_trapezoid_roi(width, height)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_array, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # (optional) visualize ROI outline
    cv2.polylines(img, roi_array, isClosed=True, color=(0, 255, 0), thickness=2)

    # --- Hough lines ---
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=6,
        theta=np.pi/60,
        threshold=90,
        minLineLength=40,
        maxLineGap=25
    )
    overlay = np.zeros_like(img)

    if lines is not None:
        x_right, y_right, x_left, y_left = [], [], [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:  # ignore nearly horizontal
                continue
            if slope > 0:
                x_right.extend([x1, x2]); y_right.extend([y1, y2])
            else:
                x_left.extend([x1, x2]);  y_left.extend([y1, y2])

        if len(x_left) >= 2 and len(x_right) >= 2:
            # Fit x = a*y + b
            left_eq  = np.poly1d(np.polyfit(y_left,  x_left,  1))
            right_eq = np.poly1d(np.polyfit(y_right, x_right, 1))

            # IMPORTANT: clip line endpoints to ROI vertical range
            y0 = bottom_y
            y1 = top_y

            # compute x for both y0 (bottom) and y1 (top), clamp to frame bounds
            left_x_bottom  = clamp(int(left_eq(y0)),  0, width - 1)
            left_x_top     = clamp(int(left_eq(y1)),  0, width - 1)
            right_x_bottom = clamp(int(right_eq(y0)), 0, width - 1)
            right_x_top    = clamp(int(right_eq(y1)), 0, width - 1)

            # Draw lane lines (their length changes with ROI now)
            cv2.line(overlay, (left_x_bottom,  y0), (left_x_top,  y1), (0, 0, 255), 10)
            cv2.line(overlay, (right_x_bottom, y0), (right_x_top, y1), (0, 0, 255), 10)

    # --- compose result ---
    result = cv2.addWeighted(img, 1.0, overlay, 1.0, 0.0)

    # HUD: show key instructions
    info1 = "ROI : W/S: Move horizon | A/D: Top width | Z/C: Bottom width"
    cv2.putText(result, info1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.imshow("Lane Detection (ROI-tuned)", result)

    # --- keyboard controls ---
    key = cv2.waitKey(5) & 0xFF
    step = 0.01
    changed = False

    if key == 27:  # ESC
        running = False
    elif key == ord('w'): horizon_ratio      = clamp(horizon_ratio - step,      0.00, 0.99); changed = True
    elif key == ord('s'): horizon_ratio      = clamp(horizon_ratio + step,      0.00, 0.99); changed = True
    elif key == ord('a'): top_width_ratio    = clamp(top_width_ratio - step,    0.05, 0.95); changed = True
    elif key == ord('d'): top_width_ratio    = clamp(top_width_ratio + step,    0.05, 0.95); changed = True
    elif key == ord('z'): bottom_width_ratio = clamp(bottom_width_ratio - step, 0.10, 0.99); changed = True
    elif key == ord('c'): bottom_width_ratio = clamp(bottom_width_ratio + step, 0.10, 0.99); changed = True

    # persist immediately on change (so last value survives crashes)
    if changed:
        save_params()

cap.release()
cv2.destroyAllWindows()