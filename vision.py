"""Module 1 - Plant Vision Intelligence: colour segmentation and canopy-shape features."""
import cv2
import numpy as np

# OpenCV HSV: H 0-179, S/V 0-255. Tune for your camera and lighting.
GREEN_LO, GREEN_HI = (35, 50, 40), (85, 255, 255)
YELLOW_LO, YELLOW_HI = (18, 60, 60), (34, 255, 255)
MIN_PLANT_FRACTION = 0.005  # below this share of the frame, report "no plant found"

FEATURE_KEYS = ["green_ratio", "yellow_ratio", "canopy_area", "aspect", "centroid_y", "scene_brightness"]


def _clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def crop_roi(bgr, roi):
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = roi
    xa, ya = int(x0 * w), int(y0 * h)
    xb, yb = max(int(x1 * w), xa + 1), max(int(y1 * h), ya + 1)
    return bgr[ya:yb, xa:xb], (xa, ya)


def analyze_frame(bgr, roi=(0.0, 0.0, 1.0, 1.0)):
    """Return (features, debug).

    features:
      green_ratio / yellow_ratio  share of plant pixels that are green / yellow
      canopy_area                 plant pixels as a fraction of the analysed region
      aspect                      canopy height / width (drops when leaves droop outward)
      centroid_y                  0 = leaf mass at top of canopy, 1 = at the bottom
                                  (rises when leaves hang down)
      scene_brightness            mean brightness, so lighting changes can be told apart
    """
    crop, offset = crop_roi(bgr, roi)
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
    green = _clean(cv2.inRange(hsv, GREEN_LO, GREEN_HI))
    yellow = _clean(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI))
    plant = cv2.bitwise_or(green, yellow)
    plant_px = int(cv2.countNonZero(plant))

    feats = {k: np.nan for k in FEATURE_KEYS}
    feats["scene_brightness"] = float(hsv[..., 2].mean() / 255.0)
    feats["plant_detected"] = False
    debug = {"green": green, "yellow": yellow, "plant": plant, "offset": offset, "bbox": None}
    if plant_px < MIN_PLANT_FRACTION * h * w:
        return feats, debug

    ys, xs = np.nonzero(plant)
    top, bottom = np.percentile(ys, [2, 98])
    left, right = np.percentile(xs, [2, 98])
    height, width = max(bottom - top, 1.0), max(right - left, 1.0)
    feats.update(
        green_ratio=cv2.countNonZero(green) / plant_px,
        yellow_ratio=cv2.countNonZero(yellow) / plant_px,
        canopy_area=plant_px / float(h * w),
        aspect=float(height / width),
        centroid_y=float(np.clip((ys.mean() - top) / height, 0.0, 1.0)),
        plant_detected=True,
    )
    debug["bbox"] = (left, top, right, bottom)
    return feats, debug


def motion_between(debug_a, debug_b):
    """Fraction of plant pixels that changed between two captures (leaf movement)."""
    a, b = debug_a["plant"], debug_b["plant"]
    if a.shape != b.shape:
        return float("nan")
    return float(cv2.countNonZero(cv2.bitwise_xor(a, b)) / max(cv2.countNonZero(a), 1))


def annotate(bgr, debug):
    """Overlay green/yellow segmentation, the ROI and the canopy box on a copy of the frame."""
    out = bgr.copy()
    ox, oy = debug["offset"]
    h, w = debug["plant"].shape
    region = out[oy:oy + h, ox:ox + w]
    overlay = region.copy()
    overlay[debug["green"] > 0] = (0, 200, 0)
    overlay[debug["yellow"] > 0] = (0, 255, 255)
    out[oy:oy + h, ox:ox + w] = cv2.addWeighted(region, 0.55, overlay, 0.45, 0)
    cv2.rectangle(out, (ox, oy), (ox + w - 1, oy + h - 1), (255, 160, 0), 1)
    if debug.get("bbox"):
        l, t, r, b = (int(v) for v in debug["bbox"])
        cv2.rectangle(out, (ox + l, oy + t), (ox + r, oy + b), (0, 0, 255), 2)
    return out
