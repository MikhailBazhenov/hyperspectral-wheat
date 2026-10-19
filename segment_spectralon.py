"""Automatic segmentation of a bright circular Spectralon target."""
from pathlib import Path
import cv2
import numpy as np


def segment_spectralon(image_path, mask_path, min_area_fraction=0.001, min_circularity=0.65,
                       edge_margin_px=5, expected_center=None):
    """Detect Spectralon, save its binary PNG mask, and return (mask, info).

    image_path: path to the input grayscale/BGR/BGRA image (8/16-bit).
    mask_path: output .png file; parent directories are created if needed.
    expected_center: optional (x, y) prior when multiple circles exist.
    edge_margin_px: erosion in pixels to exclude uncertain edge pixels.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() != ".png":
        raise ValueError("Mask must be saved as PNG (lossless binary image)")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot read input image: {image_path}")
    if edge_margin_px < 0:
        raise ValueError("edge_margin_px must be nonnegative")
    if raw.ndim == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY if raw.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError('Expected a 2D grayscale or BGR/BGRA image')
    if gray.dtype not in (np.uint8, np.uint16):
        raise ValueError('Expected uint8 or uint16 image')

    # Work at original dynamic range; Otsu supports both 8-bit and 16-bit.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold, binary = cv2.threshold(blurred, 0, int(np.iinfo(gray.dtype).max),
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < min_area_fraction * w * h or perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / perimeter**2
        if circularity < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if expected_center is None:
            score = area * circularity
        else:
            dist = np.hypot(cx - expected_center[0], cy - expected_center[1])
            score = area * circularity / (1 + (dist / max(radius, 1))**2)
        candidates.append((score, contour, area, circularity, cx, cy, radius))
    if not candidates:
        raise ValueError('No sufficiently large circular bright target found')
    _, contour, area, circularity, cx, cy, radius = max(candidates, key=lambda c: c[0])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    if edge_margin_px > 0:
        k = 2 * int(edge_margin_px) + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if not np.any(mask):
        raise ValueError('Mask empty after erosion; reduce edge_margin_px')
    info = dict(center_x=float(cx), center_y=float(cy), radius_px=float(radius),
                circularity=float(circularity), otsu_threshold=float(threshold),
                pixels=int(np.count_nonzero(mask)), contour_area=float(area))
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise IOError(f"Cannot save mask: {mask_path}")
    return mask, info


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('mask_path', help='Output PNG mask path')
    parser.add_argument('--margin', type=int, default=5)
    args = parser.parse_args()
    mask, info = segment_spectralon(args.image, args.mask_path, edge_margin_px=args.margin)
    print(info)
