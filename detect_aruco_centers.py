"""Обнаружение центров ArUco-маркеров и сохранение результатов.

Установка: pip install opencv-contrib-python numpy pandas
Запуск: python detect_aruco_centers.py image440.jpg --out results
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Contrast stretch using means of the darkest/brightest 5% of pixels.

    This only changes the 8-bit detector input, never spectral image data.
    """
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    src = image.astype(np.float32)
    good = np.isfinite(src)
    if not good.any():
        raise ValueError('Изображение не содержит конечных значений')
    values = src[good]
    # Estimate the mean of the lowest/highest 5% using partition, not sorting.
    k = max(1, int(np.ceil(values.size * 0.05)))
    low = float(np.partition(values, k - 1)[:k].mean())
    high = float(np.partition(values, values.size - k)[-k:].mean())
    if high <= low + 1e-7:
        return np.zeros(src.shape, dtype=np.uint8)
    stretch = (np.nan_to_num(src, nan=low, neginf=low, posinf=high) - low) * (255 / (high - low))
    return np.clip(stretch, 0, 255).astype(np.uint8)


def detect_centers(image: np.ndarray) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Return detected ArUco center coordinates, preserving the old interface.

    Attempts raw 8-bit, global contrast stretch, and optional CLAHE; unions IDs.
    A detection is only trusted when decoded by OpenCV's ArUco dictionary.
    """
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        original = cv2.cvtColor(image, code)
    else:
        original = image
    if original.dtype == np.uint8:
        raw8 = original
    else:
        raw8 = to_uint8_gray(original)
    enhanced = to_uint8_gray(original)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enhanced)
    candidates = (raw8, enhanced, clahe)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    found = {}
    for gray in candidates:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for marker_id, points in zip(ids.ravel(), corners):
            marker_id = int(marker_id)
            if marker_id not in found:
                found[marker_id] = points.reshape(4, 2).astype(np.float64)
    rows = []
    ordered = []
    for marker_id in sorted(found):
        pts = found[marker_id]
        center = pts.mean(axis=0)
        rows.append({'id': marker_id, 'x': float(center[0]), 'y': float(center[1])})
        ordered.append(pts)
    return pd.DataFrame(rows, columns=['id', 'x', 'y']), ordered


def main() -> None:
    parser = argparse.ArgumentParser(description='Найти центры ArUco-маркеров 6x6_250')
    parser.add_argument('image', type=Path, help='Путь к изображению')
    parser.add_argument('--out', type=Path, default=Path('aruco_results'), help='Папка результатов')
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'Не удалось прочитать {args.image}')

    df, corners = detect_centers(image)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / 'centers.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')

    gray = to_uint8_gray(image)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for row, pts in zip(df.itertuples(index=False), corners):
        polygon = np.round(pts).astype(np.int32)
        cv2.polylines(preview, [polygon], True, (0, 200, 0), 2)
        center = (round(row.x), round(row.y))
        cv2.drawMarker(preview, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(preview, f'ID {row.id}', (center[0] + 12, center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    preview_path = args.out / 'annotated.png'
    cv2.imwrite(str(preview_path), preview)

    print(f'Найдено маркеров: {len(df)}')
    print(df.to_string(index=False))
    print(f'CSV: {csv_path}\nРазметка: {preview_path}')


if __name__ == '__main__':
    main()
