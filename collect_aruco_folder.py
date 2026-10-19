"""Сбор координат ArUco-маркеров для изображений image<длина_волны>.jpg/png.

Зависимости: pip install opencv-contrib-python numpy pandas
Рядом должен находиться detect_aruco_centers.py из первого этапа.

Пример CLI:
    python collect_aruco_folder.py /path/to/images --csv centers_long.csv
"""

import argparse
import re
import warnings
from pathlib import Path

import cv2
import pandas as pd

from detect_aruco_centers import detect_centers


_FILENAME = re.compile(r"image(\d+)\.(jpg|png)", re.IGNORECASE)
_COLUMNS = ["wavelength", "file", "id", "x", "y"]


def collect_aruco_folder(folder: str | Path) -> pd.DataFrame:
    """Найти центры маркеров на всех подходящих снимках папки.

    Args:
        folder: Папка с файлами image<целая длина волны>.jpg/.png.

    Returns:
        Long-format DataFrame с колонками:
        wavelength (int), file (str), id (int), x (float), y (float).
        По одной строке на каждый распознанный маркер каждого изображения.
        Если маркеров нет, строки для этого снимка отсутствуют.

    Raises:
        NotADirectoryError: Папка не существует или путь указывает не на папку.
        ValueError: Нет подходящих файлов либо одна длина волны встречается
            в нескольких файлах.
        OSError: Не удалось прочитать изображение.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {folder}")

    images = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = _FILENAME.fullmatch(path.name)
        if match:
            images.append((int(match.group(1)), path))

    if not images:
        raise ValueError(f"В {folder} не найдены файлы image<число>.jpg или .png")

    images.sort(key=lambda item: (item[0], item[1].name.lower()))
    wavelengths = [wavelength for wavelength, _ in images]
    if len(wavelengths) != len(set(wavelengths)):
        duplicates = sorted({w for w in wavelengths if wavelengths.count(w) > 1})
        raise ValueError(f"Несколько файлов для одной длины волны: {duplicates}")

    frames = []
    for wavelength, path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Не удалось прочитать изображение: {path}")

        markers, _ = detect_centers(image)
        if markers.empty:
            warnings.warn(f"На {path.name} не найдено маркеров", stacklevel=2)
            continue

        chunk = markers.loc[:, ["id", "x", "y"]].copy()
        chunk.insert(0, "file", path.name)
        chunk.insert(0, "wavelength", wavelength)
        frames.append(chunk)

    if not frames:
        return pd.DataFrame({
            "wavelength": pd.Series(dtype="int64"),
            "file": pd.Series(dtype="str"),
            "id": pd.Series(dtype="int64"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
        })

    result = pd.concat(frames, ignore_index=True)
    result = result.loc[:, _COLUMNS]
    return result.sort_values(["wavelength", "id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать координаты ArUco-маркеров по длинам волн")
    parser.add_argument("folder", type=Path, help="Папка с изображениями")
    parser.add_argument("--csv", type=Path, default=None, help="Куда сохранить CSV (необязательно)")
    args = parser.parse_args()

    df = collect_aruco_folder(args.folder)
    print(df.to_string(index=False))
    print(f"\nВсего обнаружений: {len(df)}; длин волн: {df['wavelength'].nunique()}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False, float_format="%.4f")
        print(f"CSV сохранён: {args.csv}")


if __name__ == "__main__":
    main()
