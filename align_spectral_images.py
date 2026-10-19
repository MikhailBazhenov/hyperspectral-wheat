"""Аффинное совмещение спектральных JPG/PNG по таблице центров ArUco.

Рядом должен находиться collect_aruco_folder.py (он импортирует
ранее созданный detect_aruco_centers.py).

Пример:
    python align_spectral_images.py centers_long.csv 700 images --out aligned

Зависимости: pip install opencv-contrib-python numpy pandas
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Повторно используем правило имён из ранее созданного модуля.
from collect_aruco_folder import _FILENAME


def _fit_affine(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Метод наименьших квадратов: точки source -> target, матрица 2x3."""
    if source_xy.shape[0] < 3:
        raise ValueError("Для аффинного преобразования нужны хотя бы 3 общих маркера")
    design = np.column_stack([source_xy, np.ones(len(source_xy))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Общие маркеры лежат на одной прямой: матрицу оценить нельзя")
    coefficients, _, _, _ = np.linalg.lstsq(design, target_xy, rcond=None)
    return coefficients.T.astype(np.float64)  # (2,3)


def align_spectral_images(
    centers_csv: str | Path,
    reference_wavelength: int,
    images_folder: str | Path,
    output_folder: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Совместить все image<длина_волны>.jpg/.png с опорным изображением.

    Параметры:
        centers_csv: CSV с колонками wavelength, id, x, y (file необязателен).
        reference_wavelength: длина волны опорного кадра, целое число.
        images_folder: папка с исходными JPG/PNG.
        output_folder: папка результата; по умолчанию соседняя папка
            <images_folder>_aligned_<reference_wavelength>nm.

    Возвращает:
        (путь_к_папке_результата, diagnostics_df).

    diagnostics_df содержит число общих маркеров, матрицу преобразования,
    RMS и максимальную ошибку на центрах маркеров. Ошибки являются ошибками
    подгонки, а не независимой оценкой точности совмещения.

    Каждый результат имеет те же ширину, высоту, каналы и расширение, что
    соответствующий исходный файл; незаполненные области становятся чёрными.
    Опорный файл копируется без изменений.
    """
    centers_csv = Path(centers_csv).expanduser().resolve()
    images_folder = Path(images_folder).expanduser().resolve()
    reference_wavelength = int(reference_wavelength)
    if not centers_csv.is_file():
        raise FileNotFoundError(f"CSV не найден: {centers_csv}")
    if not images_folder.is_dir():
        raise NotADirectoryError(f"Папка изображений не найдена: {images_folder}")

    if output_folder is None:
        output_folder = images_folder.with_name(
            f"{images_folder.name}_aligned_{reference_wavelength}nm"
        )
    output_folder = Path(output_folder).expanduser().resolve()
    if output_folder == images_folder:
        raise ValueError("Папка результатов не должна совпадать с исходной папкой")

    df = pd.read_csv(centers_csv)
    required = {"wavelength", "id", "x", "y"}
    if missing := required - set(df.columns):
        raise ValueError(f"В CSV отсутствуют столбцы: {sorted(missing)}")
    for column in ("wavelength", "id", "x", "y"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[["wavelength", "id", "x", "y"]].isna().any().any():
        raise ValueError("CSV содержит пропущенные координаты или идентификаторы")
    for column in ("wavelength", "id"):
        if not np.all(df[column] == np.floor(df[column])):
            raise ValueError(f"{column} должен содержать целые числа")
        df[column] = df[column].astype(int)
    if df.duplicated(["wavelength", "id"]).any():
        raise ValueError("В CSV есть дублирующиеся пары (wavelength, id)")

    ref = df.loc[df.wavelength == reference_wavelength, ["id", "x", "y"]]
    if len(ref) < 3:
        raise ValueError(f"У опорной длины волны {reference_wavelength} нм меньше 3 маркеров")
    ref = ref.set_index("id")

    image_paths: dict[int, Path] = {}
    for path in images_folder.iterdir():
        if not path.is_file():
            continue
        match = _FILENAME.fullmatch(path.name)
        if match is None:
            continue
        wl = int(match.group(1))
        if wl in image_paths:
            raise ValueError(f"Два изображения для длины волны {wl} нм")
        image_paths[wl] = path
    if not image_paths:
        raise ValueError(f"В папке нет изображений image<число>.jpg/.png: {images_folder}")
    if reference_wavelength not in image_paths:
        raise ValueError(f"Нет опорного файла image{reference_wavelength}.jpg/.png")

    # Robust fit where available; interpolate transforms for sparsely detected
    # wavelengths. This avoids extrapolating a transform beyond measured bands.
    transforms = {}
    details = {}
    for wl, path in sorted(image_paths.items()):
        current = df.loc[df.wavelength == wl, ["id", "x", "y"]].set_index("id")
        common_ids = ref.index.intersection(current.index).sort_values()
        n = len(common_ids)
        if wl == reference_wavelength:
            matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
            err = np.zeros(n)
            method = 'reference'
        elif n >= 3:
            src = current.loc[common_ids, ["x", "y"]].to_numpy(dtype=np.float64)
            dst = ref.loc[common_ids, ["x", "y"]].to_numpy(dtype=np.float64)
            # RANSAC rejects incorrectly localized markers; LS refits inliers.
            seed = np.asarray(_fit_affine(src, dst), dtype=np.float64)
            if n >= 4:
                robust, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                    ransacReprojThreshold=3.0, maxIters=1500, confidence=0.99,
                    refineIters=10)
                if robust is not None and mask is not None and int(mask.sum()) >= 3:
                    inliers = mask.ravel().astype(bool)
                    try:
                        seed = _fit_affine(src[inliers], dst[inliers])
                    except ValueError:
                        pass
            matrix = seed
            err = np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)
            method = 'aruco'
        else:
            matrix = None
            err = np.array([])
            method = 'missing'
        if matrix is not None:
            transforms[wl] = matrix
        details[wl] = (path, n, err, method)

    observed = sorted(transforms)
    for wl, (path, n, err, method) in details.items():
        if wl in transforms:
            continue
        left = max((v for v in observed if v < wl), default=None)
        right = min((v for v in observed if v > wl), default=None)
        if left is None or right is None:
            nearest = right if left is None else left
            # Adjacent-band fallback for edge wavelengths only; never apply a
            # transform from a distant wavelength without clear diagnostics.
            if nearest is None or abs(wl - nearest) > 30:
                raise ValueError(
                    f"{path.name}: только {n} общих маркеров; "
                    "нет валидных соседних длин волн для безопасного восстановления"
                )
            transforms[wl] = transforms[nearest].copy()
            details[wl] = (path, n, err, 'nearest_band')
            continue
        t = (wl - left) / (right - left)
        transforms[wl] = (1.0 - t) * transforms[left] + t * transforms[right]
        details[wl] = (path, n, err, 'interpolated')

    rows = []
    for wl, (path, n, err, method) in sorted(details.items()):
        matrix = transforms[wl]
        rows.append({
            'wavelength': wl, 'file': path.name, 'n_markers': n,
            'rms_px': float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan,
            'max_error_px': float(np.max(err)) if err.size else np.nan,
            'method': method,
            'm00': matrix[0, 0], 'm01': matrix[0, 1], 'tx': matrix[0, 2],
            'm10': matrix[1, 0], 'm11': matrix[1, 1], 'ty': matrix[1, 2],
        })

    # Размеры кадров должны быть одинаковы; иначе одинаковый размер результата
    # и единая система координат опорного кадра будут противоречить друг другу.
    ref_image = cv2.imread(str(image_paths[reference_wavelength]), cv2.IMREAD_UNCHANGED)
    if ref_image is None:
        raise OSError(f"Не удалось прочитать {image_paths[reference_wavelength]}")
    ref_h, ref_w = ref_image.shape[:2]
    for wl, path in sorted(image_paths.items()):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Не удалось прочитать {path}")
        h, w = image.shape[:2]
        if (w, h) != (ref_w, ref_h):
            raise ValueError(f"{path.name}: размер {w}x{h}, ожидается {ref_w}x{ref_h}")

    output_folder.mkdir(parents=True, exist_ok=True)
    for wl, path in sorted(image_paths.items()):
        destination = output_folder / path.name
        if wl == reference_wavelength:
            shutil.copy2(path, destination)  # без повторного сжатия JPEG
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        aligned = cv2.warpAffine(
            image, transforms[wl], (ref_w, ref_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        options = ([cv2.IMWRITE_JPEG_QUALITY, 100] if path.suffix.lower() == ".jpg"
                   else [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not cv2.imwrite(str(destination), aligned, options):
            raise OSError(f"Не удалось записать {destination}")

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_folder / "alignment_diagnostics.csv", index=False, float_format="%.8f")
    return output_folder, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Аффинно совместить спектральные изображения")
    parser.add_argument("centers_csv", type=Path)
    parser.add_argument("reference_wavelength", type=int, help="Опорная длина волны, нм")
    parser.add_argument("images_folder", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Папка результатов")
    args = parser.parse_args()
    output, report = align_spectral_images(
        args.centers_csv, args.reference_wavelength, args.images_folder, args.out
    )
    print(f"Сохранено {len(report)} изображений: {output}")
    print(f"Медианная RMS ошибка по маркерам: {report.rms_px.median():.3f} пикселя")
    print(f"Отчёт: {output / 'alignment_diagnostics.csv'}")


if __name__ == "__main__":
    main()
