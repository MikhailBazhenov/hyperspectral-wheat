"""Коррекция гиперспектрального куба: пространственная коррекция серым
полем и спектральная коррекция Spectralon (без коррекции тёмного поля).

pip install numpy pandas opencv-python

Входы: монохромные файлы image<integer>.jpg/.jpeg/.png, CSV с
колонками wavelength и mean_intensity. Все серии должны быть геометрически
совмещены и сняты с одинаковыми параметрами экспозиции/усиления.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd

# Используем правило именования из предыдущего этапа.
from collect_aruco_folder import _FILENAME


def _index_images(folder: str | Path) -> dict[int, Path]:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {folder}")
    result = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = _FILENAME.fullmatch(path.name)
        # Предыдущий модуль поддерживает jpg/png. JPEG тоже допустим здесь.
        if match is None and path.suffix.lower() == '.jpeg':
            import re
            match = re.fullmatch(r'image(\d+)\.jpeg', path.name, re.IGNORECASE)
        if match:
            wl = int(match.group(1))
            if wl in result:
                raise ValueError(f"Несколько файлов для {wl} нм в {folder}")
            result[wl] = path
    if not result:
        raise ValueError(f"Нет image<длина_волны>.jpg/.png в {folder}")
    return dict(sorted(result.items()))


def _read_gray(path: Path, expected_shape=None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Не удалось прочитать изображение: {path}")
    if image.ndim != 2:
        raise ValueError(f"Ожидается монохромное 2D изображение: {path} (форма {image.shape})")
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(f"Размеры не совпадают: {path}, {image.shape} вместо {expected_shape}")
    return image


def correct_flat_field(
    object_folder: str | Path,
    flat_folder: str | Path,
    spectralon_csv: str | Path,
    output_folder: str | Path,
    reference_wavelength: int = 700,
    lower_percentile: float = 5,
    upper_percentile: float = 95,
    min_flat_fraction: float = 0.01,
    save_float_cube: bool = True,
) -> tuple[Path, pd.DataFrame]:
    """Скорректировать набор монохромных спектральных снимков.

    Формула по каждому спектральному каналу:
        C = O * mean(F[valid]) / F * S(reference_wavelength) / S(wavelength)

    Тёмный сигнал не вычитается: если необходима его коррекция, применяйте
    отдельную обработку к исходным кадрам до геометрического совмещения.

    valid: конечные пиксели при F > max(0, min_flat_fraction * median(F>0)),
    O >= 0; нулевые чёрные края плоского поля исключаются из статистики.
    При нормализации P5/P95 сравниваются только valid пиксели исходного O и C
    по ВСЕМ спектральным каналам. Одна линейная шкала для всего куба.

    Сохранение: файлы с исходными именами в output_folder, соответствующей
    исходной разрядности (uint8/uint16), с ограничением диапазона;
    compressed NPZ c float32 (H,W,L), wavelength и valid_mask (если включено),
    CSV с диагностикой. Некорректные пиксели в float-кубе = NaN, в снимках = 0.

    Возвращает (output_folder, diagnostics_dataframe).
    """
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("Процентили должны удовлетворять 0 <= lower < upper <= 100")
    if not (0 <= min_flat_fraction < 1):
        raise ValueError("min_flat_fraction должен быть в интервале [0,1)")

    obj_files = _index_images(object_folder)
    flat_files = _index_images(flat_folder)
    if obj_files.keys() != flat_files.keys():
        raise ValueError(f"Не совпадают длины волн: только объект={sorted(obj_files.keys()-flat_files.keys())}, "
                         f"только фон={sorted(flat_files.keys()-obj_files.keys())}")
    all_sources = [Path(object_folder).resolve(), Path(flat_folder).resolve()]
    out = Path(output_folder).expanduser().resolve()
    if out in all_sources:
        raise ValueError("Папка результата не должна совпадать с исходной папкой")

    spectrum = pd.read_csv(spectralon_csv)
    if not {'wavelength', 'mean_intensity'}.issubset(spectrum.columns):
        raise ValueError("CSV Spectralon требует столбцы wavelength, mean_intensity")
    spectrum['wavelength'] = pd.to_numeric(spectrum['wavelength'], errors='raise')
    spectrum['mean_intensity'] = pd.to_numeric(spectrum['mean_intensity'], errors='raise')
    if spectrum[['wavelength', 'mean_intensity']].isna().any().any():
        raise ValueError('Пропущенные значения в спектре Spectralon')
    if spectrum.wavelength.duplicated().any():
        raise ValueError('Повторяющиеся длины волн в спектре Spectralon')
    spectral = dict(zip(spectrum.wavelength, spectrum.mean_intensity))
    wls = list(obj_files)
    missing = set(wls + [reference_wavelength]) - spectral.keys()
    if missing:
        raise ValueError(f"Нет спектра Spectralon для длин волн: {sorted(missing)}")
    if any(not np.isfinite(spectral[wl]) or spectral[wl] <= 0 for wl in wls + [reference_wavelength]):
        raise ValueError("Значения mean_intensity Spectralon должны быть положительными и конечными")

    # Загрузка полного куба в память по требованию задачи.
    obj_imgs = []
    flat_imgs = []
    shape = None
    dtype = None
    rows = []
    for wl in wls:
        o = _read_gray(obj_files[wl], shape)
        if shape is None:
            shape, dtype = o.shape, o.dtype
        f = _read_gray(flat_files[wl], shape)
        if o.dtype != dtype or f.dtype != dtype:
            raise ValueError(f"Разная разрядность/тип изображений на {wl} нм")
        obj_imgs.append(o)
        flat_imgs.append(f)
    if dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError('Поддерживаются только uint8 и uint16 изображения')

    obj = np.stack(obj_imgs, axis=-1).astype(np.float32)
    flat = np.stack(flat_imgs, axis=-1).astype(np.float32)
    del obj_imgs, flat_imgs

    corrected = np.full(obj.shape, np.nan, dtype=np.float32)
    valid = np.zeros(obj.shape, dtype=bool)
    for idx, wl in enumerate(wls):
        f = flat[:, :, idx]
        o = obj[:, :, idx]
        positive = f[f > 0]
        if positive.size == 0:
            raise ValueError(f"Плоское поле полностью неположительно: {wl} нм")
        cutoff = float(min_flat_fraction * np.median(positive))
        # Строго > 0 даже если min_flat_fraction = 0.
        mask = (f > max(cutoff, 0.0)) & (o >= 0) & np.isfinite(o) & np.isfinite(f)
        if not np.any(mask):
            raise ValueError(f"Нет пригодных пикселей на {wl} нм")
        mean_flat = float(np.mean(f[mask], dtype=np.float64))
        spec_factor = float(spectral[reference_wavelength] / spectral[wl])
        corrected[:, :, idx][mask] = (o[mask] * (mean_flat / f[mask]) * spec_factor)
        valid[:, :, idx] = mask
        rows.append(dict(wavelength=wl, file=obj_files[wl].name, n_valid=int(mask.sum()),
                         valid_fraction=float(mask.mean()), mean_flat=mean_flat,
                         spectral_factor=spec_factor))

    original_values = obj[valid]
    corrected_values = corrected[valid]
    a, b = np.percentile(original_values, [lower_percentile, upper_percentile])
    c, d = np.percentile(corrected_values, [lower_percentile, upper_percentile])
    if not np.isfinite(d-c) or d <= c:
        raise ValueError('Невозможно нормализовать: процентили скорректированного куба совпадают')
    scale = float((b-a)/(d-c))
    offset = float(a - scale*c)
    if b == a:
        warnings.warn('Процентили исходного куба совпадают, результат будет константой')
    del original_values, corrected_values

    out.mkdir(parents=True, exist_ok=True)
    top = np.iinfo(dtype).max
    diag = pd.DataFrame(rows)
    diag['scale'] = scale
    diag['offset'] = offset
    diag['source_p_low'] = a
    diag['source_p_high'] = b
    diag['corrected_p_low'] = c
    diag['corrected_p_high'] = d

    for idx, wl in enumerate(wls):
        result = np.zeros(shape, dtype=dtype)
        layer = corrected[:, :, idx]
        mask = valid[:, :, idx]
        result[mask] = np.rint(np.clip(layer[mask] * scale + offset, 0, top)).astype(dtype)
        dst = out / obj_files[wl].name
        if dst.suffix.lower() in {'.jpg', '.jpeg'}:
            # JPEG может быть только 8-битным в OpenCV.
            if dtype != np.dtype('uint8'):
                raise ValueError('16-битные файлы необходимо сохранять в PNG')
            ok = cv2.imwrite(str(dst), result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            ok = cv2.imwrite(str(dst), result)
        if not ok:
            raise OSError(f'Не удалось сохранить {dst}')
    diag.to_csv(out / 'flat_field_diagnostics.csv', index=False)
    if save_float_cube:
        np.savez_compressed(out / 'corrected_cube_float32.npz',
                            cube=corrected, wavelengths=np.asarray(wls), valid_mask=valid)
    return out, diag
