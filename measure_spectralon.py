"""Measure mean pixel intensity inside a saved Spectralon mask."""
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

_IMAGE_NAME = re.compile(r"image(\d+)\.(jpg|jpeg|png)", re.IGNORECASE)


def measure_spectralon(mask_path, images_folder, output_csv=None):
    """Calculate mean pixel values inside one mask for a folder of images.

    Parameters
    ----------
    mask_path : str | Path
        Path to binary mask image. Nonzero pixels are included.
    images_folder : str | Path
        Folder containing image<wavelength>.jpg/.jpeg/.png files (not recursive).
        Images must already be geometrically aligned to the mask.
    output_csv : str | Path | None
        Optional path for saving the resulting DataFrame as CSV.

    Returns
    -------
    pandas.DataFrame
        Columns: wavelength, file, mean_intensity, std_intensity, n_pixels.
        Color images are converted to grayscale before measuring intensity;
        original image bit depth (uint8 / uint16) is preserved.
    """
    mask_path = Path(mask_path)
    images_folder = Path(images_folder)
    if not images_folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {images_folder}")

    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    if mask_image.ndim == 3:
        raise ValueError("Mask must be a single-channel image")
    mask = mask_image != 0
    n_pixels = int(np.count_nonzero(mask))
    if n_pixels == 0:
        raise ValueError("Mask contains no selected pixels")

    files = []
    for path in images_folder.iterdir():
        if path.is_file():
            match = _IMAGE_NAME.fullmatch(path.name)
            if match:
                files.append((int(match.group(1)), path))
    if not files:
        raise ValueError(f"No image<wavelength>.jpg/.jpeg/.png files in {images_folder}")
    files.sort(key=lambda item: (item[0], item[1].name))
    wavelengths = [w for w, _ in files]
    if len(set(wavelengths)) != len(wavelengths):
        raise ValueError("Multiple files have the same wavelength; remove duplicates")

    rows = []
    for wavelength, path in files:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unsupported channel count in {path}")
        if image.shape != mask.shape:
            raise ValueError(
                f"Size mismatch for {path.name}: image {image.shape}, mask {mask.shape}"
            )
        values = image[mask].astype(np.float64)
        rows.append({
            "wavelength": wavelength,
            "file": path.name,
            "mean_intensity": float(values.mean()),
            "std_intensity": float(values.std()),
            "n_pixels": n_pixels,
        })
    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_path")
    parser.add_argument("images_folder")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    result = measure_spectralon(args.mask_path, args.images_folder, args.output_csv)
    print(result.to_string(index=False))
