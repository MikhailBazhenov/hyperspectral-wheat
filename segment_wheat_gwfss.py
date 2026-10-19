"""Wheat plant segmentation from three hyperspectral frames using GWFSS SegFormer.

Install: pip install torch transformers pillow numpy opencv-contrib-python safetensors
"""
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "GlobalWheat/GWFSS_Segformer_b0"
_IMAGE_RE = re.compile(r"^image(\d+)\.(?:jpg|jpeg|png)$", flags=re.IGNORECASE)


def _load_band(folder: Path, wavelength: int) -> np.ndarray:
    matches = [p for p in folder.iterdir() if p.is_file() and
               (m := _IMAGE_RE.fullmatch(p.name)) and int(m.group(1)) == wavelength]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {wavelength} nm in {folder}, found {len(matches)}")
    # IMREAD_UNCHANGED preserves bit depth in case a band is uint16.
    band = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
    if band is None:
        raise OSError(f"Cannot read: {matches[0]}")
    if band.ndim == 3:
        if band.shape[2] == 4:
            band = cv2.cvtColor(band, cv2.COLOR_BGRA2GRAY)
        else:
            band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return band


def _to_uint8(band: np.ndarray, stretch_percentiles=None) -> np.ndarray:
    """Keep uint8 values unchanged unless optional percentile contrast stretch requested."""
    if stretch_percentiles is None:
        if band.dtype == np.uint8:
            return band
        if band.dtype == np.uint16:
            return np.clip(np.rint(band.astype(np.float32) / 257), 0, 255).astype(np.uint8)
        raise TypeError(f"Unsupported image dtype {band.dtype}. Use 8/16-bit PNG/JPEG.")

    low, high = map(float, stretch_percentiles)
    if not 0 <= low < high <= 100:
        raise ValueError("stretch_percentiles must satisfy 0 <= low < high <= 100")
    vals = band.astype(np.float32)
    lo, hi = np.percentile(vals, [low, high])
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    return np.clip((vals - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def segment_wheat_gwfss(
    images_folder,
    mask_path,
    rgb_wavelengths=(650, 550, 450),
    rgb_path=None,
    overlay_path=None,
    stretch_percentiles=None,
    device=None,
    model_id=MODEL_ID,
):
    """Segment the entire wheat plant (heads + stems + leaves) using GWFSS.

    Parameters
    ----------
    images_folder : str | Path
        Directory containing image<integer>.[jpg|jpeg|png] frames.
    mask_path : str | Path
        Destination for binary PNG mask (0=background, 255=plant).
    rgb_wavelengths : tuple[int, int, int]
        Wavelengths mapped to R, G, B respectively.
    rgb_path, overlay_path : str | Path | None
        Optional RGB and verification overlay files.
    stretch_percentiles : tuple[float, float] | None
        Optional per-channel contrast stretching; e.g. (1, 99).
        None leaves existing 8-bit pixel values intact.
    device : "cpu" | "cuda" | None
        None chooses CUDA when available.
    model_id : str
        Hugging Face model identifier.

    Returns
    -------
    mask : np.ndarray, uint8 (H, W)
        Binary plant mask in ORIGINAL image resolution.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    folder = Path(images_folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Images folder does not exist: {folder}")
    if len(rgb_wavelengths) != 3:
        raise ValueError("rgb_wavelengths must contain exactly R, G and B wavelengths")

    bands = [_load_band(folder, int(w)) for w in rgb_wavelengths]
    shape = bands[0].shape
    if any(b.shape != shape for b in bands):
        raise ValueError("The three wavelength images have different dimensions")
    rgb = np.stack([_to_uint8(b, stretch_percentiles) for b in bands], axis=-1)
    height, width = shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id, use_safetensors=True
    ).to(device).eval()
    inputs = processor(images=Image.fromarray(rgb, mode="RGB"), return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()

    # GWFSS labels: 0 background, 1 head, 2 stem, 3 leaf.
    mask = (np.isin(labels, [1, 2, 3]).astype(np.uint8) * 255)
    out = Path(mask_path)
    if out.suffix.lower() != ".png":
        raise ValueError("mask_path must be a .png file (lossless binary mask)")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mask):
        raise OSError(f"Cannot save mask to {out}")

    if rgb_path is not None:
        dest = Path(rgb_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(dest)
    if overlay_path is not None:
        dest = Path(overlay_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        overlay = rgb.astype(np.float32).copy()
        selected = mask > 0
        overlay[selected] = 0.55 * overlay[selected] + 0.45 * np.array([0, 255, 0])
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(dest)
    return mask
