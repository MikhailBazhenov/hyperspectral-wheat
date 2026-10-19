"""Tiled wheat segmentation with GWFSS SegFormer-B0.

Install: pip install torch torchvision transformers pillow numpy opencv-contrib-python safetensors
Reuses RGB-band loading/conversion from segment_wheat_gwfss.py.
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from segment_wheat_gwfss import _load_band, _to_uint8, MODEL_ID

# Dataset class indices: background=0, head=1, stem=2, leaf=3.
CLASS_IDS = {"head": 1, "stem": 2, "leaf": 3}
ALIASES = {
    "heads": "head", "spike": "head", "spikes": "head", "ear": "head", "ears": "head",
    "stems": "stem", "leaves": "leaf", "колосья": "head", "колос": "head",
    "стебли": "stem", "стебель": "stem", "листья": "leaf", "лист": "leaf"
}

# Distinct RGB colors for visualization.
# background = dark gray, head = yellow/orange, stem = cyan/blue, leaf = bright green
CLASS_COLORS_RGB = {
    0: (30, 30, 30),      # background
    1: (255, 200, 0),     # head
    2: (0, 180, 255),     # stem
    3: (0, 220, 0),       # leaf
}


def _starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_weights(h, w, feather):
    """Positive centre-biased weights; zero division impossible at image edges."""
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)

    def axis(n):
        k = min(feather, n // 2)
        a = np.ones(n, dtype=np.float32)
        if k > 0:
            # Ramp from 0.05 to 1 to avoid zero weight on image boundaries.
            ramp = 0.05 + 0.95 * (np.arange(k, dtype=np.float32) + 1) / (k + 1)
            a[:k] = ramp
            a[-k:] = ramp[::-1]
        return a

    return np.outer(axis(h), axis(w)).astype(np.float32)


def _labels_to_color(labels):
    """Convert class-index map HxW (uint8) to RGB color visualization HxWx3."""
    color = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS_RGB.items():
        color[labels == class_id] = rgb
    return color


def segment_wheat_gwfss_tiled(
    images_folder,
    mask_path,
    rgb_wavelengths=(650, 550, 450),
    classes=("head", "stem", "leaf"),
    tile_size=512,
    overlap=128,
    rgb_path=None,
    overlay_path=None,
    class_map_path=None,
    class_index_map_path=None,
    class_thresholds=None,
    stretch_percentiles=None,
    device=None,
    model_id=MODEL_ID,
):
    """Segment wheat using overlapping tiles; return full-resolution uint8 binary mask.

    SegFormer class logits are converted to probabilities for each tile, resized
    to that tile's original size, blended using smooth positive window weights,
    and only then argmax is applied. `classes` chooses which predicted labels
    count as foreground; it does not affect the network inference itself.

    Parameters
    ----------
    images_folder: path containing image<integer>.jpg/.jpeg/.png
    mask_path: output PNG binary mask path (0 background, 255 selected classes)
    rgb_wavelengths: (R, G, B) wavelength integers
    classes: any subset of ('leaf', 'stem', 'head')
    tile_size: tile edge length in ORIGINAL pixels (default 512)
    overlap: overlap in ORIGINAL pixels (default 128; 0 <= overlap < tile_size)
    rgb_path, overlay_path: optional PNG paths
    class_map_path: optional PNG path for COLOR visualization of classes
    class_index_map_path: optional PNG path for raw uint8 class indices 0..3
    class_thresholds: optional dict such as {"leaf": 0.75, "stem": 0.55, "head": 0.80}.
      Missing classes have no minimum confidence; accepted values are 0..1.
      Applied to the argmax class after probability blending.
    stretch_percentiles: optional per-channel (low, high) percentile stretch
    device: 'cpu', 'cuda', or None (auto)
    model_id: Hugging Face model identifier
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    tile_size, overlap = int(tile_size), int(overlap)
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size >= 32 and 0 <= overlap < tile_size")
    if len(rgb_wavelengths) != 3:
        raise ValueError("rgb_wavelengths must contain three numbers")
    if isinstance(classes, str):
        classes = [classes]
    canonical = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in classes]
    if not canonical or any(c not in CLASS_IDS for c in canonical):
        raise ValueError("classes must be a non-empty subset of head, stem, leaf")
    selected_ids = [CLASS_IDS[c] for c in canonical]

    if class_thresholds is None:
        class_thresholds = {}
    if not isinstance(class_thresholds, dict):
        raise TypeError("class_thresholds must be a dict or None")
    thresholds_by_id = {}
    for name, value in class_thresholds.items():
        key = str(name).strip().lower()
        key = ALIASES.get(key, key)
        if key not in CLASS_IDS:
            raise ValueError(f"Unknown class in class_thresholds: {name!r}")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {name!r} must be between 0 and 1")
        thresholds_by_id[CLASS_IDS[key]] = threshold

    folder = Path(images_folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"No folder: {folder}")
    bands = [_load_band(folder, int(w)) for w in rgb_wavelengths]
    if any(b.shape != bands[0].shape for b in bands):
        raise ValueError("RGB bands must have the same image size")
    rgb = np.stack([_to_uint8(b, stretch_percentiles) for b in bands], axis=-1)
    h, w = rgb.shape[:2]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    n_classes = model.config.num_labels
    if max(CLASS_IDS.values()) >= n_classes:
        raise RuntimeError(f"Model exposes {n_classes} classes, expected at least 4")

    # Store soft probabilities across all classes for correct blending at tile intersections.
    blended = np.zeros((n_classes, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    xs = _starts(w, tile_size, overlap)
    ys = _starts(h, tile_size, overlap)
    count = len(xs) * len(ys)
    n = 0
    for y in ys:
        for x in xs:
            tile = rgb[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            th, tw = tile.shape[:2]
            inputs = processor(images=Image.fromarray(tile), return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            weights = _tile_weights(th, tw, min(overlap // 2, 64))
            blended[:, y:y + th, x:x + tw] += probabilities * weights[None, :, :]
            weight_sum[y:y + th, x:x + tw] += weights
            n += 1
            print(f"\rTiles: {n}/{count}", end="", flush=True)
    print()
    probabilities = blended / weight_sum[None, :, :]
    labels = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0)
    # Apply threshold to the winning class only. Low-confidence pixels become background.
    for class_id, threshold in thresholds_by_id.items():
        reject = (labels == class_id) & (confidence < threshold)
        labels[reject] = 0
    mask = np.isin(labels, selected_ids).astype(np.uint8) * 255
    class_map_color = _labels_to_color(labels)

    def save_image(path, image):
        if path is None:
            return
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError(f"Use PNG output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Cannot save: {path}")

    save_image(mask_path, mask)
    if rgb_path is not None:
        save_image(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if class_map_path is not None:
        save_image(class_map_path, cv2.cvtColor(class_map_color, cv2.COLOR_RGB2BGR))
    if class_index_map_path is not None:
        save_image(class_index_map_path, labels)
    if overlay_path is not None:
        overlay = rgb.astype(np.float32)
        overlay[mask > 0] = 0.55 * overlay[mask > 0] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
        save_image(overlay_path, cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return mask
