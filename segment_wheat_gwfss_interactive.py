"""Interactive wheat segmentation: one tiled GWFSS inference, live class thresholds.

Dependencies: pip install numpy opencv-contrib-python pillow torch transformers safetensors
OpenCV GUI must be available (not opencv-*-headless). Works in a local desktop session.

Requires segment_wheat_gwfss.py and segment_wheat_gwfss_tiled.py next to this file.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from segment_wheat_gwfss import _load_band, _to_uint8, MODEL_ID
from segment_wheat_gwfss_tiled import _starts, _tile_weights, CLASS_IDS, ALIASES, CLASS_COLORS_RGB


KEY_UP = {2490368, 65362}
KEY_DOWN = {2621440, 65364}


def _png(path, image):
    if path is None:
        return
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Output must be PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def _canonical(name):
    key = str(name).lower().strip()
    return ALIASES.get(key, key)


def _make_labels(probabilities, enabled, thresholds):
    """Argmax first, then threshold predicted class; reject as background."""
    labels = probabilities.argmax(axis=0).astype(np.uint8)
    for name, class_id in CLASS_IDS.items():
        bad = (labels == class_id) & (
            (not enabled[name]) | (probabilities[class_id] < thresholds[name])
        )
        labels[bad] = 0
    return labels


def _infer_tiled(rgb, model_id, tile_size, overlap, device):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    if model.config.num_labels < 4:
        raise ValueError('Model must output at least four GWFSS classes')
    h, w = rgb.shape[:2]
    probs_sum = np.zeros((model.config.num_labels, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    xs, ys = _starts(w, tile_size, overlap), _starts(h, tile_size, overlap)
    total = len(xs) * len(ys)
    n = 0
    for y in ys:
        for x in xs:
            tile = rgb[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            th, tw = tile.shape[:2]
            inputs = processor(images=Image.fromarray(tile), return_tensors='pt')
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = F.interpolate(logits, size=(th, tw), mode='bilinear', align_corners=False)
                p = torch.softmax(logits, dim=1)[0].cpu().numpy()
            weights = _tile_weights(th, tw, min(overlap // 2, 64))
            probs_sum[:, y:y + th, x:x + tw] += p * weights
            weight_sum[y:y + th, x:x + tw] += weights
            n += 1
            print(f'\rGWFSS tiles: {n}/{total}', end='', flush=True)
    print()
    return probs_sum / weight_sum[None, :, :]


def segment_wheat_gwfss_interactive(
    images_folder,
    mask_path,
    rgb_wavelengths=(650, 550, 450),
    tile_size=512,
    overlap=128,
    classes=('leaf', 'stem', 'head'),
    class_thresholds=None,
    threshold_step=0.05,
    rgb_path=None,
    overlay_path=None,
    class_map_path=None,
    probabilities_path=None,
    settings_path=None,
    stretch_percentiles=None,
    device=None,
    model_id=MODEL_ID,
    max_display_width=1100,
    max_display_height=850,
):
    """Interactively adjust per-class confidence thresholds; save PNG binary mask on Enter.

    1/2/3 selects leaf/stem/head, up/down changes its threshold,
    Space enables/disables selected class, +/- changes threshold step,
    P toggles overlay and selected-class probability heatmap,
    Enter saves (and returns uint8 mask), Esc or Q cancels (returns None).
    No model re-run during interaction. Predictions can be cached via probabilities_path.

    The cache is reused ONLY if image pixel bytes, chosen bands, contrast stretch,
    model ID, tile size and overlap match. Cache uses a compressed float32 NPZ.
    Output image size equals original source image size.
    """
    tile_size, overlap = int(tile_size), int(overlap)
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError('tile_size must be >=32 and 0 <= overlap < tile_size')
    if len(rgb_wavelengths) != 3:
        raise ValueError('rgb_wavelengths must contain R,G,B wavelengths')
    if not 0 < threshold_step <= 1:
        raise ValueError('threshold_step must be in (0,1]')
    if max_display_width < 250 or max_display_height < 200:
        raise ValueError('Display dimensions are too small')

    enabled = {c: False for c in CLASS_IDS}
    if isinstance(classes, str):
        classes = (classes,)
    for name in classes:
        c = _canonical(name)
        if c not in CLASS_IDS:
            raise ValueError(f'Unknown class: {name!r}')
        enabled[c] = True
    thresholds = {c: 0.5 for c in CLASS_IDS}
    for name, value in (class_thresholds or {}).items():
        c = _canonical(name)
        if c not in CLASS_IDS:
            raise ValueError(f'Unknown threshold class: {name!r}')
        if not 0 <= float(value) <= 1:
            raise ValueError(f'Invalid threshold for {name!r}: {value}')
        thresholds[c] = float(value)

    folder = Path(images_folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    bands = [_load_band(folder, int(w)) for w in rgb_wavelengths]
    if any(b.shape != bands[0].shape for b in bands):
        raise ValueError('Input bands must have identical dimensions')
    rgb = np.stack([_to_uint8(b, stretch_percentiles) for b in bands], axis=-1)
    h, w = rgb.shape[:2]
    meta = {
        'image_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'rgb_wavelengths': [int(v) for v in rgb_wavelengths],
        'stretch_percentiles': None if stretch_percentiles is None else list(stretch_percentiles),
        'tile_size': tile_size, 'overlap': overlap, 'model_id': model_id,
        'shape': list(rgb.shape), 'cache_version': 1,
    }

    probabilities = None
    if probabilities_path is not None and Path(probabilities_path).exists():
        try:
            with np.load(probabilities_path, allow_pickle=False) as cached:
                old_meta = json.loads(str(cached['metadata'].item()))
                if old_meta == meta:
                    saved = cached['probabilities']
                    if saved.ndim == 3 and saved.shape[1:] == (h, w) and saved.shape[0] >= 4:
                        probabilities = saved.astype(np.float32)
                        print(f'Loaded cached probabilities: {probabilities_path}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'Ignoring unreadable probabilities cache: {exc}')
    if probabilities is None:
        probabilities = _infer_tiled(rgb, model_id, tile_size, overlap, device)
        if probabilities_path is not None:
            p = Path(probabilities_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, probabilities=probabilities.astype(np.float32),
                                metadata=np.array(json.dumps(meta, ensure_ascii=False)))
            print(f'Saved probabilities: {p}')

    order = ['leaf', 'stem', 'head']
    selected = 'leaf'
    view_prob = False
    step = float(threshold_step)
    window = 'GWFSS - interactive wheat segmentation'
    scale = min(1., max_display_width / w, (max_display_height - 145) / h)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    rgb_small = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    rgb_small_bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    cmap = cv2.COLORMAP_TURBO

    def render():
        labels = _make_labels(probabilities, enabled, thresholds)
        if view_prob:
            heat = np.clip(probabilities[CLASS_IDS[selected]] * 255., 0, 255).astype(np.uint8)
            heat = cv2.resize(heat, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            shown = cv2.applyColorMap(heat, cmap)
        else:
            small_labels = cv2.resize(labels, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            shown = rgb_small_bgr.copy()
            for name, class_id in CLASS_IDS.items():
                loc = small_labels == class_id
                if not np.any(loc):
                    continue
                color_bgr = np.array(CLASS_COLORS_RGB[class_id][::-1], dtype=np.float32)
                shown[loc] = np.clip(0.55 * shown[loc].astype(np.float32) + 0.45 * color_bgr, 0, 255).astype(np.uint8)
        panel = np.full((145, disp_w, 3), 26, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        s = 0.52
        def line(txt, y, color=(235, 235, 235)):
            cv2.putText(panel, txt, (9, y), font, s, color, 1, cv2.LINE_AA)
        line(f'Selected: {selected.upper()}  |  step: {step:.3f}  |  view: {"probability" if view_prob else "overlay"}', 20)
        for i, name in enumerate(order):
            color = (120, 240, 255) if name == selected else (220, 220, 220)
            line(f'{i+1} {name:<5}  {"ON " if enabled[name] else "OFF"}  threshold={thresholds[name]:.3f}', 44 + 22*i, color)
        line('UP/DOWN threshold | 1/2/3 class | SPACE toggle | +/- step', 115)
        line('P probability view | ENTER save | ESC/Q cancel', 136)
        return np.vstack((shown, panel)), labels

    # Full-resolution labels are computed from cached probabilities, without neural inference.
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while True:
            frame, labels = render()
            cv2.imshow(window, frame)
            key = cv2.waitKeyEx(0)
            if key < 0:
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        return None
                except cv2.error:
                    return None
                continue
            low = key & 0xFF
            if key in (13, 10) or low == 13:
                mask = (labels > 0).astype(np.uint8) * 255
                _png(mask_path, mask)
                if rgb_path is not None:
                    _png(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if class_map_path is not None:
                    out_color = np.zeros((h, w, 3), dtype=np.uint8)
                    for class_id, color in CLASS_COLORS_RGB.items():
                        out_color[labels == class_id] = color[::-1]
                    _png(class_map_path, out_color)
                if overlay_path is not None:
                    output = rgb.astype(np.float32)
                    for c, color in CLASS_COLORS_RGB.items():
                        if c == 0:
                            continue
                        idx = labels == c
                        output[idx] = 0.55 * output[idx] + 0.45 * np.array(color)
                    _png(overlay_path, cv2.cvtColor(np.uint8(np.clip(output, 0, 255)), cv2.COLOR_RGB2BGR))
                if settings_path is not None:
                    path = Path(settings_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'enabled_classes': [n for n in order if enabled[n]],
                                                'class_thresholds': thresholds,
                                                'rgb_wavelengths': meta['rgb_wavelengths'],
                                                'tile_size': tile_size, 'overlap': overlap,
                                                'model_id': model_id}, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
                return mask
            if low in (27, ord('q'), ord('Q')):
                return None
            if low in (ord('1'), ord('2'), ord('3')):
                selected = order[low - ord('1')]
            elif low == 32:
                enabled[selected] = not enabled[selected]
            elif key in KEY_UP:
                thresholds[selected] = min(1., round(thresholds[selected] + step, 4))
            elif key in KEY_DOWN:
                thresholds[selected] = max(0., round(thresholds[selected] - step, 4))
            elif low in (ord('+'), ord('=')):
                step = min(1., round(step * 2, 4))
            elif low in (ord('-'), ord('_')):
                step = max(0.001, round(step / 2, 4))
            elif low in (ord('p'), ord('P')):
                view_prob = not view_prob
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
