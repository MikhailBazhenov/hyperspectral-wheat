"""Create study-wide contact sheets: one PNG per vegetation index / image kind.

Sample discovery follows process_wheat_study.py: folders containing
corrected_images, excluding service parents and any path with '#'.
Output collage PNGs and a report CSV live directly in main_folder.
"""
from pathlib import Path
from math import ceil
import textwrap
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _samples(main, corrected_folder_name, markers, exclude_marker):
    samples = []
    for folder in main.rglob(corrected_folder_name):
        if not folder.is_dir() or folder.name != corrected_folder_name:
            continue
        relative = folder.relative_to(main)
        if any(exclude_marker.casefold() in part.casefold() for part in relative.parts):
            continue
        parent = folder.parent
        if any(marker.casefold() in parent.name.casefold() for marker in markers):
            continue
        samples.append((parent.relative_to(main).as_posix(), parent))
    return sorted(set(samples), key=lambda pair: pair[0].casefold())


def create_study_collages(
    main_folder,
    corrected_folder_name="corrected_images",
    index_maps_folder="index_maps",
    image_files=None,
    indices=None,
    n_columns=4,
    tile_width=360,
    tile_height=310,
    output_prefix="collage_",
    calibration_marker="calibration_target",
    spectralon_marker="spectralon",
    flat_marker="flat",
    exclude_marker="#",
    missing_color="#ededed",
    show=False,
    overwrite=True,
):
    """Build per-index and per-image-type collages for all sample folders.

    If overwrite=False and a collage PNG already exists, it is kept as is and
    not regenerated.
    """
    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(f"Главная папка не существует: {main}")
    if n_columns < 1 or tile_width < 100 or tile_height < 100:
        raise ValueError("n_columns >= 1, tile_width и tile_height >= 100")
    markers = (calibration_marker, spectralon_marker, flat_marker)
    if not all(isinstance(v, str) and v for v in (*markers, exclude_marker)):
        raise ValueError("Метки служебных папок должны быть непустыми строками")

    if image_files is None:
        image_files = {
            "wheat_mask": "wheat_mask.png",
            "wheat_rgb": "wheat_rgb.png",
            "wheat_overlay": "wheat_overlay.png",
            "wheat_classes": "wheat_classes.png",
        }
    samples = _samples(main, corrected_folder_name, markers, exclude_marker)
    if not samples:
        raise ValueError("Не найдены образцы с папкой corrected_images")

    if indices is None:
        found = set()
        for _, folder in samples:
            index_dir = folder / index_maps_folder
            if index_dir.is_dir():
                found.update(p.stem for p in index_dir.glob("*.png") if p.is_file())
        indices = sorted(found, key=str.casefold)
    else:
        indices = list(dict.fromkeys(indices))

    kinds = [("index_" + index, lambda folder, index=index: folder / index_maps_folder / (index + ".png"))
             for index in indices]
    kinds.extend((key, lambda folder, filename=filename: folder / filename)
                 for key, filename in image_files.items())

    title_font = _font(22)
    sample_font = _font(17)
    info_font = _font(15)
    margin = 15
    header = 65
    label_height = 62
    usable_image_h = tile_height - label_height - 16
    rows = []

    for kind, get_path in kinds:
        count = len(samples)
        safe_kind = "".join(c if c.isalnum() or c in "_-" else "_" for c in kind)
        output_path = main / f"{output_prefix}{safe_kind}.png"

        for sample_name, folder in samples:
            src = get_path(folder)
            status, error = ("ok", "") if src.is_file() else ("missing", "Нет файла")
            rows.append(dict(collage=kind, sample=sample_name, source=str(src),
                             status=status, error=error, output=str(output_path)))

        if output_path.exists() and not overwrite:
            print(f"{output_path.name}: reuse existing")
            continue

        nrows = ceil(count / n_columns)
        sheet = Image.new("RGB", (n_columns * tile_width, header + nrows * tile_height), "#ffffff")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 14), f"{kind}  |  {count} образцов", fill="#202020", font=title_font)
        for i, (sample_name, folder) in enumerate(samples):
            col, row = i % n_columns, i // n_columns
            x, y = col * tile_width, header + row * tile_height
            src = get_path(folder)
            draw.rectangle((x + 3, y + 3, x + tile_width - 4, y + tile_height - 4), outline="#cecece", width=1)
            status, error = "ok", ""
            if src.is_file():
                try:
                    with Image.open(src) as img:
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        img.thumbnail((tile_width - 2 * margin, usable_image_h), Image.Resampling.LANCZOS)
                        px = x + (tile_width - img.width) // 2
                        py = y + 8 + (usable_image_h - img.height) // 2
                        sheet.paste(img, (px, py))
                except Exception as exc:
                    status, error = "error", f"{type(exc).__name__}: {exc}"
            else:
                status, error = "missing", "Нет файла"
            if status != "ok":
                draw.rectangle((x + margin, y + 8, x + tile_width - margin,
                                y + usable_image_h + 8), fill=missing_color)
                draw.text((x + margin + 10, y + usable_image_h // 2),
                          "Нет изображения" if status == "missing" else "Ошибка чтения",
                          fill="#9b3737", font=info_font)
            lines = textwrap.wrap(sample_name, width=max(16, (tile_width - 20) // 9),
                                  break_long_words=True, max_lines=2, placeholder="...")
            for j, line in enumerate(lines):
                draw.text((x + margin, y + usable_image_h + 16 + j * 20), line,
                          fill="#202020", font=sample_font)
        sheet.save(output_path, format="PNG", optimize=True)
        print(f"{output_path.name}: {sum(r['status']=='ok' for r in rows if r['collage']==kind)}/{count}")
        if show:
            from IPython.display import display
            display(sheet)
        sheet.close()

    report = pd.DataFrame(rows)
    report.to_csv(main / "collage_report.csv", index=False)
    return report
