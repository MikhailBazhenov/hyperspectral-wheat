"""Optional Stage 4: collages sorted by genotype x environment (without moving data).

Names are matched case-insensitively by code substring in each sample FOLDER name.
Original image/mask folders and scientific results remain untouched.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from simple_xlsx import write_dataframe_xlsx
from hyperspectral_object_analysis import find_samples


def _font(size):
    for font_name in ('DejaVuSans.ttf', 'arial.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    # Pillow can use DejaVu Sans shipped by matplotlib in constrained installs.
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except Exception:
        return ImageFont.load_default()


def _validate_codes(codes, displays, name):
    codes = list(codes)
    displays = list(displays)
    if not codes or len(codes) != len(displays):
        raise ValueError(f'{name}: codes and display labels must be nonempty and same length')
    if any(not isinstance(v, str) or not v.strip() for v in codes + displays):
        raise ValueError(f'{name}: all codes and display labels must be non-empty strings')
    if len({c.casefold() for c in codes}) != len(codes):
        raise ValueError(f'{name}: codes must be unique (case-insensitively)')
    return codes, displays


def _match_code(sample_name, codes):
    """Exactly one case-insensitive substring match or return explicit status."""
    matches = [i for i, code in enumerate(codes) if code.casefold() in sample_name.casefold()]
    if len(matches) == 1:
        return matches[0], 'ok'
    if not matches:
        return None, 'unmatched'
    return None, 'ambiguous'


def _safe_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or 'index'


def _match_date_code(sample_name, session_name, date_codes):
    """Match the sample date first, falling back to the acquisition session.

    A session can contain multiple dates: an explicit date in the sample name
    always takes precedence. Missing or ambiguous matches are not guessed.
    """
    index, status = _match_code(str(sample_name), date_codes)
    if status == 'ok' or status == 'ambiguous':
        return index, status
    if session_name:
        return _match_code(str(session_name), date_codes)
    return None, 'unmatched'

def _safe_path_label(name):
    name = str(name).strip()
    name = re.sub(r'[\/:*?"<>|]+', '_', name)
    name = re.sub(r'\s+', ' ', name).strip(' .')
    return name or 'item'


def _variant_folder_name(row):
    return f"{_safe_path_label(row['genotype_display'])}__{_safe_path_label(row['environment_display'])}"


def _collected_map_destination(base_folder, row):
    # Session-qualified, collision-resistant paths. Identical sample and mask
    # folder names are common in separate acquisition sessions.
    import hashlib
    relative_sample = str(row['sample'])
    digest = hashlib.sha1(relative_sample.encode('utf-8')).hexdigest()[:10]
    label = _safe_path_label(relative_sample.replace('/', '__'))
    filename = f"{label}__{_safe_path_label(row['mask'])}__{digest}.png"
    return base_folder / _safe_name(row['index']) / _variant_folder_name(row) / filename


def _collect_source_maps(dest_root, items, date_index, date_code, overwrite=True):
    base = dest_root / _safe_name(date_code) / 'maps_by_index'
    if overwrite and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    collected = []
    for row, gi, ei, di, source in items:
        if di != date_index or gi is None or ei is None or not source.is_file() or row['status'] != 'ok':
            continue
        target = _collected_map_destination(base, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            shutil.copy2(source, target)
        collected.append((str(source), str(target)))
    return collected


def create_experiment_collages(
    main_folder,
    *,
    genotypes_codes,
    genotypes_display,
    environment_codes,
    environment_display,
    indices=None,
    dates_codes=None, dates_display=None,
    tile_width=240,
    tile_height=230,
    per_cell_columns=2,
    overwrite=True,
    genotypes_as_columns=False,
    sample_folders=None,
    output_folder=None,
):
    """Create date-wise sorted collages, optionally transposing the layout.

    Default: genotype rows, environment columns. With genotypes_as_columns=True:
    environment rows, genotype columns. Replicate tiles stay within their cell.
    Existing masks, source maps and separately collected map folders are untouched.
    Unmatched/ambiguous sample names are recorded, not assigned automatically.
    """
    main = Path(main_folder).expanduser().resolve()
    if not main.is_dir():
        raise NotADirectoryError(main)
    gc, gd = _validate_codes(genotypes_codes, genotypes_display, 'genotypes')
    ec, ed = _validate_codes(environment_codes, environment_display, 'environment')
    if dates_codes is None or dates_display is None:
        raise ValueError('Stage 4 requires dates_codes and dates_display')
    dc, dd = _validate_codes(dates_codes, dates_display, 'dates')
    if not isinstance(genotypes_as_columns, bool):
        raise TypeError('genotypes_as_columns must be True or False')
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
               for x in (tile_width, tile_height, per_cell_columns)):
        raise ValueError('tile_width, tile_height and per_cell_columns must be positive integers')
    # Legacy single-session calls still discover samples locally. Project-wide
    # Stage 4 passes the explicit list of sample folders and a Results folder.
    samples = find_samples(main) if sample_folders is None else sorted(
        (Path(p).expanduser().resolve() for p in sample_folders), key=lambda p: str(p).casefold())
    if any(not p.is_relative_to(main) for p in samples):
        raise ValueError('Every sample folder must be inside the project root')
    if not samples:
        raise FileNotFoundError('No corrected cubes / sample folders found')
    if indices is None:
        names = sorted({p.stem for sample in samples for p in
                        (sample / 'object_analysis' / 'index_maps').glob('*/*.png')},
                       key=str.casefold)
    else:
        names = list(dict.fromkeys(indices))
    if not names:
        raise ValueError('No index maps found; execute Stage 3 first')
    report_root = (main / 'object_analysis') if output_folder is None else Path(output_folder).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    dest = report_root / 'by_date'
    dest.mkdir(parents=True, exist_ok=True)
    for date_code in dc:
        folder=dest/_safe_name(date_code)/'collages'
        folder.mkdir(parents=True, exist_ok=True)
        expected={f'collage_{_safe_name(name)}.png' for name in names}
        for stale in folder.glob('collage_*.png'):
            if stale.name not in expected:stale.unlink()
    row_font = _font(21)
    cell_font = _font(16)
    caption_font = _font(14)
    misc_font = _font(13)
    margin = 10
    title_height = 46
    header_height = 50
    row_displays = ed if genotypes_as_columns else gd
    column_displays = gd if genotypes_as_columns else ed
    # Long experiment-condition names are common; adapt the row header width.
    row_label_width = max(180, int(max(row_font.getlength(s) for s in row_displays)) + 20)
    cell_width = per_cell_columns * tile_width + 12
    table_width = row_label_width + len(column_displays) * cell_width
    tile_text_height = 38
    items = []
    skipped = []
    for sample in samples:
        rel = sample.relative_to(main).as_posix()
        gi, gs = _match_code(sample.name, gc)
        ei, es = _match_code(sample.name, ec)
        session_name = sample.relative_to(main).parts[0] if sample_folders is not None else ''
        di, ds = _match_date_code(sample.name, session_name, dc)
        status = 'ok' if gs == es == ds == 'ok' else f'genotype_{gs};environment_{es};date_{ds}'
        maps_dir = sample / 'object_analysis' / 'index_maps'
        masks = sorted((p for p in maps_dir.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()) \
                if maps_dir.is_dir() else []
        if not masks:
            masks = [maps_dir / '__no_mask__']
        for mask_folder in masks:
            for name in names:
                source = mask_folder / f'{name}.png'
                row = dict(sample=rel, session=session_name, sample_folder=sample.name, mask=mask_folder.name,
                           genotype_code=gc[gi] if gi is not None else '',
                           genotype_display=gd[gi] if gi is not None else '',
                           environment_code=ec[ei] if ei is not None else '',
                           environment_display=ed[ei] if ei is not None else '',
                           date_code=dc[di] if di is not None else '',
                           date_display=dd[di] if di is not None else '',
                           index=name, image_path=str(source),
                           status=status if status != 'ok' else ('ok' if source.is_file() else 'missing_map'),
                           collage_path=str(dest/_safe_name(dc[di])/'collages'/f'collage_{_safe_name(name)}.png') if di is not None else '',
                           collected_map_path='')
                if di is not None and gi is not None and ei is not None:
                    row['collected_map_path'] = str(_collected_map_destination(dest/_safe_name(dc[di])/'maps_by_index', row))
                items.append((row, gi, ei, di, source))
                if status != 'ok':
                    skipped.append((rel, status))
    report = pd.DataFrame([row for row, *_ in items])
    write_dataframe_xlsx(report, report_root / 'experiment_grouping.xlsx')
    previous_settings_path = report_root / 'experiment_grouping_config.json'
    try:
        previous_settings = json.loads(previous_settings_path.read_text(encoding='utf-8')) if previous_settings_path.is_file() else {}
    except (ValueError, OSError):
        previous_settings = {}
    orientation_changed = bool(previous_settings.get('genotypes_as_columns', False)) != genotypes_as_columns
    settings = dict(genotypes_codes=gc, genotypes_display=gd,
                    environment_codes=ec, environment_display=ed,
                    dates_codes=dc, dates_display=dd,
                    indices=names, tile_width=tile_width, tile_height=tile_height,
                    per_cell_columns=per_cell_columns,
                    genotypes_as_columns=genotypes_as_columns,
                    layout_orientation=('genotypes_columns_environments_rows' if genotypes_as_columns
                                        else 'genotypes_rows_environments_columns'),
                    match_rule='case-insensitive substring in sample folder name',
                    layout=('environment rows, genotype columns' if genotypes_as_columns
                            else 'genotype rows, environment columns') +
                           ', all sample/mask maps in each cell',
                    sorted_maps_output='by_date/<date>/maps_by_index/<index>/<genotype__environment>/<session>__<sample>__<mask>.png',
                    output_directory=str(report_root),
                    original_folders_unchanged=True)
    (report_root / 'experiment_grouping_config.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')

    def cell_indices(row_number, column_number):
        """Always return (genotype, environment), independent of collage orientation."""
        if genotypes_as_columns:
            return column_number, row_number
        return row_number, column_number

    for date_index, date_code in enumerate(dc):
        collected = _collect_source_maps(dest, items, date_index, date_code, overwrite=overwrite)
        if collected:
            print(f'Sorted source maps: {len(collected)} copied for date {dd[date_index]}')
        for name in names:
            path = dest / _safe_name(date_code) / 'collages' / f'collage_{_safe_name(name)}.png'
            if path.exists() and not overwrite and not orientation_changed:
                print(f'Sorted collage retained: {path.name}')
                continue
            groups = {(g, e): [] for g in range(len(gc)) for e in range(len(ec))}
            for row, gi, ei, di, source in items:
                if row['index'] == name and gi is not None and ei is not None and di == date_index and source.is_file():
                    groups[(gi, ei)].append((row, source))
            heights = []
            for ri in range(len(row_displays)):
                max_repeats = max((len(groups[cell_indices(ri, ci)])
                                   for ci in range(len(column_displays))), default=0)
                heights.append(max(1, math.ceil(max_repeats / per_cell_columns)) * tile_height + 15)
            height = title_height + header_height + sum(heights)
            if table_width * height > 150_000_000:
                raise ValueError('Collage is too large; reduce tile_width/tile_height')
            canvas = Image.new('RGB', (table_width, height), '#ffffff')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 9), f'{name}  |  {dd[date_index]}', font=row_font, fill='#202020')
            for ci, display in enumerate(column_displays):
                x = row_label_width + ci * cell_width
                draw.rectangle((x, title_height, x + cell_width - 1, title_height + header_height - 1),
                               outline='#aaaaaa')
                draw.text((x + 8, title_height + 12), display, font=cell_font, fill='#202020')
            top = title_height + header_height
            for ri, display in enumerate(row_displays):
                draw.rectangle((0, top, row_label_width - 1, top + heights[ri] - 1), outline='#aaaaaa')
                draw.text((10, top + 14), display, font=row_font, fill='#202020')
                code = ec[ri] if genotypes_as_columns else gc[ri]
                draw.text((10, top + 45), f'[{code}]', font=misc_font, fill='#555555')
                for ci in range(len(column_displays)):
                    gi, ei = cell_indices(ri, ci)
                    left = row_label_width + ci * cell_width
                    draw.rectangle((left, top, left + cell_width - 1, top + heights[ri] - 1),
                                   outline='#cccccc')
                    group = groups[(gi, ei)]
                    if not group:
                        draw.text((left + 18, top + 22), 'No index maps', font=misc_font, fill='#888888')
                    for position, (row, image_path) in enumerate(group):
                        col = position % per_cell_columns
                        r = position // per_cell_columns
                        x = left + 6 + col * tile_width
                        y = top + 6 + r * tile_height
                        with Image.open(image_path) as im:
                            img = im.convert('RGB')
                            img.thumbnail((tile_width - 18, tile_height - tile_text_height - 12),
                                          Image.Resampling.LANCZOS)
                        canvas.paste(img, (x + (tile_width - img.width)//2,
                                           y + (tile_height - tile_text_height - img.height)//2))
                        def fit_label(value):
                            label = str(value)
                            while label and draw.textlength(label, font=caption_font) > tile_width - 16:
                                label = label[:-2] + '…' if len(label) > 2 else ''
                            return label
                        draw.text((x + 6, y + tile_height - tile_text_height),
                                  fit_label(f"{row['genotype_display']} / {row['environment_display']}"),
                                  font=caption_font, fill='#222222')
                        draw.text((x + 6, y + tile_height - tile_text_height + 18),
                                  fit_label(f"{row['sample_folder']} / {row['mask']}"),
                                  font=caption_font, fill='#555555')
                top += heights[ri]
            canvas.save(path, format='PNG', optimize=True)
            canvas.close()
            print(f'Sorted collage: {path.name}')
    if skipped:
        unique = sorted(set(skipped))
        print(f'WARNING: {len(unique)} sample folder(s) cannot be classified by given codes:')
        for sample, reason in unique:
            print(f'  {sample}: {reason}')
        print('See experiment_grouping.xlsx; unclassified samples are omitted from PNG collages.')
    return report
