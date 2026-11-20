"""Collect existing RGB mask previews and object masks from all research sessions.

Safe by default: original files are never changed and existing copies are not
replaced unless ``overwrite=True``. Copies are byte-for-byte identical.

Expected project layout (supports nested shot folders as well)::

    Project/
      Session_A/
        camera_shot_01/
          Spectral_Cube/
          Images/mask_preview.png
          masks/object_01.png
          wheat_mask.png                   # optional legacy mask
      Session_B/...

Examples
--------
Notebook::
    from collect_mask_editing_files import collect_mask_editing_files
    report = collect_mask_editing_files(PROJECT_ROOT)

Terminal::
    python collect_mask_editing_files.py "D:/WheatProject"
    python collect_mask_editing_files.py "D:/WheatProject" --overwrite

All collected files are placed in Project/Mask_Editing_Collection/ with names
containing the session, shot and type. A UTF-8 CSV manifest records original
paths so manually corrected copies can later be traced to their sources.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
from typing import Iterable

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'})
MANIFEST_NAME = 'mask_editing_manifest.csv'
DEFAULT_OUTPUT_NAME = 'Mask_Editing_Collection'
# These are outputs, not actual camera acquisitions; scanning them would copy
# old collages or duplicate archived previews from Results/by_session.
SKIP_TREE_NAMES = frozenset({
    'results', 'object_analysis', 'by_date', 'by_session',
    'corrected_images', 'pca_maps', 'index_maps', '.git', '__pycache__',
})
ROOT_RGB_NAMES = frozenset({
    'wheat_rgb.png', 'mask_preview.png', 'segmentation_rgb.png',
})
ROOT_MASK_NAMES = frozenset({'wheat_mask.png'})


@dataclass(frozen=True)
class _Entry:
    session: str
    shot: str
    kind: str
    object_id: str
    source: Path
    shot_folder: Path


def _safe_part(value: str, max_length: int = 72) -> str:
    """Retain meaningful Unicode names while removing invalid Windows chars."""
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', value)
    value = re.sub(r'\s+', ' ', value).strip(' ._')
    value = value[:max_length].rstrip(' ._') or 'unnamed'
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                         *(f'LPT{i}' for i in range(1, 10))}:
        value = '_' + value
    return value


def _file_is_color(path: Path) -> bool:
    """Accept RGB/RGBA and other real colour modes, not gray mask images."""
    try:
        with Image.open(path) as img:
            return img.mode in ('RGB', 'RGBA', 'CMYK', 'YCbCr') or (
                img.mode == 'P' and img.getpalette() is not None
            )
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def _is_shot(folder: Path) -> bool:
    """Recognize current and legacy shot folders, even if cubes were removed."""
    if (folder / 'Spectral_Cube').is_dir():
        return True
    if (folder / 'corrected_images' / 'corrected_cube.npz').is_file():
        return True
    if (folder / 'masks').is_dir():
        return True
    if (folder / 'wheat_mask.png').is_file():
        return True
    if (folder / 'Images').is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in (folder / 'Images').iterdir()
    ):
        return True
    return False


def _shot_folders(project_root: Path, output_folder: Path) -> Iterable[Path]:
    """Walk acquisitions without descending into generated results or symlinks."""
    import os
    output_resolved = output_folder.resolve()
    for current, dirs, _files in os.walk(project_root, followlinks=False):
        folder = Path(current)
        dirs[:] = [d for d in dirs
                   if not (folder / d).is_symlink()
                   and d.casefold() not in SKIP_TREE_NAMES
                   and (folder / d).resolve() != output_resolved]
        if folder == output_resolved:
            dirs[:] = []
            continue
        if _is_shot(folder):
            yield folder
            # All relevant files are directly under the shot, Images/, masks/.
            # Other nested folders (old calculation outputs) need not be scanned.
            dirs[:] = []


def _session_and_shot(project_root: Path, folder: Path) -> tuple[str, str]:
    rel = folder.relative_to(project_root)
    parts = rel.parts
    if not parts:  # caller gave a shot folder instead of the project
        return project_root.parent.name, project_root.name
    if len(parts) == 1:  # caller gave one session folder instead of the project
        return project_root.name, parts[0]
    session = parts[0]
    shot = '__'.join(parts[1:])
    return session, shot


def _entries(project_root: Path, output_folder: Path, *, include_legacy: bool):
    for shot_folder in _shot_folders(project_root, output_folder):
        session, shot = _session_and_shot(project_root, shot_folder)
        image_folder = shot_folder / 'Images'
        if image_folder.is_dir() and not image_folder.is_symlink():
            for source in sorted(image_folder.iterdir(), key=lambda p: p.name.casefold()):
                if (source.is_file() and not source.is_symlink()
                        and source.suffix.lower() in IMAGE_EXTENSIONS
                        and _file_is_color(source)):
                    yield _Entry(session, shot, 'image', source.stem, source, shot_folder)
        # Some older pipelines stored their RGB preview directly in the shot root.
        for source in sorted(shot_folder.iterdir(), key=lambda p: p.name.casefold()):
            if (source.is_file() and not source.is_symlink()
                    and source.name.casefold() in ROOT_RGB_NAMES
                    and _file_is_color(source)):
                yield _Entry(session, shot, 'image', source.stem, source, shot_folder)
        mask_folder = shot_folder / 'masks'
        if mask_folder.is_dir() and not mask_folder.is_symlink():
            for source in sorted(mask_folder.iterdir(), key=lambda p: p.name.casefold()):
                if (source.is_file() and not source.is_symlink()
                        and source.suffix.lower() in IMAGE_EXTENSIONS):
                    yield _Entry(session, shot, 'mask', source.stem, source, shot_folder)
        if include_legacy:
            for source in sorted(shot_folder.iterdir(), key=lambda p: p.name.casefold()):
                if (source.is_file() and not source.is_symlink()
                        and source.name.casefold() in ROOT_MASK_NAMES):
                    yield _Entry(session, shot, 'mask', source.stem, source, shot_folder)


def _destination_name(entry: _Entry) -> str:
    label = '__'.join((_safe_part(entry.session), _safe_part(entry.shot),
                      entry.kind.upper(), _safe_part(entry.object_id)))
    return label + entry.source.suffix.lower()


def collect_mask_editing_files(
    project_root: str | Path,
    output_folder: str | Path | None = None,
    *,
    overwrite: bool = False,
    include_legacy: bool = True,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Gather image and mask files into a *flat* output directory.

    Parameters
    ----------
    project_root : path
        The project directory containing session folders. One session directory
        also works; nested shot directories are supported.
    output_folder : path, optional
        By default ``project_root/Mask_Editing_Collection``.
    overwrite : bool
        Explicitly allow overwriting destination files. False protects any
        masks that have already been manually edited in the output directory.
    include_legacy : bool
        Also collect ``wheat_mask.png`` from the shot root.
    dry_run : bool
        Print the plan without copying anything or writing the manifest.

    Returns
    -------
    List of manifest dictionaries. CSV contains absolute original/copy paths.

    Notes
    -----
    This script copies files, not pixels: the mask PNGs are unchanged (same
    shape, polarity and values). It never restores edited masks to their
    original locations; use the generated manifest for matching them later.
    """
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root == Path(root.anchor):
        raise ValueError('Pass the actual project directory, not a disk/filesystem root')
    destination = ((root / DEFAULT_OUTPUT_NAME) if output_folder is None
                   else Path(output_folder).expanduser()).resolve()
    if destination == root:
        raise ValueError('The output folder must not be the project root')
    if root in destination.parents:
        first = destination.relative_to(root).parts[0]
        if first.lower() in {'data', 'images', 'spectral_cube'}:
            raise ValueError('Do not write collected files into protected input folders')
    # Also prevent placing output inside any actual camera shot folder.
    shots = list(_shot_folders(root, destination))
    if any(destination == s or s in destination.parents for s in shots):
        raise ValueError('Output folder must not be inside a camera shot folder')
    entries = [e for e in _entries(root, destination, include_legacy=include_legacy)]
    if not entries:
        print('No RGB previews or masks found under', root)
        return []
    entries.sort(key=lambda e: (e.session.casefold(), e.shot.casefold(),
                                e.kind, e.object_id.casefold(), str(e.source)))
    names = {}
    report: list[dict[str, str]] = []
    for entry in entries:
        proposed = _destination_name(entry)
        key = proposed.casefold()
        if key in names and names[key] != entry.source:
            # Preserve readable names but guarantee uniqueness after sanitizing.
            relative = entry.source.relative_to(root).as_posix()
            digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:10]
            stem, ext = proposed.rsplit('.', 1)
            proposed = f'{stem}__{digest}.{ext}'
            key = proposed.casefold()
            if key in names:
                raise RuntimeError(f'Unexpected destination filename collision: {proposed}')
        names[key] = entry.source
        target = destination / proposed
        if target.is_file() and not overwrite:
            status = 'skipped_existing'
        else:
            status = 'would_copy' if dry_run else 'copied'
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry.source, target)
        report.append({
            'session': entry.session,
            'camera_shot': entry.shot,
            'type': entry.kind,
            'object_or_image_name': entry.object_id,
            'source_path': str(entry.source),
            'collected_path': str(target),
            'filename': proposed,
            'status': status,
        })
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / MANIFEST_NAME).open('w', encoding='utf-8-sig', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    from collections import Counter
    counts = Counter(row['status'] for row in report)
    print(f'{len(shots)} camera-shot folders examined; {len(report)} files identified.')
    print('Status:', ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
    print('Destination:', destination)
    if not dry_run:
        print('Manifest:', destination / MANIFEST_NAME)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_root', type=Path, help='Project folder containing session directories')
    parser.add_argument('--output', type=Path, default=None, help='Flat output folder (default: <root>/Mask_Editing_Collection)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing collected copies (off by default)')
    parser.add_argument('--no-legacy', action='store_true', help='Skip legacy wheat_mask.png')
    parser.add_argument('--dry-run', action='store_true', help='Preview copying without writing any files')
    args = parser.parse_args()
    collect_mask_editing_files(args.project_root, args.output, overwrite=args.overwrite,
                               include_legacy=not args.no_legacy, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
