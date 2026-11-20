"""Safely restore manually edited masks collected by collect_mask_editing_files.py.

The collector writes Mask_Editing_Collection/mask_editing_manifest.csv with
``type``, ``source_path``, and ``collected_path`` for each file. This script uses
those exact paths (not filename guesses) and restores ONLY rows with type=mask.
RGB previews in Images are NEVER restored or modified.

Safety:
* Dry-run by default: inspect the plan before any originals are changed.
* Actual restoration requires confirmation='RESTORE MASKS'.
* Original files are backed up by default under the collection directory.
* Edited masks must retain original width/height and contain binary pixels;
  RGB grayscale-equivalent masks are accepted and saved as 8-bit 0/255 PNG.
* Only original *.png files inside a shot's masks/ folder or legacy
  wheat_mask.png files may be written. No new target files are invented.
* Malformed paths, missing files, duplicate targets, and source-newer conflicts
  are detected in the preflight. No originals are changed if any are found.
* After a successful restore, subsequent runs see unchanged masks and skip them.

Notebook example:
    from restore_mask_editing_files import restore_mask_editing_files
    report = restore_mask_editing_files(STUDY_ROOT)  # preview, no changes
    # After checking the preview:
    report = restore_mask_editing_files(
        STUDY_ROOT, dry_run=False, confirmation='RESTORE MASKS')

Terminal example:
    python restore_mask_editing_files.py "D:/WheatProject"
    python restore_mask_editing_files.py "D:/WheatProject" --apply
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

MANIFEST_NAME = 'mask_editing_manifest.csv'
DEFAULT_COLLECTION = 'Mask_Editing_Collection'
CONFIRMATION = 'RESTORE MASKS'
REQUIRED_COLUMNS = {'type', 'source_path', 'collected_path'}


def _inside(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _has_symlink(path: Path, stopping_at: Path) -> bool:
    """Check lexical path elements BEFORE calling resolve(), including target."""
    cursor = path
    while True:
        if cursor.is_symlink():
            return True
        if cursor == stopping_at or cursor == cursor.parent:
            return False
        cursor = cursor.parent


def _load_binary_mask(path: Path) -> np.ndarray:
    """Return bool array; never silently threshold gray/color pixels."""
    try:
        with Image.open(path) as img:
            if img.mode == '1':
                return np.asarray(img.convert('L')) > 0
            if img.mode in ('L', 'I;16', 'I'):
                arr = np.asarray(img)
                unique = np.unique(arr)
                if not set(int(x) for x in unique).issubset({0, 1, 255, 65535}):
                    raise ValueError('grayscale mask has nonbinary pixel values')
                return arr > 0
            if img.mode in ('P', 'RGB', 'RGBA', 'LA'):
                rgba = np.asarray(img.convert('RGBA'))
                rgb = rgba[:, :, :3]
                alpha = rgba[:, :, 3]
                if not np.all((alpha == 0) | (alpha == 255)):
                    raise ValueError('mask has partially transparent pixels')
                visible = alpha == 255
                if not np.all((rgb[visible, 0] == rgb[visible, 1]) &
                              (rgb[visible, 1] == rgb[visible, 2])):
                    raise ValueError('mask contains colored pixels, not black/white')
                if not np.all(np.isin(rgb[visible, 0], (0, 1, 255))):
                    raise ValueError('mask has nonbinary visible pixel values')
                return visible & (rgb[:, :, 0] > 0)
            raise ValueError(f'unsupported mask image mode {img.mode!r}')
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f'invalid or unreadable image: {exc}') from exc


def _validate_original_image(path: Path, edited_shape: tuple[int, int]) -> None:
    try:
        with Image.open(path) as img:
            original_shape = (img.height, img.width)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f'original mask is not an image: {exc}') from exc
    if original_shape != edited_shape:
        raise ValueError(f'image shape changed: original={original_shape}, edited={edited_shape}')


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ValueError(f'Unsupported manifest columns; required: {sorted(REQUIRED_COLUMNS)}')
        return [row for row in reader if str(row.get('type', '')).strip().lower() == 'mask']


def _plan(project_root: Path, collection: Path, *, allow_source_newer: bool) -> list[dict[str, Any]]:
    manifest = collection / MANIFEST_NAME
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f'Manifest not found or unsafe: {manifest}')
    planned: list[dict[str, Any]] = []
    raw_rows = _read_manifest(manifest)
    seen_sources: set[Path] = set()
    for index, row in enumerate(raw_rows, start=2):
        source_raw = Path(str(row.get('source_path') or '')).expanduser()
        edited_raw = Path(str(row.get('collected_path') or '')).expanduser()
        rec: dict[str, Any] = {
            'manifest_row': index, 'session': row.get('session', ''),
            'camera_shot': row.get('camera_shot', ''),
            'mask_name': row.get('object_or_image_name', ''),
            'source_path': str(source_raw), 'edited_path': str(edited_raw),
            'status': 'error', 'message': '', 'backup_path': '',
            '_source_hash': '', '_edited_hash': '',
        }
        planned.append(rec)
        try:
            if not source_raw.is_absolute() or not edited_raw.is_absolute():
                raise ValueError('manifest paths must be absolute (collector output)')
            if _has_symlink(source_raw, project_root) or _has_symlink(edited_raw, collection):
                raise ValueError('symlinks in source/collected path are forbidden')
            source = source_raw.resolve()
            edited = edited_raw.resolve()
            if not _inside(source, project_root):
                raise ValueError('original path is outside the selected project')
            if edited.parent != collection:
                raise ValueError('edited path must be directly inside the collection folder')
            if row.get('filename') and edited.name != row['filename']:
                raise ValueError('edited filename does not match the manifest')
            if not (source.suffix.lower() == '.png' and
                    (source.parent.name.casefold() == 'masks' or source.name.casefold() == 'wheat_mask.png')):
                raise ValueError('source is not a supported PNG mask destination')
            if source in seen_sources:
                raise ValueError('duplicate original mask path in manifest')
            seen_sources.add(source)
            if not edited.is_file():
                raise ValueError('edited copy is missing')
            if not source.is_file():
                raise ValueError('original mask is missing; refusing to invent a new target')
            source_digest = _hash_file(source)
            edited_digest = _hash_file(edited)
            if source_digest == edited_digest:
                rec['_source_hash'] = source_digest
                rec['_edited_hash'] = edited_digest
                rec['status'] = 'unchanged'
                rec['message'] = 'copied file is identical to original; no restoration needed'
                continue
            edited_mask = _load_binary_mask(edited)
            _validate_original_image(source, edited_mask.shape)
            rec['_source_hash'] = source_digest
            rec['_edited_hash'] = edited_digest
            try:
                same_pixels = np.array_equal(_load_binary_mask(source), edited_mask)
            except ValueError:
                same_pixels = False
            if same_pixels:
                rec['status'] = 'unchanged'
                rec['message'] = 'mask pixels are identical (file encoding may differ)'
            elif not allow_source_newer and source.stat().st_mtime_ns > edited.stat().st_mtime_ns:
                raise ValueError('original mask is newer than edited copy; resolve conflict first')
            else:
                rec['status'] = 'ready'
                rec['message'] = f'validated {edited_mask.shape[1]}x{edited_mask.shape[0]} binary mask'
        except (OSError, ValueError, TypeError) as exc:
            rec['status'] = 'error'
            rec['message'] = f'{type(exc).__name__}: {exc}'
    return planned


def _save_atomic_binary(path: Path, mask: np.ndarray) -> None:
    fd, tempname = tempfile.mkstemp(prefix='.mask_restore_', suffix='.png', dir=path.parent)
    os.close(fd)
    temp = Path(tempname)
    try:
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode='L').save(temp, format='PNG')
        os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def restore_mask_editing_files(
    project_root: str | Path,
    collection_folder: str | Path | None = None,
    *,
    dry_run: bool = True,
    confirmation: str | None = None,
    backup_originals: bool = True,
    allow_source_newer: bool = False,
) -> list[dict[str, str]]:
    """Preview or restore edited masks from the collector's manifest.

    Only modified rows of type='mask' are restored. Original masks are backed
    up before replacement by default. No RGB previews are modified.

    ``dry_run=False`` requires ``confirmation='RESTORE MASKS'``. The full
    preflight must be clean; a conflict in *any* mask blocks all restoration.
    """
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError(f'Pass the actual project directory: {root}')
    collection = (root / DEFAULT_COLLECTION if collection_folder is None else
                  Path(collection_folder).expanduser()).resolve()
    if not collection.is_dir() or collection == root:
        raise NotADirectoryError(f'Mask collection not found: {collection}')
    records = _plan(root, collection, allow_source_newer=allow_source_newer)
    ready = sum(r['status'] == 'ready' for r in records)
    unchanged = sum(r['status'] == 'unchanged' for r in records)
    errors = [r for r in records if r['status'] == 'error']
    print(f'Mask restoration preflight: {ready} modified, {unchanged} unchanged, {len(errors)} conflicts.')
    for record in records:
        print(f"  [{record['status']}] {record['edited_path']} -> {record['source_path']}")
        if record['status'] == 'error':
            print(f"        {record['message']}")
    if dry_run:
        print('DRY RUN: originals have not been changed.')
        return [{k: str(v) for k, v in r.items() if not k.startswith('_')} for r in records]
    if confirmation != CONFIRMATION:
        raise ValueError(f'Actual restoration requires confirmation={CONFIRMATION!r}')
    if errors:
        raise RuntimeError(f'{len(errors)} invalid/conflicting manifest entries; nothing restored')
    if not ready:
        print('No changed masks to restore.')
        return [{k: str(v) for k, v in r.items() if not k.startswith('_')} for r in records]
    if not backup_originals:
        raise ValueError('Backups are mandatory when replacing original masks')
    backup_dir = (collection / '_restore_backups' /
                  datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # Back up ALL targets and validate there were no changes since preflight
    # before modifying even the first original.
    for record in records:
        if record['status'] != 'ready':
            continue
        source, edited = Path(record['source_path']), Path(record['edited_path'])
        if _hash_file(source) != record['_source_hash'] or _hash_file(edited) != record['_edited_hash']:
            raise RuntimeError(f'File changed during preflight; nothing restored: {source}')
    for record in records:
        if record['status'] != 'ready':
            continue
        source = Path(record['source_path'])
        backup = backup_dir / source.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        record['backup_path'] = str(backup)
    for record in records:
        if record['status'] != 'ready':
            continue
        source, edited = Path(record['source_path']), Path(record['edited_path'])
        # Recheck the image; from now on every write is atomic per individual mask.
        if _hash_file(edited) != record['_edited_hash'] or _hash_file(source) != record['_source_hash']:
            raise RuntimeError(f'File changed during restoration; backups are in {backup_dir}')
        mask = _load_binary_mask(edited)
        _save_atomic_binary(source, mask)
        record['status'] = 'restored'
    report = [{k: str(v) for k, v in r.items() if not k.startswith('_')} for r in records]
    log_file = collection / f'mask_restoration_report_{backup_dir.name}.csv'
    with log_file.open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f'Restored {ready} mask(s). Backups: {backup_dir}')
    print(f'Restoration log: {log_file}')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_root', type=Path)
    parser.add_argument('--collection', type=Path, default=None)
    parser.add_argument('--apply', action='store_true', help='Replace originals after confirmation')
    parser.add_argument('--allow-source-newer', action='store_true',
                        help='Explicitly override original-newer conflict detection')
    args = parser.parse_args()
    confirmation = None
    if args.apply:
        confirmation = input(f'Type exactly {CONFIRMATION!r} to restore masks with backups: ')
    restore_mask_editing_files(args.project_root, args.collection, dry_run=not args.apply,
                               confirmation=confirmation, allow_source_newer=args.allow_source_newer)


if __name__ == '__main__':
    main()
