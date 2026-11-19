"""Safely remove computed files from one wheat hyperspectral study.

The only directories protected by *name* are Data, Images and Spectral_Cube
(case-insensitive). Protected directories, all their contents, and all their
ancestor directories are kept. EVERYTHING ELSE inside the study is removed:
there is deliberately no version-specific allow/blocklist of calculated files.

IMPORTANT: If an old pipeline wrote derived files into Data/Images/Spectral_Cube,
those derived files are kept too, since the program cannot tell them apart
from original camera data. This includes mask-preview RGB files in Images.

Usage (preview only):
    python clean_study_results.py "D:/Wheat/Study_01"

Deletion (explicit confirmation required):
    python clean_study_results.py "D:/Wheat/Study_01" --execute --confirm "DELETE:Study_01"

Use clean_study_results(...) in a Jupyter notebook for the same functionality.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

PROTECTED_DIR_NAMES = frozenset({'data', 'images', 'spectral_cube'})


@dataclass(frozen=True)
class CleanupPlan:
    study: Path
    protected_directories: tuple[Path, ...]
    deletion_targets: tuple[Path, ...]

    @property
    def confirmation(self) -> str:
        return f'DELETE:{self.study.name}'


def _validate_study_root(study_folder, *, allow_code_at_root: bool) -> Path:
    supplied = Path(study_folder).expanduser()
    if supplied.is_symlink():
        raise ValueError('The study root cannot itself be a symbolic link.')
    if not supplied.is_dir():
        raise NotADirectoryError(f'Study folder not found: {supplied}')
    root = supplied.resolve(strict=True)
    if root.name.casefold() in PROTECTED_DIR_NAMES:
        raise ValueError('Choose a STUDY folder, not Data, Images or Spectral_Cube itself.')
    if root == Path(root.anchor):
        raise ValueError('Refusing to clean a filesystem/drive root.')
    home = Path.home().resolve()
    if root == home or root in home.parents:
        raise ValueError('Refusing to clean the home directory or its ancestors.')
    cwd = Path.cwd().resolve()
    if root == cwd or root in cwd.parents:
        raise ValueError('Refusing to clean the current working directory or its ancestors. '
                         'Select a specific study subdirectory.')
    if not allow_code_at_root:
        source_files = [p.name for p in root.iterdir()
                        if p.is_file() and p.suffix.casefold() in {'.py', '.ipynb'}]
        if source_files:
            raise ValueError('This folder contains Python/Jupyter files at its top level: '
                             f'{source_files[:6]}. This often means you selected the PIPELINE '
                             'project root instead of a study. If intentional, pass '
                             'allow_code_at_root=True (these files would be deleted).')
    return root


def _scan_folder(folder: Path) -> tuple[bool, list[Path], list[Path]]:
    """Find protected descendants and minimal deletion branches; never follow links.

    Returns (has_protected_descendant, deletion_targets, protected_dirs).
    If a child folder contains no protected directory anywhere underneath it,
    schedule that child as ONE rmtree target, not its individual contents.
    """
    has_protected = False
    deletions: list[Path] = []
    protected: list[Path] = []
    with os.scandir(folder) as scanner:
        entries = sorted(list(scanner), key=lambda e: e.name.casefold())
    for entry in entries:
        child = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            if entry.name.casefold() in PROTECTED_DIR_NAMES:
                has_protected = True
                protected.append(child)
                # Deliberately do NOT inspect any descendant of these folders.
            else:
                child_has_protected, child_deletions, child_protected = _scan_folder(child)
                if child_has_protected:
                    has_protected = True
                    deletions.extend(child_deletions)
                    protected.extend(child_protected)
                else:
                    deletions.append(child)
        else:
            # Includes regular files and links, which are never followed.
            # A link named Data/Images/Spectral_Cube is kept to avoid silently
            # deleting a user-designated protected-looking source reference.
            if entry.is_symlink() and entry.name.casefold() in PROTECTED_DIR_NAMES:
                has_protected = True
                protected.append(child)
            else:
                deletions.append(child)
    return has_protected, deletions, protected


def plan_study_cleanup(study_folder, *, allow_code_at_root=False) -> CleanupPlan:
    """Create a deletion plan WITHOUT changing any file.

    Require a real Spectral_Cube folder. This prevents an unrecognized or
    wrongly selected directory from being emptied by accident.
    """
    root = _validate_study_root(study_folder, allow_code_at_root=allow_code_at_root)
    _, deletions, protected = _scan_folder(root)
    real_cubes = [p for p in protected if p.name.casefold() == 'spectral_cube'
                  and not p.is_symlink()]
    if not real_cubes:
        raise ValueError('No real Spectral_Cube directory was found under this path. '
                         'To prevent accidental data loss, cleanup was refused.')
    if not protected:
        raise ValueError('No protected camera directories found; refusing cleanup.')
    return CleanupPlan(root, tuple(protected), tuple(deletions))


def _show_plan(plan: CleanupPlan, *, max_listed=60) -> None:
    print(f'\nSTUDY: {plan.study}')
    print('PROTECTED: Data / Images / Spectral_Cube, their contents and parent directories')
    print(f'Protected directories/links found: {len(plan.protected_directories)}')
    print(f'Deletion targets: {len(plan.deletion_targets)} '
          '(one target may contain an entire calculated-data subtree)')
    for path in plan.deletion_targets[:max_listed]:
        print('  DELETE:', path.relative_to(plan.study).as_posix())
    remaining = len(plan.deletion_targets) - max_listed
    if remaining > 0:
        print(f'  ... and {remaining} more target(s)')


def clean_study_results(study_folder, *, dry_run=True, confirmation=None,
                        allow_code_at_root=False, max_listed=60) -> CleanupPlan:
    """Preview or remove ALL non-protected files/directories within a study.

    By default dry_run=True: list planned deletions but make no changes.
    To delete, pass dry_run=False and confirmation='DELETE:<study folder name>'.
    Links are NOT followed; files outside the study are never deleted.
    """
    plan = plan_study_cleanup(study_folder, allow_code_at_root=allow_code_at_root)
    _show_plan(plan, max_listed=max_listed)
    if dry_run:
        print('DRY RUN: no file has been deleted.')
        print(f'To execute, explicitly confirm: {plan.confirmation!r}')
        return plan
    if confirmation != plan.confirmation:
        raise ValueError(f'Cleanup NOT executed. Required confirmation: {plan.confirmation!r}')

    # Rescanning immediately before removal guards against a changed folder
    # structure since the preview. The confirmation is checked before any delete.
    current = plan_study_cleanup(plan.study, allow_code_at_root=allow_code_at_root)
    if current.deletion_targets != plan.deletion_targets or current.protected_directories != plan.protected_directories:
        raise RuntimeError('Directory tree changed while preparing cleanup; no files deleted. '
                           'Re-run the preview and retry.')

    deleted = 0
    for target in current.deletion_targets:
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted += 1
    print(f'DONE: {deleted} deletion target(s) removed; protected directories untouched.')
    return current


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('study_folder', type=Path)
    parser.add_argument('--execute', action='store_true', help='Actually delete results (default is preview only)')
    parser.add_argument('--confirm', default=None, help='Required exact text: DELETE:<study folder name>')
    parser.add_argument('--allow-code-at-root', action='store_true',
                        help='Permit deletion of .py/.ipynb files located directly in study folder')
    parser.add_argument('--max-listed', type=int, default=60)
    args = parser.parse_args(argv)
    clean_study_results(args.study_folder, dry_run=not args.execute,
                        confirmation=args.confirm, allow_code_at_root=args.allow_code_at_root,
                        max_listed=max(0, args.max_listed))


if __name__ == '__main__':
    main()
