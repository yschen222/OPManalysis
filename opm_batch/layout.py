from __future__ import annotations
from pathlib import Path
from .config import NOISE_SUBDIR
from .io_utils import pick_scan_csv

def folder_has_scan_csv(folder: Path, root_for_pick: Path) -> bool:
    try:
        return pick_scan_csv(folder, root=root_for_pick) is not None
    except Exception:
        return False

def detect_layout(root_dir: Path):
    # (C) root itself is one dataset
    if folder_has_scan_csv(root_dir, root_for_pick=root_dir):
        return ('single', [root_dir], 'all_only')

    lvl1 = [p for p in sorted(root_dir.iterdir()) if p.is_dir() and p.name != NOISE_SUBDIR]
    if not lvl1:
        return ('single', [root_dir], 'all_only')

    # (A) flat: first-level dirs are datasets
    if any(folder_has_scan_csv(d, root_for_pick=root_dir) for d in lvl1):
        return ('flat', [root_dir], 'all_only')

    # (B) session: first-level are sessions; second-level contain datasets
    session_dirs = []
    for s in lvl1:
        subdirs = [p for p in sorted(s.iterdir()) if p.is_dir() and p.name != NOISE_SUBDIR]
        if any(folder_has_scan_csv(sd, root_for_pick=s) for sd in subdirs):
            session_dirs.append(s)
    if session_dirs:
        return ('session', session_dirs, 'per_session')

    return ('single', [root_dir], 'all_only')
