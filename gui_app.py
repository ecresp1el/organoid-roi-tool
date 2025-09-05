#!/usr/bin/env python
import sys, os, csv, json, traceback, time
from pathlib import Path
import numpy as np
import tifffile as tiff
import pandas as pd
print("[gui] Starting Organoid ROI Tool v9...")
print(f"[gui] Python: {sys.version}")
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    import napari
    print(f"[gui] PySide6 imported, napari imported.")
except Exception as e:
    print("[gui] ERROR importing GUI libs:", e)
    sys.exit(1)
# Pillow fallback
try:
    from PIL import Image
    _HAS_PIL = True
    print("[gui] Pillow available for TIFF fallback.")
except Exception:
    _HAS_PIL = False
    print("[gui] Pillow NOT available; no TIFF fallback.")
from roi_utils import compute_roi, save_roi_json

# -----------------------------------------------------------------------------
# Session persistence helpers
# -----------------------------------------------------------------------------
_APP_STATE_DIR = Path.home() / ".organoid-roi-tool"
_APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

def _global_state_path() -> Path:
    return _APP_STATE_DIR / "state.json"

def _read_json(path: Path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[gui] Failed to write JSON {path}: {e}")

def _merge_global_state(updates: dict):
    st = _read_json(_global_state_path()) or {}
    st.update(updates)
    _write_json(_global_state_path(), st)

def infer_project_root_from_path(p: Path) -> Path | None:
    parts = p.resolve().parts
    if 'wells' in parts:
        i = parts.index('wells')
        if i > 0:
            return Path(*parts[:i])
    return None

class ProjectDashboard(QtWidgets.QDialog):
    def __init__(self, parent, project_root: Path):
        super().__init__(parent)
        self.setWindowTitle('Project Progress Dashboard')
        self.resize(800, 500)
        self.parent_app = parent  # OrganoidROIApp
        self.project_root = project_root
        v = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.lbl_root = QtWidgets.QLabel(str(self.project_root))
        self.btn_change = QtWidgets.QPushButton('Change Project Root')
        self.btn_refresh = QtWidgets.QPushButton('Refresh')
        top.addWidget(QtWidgets.QLabel('Project:'))
        top.addWidget(self.lbl_root, 1)
        top.addWidget(self.btn_change)
        top.addWidget(self.btn_refresh)
        v.addLayout(top)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Name', 'Done', 'Total', '%', 'Next Unlabeled'])
        self.tree.setColumnWidth(0, 260)
        v.addWidget(self.tree, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.btn_go = QtWidgets.QPushButton('Next Unlabeled (Scope)')
        self.btn_close = QtWidgets.QPushButton('Close')
        bottom.addStretch(1)
        bottom.addWidget(self.btn_go)
        bottom.addWidget(self.btn_close)
        v.addLayout(bottom)
        self.btn_close.clicked.connect(self.close)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_change.clicked.connect(self.change_root)
        self.btn_go.clicked.connect(self.open_selected_unlabeled)
        self.tree.itemDoubleClicked.connect(lambda *_: self.open_selected_unlabeled())
        self.data = None
        self.refresh()

    def change_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Project Root', str(self.project_root))
        if not d:
            return
        p = Path(d)
        self.project_root = p
        self.lbl_root.setText(str(p))
        _merge_global_state({'last_project_root': str(p.resolve())})
        self.refresh()

    def scan(self):
        root = self.project_root
        wells_dir = root / 'wells'
        if not wells_dir.exists():
            return {'error': f'No wells/ directory under {root}', 'root': str(root), 'wells': {}}
        wells = {}
        total_all = 0
        done_all = 0
        for well_dir in sorted([p for p in wells_dir.iterdir() if p.is_dir()]):
            well_name = well_dir.name
            well_total = 0
            well_done = 0
            days = {}
            for day_dir in sorted([p for p in well_dir.iterdir() if p.is_dir()]):
                day_name = day_dir.name
                day_total = 0
                day_done = 0
                unlabeled = []
                # recurse one more level (times) then files, excluding derived ROI outputs
                all_tifs = sorted(day_dir.rglob('*.tif')) + sorted(day_dir.rglob('*.tiff'))
                tifs = [sub for sub in all_tifs if not is_derived_tiff_name(sub.name)]
                for sub in tifs:
                    day_total += 1
                    base = sub.stem
                    roi_json = sub.parent / f"{base}_roi.json"
                    if roi_json.exists():
                        day_done += 1
                    else:
                        unlabeled.append(sub)
                days[day_name] = {
                    'total': day_total,
                    'done': day_done,
                    'next_unlabeled': str(unlabeled[0]) if unlabeled else None,
                }
                well_total += day_total
                well_done += day_done
            wells[well_name] = {
                'total': well_total,
                'done': well_done,
                'days': days,
                'next_unlabeled': self._first_unlabeled(days),
            }
            total_all += well_total
            done_all += well_done
        return {'root': str(root), 'total': total_all, 'done': done_all, 'wells': wells}

    def _first_unlabeled(self, days_dict):
        for dname in sorted(days_dict.keys()):
            nu = days_dict[dname].get('next_unlabeled')
            if nu:
                return nu
        return None

    def refresh(self):
        self.tree.clear()
        self.data = self.scan()
        if not self.data or 'wells' not in self.data:
            QtWidgets.QMessageBox.warning(self, 'Scan Failed', 'Could not scan project root.')
            return
        # Project root row
        total = self.data.get('total', 0) or 0
        done = self.data.get('done', 0) or 0
        pct = f"{int(round(100.0*done/total)) if total else 0}"
        root_item = QtWidgets.QTreeWidgetItem(['ALL', str(done), str(total), pct, ''])
        self.tree.addTopLevelItem(root_item)
        root_item.setData(0, QtCore.Qt.UserRole, {'type': 'project', 'path': str(self.project_root)})
        # Wells
        wells_dir = self.project_root / 'wells'
        for well_name, winfo in sorted(self.data['wells'].items()):
            w_total = winfo.get('total', 0)
            w_done = winfo.get('done', 0)
            w_pct = f"{int(round(100.0*w_done/w_total)) if w_total else 0}"
            w_next = winfo.get('next_unlabeled') or ''
            w_item = QtWidgets.QTreeWidgetItem([well_name, str(w_done), str(w_total), w_pct, w_next])
            root_item.addChild(w_item)
            w_item.setData(0, QtCore.Qt.UserRole, {'type': 'well', 'path': str((wells_dir / well_name).resolve())})
            # Days
            for day_name, dinfo in sorted(winfo['days'].items()):
                d_total = dinfo.get('total', 0)
                d_done = dinfo.get('done', 0)
                d_pct = f"{int(round(100.0*d_done/d_total)) if d_total else 0}"
                d_next = dinfo.get('next_unlabeled') or ''
                d_item = QtWidgets.QTreeWidgetItem([day_name, str(d_done), str(d_total), d_pct, d_next])
                w_item.addChild(d_item)
                d_item.setData(0, QtCore.Qt.UserRole, {'type': 'day', 'path': str((wells_dir / well_name / day_name).resolve())})
        self.tree.expandAll()

    def open_selected_unlabeled(self):
        it = self.tree.currentItem()
        if not it:
            QtWidgets.QMessageBox.information(self, 'Select a row', 'Select ALL, a well, or a day to set scope.')
            return
        meta = it.data(0, QtCore.Qt.UserRole)
        if not meta:
            QtWidgets.QMessageBox.information(self, 'No scope', 'Could not determine scope from selection.')
            return
        try:
            self.parent_app.set_nav_scope(meta['type'], Path(meta['path']))
            if not self.parent_app.next_unlabeled_scope(from_start=True):
                QtWidgets.QMessageBox.information(self, 'Done', 'No unlabeled images found in the selected scope.')
            else:
                self.parent_app.raise_(); self.parent_app.activateWindow()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Open Failed', f'Failed to open next unlabeled in scope:\n{e}')
def read_pixel_size_um(tif_path: Path):
    try:
        with tiff.TiffFile(str(tif_path)) as tf:
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags.values()}
            xres = tags.get('XResolution', None)
            if xres:
                if isinstance(xres, tuple) and len(xres) == 2 and xres[1] != 0:
                    ppi = xres[0] / xres[1]
                elif isinstance(xres, (int, float)):
                    ppi = float(xres)
                else:
                    ppi = None
                if ppi and ppi > 0:
                    um_per_px = 25400.0 / ppi
                    return um_per_px
    except Exception:
        pass
    return None
def parse_from_path(p: Path):
    parts = p.parts
    if 'wells' in parts:
        i = parts.index('wells')
        try:
            well = parts[i+1]; day = parts[i+2]; time = parts[i+3]
            return well, day, time
        except Exception:
            return None, None, None
    return None, None, None
def _load_with_tifffile(path: Path):
    """Robust TIFF reader using tifffile.

    - Tries to stack all pages.
    - If pages have mismatched shapes, falls back to stacking only pages that
      match the first page's height/width.
    - If still not stackable, returns the first page.
    """
    try:
        return tiff.imread(str(path))
    except Exception as e:
        # Handle mismatched shapes between pages by manual reading
        msg = str(e).lower()
        if 'same shape' in msg or 'must have the same shape' in msg:
            try:
                with tiff.TiffFile(str(path)) as tf:
                    first = tf.pages[0].asarray()
                    frames = [first]
                    h, w = first.shape[:2]
                    for page in tf.pages[1:]:
                        try:
                            arr = page.asarray()
                            if arr.shape[:2] == (h, w):
                                frames.append(arr)
                        except Exception:
                            pass
                    if len(frames) > 1:
                        try:
                            return np.stack(frames, axis=0)
                        except Exception:
                            return first
                    return first
            except Exception:
                pass
        # Re-raise for upstream fallback (Pillow) to take over
        raise
def _load_with_pillow(path: Path):
    img = Image.open(str(path))
    frames = []
    try:
        i = 0
        while True:
            img.seek(i)
            frames.append(np.array(img))
            i += 1
    except EOFError:
        pass
    if len(frames) == 0:
        frames = [np.array(img)]
    # Some multi-page TIFFs may have varying shapes; keep only frames
    # matching the first frame's HxW before stacking.
    if len(frames) > 1:
        base = frames[0]
        h, w = base.shape[:2]
        frames = [f for f in frames if f.shape[:2] == (h, w)] or [base]
    arr = np.stack(frames, axis=0)
    if arr.ndim == 4:
        rgb = arr[..., :3].astype(np.float32)
        gray = (0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2])
        arr = gray
    return arr

def ensure_2d_image(img: np.ndarray) -> np.ndarray:
    """Convert various TIFF shapes to a 2D grayscale image for display/ROI.

    Heuristics:
    - If 2D: return as-is
    - If 3D and last dim is RGB/RGBA: convert to grayscale
    - If 3D and first dim looks like a small stack (T/Z): max project axis 0
    - If 3D and last dim is 1: squeeze channel
    - If >3D: max project across non-spatial axes; then handle color if present
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Color images: (H,W,3 or 4)
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            return gray
        # Stack: (T,Z, H, W) → typically first dim is smaller than spatial
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            try:
                return np.max(arr, axis=0)
            except Exception:
                return np.squeeze(arr)
        # Single-channel last-dim
        if arr.shape[-1] == 1:
            return np.squeeze(arr, axis=-1)
        # Fallback: max project first axis
        try:
            return np.max(arr, axis=0)
        except Exception:
            return np.squeeze(arr)
    # Higher dimensions: reduce non-spatial axes by max
    axes_sorted = np.argsort(arr.shape)
    yx_axes = axes_sorted[-2:]
    other_axes = tuple(ax for ax in range(arr.ndim) if ax not in yx_axes)
    try:
        reduced = np.max(arr, axis=other_axes)
    except Exception:
        reduced = np.squeeze(arr)
    if reduced.ndim == 3 and reduced.shape[-1] in (3, 4):
        rgb = reduced[..., :3].astype(np.float32)
        return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return np.squeeze(reduced)

# -----------------------------------------------------------------------------
# File listing helpers to avoid navigating into derived ROI outputs
# -----------------------------------------------------------------------------
def is_derived_tiff_name(name: str) -> bool:
    n = name.lower()
    return (
        n.endswith('_mask.tif') or n.endswith('_mask.tiff') or
        n.endswith('_roi_masked.tif') or n.endswith('_roi_masked.tiff') or
        n.endswith('_roi_masked_cropped.tif') or n.endswith('_roi_masked_cropped.tiff')
    )

def list_original_tiffs(directory: Path):
    files = sorted(directory.glob('*.tif')) + sorted(directory.glob('*.tiff'))
    return [p for p in files if not is_derived_tiff_name(p.name)]

def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def parse_day_name(name: str) -> int:
    # expects 'day_XX'
    try:
        if name.lower().startswith('day_'):
            return int(name.split('_', 1)[1])
    except Exception:
        pass
    return 0

def parse_time_name(name: str) -> int:
    # expects 'HHhMMm'
    try:
        s = name.lower()
        if 'h' in s and 'm' in s:
            hh = int(s.split('h')[0])
            mm = int(s.split('h')[1].split('m')[0])
            return hh * 60 + mm
    except Exception:
        pass
    return 0

def sorted_originals_under(root: Path) -> list[Path]:
    # Walk wells/<well>/day_XX/HHhMMm/*.tif in a stable order
    images: list[Path] = []
    wells_dir = root
    # root may be project or well or day; normalize below
    if (root / 'wells').exists():  # project root
        wells_dir = root / 'wells'
    # If root is a well dir, its parent is 'wells'
    # If root is a day dir, we handle specially later
    if wells_dir.name == 'wells':
        # project scope: iterate wells
        for well_dir in sorted([p for p in wells_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            for day_dir in sorted([p for p in well_dir.iterdir() if p.is_dir()], key=lambda p: parse_day_name(p.name)):
                for time_dir in sorted([p for p in day_dir.iterdir() if p.is_dir()], key=lambda p: parse_time_name(p.name)):
                    images.extend(list_original_tiffs(time_dir))
        return images
    # If root is a well directory (contains day_* subfolders)
    if any(p.is_dir() and p.name.lower().startswith('day_') for p in root.iterdir()):
        for day_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: parse_day_name(p.name)):
            for time_dir in sorted([p for p in day_dir.iterdir() if p.is_dir()], key=lambda p: parse_time_name(p.name)):
                images.extend(list_original_tiffs(time_dir))
        return images
    # If root is a day directory (contains time subfolders)
    if any(p.is_dir() and ('h' in p.name.lower()) for p in root.iterdir()):
        for time_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: parse_time_name(p.name)):
            images.extend(list_original_tiffs(time_dir))
        return images
    # Fallback: just list originals directly in root
    return list_original_tiffs(root)
def load_image_any(path: Path):
    try:
        img = _load_with_tifffile(path)
        return img, 'tifffile'
    except Exception as e:
        msg = str(e)
        print(f"[gui] tifffile failed: {msg}")
        if _HAS_PIL:
            try:
                img = _load_with_pillow(path)
                print('[gui] Pillow fallback succeeded.')
                return img, 'pillow'
            except Exception as e2:
                print(f'[gui] Pillow fallback failed: {e2}')
                raise
        raise

def upsert_row(csv_path: Path, row: dict, key="image_path"):
    """Append or update a single row keyed by `key` (default: image_path).
    Ensures exactly one row per image in the CSV.
    """
    cols = ["image_path", "well", "day", "time", "area_px", "perimeter_px", "centroid_yx", "pixel_size_um"]
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Drop any existing rows with the same key
            if key in df.columns:
                df = df[df[key] != row[key]]
            # Ensure all columns exist; add missing ones
            for c in cols:
                if c not in df.columns:
                    df[c] = "" if c in ("centroid_yx", "pixel_size_um") else None
            # Reorder columns
            df = df[[c for c in cols if c in df.columns]] if set(cols).issubset(df.columns) else df
            # Append new row
            df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=cols)
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[gui] upsert_row failed for {csv_path}: {e}")
        # Fallback: write a minimal CSV to avoid blocking UX
        try:
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerow(row)
        except Exception as e2:
            print(f"[gui] fallback CSV write failed for {csv_path}: {e2}")

class OrganoidROIApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Organoid ROI Tool v9')
        self.resize(1200, 800)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        btn_bar = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton('Open Image')
        self.btn_init_project = QtWidgets.QPushButton('Initialize Project')
        self.btn_open_folder = QtWidgets.QPushButton('Import Project')
        self.btn_open_dashboard = QtWidgets.QPushButton('Open Dashboard')
        self.btn_prev = QtWidgets.QPushButton('Prev')
        self.btn_next = QtWidgets.QPushButton('Next')
        self.btn_next_unlabeled = QtWidgets.QPushButton('Next Unlabeled (Scope)')
        self.btn_save = QtWidgets.QPushButton('Save ROI')
        self.btn_delete = QtWidgets.QPushButton('Delete ROI')
        self.btn_stop = QtWidgets.QPushButton('Stop / Save Session')
        self.chk_auto_advance = QtWidgets.QCheckBox('Auto-advance')
        self.chk_auto_advance.setChecked(True)
        btn_bar.addWidget(self.btn_open)
        btn_bar.addWidget(self.btn_init_project)
        btn_bar.addWidget(self.btn_open_folder)
        btn_bar.addWidget(self.btn_open_dashboard)
        btn_bar.addStretch(1)
        btn_bar.addWidget(self.btn_prev)
        btn_bar.addWidget(self.btn_next)
        btn_bar.addWidget(self.btn_next_unlabeled)
        btn_bar.addWidget(self.btn_save)
        btn_bar.addWidget(self.btn_delete)
        btn_bar.addWidget(self.btn_stop)
        btn_bar.addWidget(self.chk_auto_advance)
        layout.addLayout(btn_bar)
        self.viewer = napari.Viewer()
        self.viewer.window._qt_window.setWindowFlag(QtCore.Qt.Widget, True)
        layout.addWidget(self.viewer.window._qt_window)
        # Progress UI
        prog_bar_row = QtWidgets.QHBoxLayout()
        self.progress_label = QtWidgets.QLabel('Progress: 0 / 0')
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.scope_label = QtWidgets.QLabel('Scope: Folder')
        prog_bar_row.addWidget(self.progress_label)
        prog_bar_row.addWidget(self.progress_bar)
        prog_bar_row.addWidget(self.scope_label)
        layout.addLayout(prog_bar_row)

        print('[gui] Napari viewer created.')
        self.current_dir = None; self.file_list = []; self.file_index = -1
        self.image_layer = None; self.shapes = None
        self.btn_open.clicked.connect(self.open_image_dialog)
        self.btn_init_project.clicked.connect(self.open_import_project_dialog)
        self.btn_open_folder.clicked.connect(self.open_project_dialog)
        self.btn_open_dashboard.clicked.connect(self.open_dashboard)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next_unlabeled.clicked.connect(lambda: self.next_unlabeled_scope(from_start=False))
        self.btn_save.clicked.connect(self.confirm_save_roi)
        self.btn_delete.clicked.connect(self.confirm_delete_roi)
        self.btn_stop.clicked.connect(self.stop_and_save_session)
        self.setAcceptDrops(True)
        # Shortcuts: keep references on self
        self.sc_save = QtGui.QShortcut(QtGui.QKeySequence('S'), self)
        self.sc_save.activated.connect(self.confirm_save_roi)
        self.sc_save2 = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save), self)
        self.sc_save2.activated.connect(self.confirm_save_roi)
        self.sc_next = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self.sc_next.activated.connect(self.next_image)
        self.sc_prev = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self.sc_prev.activated.connect(self.prev_image)
        self.sc_delete = QtGui.QShortcut(QtGui.QKeySequence('D'), self)
        self.sc_delete.activated.connect(self.confirm_delete_roi)
        self.sc_quit = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit), self)
        self.sc_quit.activated.connect(self.close)
        self.statusBar().showMessage('Ready')

        # Menu bar actions
        self._init_menubar()

        # Project root tracking
        self.project_root: Path | None = None
        # Navigation scope: one of 'folder', 'day', 'well', 'project'
        self.nav_scope_type: str = 'folder'
        self.nav_scope_root: Path | None = None
        # Dashboard instance holder
        self.dashboard: ProjectDashboard | None = None

        # Offer resume if we have a previous session
        QtCore.QTimer.singleShot(0, self.offer_resume_last_session)

    # ---------------------------
    # Menu bar
    # ---------------------------
    def _init_menubar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu('&File')
        act_open = QtGui.QAction('Open Image…', self)
        act_open.triggered.connect(self.open_image_dialog)
        file_menu.addAction(act_open)
        act_open_folder = QtGui.QAction('Open Folder…', self)
        act_open_folder.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open_folder)
        file_menu.addSeparator()
        act_resume = QtGui.QAction('Resume Last Session', self)
        act_resume.triggered.connect(self.resume_last_session)
        file_menu.addAction(act_resume)
        file_menu.addSeparator()
        act_quit = QtGui.QAction('Quit', self)
        act_quit.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        proj_menu = mb.addMenu('&Project')
        act_set_root = QtGui.QAction('Set Project Root…', self)
        act_set_root.triggered.connect(self.set_project_root_dialog)
        proj_menu.addAction(act_set_root)
        act_init = QtGui.QAction('Initialize Project (Reorganize)…', self)
        act_init.triggered.connect(self.open_import_project_dialog)
        proj_menu.addAction(act_init)
        act_open_proj = QtGui.QAction('Import Existing Project…', self)
        act_open_proj.triggered.connect(self.open_project_dialog)
        proj_menu.addAction(act_open_proj)
        act_open_dashboard = QtGui.QAction('Open Progress Dashboard', self)
        act_open_dashboard.triggered.connect(self.open_dashboard)
        proj_menu.addAction(act_open_dashboard)
    def confirm_save_roi(self):
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Save', 'Are you sure you want to save this ROI?',
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.save_roi()

    def confirm_delete_roi(self):
        # Enhanced: also offer to remove saved ROI files for accurate progress tracking
        current_file = None
        if self.current_dir and 0 <= self.file_index < len(self.file_list):
            current_file = self.file_list[self.file_index]
        files_removed = []
        if self.shapes is None or len(self.shapes.data) == 0:
            # Even if no shapes exist in the viewer, allow deleting any saved ROI files
            if current_file:
                base = current_file.stem
                roi_json = self.current_dir / f"{base}_roi.json"
                mask_tif = self.current_dir / f"{base}_mask.tif"
                masked_full = self.current_dir / f"{base}_roi_masked.tif"
                masked_crop = self.current_dir / f"{base}_roi_masked_cropped.tif"
                if any(p.exists() for p in (roi_json, mask_tif, masked_full, masked_crop)):
                    reply = QtWidgets.QMessageBox.question(
                        self, 'Delete Saved ROI Files',
                        'No ROI drawn. Delete any existing saved ROI files for this image?',
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    if reply == QtWidgets.QMessageBox.Yes:
                        for p in (roi_json, mask_tif, masked_full, masked_crop):
                            try:
                                if p.exists():
                                    p.unlink(); files_removed.append(p.name)
                            except Exception as e:
                                print(f'[gui] Failed to delete {p}: {e}')
                        if files_removed:
                            QtWidgets.QMessageBox.information(self, 'Deleted', 'Removed: ' + ", ".join(files_removed))
                            self.update_progress_ui()
                            self.persist_session()
                else:
                    QtWidgets.QMessageBox.information(self, 'No ROI', 'No ROI to delete.')
            else:
                QtWidgets.QMessageBox.information(self, 'No ROI', 'No ROI to delete.')
            return
        reply = QtWidgets.QMessageBox.question(
            self, 'Delete ROI',
            'Delete current ROI from viewer and remove any saved ROI files for this image?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.shapes.data = []
            if current_file:
                base = current_file.stem
                roi_json = self.current_dir / f"{base}_roi.json"
                mask_tif = self.current_dir / f"{base}_mask.tif"
                masked_full = self.current_dir / f"{base}_roi_masked.tif"
                masked_crop = self.current_dir / f"{base}_roi_masked_cropped.tif"
                for p in (roi_json, mask_tif, masked_full, masked_crop):
                    try:
                        if p.exists():
                            p.unlink(); files_removed.append(p.name)
                    except Exception as e:
                        print(f'[gui] Failed to delete {p}: {e}')
            msg = 'ROI deleted.' + (f" Removed: {', '.join(files_removed)}" if files_removed else '')
            QtWidgets.QMessageBox.information(self, 'Deleted', msg)
            self.update_progress_ui()
            self.persist_session()
            try:
                if self.dashboard is not None:
                    self.dashboard.refresh()
            except Exception:
                pass
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()
    def dropEvent(self, event):
        urls = [u.toLocalFile() for u in event.mimeData().urls()]
        if urls: self.load_image(Path(urls[0]))
    def open_image_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open TIFF', '', 'TIFF images (*.tif *.tiff)')
        if fn: self.load_image(Path(fn))
    def open_folder_dialog(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', '')
        if dir_path:
            self.load_directory(Path(dir_path))

    # ---------------------------
    # Session & progress helpers
    # ---------------------------
    def local_session_path(self, directory: Path) -> Path:
        return directory / '.organoid_roi_session.json'

    def persist_session(self):
        if not self.current_dir or not self.file_list:
            return
        info = {
            'dir': str(self.current_dir.resolve()),
            'last_index': int(self.file_index),
            'last_file': self.file_list[self.file_index].name if 0 <= self.file_index < len(self.file_list) else None,
            'timestamp': time.time(),
            'total': len(self.file_list),
            'done': self.count_done_in_dir(self.current_dir),
        }
        # Write local session
        _write_json(self.local_session_path(self.current_dir), info)
        # Merge global pointer
        updates = {'last_session': info}
        if self.project_root:
            updates['last_project_root'] = str(self.project_root.resolve())
        _merge_global_state(updates)
        self.statusBar().showMessage('Session saved')

    def offer_resume_last_session(self):
        st = _read_json(_global_state_path())
        if not st or 'last_session' not in st:
            return
        info = st['last_session']
        last_dir = Path(info.get('dir', ''))
        last_file = info.get('last_file', None)
        if not last_dir or not last_dir.exists():
            return
        msg = f"Resume last session in:\n{last_dir}\n"
        if last_file:
            msg += f"Last image: {last_file}"
        reply = QtWidgets.QMessageBox.question(
            self, 'Resume?', msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.load_directory(last_dir, prefer_file=last_file)

    def resume_last_session(self):
        st = _read_json(_global_state_path())
        if not st or 'last_session' not in st:
            QtWidgets.QMessageBox.information(self, 'No Session', 'No previous session found.')
            return
        info = st['last_session']
        last_dir = Path(info.get('dir', ''))
        last_file = info.get('last_file', None)
        if not last_dir or not last_dir.exists():
            QtWidgets.QMessageBox.warning(self, 'Not Found', 'Last session directory not found.')
            return
        self.load_directory(last_dir, prefer_file=last_file)

    def update_progress_ui(self):
        if not self.current_dir or not self.file_list:
            self.progress_label.setText('Progress: 0 / 0')
            self.progress_bar.setValue(0)
            return
        total = len(self.file_list)
        done = self.count_done_in_dir(self.current_dir)
        pct = int(round(100.0 * done / total)) if total > 0 else 0
        self.progress_label.setText(f'Progress: {done} / {total}')
        self.progress_bar.setValue(pct)
        self.statusBar().showMessage(f'Processed {done}/{total}')

    def count_done_in_dir(self, directory: Path) -> int:
        tif_files = list_original_tiffs(directory)
        cnt = 0
        for f in tif_files:
            base = f.stem
            roi_json = directory / f"{base}_roi.json"
            if roi_json.exists():
                cnt += 1
        return cnt

    def find_next_unlabeled(self, start_index: int = 0) -> int:
        if not self.file_list:
            return -1
        n = len(self.file_list)
        for k in range(n):
            idx = (start_index + k) % n
            base = self.file_list[idx].stem
            if not (self.current_dir / f"{base}_roi.json").exists():
                return idx
        return -1

    # ---------------------------
    # Directory loader
    # ---------------------------
    def load_directory(self, directory: Path, prefer_file: str | None = None):
        directory = Path(directory)
        tif_files = list_original_tiffs(directory)
        if not tif_files:
            QtWidgets.QMessageBox.information(self, 'Empty Folder', 'No TIFF images found in this folder.')
            return
        self.current_dir = directory
        self.file_list = tif_files
        pr = infer_project_root_from_path(directory)
        if pr:
            self.project_root = pr
            _merge_global_state({'last_project_root': str(pr.resolve())})
        # Determine starting index
        idx = 0
        # If a specific file is preferred (from resume)
        if prefer_file:
            for i, f in enumerate(self.file_list):
                if f.name == prefer_file:
                    idx = i; break
        else:
            # Try local session index
            sess = _read_json(self.local_session_path(directory)) or {}
            idx = int(sess.get('last_index', 0))
            if not (0 <= idx < len(self.file_list)):
                idx = 0
            # Prefer first unlabeled if available
            nu = self.find_next_unlabeled(start_index=idx)
            if nu != -1:
                idx = nu
        self.file_index = idx
        self.load_image(self.file_list[self.file_index])
        # Progress will be updated in load_image

    def next_unlabeled(self):
        # Backward compatibility: delegate to scope-aware navigation
        if not self.next_unlabeled_scope(from_start=False):
            QtWidgets.QMessageBox.information(self, 'Done', 'No unlabeled images found in the current scope.')

    def set_nav_scope(self, scope_type: str, root: Path | None):
        scope_type = scope_type if scope_type in ('folder', 'day', 'well', 'project') else 'folder'
        self.nav_scope_type = scope_type
        self.nav_scope_root = root
        # Build a friendly scope label
        label = scope_type.capitalize()
        try:
            if scope_type == 'well' and root is not None:
                label = f"Well ({root.name})"
            elif scope_type == 'day' and root is not None:
                label = f"Day ({root.name})"
            elif scope_type == 'project' and root is not None:
                label = f"Project ({Path(root).name})"
        except Exception:
            pass
        self.scope_label.setText(f'Scope: {label}')
        print(f'[gui] Navigation scope set to {scope_type} root={root}')

    def _scope_root_for_current(self) -> Path:
        if self.nav_scope_type == 'folder' or self.nav_scope_root is None:
            return self.current_dir if self.current_dir else Path('.')
        return self.nav_scope_root

    def _images_from_manifest(self, root: Path) -> list[Path]:
        proj = self.project_root or infer_project_root_from_path(root) or root
        manifest = Path(proj) / 'manifest.csv'
        if not manifest.exists():
            return []
        seen = set()
        out: list[Path] = []
        try:
            with manifest.open('r', newline='') as f:
                r = csv.DictReader(f)
                for row in r:
                    p = Path(row.get('new_path', '')).expanduser()
                    if not p or not p.exists():
                        continue
                    if is_derived_tiff_name(p.name):
                        continue
                    if not is_subpath(p, root):
                        continue
                    if p in seen:
                        continue
                    seen.add(p)
                    out.append(p)
        except Exception as e:
            print(f'[gui] Warning: failed to read manifest: {e}')
            return []
        return out

    def _scope_images(self) -> list[Path]:
        if self.nav_scope_type == 'folder' or self.nav_scope_root is None:
            return list_original_tiffs(self.current_dir) if self.current_dir else []
        root = self._scope_root_for_current()
        if self.nav_scope_type == 'project' and self.project_root:
            root = self.project_root
        images = self._images_from_manifest(root)
        if images:
            return images
        return sorted_originals_under(root)

    def next_unlabeled_scope(self, from_start: bool) -> bool:
        images = self._scope_images()
        if not images:
            return False
        start_idx = 0
        if not from_start and self.image_layer is not None and self.current_dir is not None:
            try:
                cur_path = (self.current_dir / self.image_layer.name).resolve()
                start_idx = images.index(cur_path) + 1
            except Exception:
                start_idx = 0
        n = len(images)
        for k in range(n):
            idx = (start_idx + k) % n
            p = images[idx]
            base = p.stem
            if not (p.parent / f"{base}_roi.json").exists():
                self.load_image(p)
                return True
        return False

    def stop_and_save_session(self):
        self.persist_session()
        QtWidgets.QMessageBox.information(self, 'Session Saved', 'Your session has been saved. You can safely close the app and resume later from File → Resume Last Session.')
    def load_image(self, path: Path):
        print(f'[gui] Loading image: {path}')
        # If a derived ROI output was selected, try to redirect to the original image
        if is_derived_tiff_name(path.name):
            base = path.name
            for suf in ('_roi_masked_cropped', '_roi_masked', '_mask'):
                if base.lower().endswith(suf + '.tif') or base.lower().endswith(suf + '.tiff'):
                    orig = path.with_name(path.name[:-(len(suf) + 5)])  # remove suffix + .tif(f)
                    if orig.exists():
                        print(f'[gui] Redirecting derived output to original: {path.name} -> {orig.name}')
                        path = orig
                        break
        try:
            img, backend = load_image_any(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to read image:\n{e}')
            print(f'[gui] ERROR reading image: {e}')
            return
        img = ensure_2d_image(img)
        if img.ndim != 2:
            img = np.squeeze(img)
        print(f'[gui] Standardized to 2D for display. shape={img.shape}')
        self.viewer.layers.clear()
        low, high = float(np.percentile(img,2)), float(np.percentile(img,98))
        self.image_layer = self.viewer.add_image(img, name=path.name, contrast_limits=(low, high))
        self.shapes = self.viewer.add_shapes(name='ROI', shape_type='polygon')
        try:
            # Prefer drawing mode by default for quick ROI creation
            self.shapes.mode = 'add_polygon'
        except Exception:
            pass
        print(f'[gui] Image layer added via {backend}: shape={img.shape}, CL=({low:.2f},{high:.2f})')
        self.current_dir = path.parent
        self.file_list = list_original_tiffs(self.current_dir)
        try: self.file_index = self.file_list.index(path)
        except ValueError: self.file_index = 0
        self.setWindowTitle(f'Organoid ROI Tool — {path.name}')
        print(f'[gui] File browsing initialized in directory: {self.current_dir} ({len(self.file_list)} tif files)')
        # Set default scope to Well when under a project structure
        try:
            parts = path.resolve().parts
            if 'wells' in parts:
                i = parts.index('wells')
                if i + 1 < len(parts):
                    well_dir = Path(*parts[:i+2])
                    # If user hasn't explicitly set Day scope, prefer Well scope
                    if self.nav_scope_type in ('folder', 'project') or not self.nav_scope_root or not is_subpath(path, self.nav_scope_root):
                        self.set_nav_scope('well', well_dir)
        except Exception:
            pass
        # If an ROI JSON already exists, preload it
        base = Path(path).stem
        roi_json = self.current_dir / f"{base}_roi.json"
        if roi_json.exists():
            try:
                with roi_json.open('r') as f:
                    data = json.load(f)
                verts = data.get('vertices_yx')
                if verts and isinstance(verts, list):
                    self.shapes.data = [np.asarray(verts, dtype=float)]
                    self.statusBar().showMessage(f'Preloaded ROI from {roi_json.name}')
                    print(f'[gui] Preloaded ROI from {roi_json}')
            except Exception as e:
                print(f'[gui] Failed to preload ROI {roi_json}: {e}')
        # Update progress and persist current position
        self.update_progress_ui()
        self.persist_session()
    def step(self, delta: int):
        if not self.file_list:
            print('[gui] Step ignored; no files in directory.'); return
        self.file_index = (self.file_index + delta) % len(self.file_list)
        self.load_image(self.file_list[self.file_index])
    def prev_image(self): self.step(-1)
    def next_image(self): self.step(+1)
    def save_roi(self):
        if self.image_layer is None or self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'No image', 'Open an image and draw a polygon first.')
            print('[gui] Save aborted: no image or shapes layer.'); return
        if len(self.shapes.data) == 0:
            QtWidgets.QMessageBox.warning(self, 'No ROI', 'Draw a polygon ROI using the Shapes tool.')
            print('[gui] Save aborted: no polygon present.'); return
        vertices = self.shapes.data[0]
        img = self.image_layer.data
        res = compute_roi(vertices.tolist(), img.shape[:2])
        print(f"[gui] ROI computed: area_px={res.area_px:.2f}, perimeter_px={res.perimeter_px:.2f}, centroid={res.centroid_yx}")
        current_name = self.image_layer.name
        img_dir = self.current_dir
        base = Path(current_name).stem
        roi_json = img_dir / f"{base}_roi.json"
        mask_tif = img_dir / f"{base}_mask.tif"
        tiff.imwrite(str(mask_tif), res.mask.astype(np.uint8)*255)
        save_roi_json(str(roi_json), res.vertices_yx, str(img_dir / Path(current_name)))
        print(f'[gui] Saved: {mask_tif.name}, {roi_json.name}')
        time.sleep(0.1)  # Small delay to ensure file system update
        if not (roi_json.exists() and mask_tif.exists()):
            QtWidgets.QMessageBox.critical(self, 'Save Failed', 'ROI or mask file was not saved correctly!')
            print('[gui] Save failed: file(s) missing.'); return
        # === Save masked TIFFs (full-size + cropped) with associated alpha ===
        # Keeps original pixel dtype. Outside-ROI pixels are NOT set to 0; they are marked "missing"
        # via an associated alpha channel (EXTRASAMPLES=ASSOCALPHA). Many tools (Napari, ImageJ)
        # will honor this as transparency/mask.
        masked_full = img_dir / f"{base}_roi_masked.tif"
        masked_crop = img_dir / f"{base}_roi_masked_cropped.tif"

        img_data = self.image_layer.data
        dtype = img_data.dtype

        # Build alpha channel in SAME dtype as image so dtype is preserved.
        # Use 0 outside ROI, max inside ROI (alpha semantics: 0=transparent, max=opaque).
        if np.issubdtype(dtype, np.integer):
            alpha_max = np.iinfo(dtype).max
        else:
            alpha_max = 1.0
        alpha = (res.mask.astype(dtype) * alpha_max)

        # Stack into (H, W, 2): [image, alpha]
        stacked_full = np.stack([img_data.astype(dtype, copy=False), alpha], axis=-1)

        # Write full-size masked TIFF with associated alpha
        # Note: 'assocalpha' marks the last sample as alpha (missing outside ROI).
        tiff.imwrite(
            str(masked_full),
            stacked_full,
            photometric="minisblack",
            extrasamples="assocalpha",
        )
        print(f"[gui] Saved full-size masked TIFF: {masked_full.name}")

        # Tight crop to ROI bounding box (for smaller files)
        ys, xs = np.where(res.mask)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        cropped_img = img_data[y0:y1, x0:x1]
        cropped_alpha = alpha[y0:y1, x0:x1]
        stacked_crop = np.stack([cropped_img, cropped_alpha], axis=-1)

        tiff.imwrite(
            str(masked_crop),
            stacked_crop,
            photometric="minisblack",
            extrasamples="assocalpha",
        )
        print(f"[gui] Saved cropped masked TIFF: {masked_crop.name}")
        well, day, time_ = parse_from_path(img_dir / current_name)
        parts = (img_dir / current_name).resolve().parts
        proj_root = None
        if 'wells' in parts:
            i = parts.index('wells'); proj_root = Path(*parts[:i])
            self.project_root = proj_root
            _merge_global_state({'last_project_root': str(proj_root.resolve())})
        px_um = read_pixel_size_um(img_dir / current_name)
        row = {'image_path': str((img_dir / current_name).resolve()), 'well': well, 'day': day, 'time': time_,
               'area_px': res.area_px, 'perimeter_px': res.perimeter_px, 'centroid_yx': json.dumps(res.centroid_yx),
               'pixel_size_um': px_um if px_um is not None else ''}
        # Upsert (one row per image) into local and project-level CSVs
        local_csv = img_dir / 'roi_measurements.csv'
        upsert_row(local_csv, row)
        print(f'[gui] Upserted measurements into {local_csv}')
        if proj_root:
            proj_csv = proj_root / 'roi_measurements.csv'
            upsert_row(proj_csv, row)
            print(f'[gui] Upserted measurements into {proj_csv}')
        self.statusBar().showMessage(f'Saved ROI: {roi_json.name}, {mask_tif.name}')
        QtWidgets.QMessageBox.information(self, 'Saved', f'ROI saved:\n{roi_json.name}\n{mask_tif.name}\nMeasurements appended.')
        self.update_progress_ui()
        self.persist_session()
        # Refresh dashboard if open
        try:
            if self.dashboard is not None:
                self.dashboard.refresh()
        except Exception:
            pass
        if self.chk_auto_advance.isChecked():
            self.next_image()
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.persist_session()
        except Exception:
            pass
        super().closeEvent(event)

    # ---------------------------
    # Project root & dashboard
    # ---------------------------
    def set_project_root_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Project Root', str(self.project_root) if self.project_root else '')
        if not d:
            return
        self.project_root = Path(d)
        _merge_global_state({'last_project_root': str(self.project_root.resolve())})

    def open_dashboard(self):
        pr = self.project_root
        if pr is None:
            # Try from global state
            st = _read_json(_global_state_path()) or {}
            pr_txt = st.get('last_project_root', '')
            if pr_txt:
                pr = Path(pr_txt)
        if pr is None or not pr.exists():
            QtWidgets.QMessageBox.information(self, 'Select Project', 'Please set the project root (Project → Set Project Root…)')
            return
        self.dashboard = ProjectDashboard(self, pr)
        self.dashboard.show()

    # ---------------------------
    # Import Project (wrap reorganize.py)
    # ---------------------------
    def open_import_project_dialog(self):
        dlg = ImportProjectDialog(self)
        dlg.exec()

    def open_project_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Project Root (with wells/)', str(self.project_root) if self.project_root else '')
        if not d:
            return
        pr = Path(d)
        if not (pr / 'wells').exists():
            QtWidgets.QMessageBox.warning(self, 'Invalid Project', 'Selected folder does not contain a wells/ directory.')
            return
        self.project_root = pr
        _merge_global_state({'last_project_root': str(pr.resolve())})
        self.set_nav_scope('project', pr)
        # Try to jump to first unlabeled; else open dashboard
        nu = self.find_first_unlabeled_in_project(pr)
        if nu is not None:
            try:
                self.load_image(nu)
                return
            except Exception:
                pass
        self.open_dashboard()

    def find_first_unlabeled_in_project(self, project_root: Path) -> Path | None:
        wells_dir = project_root / 'wells'
        if not wells_dir.exists():
            return None
        for well_dir in sorted([p for p in wells_dir.iterdir() if p.is_dir()]):
            all_tifs = sorted(well_dir.rglob('*.tif')) + sorted(well_dir.rglob('*.tiff'))
            for tif in [p for p in all_tifs if not is_derived_tiff_name(p.name)]:
                base = tif.stem
                if not (tif.parent / f"{base}_roi.json").exists():
                    return tif
        return None

class ImportProjectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Import Project (Reorganize Raw Files)')
        self.resize(700, 520)
        v = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QGridLayout()
        self.ed_raw = QtWidgets.QLineEdit()
        self.btn_raw = QtWidgets.QPushButton('Browse…')
        self.ed_out = QtWidgets.QLineEdit()
        self.btn_out = QtWidgets.QPushButton('Browse…')
        self.sp_min_col = QtWidgets.QSpinBox(); self.sp_min_col.setRange(1, 12); self.sp_min_col.setValue(1)
        self.ed_rows = QtWidgets.QLineEdit('ABCDEFGH')
        self.chk_copy = QtWidgets.QCheckBox('Copy files (default: move)')
        self.chk_dry = QtWidgets.QCheckBox('Dry run (no changes)')
        r = 0
        form.addWidget(QtWidgets.QLabel('Raw folder (flat):'), r, 0)
        form.addWidget(self.ed_raw, r, 1)
        form.addWidget(self.btn_raw, r, 2); r += 1
        form.addWidget(QtWidgets.QLabel('Project root (output):'), r, 0)
        form.addWidget(self.ed_out, r, 1)
        form.addWidget(self.btn_out, r, 2); r += 1
        form.addWidget(QtWidgets.QLabel('Rows (A–H):'), r, 0)
        form.addWidget(self.ed_rows, r, 1); r += 1
        form.addWidget(QtWidgets.QLabel('Min column:'), r, 0)
        form.addWidget(self.sp_min_col, r, 1); r += 1
        form.addWidget(self.chk_copy, r, 1); r += 1
        form.addWidget(self.chk_dry, r, 1); r += 1
        v.addLayout(form)
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        v.addWidget(self.log, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton('Run Import')
        self.btn_close = QtWidgets.QPushButton('Close')
        buttons.addStretch(1)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_close)
        v.addLayout(buttons)
        self.btn_close.clicked.connect(self.close)
        self.btn_raw.clicked.connect(self.pick_raw)
        self.btn_out.clicked.connect(self.pick_out)
        self.btn_run.clicked.connect(self.run)
        # Suggest defaults based on parent state
        try:
            st = _read_json(_global_state_path()) or {}
            last_pr = st.get('last_project_root')
            if last_pr:
                self.ed_out.setText(last_pr)
        except Exception:
            pass
        self.thread = None

    def pick_raw(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Pick Raw Folder (flat)')
        if d:
            self.ed_raw.setText(d)

    def pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Pick Project Root (output)')
        if d:
            self.ed_out.setText(d)

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    def run(self):
        raw = self.ed_raw.text().strip()
        out = self.ed_out.text().strip()
        if not raw or not out:
            QtWidgets.QMessageBox.information(self, 'Missing', 'Please select both raw and output folders.')
            return
        rows = self.ed_rows.text().strip() or 'ABCDEFGH'
        min_col = int(self.sp_min_col.value())
        copy = self.chk_copy.isChecked()
        dry = self.chk_dry.isChecked()
        self.log.clear()
        self.append_log('[import] Starting...')
        # Run in a thread to keep UI responsive
        self.thread = ImportWorker(raw, out, copy, dry, min_col, rows)
        self.thread.sig_log.connect(self.append_log)
        self.thread.sig_done.connect(self.on_done)
        self.btn_run.setEnabled(False)
        self.thread.start()

    def on_done(self, ok: bool, result: dict | None, error: str | None):
        self.btn_run.setEnabled(True)
        if not ok:
            QtWidgets.QMessageBox.critical(self, 'Import Failed', error or 'Unknown error')
            return
        self.append_log('[import] Completed.')
        # Update parent project root and offer to open dashboard
        try:
            out_root = Path(result.get('out_root'))
            parent = self.parent() if isinstance(self.parent(), OrganoidROIApp) else None
            if parent is not None:
                parent.project_root = out_root
                _merge_global_state({'last_project_root': str(out_root.resolve())})
                parent.set_nav_scope('project', out_root)
                if QtWidgets.QMessageBox.question(self, 'Open Dashboard?', 'Import complete. Open project dashboard?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
                    parent.open_dashboard()
        except Exception:
            pass

class ImportWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)
    sig_done = QtCore.Signal(bool, dict, str)
    def __init__(self, raw, out, copy, dry, min_col, rows):
        super().__init__()
        self.raw = raw; self.out = out; self.copy = copy; self.dry = dry; self.min_col = min_col; self.rows = rows
    def run(self):
        try:
            # Import here to avoid top-level dependency at module import
            import reorganize as r
            result = r.organize(self.raw, self.out, copy=self.copy, dry_run=self.dry, min_col=self.min_col, rows=self.rows, log=lambda m: self.sig_log.emit(m))
            self.sig_done.emit(True, result, '')
        except Exception as e:
            self.sig_log.emit(f'[import] ERROR: {e}')
            self.sig_done.emit(False, None, str(e))
def main():
    print('[gui] Launching Qt app...')
    app = QtWidgets.QApplication(sys.argv)
    win = OrganoidROIApp(); win.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()
