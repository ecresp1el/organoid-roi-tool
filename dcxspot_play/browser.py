"""Interactive viewer for DCX spot outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import segmentation

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .utils import _read_tiff

CONFIG_PATH = Path(__file__).resolve().parent.parent / "dcxspot_config.json"


def load_default_target() -> Optional[Path]:
    if not CONFIG_PATH.exists():
        return None
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except Exception:
        return None
    project_root = cfg.get("project_root")
    if not project_root:
        return None
    return Path(project_root)


@dataclass
class DCXSet:
    base: str
    folder: Path
    labels_path: Path
    mcherry_path: Path
    overlay_path: Path
    spots_path: Path


def _find_sets_in_folder(folder: Path) -> List[DCXSet]:
    sets: List[DCXSet] = []
    for labels_path in sorted(folder.glob("*_labels.tif")):
        base = labels_path.stem.replace("_labels", "")
        mcherry = labels_path.with_name(f"{base}_mcherry_masked.tif")
        overlay = labels_path.with_name(f"{base}_overlay_ids.png")
        spots = labels_path.with_name(f"{base}_spots.csv")
        if not (mcherry.exists() and overlay.exists() and spots.exists()):
            continue
        sets.append(
            DCXSet(
                base=base,
                folder=folder,
                labels_path=labels_path,
                mcherry_path=mcherry,
                overlay_path=overlay,
                spots_path=spots,
            )
        )
    return sets


def discover_sets(target: Optional[Path]) -> List[DCXSet]:
    if target is None:
        return []
    target = target.expanduser().resolve()
    if target.is_file():
        if target.name.endswith("_labels.tif"):
            return _find_sets_in_folder(target.parent)
        return []
    if target.is_dir():
        if any(p.name.endswith("_labels.tif") for p in target.glob("*_labels.tif")):
            return _find_sets_in_folder(target)
        parents = {p.parent for p in target.rglob("*_labels.tif")}
        sets: List[DCXSet] = []
        for parent in sorted(parents):
            sets.extend(_find_sets_in_folder(parent))
        return sets
    return []


def _percentile_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    stretched = (img - vmin) / (vmax - vmin)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


class DCXBrowserApp(tk.Tk):
    def __init__(self, sets: List[DCXSet], origin: Optional[Path]):
        super().__init__()
        self.title("DCX Spot Browser")
        self.geometry("1320x780")

        self.origin_path = origin
        self.sets = sets
        self.current_set: Optional[DCXSet] = None
        self.labels: Optional[np.ndarray] = None
        self.mcherry: Optional[np.ndarray] = None
        self.boundaries: Optional[np.ndarray] = None
        self.overlay_img: Optional[np.ndarray] = None
        self.measurements: Optional[pd.DataFrame] = None

        self._build_ui()
        if self.sets:
            self._populate_listbox()
            self.listbox.selection_set(0)
            self._load_selected_set(0)
        else:
            self.status_var.set("Use Open folder… to select a dcxspot directory")

    def _build_ui(self):
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(main, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        content = ttk.Frame(main)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(sidebar)
        btn_row.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_row, text="Open folder…", command=self._open_dialog).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Refresh", command=self._refresh_current).pack(side=tk.RIGHT)

        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.listbox = tk.Listbox(list_frame, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_set)

        ctrl = ttk.Frame(content)
        ctrl.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(ctrl, text="Cluster:").pack(side=tk.LEFT)
        self.cluster_var = tk.IntVar(value=1)
        self.cluster_spin = ttk.Spinbox(ctrl, from_=1, to=1, width=6, textvariable=self.cluster_var, command=self._on_cluster_change)
        self.cluster_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Prev", command=lambda: self._step_cluster(-1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Next", command=lambda: self._step_cluster(1)).pack(side=tk.LEFT, padx=(2, 10))

        self.info_var = tk.StringVar(value="Select an entry")
        ttk.Label(ctrl, textvariable=self.info_var).pack(side=tk.LEFT, padx=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def _open_dialog(self):
        path = filedialog.askdirectory(title="Select a dcxspot folder")
        if path:
            self.load_from_path(Path(path))

    def _refresh_current(self):
        if self.origin_path is None:
            return
        self.load_from_path(self.origin_path)

    def load_from_path(self, path: Path):
        sets = discover_sets(path)
        if not sets:
            messagebox.showerror("No DCX outputs", f"Could not find DCX results under {path}")
            return
        self.origin_path = path if path.is_dir() else path.parent
        self.sets = sets
        self._populate_listbox()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(0)
        self._load_selected_set(0)
        self.status_var.set(str(self.origin_path))

    def _populate_listbox(self):
        self.listbox.delete(0, tk.END)
        origin = self.origin_path
        for item in self.sets:
            label_path = item.labels_path
            if origin and origin in label_path.parents:
                display = str(label_path.relative_to(origin))
            else:
                display = str(label_path)
            self.listbox.insert(tk.END, display)

    def _on_select_set(self, _event=None):
        sel = self.listbox.curselection()
        if sel:
            self._load_selected_set(sel[0])

    def _load_selected_set(self, index: int):
        if index < 0 or index >= len(self.sets):
            return
        item = self.sets[index]
        self.current_set = item
        self.labels = tifffile.imread(item.labels_path).astype(np.int32)
        self.mcherry = _read_tiff(item.mcherry_path).astype(np.float32)
        self.boundaries = segmentation.find_boundaries(self.labels, mode="inner")
        try:
            self.overlay_img = plt.imread(item.overlay_path)
        except Exception:
            self.overlay_img = None
        try:
            self.measurements = pd.read_csv(item.spots_path)
        except Exception:
            self.measurements = None

        max_cluster = int(self.labels.max()) if self.labels is not None else 1
        if max_cluster < 1:
            max_cluster = 1
        self.cluster_spin.configure(to=max_cluster)
        self.cluster_var.set(1)
        self._update_display()

    def _on_cluster_change(self):
        self._update_display()

    def _step_cluster(self, delta: int):
        current = self.cluster_var.get()
        maximum = int(self.cluster_spin.cget("to"))
        new_value = current + delta
        if new_value < 1:
            new_value = maximum
        elif new_value > maximum:
            new_value = 1
        self.cluster_var.set(new_value)
        self._update_display()

    def _update_display(self):
        if self.labels is None or self.mcherry is None:
            return
        cluster_id = self.cluster_var.get()
        max_cluster = int(self.cluster_spin.cget("to"))
        if cluster_id < 1 or cluster_id > max_cluster:
            return

        mcherry_disp = _percentile_stretch(self.mcherry)
        mask = self.labels == cluster_id

        self.axes[0].clear()
        self.axes[0].imshow(mcherry_disp, cmap="gray")
        if self.boundaries is not None:
            self.axes[0].imshow(np.ma.masked_where(~self.boundaries, self.boundaries), cmap="autumn", alpha=0.6)
        self.axes[0].imshow(np.ma.masked_where(~mask, mask), cmap="cool", alpha=0.5)
        self.axes[0].set_title("mCherry + cluster highlight")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(mask, cmap="viridis")
        self.axes[1].set_title(f"Cluster {cluster_id} mask")
        self.axes[1].axis("off")

        self.axes[2].clear()
        if self.overlay_img is not None:
            self.axes[2].imshow(self.overlay_img)
            self.axes[2].imshow(np.ma.masked_where(~mask, mask), cmap="cool", alpha=0.3)
            self.axes[2].set_title("Overlay (IDs)")
        else:
            self.axes[2].text(0.5, 0.5, "Overlay unavailable", ha="center", va="center")
            self.axes[2].axis("off")
        self.axes[2].axis("off")

        self.canvas.draw_idle()

        info = f"Cluster {cluster_id}/{max_cluster}"
        if self.measurements is not None and not self.measurements.empty:
            row = self.measurements[self.measurements["cluster_id"] == cluster_id]
            if not row.empty:
                r = row.iloc[0]
                info += (
                    f" | area={r['area_px']:.0f}px"
                    f" mean={r['mean_intensity']:.1f}"
                    f" sum={r['sum_intensity']:.1f}"
                )
        self.info_var.set(info)


def launch_browser(path: Optional[Path] = None):
    target = path or load_default_target()
    sets = discover_sets(target)
    origin = None
    if target is not None:
        origin = target if target.is_dir() else target.parent
    app = DCXBrowserApp(sets, origin)
    if not sets and target is not None:
        app.status_var.set(f"No DCX outputs found in {target}")
    app.mainloop()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, help="Optional path to dcxspot folder or labels file")
    return ap.parse_args()


def main():
    args = parse_args()
    launch_browser(args.path)


if __name__ == "__main__":
    main()
