"""Interactive viewer for DCX spot outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import tifffile

from .utils import _read_tiff  # reuse tif loader


@dataclass
class DCXSet:
    base: str
    folder: Path
    labels_path: Path
    mcherry_path: Path
    bf_path: Path | None
    overlay_path: Path
    spots_path: Path


def _is_labels_file(p: Path) -> bool:
    return p.name.endswith("_labels.tif")


def _find_sets_in_folder(folder: Path) -> List[DCXSet]:
    items: List[DCXSet] = []
    for labels_path in sorted(folder.glob("*_labels.tif")):
        base = labels_path.stem.replace("_labels", "")
        mcherry = labels_path.with_name(f"{base}_mcherry_masked.tif")
        bf = labels_path.with_name(f"{base}_bf_masked.tif")
        overlay = labels_path.with_name(f"{base}_overlay_ids.png")
        spots = labels_path.with_name(f"{base}_spots.csv")
        if not (mcherry.exists() and overlay.exists() and spots.exists()):
            continue
        if not bf.exists():
            bf = None
        items.append(DCXSet(
            base=base,
            folder=folder,
            labels_path=labels_path,
            mcherry_path=mcherry,
            bf_path=bf,
            overlay_path=overlay,
            spots_path=spots,
        ))
    return items


def discover_sets(target: Path) -> List[DCXSet]:
    target = target.expanduser().resolve()
    if target.is_file():
        if _is_labels_file(target):
            return _find_sets_in_folder(target.parent)
        return []
    if target.is_dir():
        if any(_is_labels_file(p) for p in target.glob("*_labels.tif")):
            return _find_sets_in_folder(target)
        # search recursively for dcxspot folders (unique parents)
        parents = {p.parent for p in target.rglob("*_labels.tif")}
        items: List[DCXSet] = []
        for parent in sorted(parents):
            items.extend(_find_sets_in_folder(parent))
        return items
    return []


# GUI implementation will follow later in this module

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import segmentation


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
    def __init__(self, initial_sets: list[DCXSet] | None = None):
        super().__init__()
        self.title("DCX Spot Browser")
        self.geometry("1200x720")

        self.sets: list[DCXSet] = initial_sets or []
        self.current_set: DCXSet | None = None
        self.labels: Optional[np.ndarray] = None
        self.mcherry: Optional[np.ndarray] = None
        self.boundaries: Optional[np.ndarray] = None
        self.measurements: Optional[pd.DataFrame] = None
        self.overlay_img: Optional[np.ndarray] = None

        self._build_ui()
        if self.sets:
            self._populate_listbox(self.sets)
            self.listbox.selection_set(0)
            self._load_selected_set(0)

    def _build_ui(self):
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Open…", command=self._open_dialog).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_current_path).pack(side=tk.RIGHT)

        self.listbox = tk.Listbox(left, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.listbox.bind("<<ListboxSelect>>", self._on_select_set)

        control = ttk.Frame(right)
        control.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(control, text="Cluster ID:").pack(side=tk.LEFT)
        self.cluster_var = tk.IntVar(value=1)
        self.cluster_spin = ttk.Spinbox(
            control,
            from_=1,
            to=1,
            textvariable=self.cluster_var,
            command=self._on_cluster_change,
            width=6,
            wrap=True,
        )
        self.cluster_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Prev", command=lambda: self._bump_cluster(-1)).pack(side=tk.LEFT)
        ttk.Button(control, text="Next", command=lambda: self._bump_cluster(1)).pack(side=tk.LEFT)

        self.info_var = tk.StringVar(value="Select an image")
        ttk.Label(control, textvariable=self.info_var).pack(side=tk.RIGHT)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.fig.tight_layout()

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="Select a labels file",
            filetypes=[("Labels", "*_labels.tif"), ("All", "*")],
        )
        if path:
            self.load_from_path(Path(path))

    def _refresh_current_path(self):
        if self.current_set is None:
            return
        self.load_from_path(self.current_set.labels_path)

    def load_from_path(self, path: Path):
        sets = discover_sets(path)
        if not sets:
            messagebox.showerror("No DCX outputs found", f"Could not find DCX outputs under {path}")
            return
        self.sets = sets
        self._populate_listbox(self.sets)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(0)
        self._load_selected_set(0)
        parent = sets[0].folder
        self.status_var.set(str(parent))

    def _populate_listbox(self, sets: list[DCXSet]):
        self.listbox.delete(0, tk.END)
        for item in sets:
            rel = item.folder.name if item.folder.name not in ("dcxspot",) else item.base
            display = f"{item.base}"
            self.listbox.insert(tk.END, display)

    def _on_select_set(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self._load_selected_set(index)

    def _load_selected_set(self, index: int):
        if index < 0 or index >= len(self.sets):
            return
        item = self.sets[index]
        self.current_set = item
        self.labels = tifffile.imread(item.labels_path).astype(np.int32)
        if self.labels.max() == 0:
            messagebox.showinfo("No clusters", f"No clusters found in {item.labels_path.name}")
        self.mcherry = _read_tiff(item.mcherry_path).astype(np.float32)
        self.measurements = pd.read_csv(item.spots_path)
        self.boundaries = segmentation.find_boundaries(self.labels, mode="inner")
        try:
            self.overlay_img = plt.imread(item.overlay_path)
        except Exception:
            self.overlay_img = None

        max_cluster = int(self.labels.max()) if self.labels is not None else 1
        if max_cluster < 1:
            max_cluster = 1
        self.cluster_spin.configure(to=max_cluster)
        self.cluster_var.set(1)
        self._update_display()

    def _on_cluster_change(self):
        self._update_display()

    def _bump_cluster(self, delta: int):
        current = self.cluster_var.get()
        new_value = current + delta
        maximum = int(self.cluster_spin.cget("to"))
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
        self.axes[0].clear()
        self.axes[0].imshow(mcherry_disp, cmap="gray")
        if self.boundaries is not None:
            self.axes[0].imshow(np.ma.masked_where(~self.boundaries, self.boundaries), cmap="autumn", alpha=0.6)
        mask = self.labels == cluster_id
        self.axes[0].imshow(np.ma.masked_where(~mask, mask), cmap="cool", alpha=0.6)
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
            self.axes[2].set_title("Overlay with IDs")
        else:
            self.axes[2].text(0.5, 0.5, "Overlay unavailable", ha="center", va="center")
            self.axes[2].set_title("Overlay with IDs")
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
    sets: list[DCXSet] = []
    if path is not None:
        sets = discover_sets(path)
    app = DCXBrowserApp(sets)
    if not sets and path is not None:
        app.status_var.set(f"No DCX outputs found in {path}")
    app.mainloop()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, help="Optional path to a dcxspot folder or labels file")
    return ap.parse_args()


def main():
    args = parse_args()
    launch_browser(args.path)


if __name__ == "__main__":
    main()
