"""Interactive viewer for DCX spot outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import segmentation

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .utils import _read_tiff

CONFIG_PATH = Path(__file__).resolve().parent.parent / "dcxspot_config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def load_default_target(cfg: Optional[dict] = None) -> Optional[Path]:
    cfg = cfg or load_config()
    project_root = cfg.get("project_root")
    if not project_root:
        return None
    project_root = Path(project_root)
    store_within = cfg.get("store_within_project", True)
    if store_within:
        return project_root
    output_root = cfg.get("output_root")
    if output_root:
        return Path(output_root)
    return project_root


@dataclass
class DCXSet:
    base: str
    folder: Path
    labels_path: Path
    mcherry_path: Path
    overlay_path: Path
    spots_path: Path
    qc_path: Path


def _find_sets_in_folder(folder: Path) -> List[DCXSet]:
    sets: List[DCXSet] = []
    for labels_path in sorted(folder.glob("*_labels.tif")):
        base = labels_path.stem.replace("_labels", "")
        mcherry = labels_path.with_name(f"{base}_mcherry_masked.tif")
        overlay = labels_path.with_name(f"{base}_overlay_ids.png")
        spots = labels_path.with_name(f"{base}_spots.csv")
        qc = labels_path.with_name(f"{base}_qc.json")
        if not (mcherry.exists() and overlay.exists() and spots.exists() and qc.exists()):
            continue
        sets.append(
            DCXSet(
                base=base,
                folder=folder,
                labels_path=labels_path,
                mcherry_path=mcherry,
                overlay_path=overlay,
                spots_path=spots,
                qc_path=qc,
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


def _normalize_for_segmentation(data: np.ndarray, roi: np.ndarray, percentiles: Tuple[float, float]) -> np.ndarray:
    lo, hi = percentiles
    img = np.nan_to_num(data, nan=0.0)
    if lo == 0 and hi == 0:
        return img.astype(np.float32)
    roi_vals = img[roi]
    if roi_vals.size == 0:
        return img.astype(np.float32)
    lo_val, hi_val = np.percentile(roi_vals, [lo, hi])
    if hi_val <= lo_val:
        return img.astype(np.float32)
    seg = (img - lo_val) / (hi_val - lo_val)
    return np.clip(seg, 0.0, 1.0).astype(np.float32)


def _cluster_crop(mask: np.ndarray, pad: int = 5) -> Tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]))
    y0 = max(0, ys.min() - pad)
    y1 = min(mask.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask.shape[1], xs.max() + pad + 1)
    return (slice(y0, y1), slice(x0, x1))


def _upsample(arr: np.ndarray, factor: int = 4) -> np.ndarray:
    if arr.size == 0:
        return arr
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


class DCXBrowserApp(tk.Tk):
    def __init__(self, sets: List[DCXSet], origin: Optional[Path], config: Optional[dict] = None):
        super().__init__()
        self.title("DCX Spot Browser")
        self.geometry("1320x820")

        self.origin_path = origin
        self.sets = sets
        self.current_set: Optional[DCXSet] = None
        self.labels: Optional[np.ndarray] = None
        self.mcherry: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.segmentation: Optional[np.ndarray] = None
        self.otsu_threshold: float = 0.0
        self.binary_map: Optional[np.ndarray] = None
        self.measurements: Optional[pd.DataFrame] = None
        self.config_data = config or {}
        self.upsample_factor = 4
        self.scale_bar_px = int(self.config_data.get("scale_bar_px", 100))
        self.scale_bar_label = self.config_data.get("scale_bar_label")

        self._build_ui()
        if self.sets:
            self._populate_listbox()
            self.listbox.selection_set(0)
            self._load_selected_set(0)
        else:
            default_target = load_default_target(self.config_data)
            if default_target is not None:
                self.status_var.set(f"No dcxspot folders found under {default_target}")
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

        self.fit_to_roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl,
            text="Fit to ROI",
            variable=self.fit_to_roi_var,
            command=self._update_display,
        ).pack(side=tk.LEFT, padx=12)

        spacer = ttk.Frame(ctrl)
        spacer.pack(side=tk.LEFT, expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12))
        self.fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.9, hspace=0.25, wspace=0.25)
        self.cbar_axes = [
            self.fig.add_axes([0.08, 0.92, 0.37, 0.02]),  # raw
            self.fig.add_axes([0.55, 0.92, 0.37, 0.02]),  # otsu
        ]
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
        self.roi_mask = np.isfinite(self.mcherry)
        roi_mask = self.roi_mask

        try:
            qc = json.loads(item.qc_path.read_text())
        except Exception:
            qc = {}
        self.otsu_threshold = float(qc.get("otsu_threshold", 0.0))
        normalize_percentiles = tuple(qc.get("normalize_percentiles", (0.0, 0.0)))

        lo, hi = normalize_percentiles
        roi_vals = self.mcherry[roi_mask]
        lo_val = hi_val = None
        if roi_vals.size > 0 and not (lo == 0 and hi == 0):
            lo_val, hi_val = np.percentile(roi_vals, [lo, hi])
            if hi_val <= lo_val:
                lo_val = hi_val = None

        if lo_val is not None and hi_val is not None:
            seg_norm = (self.mcherry - lo_val) / (hi_val - lo_val)
            seg_norm = np.clip(seg_norm, 0.0, 1.0)
            seg_disp = seg_norm * (hi_val - lo_val) + lo_val
        else:
            seg_norm = self.mcherry.astype(np.float32)
            seg_disp = self.mcherry.astype(np.float32)

        self.segmentation_norm = seg_norm
        self.segmentation_display = seg_disp
        self.binary_map = (self.segmentation_norm > self.otsu_threshold) & roi_mask

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

    def _get_roi_crop(self, pad: int = 10) -> Tuple[slice, slice]:
        if self.roi_mask is None:
            return (slice(None), slice(None))
        ys, xs = np.where(self.roi_mask)
        if ys.size == 0 or xs.size == 0:
            return (slice(None), slice(None))
        y0 = max(0, ys.min() - pad)
        y1 = min(self.roi_mask.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad)
        x1 = min(self.roi_mask.shape[1], xs.max() + pad + 1)
        return (slice(y0, y1), slice(x0, x1))

    def _draw_scale_bar(self, ax, array_shape):
        if self.scale_bar_px <= 0:
            return
        upsample = self.upsample_factor
        length = self.scale_bar_px * upsample
        height = max(4, upsample)
        margin = 20
        img_h, img_w = array_shape
        x0 = margin
        y0 = img_h - margin - height
        rect = patches.Rectangle((x0, y0), length, height, color="white", linewidth=0)
        ax.add_patch(rect)
        label = self.scale_bar_label or f"{self.scale_bar_px} px"
        ax.text(
            x0 + length / 2,
            y0 - 8,
            label,
            color="white",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
        )

    def _update_display(self):
        if self.labels is None or self.mcherry is None or self.segmentation_display is None:
            return
        cluster_id = self.cluster_var.get()
        max_cluster = int(self.cluster_spin.cget("to"))
        if cluster_id < 1 or cluster_id > max_cluster:
            return

        cluster_mask = self.labels == cluster_id

        fit_roi = bool(self.fit_to_roi_var.get())
        crop = self._get_roi_crop(pad=10) if fit_roi else (slice(None), slice(None))
        suffix = "ROI crop ×4" if fit_roi else "×4"

        raw_disp = np.nan_to_num(self.mcherry[crop], nan=0.0)
        seg_disp = np.nan_to_num(self.segmentation_display[crop], nan=0.0)
        boundary = segmentation.find_boundaries(cluster_mask, mode="inner").astype(float)[crop]

        factor = self.upsample_factor
        raw_zoom = _upsample(raw_disp, factor=factor)
        seg_zoom = _upsample(seg_disp, factor=factor)
        boundary_zoom = _upsample(boundary, factor=factor)

        ax_raw = self.axes[0, 0]
        ax_seg = self.axes[0, 1]
        ax_mask = self.axes[1, 0]
        ax_overlay = self.axes[1, 1]

        ax_raw.clear()
        raw_min = float(np.min(raw_disp))
        raw_max = float(np.max(raw_disp))
        if raw_min == raw_max:
            raw_max = raw_min + 1.0
        im_raw = ax_raw.imshow(raw_zoom, cmap="gray", vmin=raw_min, vmax=raw_max)
        ax_raw.set_title(f"Raw ({suffix})")
        ax_raw.axis("off")
        ax_raw.set_anchor("C")
        self._draw_scale_bar(ax_raw, raw_zoom.shape[:2])

        ax_seg.clear()
        seg_min = float(np.min(seg_disp))
        seg_max = float(np.max(seg_disp))
        if seg_min == seg_max:
            seg_max = seg_min + 1.0
        im_seg = ax_seg.imshow(seg_zoom, cmap="magma", vmin=seg_min, vmax=seg_max)
        ax_seg.set_title(f"Otsu normalization ({suffix})")
        ax_seg.axis("off")
        ax_seg.set_anchor("C")
        self._draw_scale_bar(ax_seg, seg_zoom.shape[:2])

        ax_mask.clear()
        ax_mask.imshow(boundary_zoom, cmap="viridis")
        ax_mask.set_title(f"DCX mask boundary ({suffix})")
        ax_mask.axis("off")
        ax_mask.set_anchor("C")

        ax_overlay.clear()
        ax_overlay.imshow(seg_zoom, cmap="magma", vmin=seg_min, vmax=seg_max)
        ax_overlay.imshow(np.ma.masked_where(boundary_zoom == 0, boundary_zoom), cmap="cool", alpha=0.6)
        ax_overlay.set_title(f"Mask on Otsu ({suffix})")
        ax_overlay.axis("off")
        ax_overlay.set_anchor("C")
        self._draw_scale_bar(ax_overlay, seg_zoom.shape[:2])

        cbar_raw_ax, cbar_seg_ax = self.cbar_axes
        for cax in self.cbar_axes:
            cax.cla()
        cbar_raw = self.fig.colorbar(im_raw, cax=cbar_raw_ax, orientation="horizontal")
        cbar_raw.ax.set_xlabel("Raw intensity (pixel value)")
        cbar_seg = self.fig.colorbar(im_seg, cax=cbar_seg_ax, orientation="horizontal")
        cbar_seg.ax.set_xlabel("Otsu input intensity")

        self.canvas.draw_idle()

        info = f"Cluster {cluster_id}/{max_cluster} | otsu={self.otsu_threshold:.4g}"
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
    config = load_config()
    target = path or load_default_target(config)
    sets = discover_sets(target)
    origin = None
    if target is not None:
        origin = target if target.is_dir() else target.parent
    app = DCXBrowserApp(sets, origin, config=config)
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
