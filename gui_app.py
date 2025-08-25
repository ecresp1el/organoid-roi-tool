#!/usr/bin/env python
import sys, os, csv, json, traceback
from pathlib import Path
import numpy as np
import tifffile as tiff
import pandas as pd
print("[gui] Starting Organoid ROI Tool v8...")
print(f"[gui] Python: {sys.version}")
try:
    from PySide6 import QtWidgets, QtCore
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
    return tiff.imread(str(path))
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
    arr = np.stack(frames, axis=0)
    if arr.ndim == 4:
        rgb = arr[..., :3].astype(np.float32)
        gray = (0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2])
        arr = gray
    return arr
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
class OrganoidROIApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Organoid ROI Tool v8')
        self.resize(1200, 800)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        btn_bar = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton('Open Image')
        self.btn_prev = QtWidgets.QPushButton('Prev')
        self.btn_next = QtWidgets.QPushButton('Next')
        self.btn_save = QtWidgets.QPushButton('Save ROI')
        btn_bar.addWidget(self.btn_open); btn_bar.addStretch(1)
        btn_bar.addWidget(self.btn_prev); btn_bar.addWidget(self.btn_next); btn_bar.addWidget(self.btn_save)
        layout.addLayout(btn_bar)
        self.viewer = napari.Viewer()
        self.viewer.window._qt_window.setWindowFlag(QtCore.Qt.Widget, True)
        layout.addWidget(self.viewer.window._qt_window)
        print('[gui] Napari viewer created.')
        self.current_dir = None; self.file_list = []; self.file_index = -1
        self.image_layer = None; self.shapes = None
        self.btn_open.clicked.connect(self.open_image_dialog)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save.clicked.connect(self.save_roi)
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()
    def dropEvent(self, event):
        urls = [u.toLocalFile() for u in event.mimeData().urls()]
        if urls: self.load_image(Path(urls[0]))
    def open_image_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open TIFF', '', 'TIFF images (*.tif *.tiff)')
        if fn: self.load_image(Path(fn))
    def load_image(self, path: Path):
        print(f'[gui] Loading image: {path}')
        try:
            img, backend = load_image_any(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to read image:\n{e}')
            print(f'[gui] ERROR reading image: {e}')
            return
        if img.ndim > 2:
            try:
                img = np.max(img, axis=0); print('[gui] Max projection applied.')
            except Exception:
                img = img.squeeze(); print('[gui] Squeeze fallback used.')
        self.viewer.layers.clear()
        low, high = float(np.percentile(img,2)), float(np.percentile(img,98))
        self.image_layer = self.viewer.add_image(img, name=path.name, contrast_limits=(low, high))
        self.shapes = self.viewer.add_shapes(name='ROI', shape_type='polygon')
        print(f'[gui] Image layer added via {backend}: shape={img.shape}, CL=({low:.2f},{high:.2f})')
        self.current_dir = path.parent
        self.file_list = sorted([p for p in self.current_dir.glob('*.tif')])
        try: self.file_index = self.file_list.index(path)
        except ValueError: self.file_index = 0
        self.setWindowTitle(f'Organoid ROI Tool — {path.name}')
        print(f'[gui] File browsing initialized in directory: {self.current_dir} ({len(self.file_list)} tif files)')
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
        base = Path(current_name).with_suffix('')
        roi_json = img_dir / f"{base}_roi.json"
        mask_tif = img_dir / f"{base}_mask.tif"
        tiff.imwrite(str(mask_tif), res.mask.astype(np.uint8)*255)
        save_roi_json(str(roi_json), res.vertices_yx, str(img_dir / Path(current_name)))
        print(f'[gui] Saved: {mask_tif.name}, {roi_json.name}')
        well, day, time = parse_from_path(img_dir / current_name)
        parts = (img_dir / current_name).resolve().parts
        proj_root = None
        if 'wells' in parts:
            i = parts.index('wells'); proj_root = Path(*parts[:i])
        px_um = read_pixel_size_um(img_dir / current_name)
        row = {'image_path': str((img_dir / current_name).resolve()), 'well': well, 'day': day, 'time': time,
               'area_px': res.area_px, 'perimeter_px': res.perimeter_px, 'centroid_yx': json.dumps(res.centroid_yx),
               'pixel_size_um': px_um if px_um is not None else ''}
        def append_csv(csv_path: Path):
            write_header = not csv_path.exists()
            with csv_path.open('a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header: w.writeheader()
                w.writerow(row)
            print(f'[gui] Appended measurements to {csv_path}')
        append_csv(img_dir / 'roi_measurements.csv')
        if proj_root: append_csv(proj_root / 'roi_measurements.csv')
        QtWidgets.QMessageBox.information(self, 'Saved', f'ROI saved:\n{roi_json.name}\n{mask_tif.name}\nMeasurements appended.')
def main():
    print('[gui] Launching Qt app...')
    app = QtWidgets.QApplication(sys.argv)
    win = OrganoidROIApp(); win.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()