import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Optional imports for flexible mask rasterization
try:
    from skimage.draw import polygon as draw_polygon  # type: ignore
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - environment may lack skimage
    draw_polygon = None  # type: ignore
    _HAS_SKIMAGE = False

try:
    from PIL import Image, ImageDraw  # type: ignore
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = ImageDraw = None  # type: ignore
    _HAS_PIL = False

@dataclass
class ROIResult:
    mask: np.ndarray
    area_px: float
    perimeter_px: float
    centroid_yx: Tuple[float, float]
    vertices_yx: List[Tuple[float, float]]

def _polygon_to_mask_skimage(vertices_yx, shape):
    yy = np.array([v[0] for v in vertices_yx], dtype=float)
    xx = np.array([v[1] for v in vertices_yx], dtype=float)
    rr, cc = draw_polygon(yy, xx, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _polygon_to_mask_pillow(vertices_yx, shape):
    # Pillow expects x,y ordering
    xy = [(float(x), float(y)) for (y, x) in vertices_yx]
    h, w = int(shape[0]), int(shape[1])
    img = Image.new("L", (w, h), 0)
    drw = ImageDraw.Draw(img)
    drw.polygon(xy, outline=1, fill=1)
    return (np.array(img) > 0)


def _polygon_to_mask_numpy(vertices_yx, shape):
    # Numpy-only fallback using even-odd rule (ray casting) per scanline.
    # Not the fastest, but avoids external deps.
    h, w = int(shape[0]), int(shape[1])
    pts = np.asarray(vertices_yx, dtype=float)
    if len(pts) < 3:
        return np.zeros((h, w), dtype=bool)
    y = pts[:, 0]
    x = pts[:, 1]
    # Close polygon
    y2 = np.roll(y, -1)
    x2 = np.roll(x, -1)
    mask = np.zeros((h, w), dtype=bool)
    # For each row, compute intersections of edges with the horizontal scanline
    for yi in range(h):
        # Find edges that cross this scanline (exclusive of the upper endpoint)
        cond = ((y <= yi) & (y2 > yi)) | ((y2 <= yi) & (y > yi))
        if not np.any(cond):
            continue
        x_intersections = x[cond] + (yi - y[cond]) * (x2[cond] - x[cond]) / (y2[cond] - y[cond])
        xs = np.sort(x_intersections)
        # Fill between pairs of intersections
        for i in range(0, xs.size, 2):
            x_start = int(np.ceil(xs[i]))
            if i + 1 >= xs.size:
                break
            x_end = int(np.floor(xs[i + 1]))
            if x_end >= 0 and x_start < w and x_end >= x_start:
                mask[yi, max(0, x_start):min(w, x_end + 1)] = True
    return mask


def polygon_to_mask(vertices_yx, shape):
    """Rasterize a polygon into a boolean mask of given ``shape``.

    Tries scikit-image if available, otherwise falls back to Pillow, then
    a numpy-only scanline implementation. This flexibility streamlines
    setup and testing in constrained environments.
    """
    if _HAS_SKIMAGE and draw_polygon is not None:
        return _polygon_to_mask_skimage(vertices_yx, shape)
    if _HAS_PIL:
        return _polygon_to_mask_pillow(vertices_yx, shape)
    return _polygon_to_mask_numpy(vertices_yx, shape)

def polygon_area_perimeter(vertices_yx):
    pts = np.array(vertices_yx, dtype=float)
    if len(pts) < 3: return 0.0, 0.0
    x = pts[:,1]; y = pts[:,0]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    diffs = np.diff(pts, axis=0, append=pts[:1])
    perimeter = np.sum(np.linalg.norm(diffs, axis=1))
    return float(area), float(perimeter)

def polygon_centroid(vertices_yx):
    pts = np.array(vertices_yx, dtype=float)
    if len(pts) < 3: return (float(np.mean(pts[:,0])), float(np.mean(pts[:,1])))
    x = pts[:,1]; y = pts[:,0]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    cx = np.dot((x + np.roll(x, -1)), (x*np.roll(y, -1) - np.roll(x, -1)*y))
    cy = np.dot((y + np.roll(y, -1)), (x*np.roll(y, -1) - np.roll(x, -1)*y))
    A = a / 2.0
    if A == 0: return (float(np.mean(y)), float(np.mean(x)))
    return (float(cy/(6*A)), float(cx/(6*A)))

def save_roi_json(path, vertices_yx, image_path, extra=None):
    data = {"image_path": image_path, "vertices_yx": vertices_yx}
    if extra: data.update(extra)
    with open(path, "w") as f: json.dump(data, f, indent=2)

def compute_roi(vertices_yx, shape):
    mask = polygon_to_mask(vertices_yx, shape)
    area, perimeter = polygon_area_perimeter(vertices_yx)
    pts = np.array(vertices_yx, dtype=float)
    cy, cx = float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))
    try:
        _cy, _cx = polygon_centroid(vertices_yx); cy, cx = _cy, _cx
    except Exception: pass
    return ROIResult(mask=mask, area_px=area, perimeter_px=perimeter, centroid_yx=(cy, cx), vertices_yx=vertices_yx)
