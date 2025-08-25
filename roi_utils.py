import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from skimage.draw import polygon as draw_polygon

@dataclass
class ROIResult:
    mask: np.ndarray
    area_px: float
    perimeter_px: float
    centroid_yx: Tuple[float, float]
    vertices_yx: List[Tuple[float, float]]

def polygon_to_mask(vertices_yx, shape):
    yy = np.array([v[0] for v in vertices_yx], dtype=float)
    xx = np.array([v[1] for v in vertices_yx], dtype=float)
    rr, cc = draw_polygon(yy, xx, shape)
    mask = np.zeros(shape, dtype=bool); mask[rr, cc] = True
    return mask

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
