import math
import numpy as np

from roi_utils import polygon_area_perimeter, polygon_centroid, compute_roi


def test_rectangle_area_perimeter_centroid_and_mask():
    # Rectangle with corners: (y,x) = (1,1) -> (1,4) -> (3,4) -> (3,1)
    verts = [(1, 1), (1, 4), (3, 4), (3, 1)]
    shape = (6, 7)  # image height x width

    area, perim = polygon_area_perimeter(verts)
    # width=3, height=2 => area=6, perimeter=2*(3+2)=10
    assert area == 6.0
    assert perim == 10.0

    cy, cx = polygon_centroid(verts)
    # Centroid should be average of min/max: y=(1+3)/2=2, x=(1+4)/2=2.5
    assert math.isclose(cy, 2.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(cx, 2.5, rel_tol=0, abs_tol=1e-6)

    res = compute_roi(verts, shape)
    # Mask sum should be close to geometric area for axis-aligned rectangle
    assert np.isclose(res.mask.sum(), area)


def test_triangle_basic_properties():
    # Right triangle with legs length 4 and 3 => area = 6
    verts = [(1, 1), (1, 5), (4, 1)]
    shape = (8, 8)

    area, perim = polygon_area_perimeter(verts)
    assert math.isclose(area, 6.0, abs_tol=1e-6)
    # Perimeter ~ 3 + 4 + 5 = 12
    assert math.isclose(perim, 12.0, abs_tol=1e-6)

    res = compute_roi(verts, shape)
    # Mask sum should be within ~1px of geometric area for simple triangles
    assert abs(int(res.mask.sum()) - int(round(area))) <= 1

