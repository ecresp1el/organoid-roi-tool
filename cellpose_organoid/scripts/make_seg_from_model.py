#!/usr/bin/env python3
"""
Batch-create Cellpose *_seg.npy files for all TIFFs in one or more folders.

- Uses a large diameter so the whole organoid is one ROI.
- Saves the standard Cellpose *_seg.npy next to each image.
"""
import argparse
import pathlib
from time import perf_counter

import numpy as np  # noqa: F401  # ensures dependency when scripts are bundled
from cellpose import io, models


def run_one_dir(in_dir, model_path, diameter, chan, chan2, flow_thresh, cellprob_thresh):
    in_dir = pathlib.Path(in_dir)
    if not in_dir.exists():
        print(f"[WARN] Skipping missing dir: {in_dir}")
        return 0, 0, False

    dir_start = perf_counter()
    print(f"[INFO] Scanning: {in_dir}")
    images = sorted(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
    if not images:
        print(f"[WARN] No TIFF images found in {in_dir}")
        return 0, 0, True

    print(f"[INFO] Found {len(images)} candidate images")
    m = models.CellposeModel(pretrained_model=model_path) if model_path not in ["cyto3", "cyto2", "cyto3_cp3", "cyto2_cp3"] else models.CellposeModel(model_type=model_path)
    created = 0

    for idx, img_path in enumerate(images, start=1):
        seg_path = img_path.with_name(img_path.stem + "_seg.npy")
        if seg_path.exists():
            print(f"[SKIP] [{idx}/{len(images)}] {img_path.name} already has segmentation")
            continue  # already labeled (GUI or previous run)
        img_start = perf_counter()
        print(f"[STEP] [{idx}/{len(images)}] Segmenting {img_path.name}")
        img = io.imread(img_path)
        masks, flows, styles = m.eval(
            img,
            channels=[chan, chan2],
            diameter=diameter,
            flow_threshold=flow_thresh,
            cellprob_threshold=cellprob_thresh,
        )
        io.masks_flows_to_seg(img_path, masks, flows, img.shape, img=img)
        elapsed = perf_counter() - img_start
        print(f"[OK] {img_path.name} -> {seg_path.name} ({elapsed:.2f}s)")
        created += 1

    dir_elapsed = perf_counter() - dir_start
    print(f"[INFO] Completed {in_dir} in {dir_elapsed:.2f}s ({created}/{len(images)} new segmentations)")
    return len(images), created, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True, help="One or more folders with .tif images")
    p.add_argument("--model", default="cyto3", help="cyto3 (default) or path to custom model")
    p.add_argument("--diameter", type=float, default=1500.0)
    p.add_argument("--chan", type=int, default=0)
    p.add_argument("--chan2", type=int, default=0)
    p.add_argument("--flow_threshold", type=float, default=0.1)
    p.add_argument("--cellprob_threshold", type=float, default=-6.0)
    args = p.parse_args()

    overall_start = perf_counter()
    print(
        "[START] Generating Cellpose segmentations "
        f"for {len(args.dirs)} directories | model={args.model} | diameter={args.diameter}"
    )
    total_dirs = 0
    total_images = 0
    total_created = 0

    for d in args.dirs:
        images, created, ok = run_one_dir(
            d,
            args.model,
            args.diameter,
            args.chan,
            args.chan2,
            args.flow_threshold,
            args.cellprob_threshold,
        )
        if ok:
            total_dirs += 1
        total_images += images
        total_created += created

    overall_elapsed = perf_counter() - overall_start
    print(
        "[DONE] Directories scanned: "
        f"{len(args.dirs)} (existing: {total_dirs}) | images seen: {total_images} | "
        f"new segmentations: {total_created} | elapsed: {overall_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
