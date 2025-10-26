#!/usr/bin/env python3
"""
Batch-create Cellpose *_seg.npy files for all TIFFs in one or more folders.

- Uses a large diameter so the whole organoid is one ROI.
- Saves the standard Cellpose *_seg.npy next to each image.
"""
import argparse
import logging
import pathlib
import sys
from time import perf_counter

import numpy as np  # noqa: F401  # ensures dependency when scripts are bundled
from cellpose import io, models
try:  # torch is optional until GPU detection is needed
    import torch
except ImportError:  # pragma: no cover - environments without torch
    torch = None


def resolve_device():
    """Detect the best available device (MPS on Apple Silicon if present)."""
    if torch is None:
        return {"gpu": False, "device": None, "label": "cpu", "reason": "torch not installed"}

    # CUDA is unlikely on this workstation, but keep the check for completeness.
    if torch.cuda.is_available():  # pragma: no cover - depends on NVIDIA hardware
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        return {"gpu": True, "device": torch.device("cuda"), "label": f"cuda:{name}", "reason": "CUDA available"}

    # Apple Silicon / Metal Performance Shaders backend
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available and mps_available.is_available() and mps_available.is_built():
        return {"gpu": True, "device": torch.device("mps"), "label": "mps", "reason": "MPS backend available"}

    return {"gpu": False, "device": None, "label": "cpu", "reason": "no GPU backend detected"}


def configure_cellpose_logging(verbose: bool) -> None:
    """Stream Cellpose logging output to stdout when verbose mode is enabled."""
    if not verbose:
        return

    fmt = logging.Formatter("[CELLPOSE] %(message)s")
    target_loggers = [
        logging.getLogger("cellpose"),
        logging.getLogger("cellpose.models"),
    ]
    for logger in target_loggers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(getattr(handler, "_cellpose_helper", False) for handler in logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            handler.setFormatter(fmt)
            handler._cellpose_helper = True  # mark so we do not duplicate
            logger.addHandler(handler)


def run_one_dir(in_dir, model_path, diameter, chan, chan2, flow_thresh, cellprob_thresh, model_kwargs):
    """Segment every TIFF inside ``in_dir`` and save Cellpose outputs next to the images.

    Parameters are intentionally explicit so non-programmers can map them back to
    the command-line flags defined in ``main``. The function prints progress
    messages for every major action and returns simple counters so the caller can
    build a final summary (total images seen, how many new ``*_seg.npy`` files
    were created, and whether the directory existed).
    """

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
    # ``model_path`` accepts either one of the built-in Cellpose presets (cyto3, cyto2, …)
    # or an absolute path to a custom model folder produced by training.
    model_kwargs = model_kwargs or {}
    if model_path in ["cyto3", "cyto2", "cyto3_cp3", "cyto2_cp3"]:
        m = models.CellposeModel(model_type=model_path, **model_kwargs)
    else:
        m = models.CellposeModel(pretrained_model=model_path, **model_kwargs)
    print(f"[INFO] Cellpose model device: {m.device} (gpu flag={m.gpu})")
    created = 0

    for idx, img_path in enumerate(images, start=1):
        seg_path = img_path.with_name(img_path.stem + "_seg.npy")
        if seg_path.exists():
            print(f"[SKIP] [{idx}/{len(images)}] {img_path.name} already has segmentation")
            continue  # already labeled (GUI or previous run)
        img_start = perf_counter()
        print(f"[STEP] [{idx}/{len(images)}] Segmenting {img_path.name}")
        img = io.imread(img_path)
        # ``channels`` tells Cellpose which imaging channel to treat as signal/background.
        # With single-channel projections, ``chan=0`` and ``chan2=0`` means "use grayscale"
        # and "no second channel".
        print(
            f"[DEBUG] Calling Cellpose eval | diameter={diameter} "
            f"| chan={chan} | chan2={chan2} | flow_threshold={flow_thresh} "
            f"| cellprob_threshold={cellprob_thresh}"
        )
        masks, flows, styles = m.eval(
            img,
            channels=[chan, chan2],
            diameter=diameter,
            flow_threshold=flow_thresh,
            cellprob_threshold=cellprob_thresh,
        )
        print(
            f"[DEBUG] Eval complete | mask_count={int(masks.max()) if masks.size else 0} "
            f"| masks_shape={getattr(masks, 'shape', None)}"
        )
        # This helper writes the standard Cellpose results (``*_seg.npy`` plus flow files)
        # alongside ``img_path`` so any downstream tool can load them.
        seg_start = perf_counter()
        # ``save_masks`` persists the Cellpose segmentation results to disk. It writes
        # the ``*_seg.npy`` file plus the companion ``*_cp_output.npy`` and ``*_mask.npy``.
        # Older examples used ``masks_flows_to_seg`` but the signature changed in v3,
        # so we call ``save_masks`` directly to avoid version mismatches.
        io.save_masks(img_path, masks, flows, img)
        save_elapsed = perf_counter() - seg_start
        elapsed = perf_counter() - img_start
        print(
            f"[OK] {img_path.name} -> {seg_path.name} "
            f"(eval={elapsed - save_elapsed:.2f}s, save={save_elapsed:.2f}s)"
        )
        created += 1

    dir_elapsed = perf_counter() - dir_start
    print(f"[INFO] Completed {in_dir} in {dir_elapsed:.2f}s ({created}/{len(images)} new segmentations)")
    return len(images), created, True


def main():
    p = argparse.ArgumentParser(
        description=(
            "Create Cellpose *_seg.npy files for every TIFF in the provided folders. "
            "Defaults assume single-channel organoid projections where one mask should "
            "cover the entire structure."
        )
    )
    p.add_argument("--dirs", nargs="+", required=True, help="One or more folders with .tif images")
    p.add_argument("--model", default="cyto3", help="cyto3 (default) or path to custom model")
    p.add_argument("--diameter", type=float, default=1500.0, help="Approximate object size in pixels (use the same value used during training).")
    p.add_argument("--chan", type=int, default=0, help="Primary imaging channel (0 = grayscale/single-channel TIFF).")
    p.add_argument("--chan2", type=int, default=0, help="Secondary channel (0 = none).")
    p.add_argument("--flow_threshold", type=float, default=0.1, help="Cellpose flow threshold; lower values relax the shape filtering.")
    p.add_argument("--cellprob_threshold", type=float, default=-6.0, help="Cellpose cell probability threshold; negative values favour 'keep everything'.")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the native Cellpose progress messages (useful for debugging long runs).",
    )
    args = p.parse_args()

    configure_cellpose_logging(args.verbose)

    device_info = resolve_device()
    device_label = device_info["label"]
    print(
        "[INFO] Cellpose device selection: "
        f"{device_label} (gpu={device_info['gpu']} | {device_info['reason']})"
    )
    model_kwargs = {"gpu": device_info["gpu"]}
    if device_info["device"] is not None:
        model_kwargs["device"] = device_info["device"]

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
            model_kwargs,
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
