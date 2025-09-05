#!/usr/bin/env python
import argparse, re, shutil, csv
from pathlib import Path

# Match filenames like "..._C12_1_03d12h00m.tif" (row A-H, col 1-12)
PATTERN = re.compile(
    r"_(?P<well>[A-Ha-h](?P<col>\d{1,2}))_\d+_(?P<day>\d{2})d(?P<hour>\d{2})h(?P<minute>\d{2})m\.(?:tif|tiff)$",
    re.IGNORECASE
)

def parse_meta(name: str):
    m = PATTERN.search(name)
    if not m:
        return None
    col = int(m.group("col"))
    well = m.group("well").upper()
    return {
        "well": well,            # e.g., "C12"
        "row": well[0],          # e.g., "C"
        "col": col,              # e.g., 12
        "day": int(m.group("day")),
        "hour": int(m.group("hour")),
        "minute": int(m.group("minute")),
    }

def organize(raw: str | Path, out: str | Path, copy: bool = False, dry_run: bool = False,
             min_col: int = 1, rows: str = "ABCDEFGH", log=None):
    """Programmatic API to reorganize files.

    Returns a dict with summary and a list of action rows.
    """
    def _log(msg: str):
        if log:
            try:
                log(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    rows_set = set(rows.upper())

    raw_dir = Path(raw).expanduser().resolve()
    out_root = Path(out).expanduser().resolve()
    (out_root / "wells").mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    _log(f"[organize] RAW DIR : {raw_dir}")
    _log(f"[organize] OUT ROOT: {out_root}")
    _log(f"[organize] Writing manifest to: {manifest_path}")
    _log(f"[organize] Filters -> rows: {''.join(sorted(rows_set))}, min_col >= {min_col}")
    _log("[organize] Scanning *.tif / *.tiff ...")

    processed = 0
    skipped_pattern = 0
    skipped_row = 0
    skipped_col = 0
    rows_out = []

    files = list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    for p in sorted(files):
        meta = parse_meta(p.name)
        if not meta:
            skipped_pattern += 1
            _log(f"  - SKIP (name pattern): {p.name}")
            continue

        if meta["row"].upper() not in rows_set:
            skipped_row += 1
            _log(f"  - SKIP (row filter): {p.name} (row={meta['row']})")
            continue

        if meta["col"] < min_col:
            skipped_col += 1
            _log(f"  - SKIP (col<{min_col}): {p.name} (col={meta['col']})")
            continue

        day_str = f"day_{meta['day']:02d}"
        time_str = f"{meta['hour']:02d}h{meta['minute']:02d}m"
        dest_dir = out_root / "wells" / meta["well"] / day_str / time_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name

        if dry_run:
            action = "would copy" if copy else "would move"
            _log(f"  ~ {action}: {p.name}  ->  {dest}")
        else:
            if copy:
                shutil.copy2(p, dest); action = "copied"
            else:
                shutil.move(p, dest); action = "moved"
            processed += 1
            _log(f"  + {action}: {p.name}  ->  {dest}")

        rows_out.append({
            "orig_path": str(p),
            "new_path": str(dest),
            "new_rel": str(dest.relative_to(out_root)),
            "well": meta["well"],
            "row": meta["row"],
            "col": meta["col"],
            "day": meta["day"],
            "hour": meta["hour"],
            "minute": meta["minute"],
        })

    header = ["orig_path","new_path","new_rel","well","row","col","day","hour","minute"]
    if not dry_run:
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for r in rows_out:
                w.writerow(r)

    _log("\n[organize] SUMMARY")
    _log(f"  processed: {processed}")
    _log(f"  skipped (name pattern): {skipped_pattern}")
    _log(f"  skipped (row filter): {skipped_row}")
    _log(f"  skipped (col filter): {skipped_col}")
    _log(f"[organize] Done. Manifest: {manifest_path}")

    return {
        "processed": processed,
        "skipped_pattern": skipped_pattern,
        "skipped_row": skipped_row,
        "skipped_col": skipped_col,
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "raw_dir": str(raw_dir),
        "rows": rows,
        "min_col": min_col,
        "dry_run": dry_run,
        "copy": copy,
        "actions": rows_out,
    }

def main():
    ap = argparse.ArgumentParser(description="Reorganize raw TIFFs into wells/<ROWCOL>/day_XX/HHhMMm")
    ap.add_argument("--raw", required=True, help="Folder with raw .tif/.tiff images (flat)")
    ap.add_argument("--out", required=True, help="Project output root")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move (default: move)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying/moving files")
    ap.add_argument("--min_col", type=int, default=1, help="Only include wells with column >= this number (default: 1)")
    ap.add_argument("--rows", default="ABCDEFGH", help="Only include wells whose row letter is in this set (default: ABCDEFGH)")
    args = ap.parse_args()

    organize(args.raw, args.out, copy=args.copy, dry_run=args.dry_run, min_col=args.min_col, rows=args.rows)

if __name__ == "__main__":
    main()
