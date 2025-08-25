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

def main():
    ap = argparse.ArgumentParser(description="Reorganize raw TIFFs into wells/<ROWCOL>/day_XX/HHhMMm")
    ap.add_argument("--raw", required=True, help="Folder with raw .tif/.tiff images (flat)")
    ap.add_argument("--out", required=True, help="Project output root")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move (default: move)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying/moving files")
    ap.add_argument("--min_col", type=int, default=1, help="Only include wells with column >= this number (default: 1)")
    ap.add_argument("--rows", default="ABCDEFGH", help="Only include wells whose row letter is in this set (default: ABCDEFGH)")
    args = ap.parse_args()

    rows_set = set(args.rows.upper())

    raw_dir = Path(args.raw).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    (out_root / "wells").mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    print(f"[organize] RAW DIR : {raw_dir}")
    print(f"[organize] OUT ROOT: {out_root}")
    print(f"[organize] Writing manifest to: {manifest_path}")
    print(f"[organize] Filters -> rows: {''.join(sorted(rows_set))}, min_col >= {args.min_col}")
    print("[organize] Scanning *.tif / *.tiff ...")

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
            print(f"  - SKIP (name pattern): {p.name}")
            continue

        if meta["row"].upper() not in rows_set:
            skipped_row += 1
            print(f"  - SKIP (row filter): {p.name} (row={meta['row']})")
            continue

        if meta["col"] < args.min_col:
            skipped_col += 1
            print(f"  - SKIP (col<{args.min_col}): {p.name} (col={meta['col']})")
            continue

        day_str = f"day_{meta['day']:02d}"
        time_str = f"{meta['hour']:02d}h{meta['minute']:02d}m"
        dest_dir = out_root / "wells" / meta["well"] / day_str / time_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name

        if args.dry_run:
            action = "would copy" if args.copy else "would move"
            print(f"  ~ {action}: {p.name}  ->  {dest}")
        else:
            if args.copy:
                shutil.copy2(p, dest); action = "copied"
            else:
                shutil.move(p, dest); action = "moved"
            processed += 1
            print(f"  + {action}: {p.name}  ->  {dest}")

        rows_out.append({
            "orig_path": str(p),
            "new_path": str(dest),
            "well": meta["well"],
            "row": meta["row"],
            "col": meta["col"],
            "day": meta["day"],
            "hour": meta["hour"],
            "minute": meta["minute"],
        })

    header = ["orig_path","new_path","well","row","col","day","hour","minute"]
    if not args.dry_run:
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for r in rows_out:
                w.writerow(r)

    print("\n[organize] SUMMARY")
    print(f"  processed: {processed}")
    print(f"  skipped (name pattern): {skipped_pattern}")
    print(f"  skipped (row filter): {skipped_row}")
    print(f"  skipped (col filter): {skipped_col}")
    print(f"[organize] Done. Manifest: {manifest_path}")

if __name__ == "__main__":
    main()