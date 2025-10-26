# Organoid ROI Tool (v9)

A simple, reliable tool to organize Incucyte TIFF images, draw a single polygon ROI per image, and export:

- A clean ROI mask (full-size and cropped) as TIFFs
- A compact ROI JSON file (the polygon you drew)
- A running CSV of measurements (area, perimeter, centroid, pixel size)

This guide is written for non‑programmers. It assumes Anaconda is already installed.

---

## 1) Quick Start (macOS)

1. Download or clone this folder to your Mac.
2. First run (one‑time): double‑click `RUN_FIRST_mac_Conda_Create_And_Launch.command`.
   - This creates the Conda environment, checks required packages (including imagecodecs and pytest), and launches the app.
   - If macOS blocks the file, right‑click it, choose Open, then Open again. If needed, allow in System Settings → Privacy & Security.
3. Next time: double‑click `RUN_GUI_mac_Conda.command` to launch the app directly.

Manual run (optional):

```
conda env create -n organoid_roi_incucyte_imaging -f environment.yml
conda activate organoid_roi_incucyte_imaging
python gui_app.py
```

> Need CellProfiler or Cellpose? Use the dedicated `cellprofiler_env` described in [Section 9](#9-cellprofiler--cellpose-environment) so the legacy scientific stack required by CellProfiler stays separate from the modern Napari/Qt stack that powers the ROI tool.

---

## 2) Organizing Your Images (Optional but Recommended)

If your raw images are all in one folder and named like Incucyte exports:

Example filename pattern accepted:
- Something like: `..._C12_1_03d12h00m.tif`
  - `C12` = well (row C, column 12)
  - `03d12h00m` = day 03, hour 12, minute 00

Use the organizer to place files into a neat project structure:

```
<project-root>/
  wells/
    C12/
      day_03/
        12h00m/
          your_image.tif
```

How to run on macOS:

- Double‑click `RUN_Reorganize_mac_Conda.command`, then answer the prompts:
  - Raw folder: folder containing your `.tif`/`.tiff` images
  - Output folder: your new project folder
  - Min column (optional): only include wells with column ≥ this number (e.g., 4)
  - Rows (optional): only include which rows (default `ABCDEFGH`)

Advanced (terminal):

```
conda run -n organoid_roi_incucyte_imaging \
  python reorganize.py --raw /path/to/raw --out /path/to/project --min_col 4 --rows ABCD
```

The organizer writes a `manifest.csv` at the project root so you can track moved/copied files.

---

## 3) Using The ROI App

Launch the app (mac):
- Double‑click `RUN_GUI_mac_Conda.command`

Open or initialize a project:
- Initialize Project: Project → “Initialize Project (Reorganize)…” or click the “Initialize Project” button. Pick your raw folder (flat TIFFs) and a project output root. Optionally filter rows/columns and choose copy vs move. Files are organized under `wells/<well>/day_XX/HHhMMm/` and a `manifest.csv` is written.
- Import Project: Click the “Import Project” button (or Project → “Import Existing Project…”). Select a project root that contains `wells/`. The app will jump to the first unlabeled image if available, and you can also open the Dashboard to navigate.
- Open Dashboard: Click the “Open Dashboard” button (or Project → “Open Progress Dashboard”) to see project-wide progress and jump to the next unlabeled per well/day.
- Migrate Project: Click the “Migrate Project” button (or Project → “Migrate Project for Portability…”) to convert older projects to portable paths and set the current user.
- Validate Project: Click the “Validate Project” button (or Project → “Validate Project”) to confirm structure, portable paths, and CSVs are OK. The app also validates automatically when you import/initialize a project and will say “Project validation OK” if everything looks good.

Set current user (project-local):
- On first import/initialize, you’ll be prompted to enter your name/initials; this is stored in `<project>/.roi_project.json` and shown in the toolbar.
- Change anytime via Project → “Set Current User…”.
- Each save logs `user` + `timestamp` into ROI JSON and `roi_measurements.csv`, and appends an entry to `<project>/roi_activity_log.csv` (action log).
- Open Image: Click “Open Image” for a one-off single file, or drag-and-drop a `.tif`/`.tiff`.

If your image has multiple frames, the app automatically applies a maximum intensity projection so you see a 2D image.

Behind the scenes (plain English):
- The app treats the folder you select as a self‑contained “project”. Images live under `wells/<well>/day_XX/HHhMMm/`.
- Progress is tracked by the presence of a small `*_roi.json` file next to each image. If it exists, that image is considered “done”.
- Portable paths: `manifest.csv` and `roi_measurements.csv` use paths relative to the project so you can copy the folder anywhere (external drive, another computer) and keep going.
- User and time: each save records who labeled the image and when (shown in CSV and in per‑image ROI JSON). There is also a simple `roi_activity_log.csv` with one line per save/delete.
- Scope navigation: “Next Unlabeled (Scope)” moves through images within your chosen scope (Project / Well / Day). The default scope is the current Well.

Adjust how it looks:
- The contrast is auto‑stretched to the 2nd–98th percentile; you can tweak layer contrast in Napari if needed.
- Use the mouse wheel or trackpad to zoom; drag to pan.

Draw an ROI:
- In the left layer list, click the “ROI” layer (already selected) and ensure it’s in “Add Polygon” mode.
- Click to add vertices around your organoid. Click the first point (or double‑click) to close.
- Tip: Your goal is one polygon per image.

Save the ROI:
- Click “Save ROI” (or press S / Cmd+S).
- If “Auto‑advance (Scope)” is checked, the app will move to the next unlabeled image within the current scope after saving.

Delete the ROI:
- Click “Delete ROI” (or press D), then confirm.

Browse images:
- “Next Unlabeled (Scope)” advances to the next image without a saved ROI within the current navigation scope (Right Arrow shortcut).
- “Prev Saved ROI” jumps back through the history of images you’ve saved in this session (Left Arrow shortcut).
- Default scope when working in a project is the current Well (across its days and times). You can switch scope via the Project Dashboard by selecting ALL (project), a specific well, or a day before clicking “Next Unlabeled (Scope)”. The active scope is shown next to the progress bar.
- Use the Project Dashboard for project-wide navigation across wells/days.
- Note: Navigation ignores derived ROI outputs (`*_mask.tif`, `*_roi_masked*.tif`) so you only step through original images.

Session & progress:
- The app shows a progress bar (done / total) for the current folder.
- Your place is saved automatically (folder, last image). Use File → “Resume Last Session” to pick up where you left off.
- “Stop / Save Session” saves your state so you can safely close the app.

Project‑wide progress dashboard:
- Open via Project → “Open Progress Dashboard” or the “Open Dashboard” button. The project root can be set from there or is inferred when you Import/Initialize a project.
- Shows totals per well and per day: Done, Total, and %.
- “Next Unlabeled (Scope)” uses the selected row (ALL/well/day) as the scope and opens the next image without a saved ROI in that group. The main window then continues within the same scope.
- “Refresh” rescans; it also auto‑refreshes after ROI saves/deletes.

Initialize project (inside the GUI):
- Use Project → “Initialize Project (Reorganize)…” or the “Initialize Project” button.
- Choose a raw input folder (flat TIFFs) and an output project root.
- Optional filters: rows (A–H), min column, copy vs move, dry run.
- Organizes files as `wells/<well>/day_XX/HHhMMm/` and appends to `<project>/manifest.csv`.
- After completion, you can open the progress dashboard immediately.

Portability & collaboration:
- Project folder is self-contained: `wells/`, `manifest.csv`, `roi_measurements.csv`, per-image ROI JSONs, plus `.roi_project.json` (user list/current user) and `.roi_session.json` (last image/scope).
- Paths in `manifest.csv` and `roi_measurements.csv` are stored relative to project root so you can move the folder across machines/drives.
- The app auto-resumes from `.roi_session.json` when importing a project. Use “Set Current User…” to switch users between sessions.

Migrate an existing project for portability:
- Project → “Migrate Project for Portability…” updates `manifest.csv` to include relative paths (`new_rel`) and adds `image_relpath`, `user`, and `timestamp_iso` columns to measurement CSVs (leaves past user/timestamp blank if unknown).
- The migrator also asks if you’re one of the previous users; you can pick from the list or add a new one. Your choice is saved in `.roi_project.json` as the current user.

Moving a project to another drive/computer (non‑programmer steps):
- In the app, open your project and click “Migrate Project” once (adds portable paths and sets/chooses the user).
- Click “Validate Project” and confirm you see “Project validation OK. No issues found.”
- Close the app and copy the entire project folder (includes `wells/`, `manifest.csv`, `roi_measurements.csv`, `.roi_project.json`, `.roi_session.json`, per‑image ROI JSONs) to your drive/computer.
- On the other machine, click “Import Project”, pick the folder. The app validates automatically, prompts for user if needed, and resumes where you left off.

Preloading existing ROI:
- If a matching ROI JSON file is already present for the image, the app auto‑loads it so you can review or edit.

---

## 3A) Common Scenarios (Quick Flows)

- New user, new project (first time)
  - Click “Initialize Project (Reorganize)…”, pick your raw folder and a new project folder.
  - When prompted, enter your name/initials (current user).
  - Click “Validate Project” → expect “Project validation OK. No issues found.”
  - Click “Open Dashboard”, select ALL or a specific well, then “Next Unlabeled (Scope)” to start.

- Continue an existing project (same computer)
  - Click “Import Project”, choose the project folder (contains `wells/`).
  - The app validates automatically and resumes where you left off (from `.roi_session.json`).
  - If you want to switch person, use Project → “Set Current User…”.

- Continue an existing project (different computer / external drive)
  - On the original computer: click “Migrate Project” once, then “Validate Project” (expect OK). Close the app.
  - Copy the whole project folder to your drive/computer.
  - On the new computer: click “Import Project”, select the folder. The app validates and resumes. If asked, pick your user or add a new one.

- Add or switch user on a project
  - Project → “Set Current User…”. Pick from the list or add a new name. Subsequent saves record that user.

- Start over at the first unlabeled (optional)
  - Delete `<project>/.roi_session.json`, then “Import Project” again. Or open the Dashboard, select ALL (project), and click “Next Unlabeled (Scope)”.

- How progress is counted
  - An image is “done” if a small `*_roi.json` exists next to it. The Dashboard totals use that rule.

---

## 4) What Gets Saved

When you save an ROI for `image.tif` in a folder, these files are written in the same folder:

- `image_mask.tif`: a binary mask (0/255) showing the ROI area.
- `image_roi.json`: coordinates of the polygon you drew.
- `image_roi_masked.tif`: full‑size TIFF with an associated alpha channel marking the ROI. Original pixels are preserved; the alpha marks outside‑ROI as transparent.
- `image_roi_masked_cropped.tif`: a tighter crop around the ROI, also with alpha.
- Plot outputs (via `python -m dcxspot_play.plot_growth`): saved in `<project-root>/plots/` as `<prefix>_area_boxplot.(png|pdf|svg)`, `<prefix>_growth_boxplot.(png|pdf|svg)`, `<prefix>_fluor_total_boxplot.(png|pdf|svg)`, `<prefix>_fluor_total_growth_boxplot.(png|pdf|svg)`, `<prefix>_fluor_density_boxplot.(png|pdf|svg)`, `<prefix>_fluor_density_growth_boxplot.(png|pdf|svg)`, plus per-well image panels `<prefix>_well_<WELL>_panel.(png|pdf|svg)` (row 1: brightfield ROI, row 2: raw mCherry with absolute color scale, row 3: ROI-scaled mCherry with 0–1 scale).

Measurements CSVs (upsert one row per image):
- Local folder CSV: `<image-folder>/roi_measurements.csv`
- If your image path looks like `<project-root>/wells/...`, a project CSV is also updated: `<project-root>/roi_measurements.csv`

---

## 5) Automated Plotting & Reporting CLI Tools

Each helper script below runs inside the Conda environment (`conda activate organoid_roi_incucyte_imaging`). Paths can be absolute or relative. All scripts read project-level channel aliases from `<project>/channel_aliases.json` (optional) and accept `--alias token=Label` overrides.

### 5.1 Panel A strip generator

```
python tools/generate_panel_a_figures.py /path/to/project_or_nd2 \
  --limit 5 \
  --alias gfp=LHX6 --alias cy5=SOX2 \
  --project-root /path/to/project  # optional override
  --output-dir /path/to/custom/panel_a  # optional
```

Creates one 1×N PNG per ND2 file showing:
- merged RGB (DAPI→blue, Cy5→red, GFP→green) and up to two single-channel pseudo-colour panels,
- 10 px internal colourbars and scale bars,
- saved in `<project>/panel_a/<nd2_stem>_panelA.png` by default.

Options:
- `--project-root` force where channel aliases/outputs are read and written.
- `--output-dir` change the destination folder.
- `--limit` limit how many ND2 files to process.
- `--alias token=Label` (repeatable) map raw channel names to biology (e.g. `cy5=SOX2`).

### 5.2 Grouped Panel A summary

```
python tools/build_panel_a_summary.py /path/to/project \
  --groups "DIV 18" "DIV 23" \
  --alias gfp=LHX6 --alias cy5=SOX2
```

Generates individual Panel A strips (if not already present) and combines them into a grouped PNG (per timepoint row). Output saved to `<project>/panel_a/panel_a_summary.png` unless `--output-dir` is specified.

### 5.3 Workflow figure (8-panel analytic report)

```
python tools/generate_workflow_figure.py \
  "/path/to/file.nd2" \
  --alias gfp=LHX6 --alias cy5=SOX2 \
  --smoothing-sigma 0.8 \
  --otsu-offset 0.06 \
  --footprint-radius 1 \
  --min-distance 4 \
  --min-size 80 \
  --max-size 1500 \
  --open-radius 1 \
  --tophat-radius 25 \
  --ring-iterations 16
```

Outputs a fixed-layout 1920 × 1164 PNG to `<project>/cellcount_project/figures/<nd2_stem>_workflow.png` and the matching per-cell CSV to `<project>/cellcount_project/cells/`. Panel layout:

- **A**: raw ND2 overview (RGB + individual channels).
- **B**: stepwise DAPI segmentation (six subpanels with internal annotations).
- **C**: 2×2 grid of sampled nuclei (IDs 1/1171/2336/3506 when available) with cyan masks and translucent orange rings.
- **D**: per-channel patches for one sampled cell (DAPI + markers).
- **E**: marker intensity histograms.
- **F**: cumulative distributions (ECDF).
- **G**: scatter plot (marker vs marker, colour-coded by cell area).
- **H**: FACS-style density (log-scaled hexbin).

Segmentation options:

| Flag | Purpose |
|------|---------|
| `--smoothing-sigma` | Gaussian blur σ applied to DAPI.
| `--otsu-offset` | Additive tweak to Otsu threshold (positive trims background).
| `--min-size` | Remove connected components smaller than this (pixels).
| `--max-size` | Remove components larger than this (pixels).
| `--min-distance` | Minimum separation between watershed seeds.
| `--footprint-radius` | Seed footprint radius (`2r + 1` square) for `peak_local_max`.
| `--open-radius` / `--close-radius` | Morphological opening/closing after thresholding.
| `--tophat-radius` | White top-hat radius to correct illumination before smoothing.
| `--ring-iterations` | Binary dilation iterations for visualising ROI rings (Panel C/D only).

### 5.4 Timepoint grids for LHX6 or Nestin/DCX projects

```
python tools/plot_lhx6_grid.py /path/to/project \
  --marker-token gfp --div18-token "DIV 18" --div23-token "DIV 23"

python tools/plot_nestin_dcx.py /path/to/project \
  --nestin-token gfp --dcx-token cy5 \
  --div18-token "DIV 18" --div23-token "DIV 23"
```

Both scripts rescale DIV 18 marker intensities to the DIV 23 range and output grids to `<project>/panel_a/` (one PNG per marker). Options let you rename tokens, adjust grouping tokens, and override output names.

### 5.5 Marker intensity summaries

```
python tools/plot_marker_timepoints.py /path/to/project \
  --timepoint "DIV 18" --timepoint "DIV 23" \
  --alias gfp=LHX6 --alias cy5=SOX2
```

Creates one boxplot+scatter figure per marker showing mean raw intensity distributions per requested timepoint. Saved in `<project>/analysis/<Marker>_intensity.png`. Use `--skip-token` to exclude channels (default skips DAPI).

All scripts accept `--alias` overrides; if no aliases are provided, raw channel names are used. For reproducible outputs, ensure the Conda environment (`organoid_roi_incucyte_imaging`) is active before running these commands.

Each row includes:
- image_relpath (portable key), image_path (absolute, for convenience)
- well, day, time
- area_px, perimeter_px, centroid_yx
- pixel_size_um (if readable from TIFF tags)
- user (who saved), timestamp_iso (when saved)

Note on pixel size:
- The app tries to read pixel size from the TIFF XResolution tag and converts to micrometers per pixel. If missing, the field is left blank in the CSV.

Activity log (project-level):
- `<project>/roi_activity_log.csv` records one line per save/delete with: timestamp_iso, user, image_relpath, image_path, action (save/delete), well, day, time.

### Plotting ROI Growth

Run the plotting helper to generate consistent box-plot summaries (area + fluorescence, raw + fold-change) and ROI-only image panels with the shared minimal style:

```
python -m dcxspot_play.plot_growth --prefix exp1 --div-start 11
```

The script reads `<project-root>/roi_measurements.csv` by default (using `dcxspot_config.json`) and writes outputs to `<project-root>/plots/`. Each run emits PNG/PDF/SVG triplets for area, area fold-change, total mCherry intensity, total mCherry fold-change, area-normalised (per-pixel) intensity, and the corresponding fold-change. Intensities are measured inside each ROI on the matching fluorescence images; fold-change normalises to the first time-point per well. It also builds per-well image panels cropped to the ROI (row 1: brightfield + outline, row 2: raw mCherry with a shared absolute colorbar, row 3: ROI-normalised mCherry with a shared 0–1 colorbar) so you can visually track each organoid over time in a single figure. Set `div_start` in `dcxspot_config.json` (or pass `--div-start`) so day_00 maps to the DIV value you expect (e.g., DIV11). Use `--prefix` to label different experiments; override `--output-dir` only when you intentionally want a different destination.

### Spatial Autocorrelation (Moran's I)

```
python -m dcxspot_play.morans_batch --project-root /path/to/project --permutations 999
```
python -m dcxspot_play.morans_batch --project-root /path/to/project --permutations 999
```

**Inputs & assumptions**

- Each entry in `roi_measurements.csv` must have the companion files saved by the ROI/DCX workflow: `*_mask.tif`, `fluorescence/*_mcherry.tif`, and (optionally) `dcxspot/*_labels.tif` containing the detected islands.
- Moran’s I statistics are computed on the raw mCherry intensities inside the ROI (mean centred, no percentile scaling). Display scaling is applied only when rendering figures.
- Adjacency weights are zeroed outside the ROI, so only intra-ROI neighbours contribute (queen or rook connectivity via `--neighbors`).

**Outputs (default under `<project>/plots/morans/`):**

- `morans_global.csv` – global Moran’s I + permutation summary for every ROI.
- `heatmaps/<image>_local_moran.png` – Local Moran Ii (raw values) restricted to the ROI.
- `overlays/<image>_local_overlay.png` – raw fluorescence blended with the Local Moran colour map.
- `heatmaps/p_values/<image>_local_pvals.png` – Local Moran permutation p-values (default `--local-permutations 199`; set to 0 to skip).
- `panels/<prefix>_well_<WELL>_moran_panel.(png|pdf|svg)` – six-row workflow montage per well (raw fluorescence → ROI mask → detected islands → Local Moran raw → Local Moran scaled → p<0.05 significance overlay).

Use `--wells H11 H12` to restrict the run to specific wells, adjust `--output-csv`, `--heatmap-dir`, `--overlay-dir`, and `--panel-dir` to customise destinations, and tune `--permutations` / `--local-permutations` for the desired runtime vs. precision. The batch runner processes wells sequentially and prints progress so you can monitor long analyses.

---

## 5) Keyboard Shortcuts

- Save ROI: `S` or `Cmd+S`
- Next Unlabeled (Scope): Right Arrow
- Previous Saved ROI: Left Arrow
- Delete ROI: `D`
- Quit: `Cmd+Q`

 “Auto‑advance (Scope)” moves to the next unlabeled image in the current scope after a successful save.

---

## 6) Tips For Clean ROIs

- Zoom in before placing vertices for better precision.
- Place points sparingly; you can adjust vertices after closing the polygon.
- Avoid self‑intersecting polygons.
- If you made a mistake, delete and redraw—it’s often faster.

---

## 7) Troubleshooting

The .command file won’t open on macOS
- Right‑click the file → Open → Open. Or allow it under System Settings → Privacy & Security.

Conda not found
- Open a Terminal where `conda` works. If necessary, run `conda init zsh` then restart Terminal.

Imports fail or the app won’t launch
- Update the environment: `conda env update -n organoid_roi_incucyte_imaging -f environment.yml`
- Then try: `./RUN_GUI_mac_Conda.command`

Images won’t open or look wrong
- We use `tifffile` with a Pillow fallback. Make sure your files are standard TIFFs.
- Multi‑frame images are max‑projected to 2D for ROI drawing.

No ROI is saved / file missing
- The app writes a mask TIFF and a small ROI JSON next to your image. If it reports a save problem, ensure you have write permission to the folder and enough disk space.

CSV not updating
- The tool “upserts” one row per image using `image_relpath` so there are no duplicates. Close the CSV in Excel while saving from the app.

“No unlabeled images found” on a new project
- Make sure you clicked “Import Project” (or used Initialize Project) so the app knows your project root.
- Check the scope label next to the progress bar. If it’s not “Project (…)” or “Well (…)”, open the Dashboard, select “ALL” or a specific well, then click “Next Unlabeled (Scope)”.
- Verify your images have `.tif` or `.tiff` extensions. The app detects both in any letter case (e.g., `.TIF`).

---

## 8) Advanced / Optional

Run automated tests (inside the environment):

```
conda run -n organoid_roi_incucyte_imaging pytest -q
```

Re‑run the organizer with filters:

```
conda run -n organoid_roi_incucyte_imaging \
  python reorganize.py --raw /path/to/raw --out /path/to/project --min_col 4 --rows ABCD
```

Manual launch with logs:

```
conda activate organoid_roi_incucyte_imaging
python gui_app.py
```

---

## 9) CellProfiler + Cellpose Environment

CellProfiler 4.x and the latest Cellpose GUI require a different (older) set of scientific libraries than the Napari ROI tool. To avoid dependency conflicts, keep them in their own Conda environment built from `environment.cellprofiler.yml`.

### Create the environment

```
conda env create -n cellprofiler_env -f environment.cellprofiler.yml
```

This installs:

- Legacy imaging stack pinned for CellProfiler (`scipy 1.9.0`, `scikit-image 0.18.3`, `wxPython 4.2.3`, etc.)
- PyTorch 2.1 CPU build from the official `pytorch` channel
- CellProfiler 4.2.8 and Cellpose 3.0.5 (with GUI extras)

### Launch CellProfiler

```
conda activate cellprofiler_env
cellprofiler          # launches the CellProfiler GUI
```

Use this environment whenever you need to create or run CellProfiler pipelines that depend on the classic wxPython interface.

### Launch Cellpose

```
conda activate cellprofiler_env
cellpose              # launches the Cellpose GUI
```

### Export projection stacks for CellProfiler / Cellpose

Run the helper in `simple_projection_tool/prepare_for_cellprofiler_cellpose.py`
once your `analysis_results/` folders are ready:

```
conda activate organoid_roi_incucyte_imaging
python simple_projection_tool/prepare_for_cellprofiler_cellpose.py \
    --base-path /path/to/project_folder \
    --analysis PCDHvsLHX6_WTvsKO_IHC
```

The exporter now mirrors the DAPI (nuclear) projections alongside every marker,
writes Cellpose-friendly multichannel TIFF stacks named
`…__LHX6+PCDH19+DAPI.tif` (stored as Z×C×Y×X when multiple planes exist, or
flattened to C×Y×X for 2-D projections so the GUI loads without `--Zstack`),
and logs a `[check]` message after each write confirming that all expected
channels made it into the stack.

The same environment also exposes the command-line tools, e.g. `cellprofiler --help`, `python -m cellpose --help`, etc.

### Train the organoid Cellpose model

After the projection exporter runs, follow the dedicated guide in
`cellpose_organoid/README.md`. In short:

1. Edit `cellpose_organoid/train_organoid_model.sh` so `PROJECT_ROOT`,
   `ANALYSIS`, `PROJECTION_TYPES`, and `EXPERIMENT_GROUPS` match your dataset.  
   The script only references helper code inside the repo; all training outputs
   live next to your project data under
   `<project>/cellpose_organoid_workspace/`.
2. Activate the CellProfiler/Cellpose environment:
   ```
   conda activate cellprofiler_env
   ```
3. Run the end-to-end pipeline:
   ```
   bash cellpose_organoid/train_organoid_model.sh
   ```
   The script:
   - Symlinks TIFFs referenced in `cellpose_multichannel_metadata.csv`
     into the external workspace
   - Auto-generates `_seg.npy` labels if they are missing (using either a
     built-in Cellpose model or your last custom model)
   - Trains a custom “whole organoid” model and archives a timestamped copy
     under `cellpose_organoid_workspace/models/`

To re-run segmentation on new folders, call
`python cellpose_organoid/scripts/make_seg_from_model.py --dirs <folders> --model <model_path>`
from the same environment.

### Switching between environments

- ROI labeling and project management → `organoid_roi_incucyte_imaging`
- CellProfiler / Cellpose workflows → `cellprofiler_env`

Deactivate (`conda deactivate`) before activating the other environment to avoid path mix-ups.

---

## 10) File Reference

- `gui_app.py` – The Napari‑based GUI for loading images, drawing ROI, and saving outputs.
- `roi_utils.py` – ROI math: polygon mask, area, perimeter, centroid. Robust fallbacks (scikit‑image → Pillow → NumPy) for mask rasterization.
- `reorganize.py` – CLI to organize raw `.tif/.tiff` into `wells/<well>/day_XX/HHhMMm` and append to `manifest.csv`.
- `environment.yml` – Conda environment spec. Includes `pytest` for tests and `imagecodecs` for TIFF performance.
- `RUN_FIRST_mac_Conda_Create_And_Launch.command` – One‑time setup and launch (macOS).
- `RUN_GUI_mac_Conda.command` – Launch the app (macOS).
- `RUN_Reorganize_mac_Conda.command` – Run the organizer (macOS).

---

## 11) FAQ

Does saving an ROI change my original image?
- No. The original image is not modified. New files are created alongside it.

Why two masked TIFFs?
- One is full‑size; one is cropped to the ROI’s bounding box to save space. Both carry an alpha channel to mark the ROI.

What if I already have an ROI?
- If an `*_roi.json` exists, the app preloads it when the image opens so you can review/edit.

Can I save multiple ROIs per image?
- The intended workflow is one polygon per image. If you draw more than one, only the first polygon is used for saving.

I need help adjusting the workflow for my lab
- Open an issue or share details of your directory layout and file naming. The organizer is customizable.

---

Happy analyzing!
 
---
 
## 12) Sample Data Quick Test (Works Out‑of‑the‑Box)
 
Use this quick recipe to prove everything is working using synthetic TIFFs.
 
1) Generate synthetic images (flat folder):
 
```
conda run -n organoid_roi_incucyte_imaging \
  python tools/make_fake_data.py --raw sample_raw --wells A01 A02 A03 --days 01 --times "00:00" "12:00"

Tip: The generator accepts times as HH:MM, HHMM, or HH (minutes default to 00). These are equivalent:

```
--times "00:00" "12:00"
--times 0000 1200
--times 00 12
```
```
 
2) Reorganize into a project layout:
 
```
conda run -n organoid_roi_incucyte_imaging \
  python reorganize.py --raw sample_raw --out sample_project
```
 
This creates:
 
```
sample_project/
  wells/
    A01/day_01/00h00m/*.tif
    A01/day_01/12h00m/*.tif
    A02/day_01/...  A03/day_01/...
  manifest.csv
```
 
3) Launch the GUI and draw your first ROI:
 
```
conda run -n organoid_roi_incucyte_imaging python gui_app.py
```
 
Open one of the images under `sample_project/wells/...`, draw a polygon, and click “Save ROI”. You should see:
 
- `*_mask.tif`, `*_roi.json`, `*_roi_masked.tif`, `*_roi_masked_cropped.tif` next to the image
- `roi_measurements.csv` in the image folder and at `sample_project/roi_measurements.csv`
 
If all of that appears, your setup is good to go.

Shortcut: one‑click sample data on macOS

- Double‑click `RUN_MakeSampleData_mac_Conda.command` to generate sample data and reorganize it into `sample_project` with sensible defaults (wells A01–A03; day 01; times 00:00 and 12:00).
- You can override via environment variables before running (in Terminal):

```
export TIMES="00:00 06:00 12:00"; export DAYS="01 02"; ./RUN_MakeSampleData_mac_Conda.command
```
