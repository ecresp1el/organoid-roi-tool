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

Open images:
- Click “Open Image” to pick a single `.tif`/`.tiff`, or
- Click “Open Folder” to browse all TIFFs in a folder, or
- Drag and drop a file into the window.

If your image has multiple frames, the app automatically applies a maximum intensity projection so you see a 2D image.

Adjust how it looks:
- The contrast is auto‑stretched to the 2nd–98th percentile; you can tweak layer contrast in Napari if needed.
- Use the mouse wheel or trackpad to zoom; drag to pan.

Draw an ROI:
- In the left layer list, click the “ROI” layer (already selected) and ensure it’s in “Add Polygon” mode.
- Click to add vertices around your organoid. Click the first point (or double‑click) to close.
- Tip: Your goal is one polygon per image.

Save the ROI:
- Click “Save ROI” (or press S / Cmd+S).
- If “Auto‑advance” is checked, the app will move to the next image in the same folder after saving.

Delete the ROI:
- Click “Delete ROI” (or press D), then confirm.

Browse images in the same folder:
- Use the Prev / Next buttons, or press Left/Right arrow keys.
- “Next Unlabeled” jumps to the next image without a saved ROI.

Session & progress:
- The app shows a progress bar (done / total) for the current folder.
- Your place is saved automatically (folder, last image). Use File → “Resume Last Session” to pick up where you left off.
- “Stop / Save Session” saves your state so you can safely close the app.

Project‑wide progress dashboard:
- Set the project root via Project → “Set Project Root…”. The project is expected to have a `wells/` directory (e.g., `project/wells/C12/day_03/...`).
- Open the dashboard via Project → “Open Progress Dashboard”.
- The dashboard shows totals per well and per day: Done, Total, and %.
- “Next Unlabeled” in each row lets you jump straight to the next image without a saved ROI in that group.
- Use “Refresh” to rescan after making changes; it also auto‑refreshes after you save an ROI.

Preloading existing ROI:
- If a matching ROI JSON file is already present for the image, the app auto‑loads it so you can review or edit.

---

## 4) What Gets Saved

When you save an ROI for `image.tif` in a folder, these files are written in the same folder:

- `image_mask.tif`: a binary mask (0/255) showing the ROI area.
- `image_roi.json`: coordinates of the polygon you drew.
- `image_roi_masked.tif`: full‑size TIFF with an associated alpha channel marking the ROI. Original pixels are preserved; the alpha marks outside‑ROI as transparent.
- `image_roi_masked_cropped.tif`: a tighter crop around the ROI, also with alpha.

Measurements CSVs (upsert one row per image):
- Local folder CSV: `<image-folder>/roi_measurements.csv`
- If your image path looks like `<project-root>/wells/...`, a project CSV is also updated: `<project-root>/roi_measurements.csv`

Each row includes:
- image_path, well, day, time
- area_px, perimeter_px, centroid_yx
- pixel_size_um (if readable from TIFF tags)

Note on pixel size:
- The app tries to read pixel size from the TIFF XResolution tag and converts to micrometers per pixel. If missing, the field is left blank in the CSV.

---

## 5) Keyboard Shortcuts

- Save ROI: `S` or `Cmd+S`
- Next image: Right Arrow
- Previous image: Left Arrow
- Delete ROI: `D`
- Quit: `Cmd+Q`

“Auto‑advance” toggle moves to the next image automatically after a successful save.

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
- After saving, the app checks that files were written. If something’s missing, it shows an error. Ensure you have write permission to the folder.

CSV not updating
- The tool “upserts” one row per image (no duplicates). Close the CSV in Excel while saving from the app.

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

## 9) File Reference

- `gui_app.py` – The Napari‑based GUI for loading images, drawing ROI, and saving outputs.
- `roi_utils.py` – ROI math: polygon mask, area, perimeter, centroid. Robust fallbacks (scikit‑image → Pillow → NumPy) for mask rasterization.
- `reorganize.py` – CLI to organize raw `.tif/.tiff` into `wells/<well>/day_XX/HHhMMm` and append to `manifest.csv`.
- `environment.yml` – Conda environment spec. Includes `pytest` for tests and `imagecodecs` for TIFF performance.
- `RUN_FIRST_mac_Conda_Create_And_Launch.command` – One‑time setup and launch (macOS).
- `RUN_GUI_mac_Conda.command` – Launch the app (macOS).
- `RUN_Reorganize_mac_Conda.command` – Run the organizer (macOS).

---

## 10) FAQ

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
 
## 11) Sample Data Quick Test (Works Out‑of‑the‑Box)
 
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
