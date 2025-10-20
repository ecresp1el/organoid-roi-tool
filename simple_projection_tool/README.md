# Simple Projection Export Tool

This utility creates a standardized set of projections and visualizations for each Imaris `.ims` file in a folder. It lives in its own subdirectory (`simple_projection_tool/`) so it can evolve independently while still relying on the shared `imaris_tools` package in this repository (exposed here via a symlink).

---

## 1. What the script does

1. **Discovers `.ims` files** in the folder you provide (optionally recursive).
2. **Reads channel metadata** (names, colours, wavelengths) through `imaris_tools.metadata.read_metadata`.  
   *Only this helper from the main project is required; it is symlinked into the folder so no installation steps are needed beyond activating the existing Conda environment.*
3. **Streams each channel volume** with `h5py` and computes the raw:
   - Max intensity projection (brightest voxels along Z)
   - Mean projection (average intensity along Z)
   - Median projection (middle intensity along Z)
4. **Exports three sets of outputs** per channel:

   | Location | Description |
   | -------- | ----------- |
   | `simple_projections/<ims_stem>/16bit/` | Max/mean/median TIFFs stored as unsigned 16-bit. If the source data already fits in ≤16-bit it is preserved verbatim. |
   | `simple_projections/<ims_stem>/8bit/`  | Max/mean/median TIFFs normalised into unsigned 8-bit for quick viewing in generic software. |
   | `simple_projections/<ims_stem>/figures/` | PNG panels coloured with the Imaris channel colour and annotated with a colour bar. Subfolders describe the intensity scaling used: |
| `figures/raw_min_max/` | Raw minimum → maximum stretch (no scaling). |
| `figures/percentile_95/` | Minimum → 95th percentile stretch. `vmin = data.min()`, `vmax = min(data.max(), np.percentile(data, 95.0))` (see `_determine_scale(..., mode="percentile")`). |
| `figures/median_mad/` | Median ± 3×MAD (median absolute deviation). `median = np.median(data)` and `mad = np.median(|data - median|)`, then `vmin = max(raw_min, median - 3×mad)`, `vmax = min(raw_max, median + 3×mad)` (implemented in `_determine_scale(..., mode="mad")`). |

Both percentile- and MAD-based limits are calculated per image, so the PNGs faithfully reflect the raw dynamic range while emphasising structures of interest. The TIFFs remain untouched.

For reference:

- Raw TIFFs live in `16bit/` and `8bit/` exactly as returned by `_to_uint16` and `_to_uint8`.
- Figure PNGs use the same projections but display data with the alternate scaling above.
- Channel names in every filename come from Imaris metadata (sanitised to be filesystem-safe).

   Each `figures/` directory also carries a `README.txt` that explains the scaling strategies so collaborators can interpret the outputs without reading the code.

---

## 2. Requirements

Activate the project’s Conda environment so the shared dependencies are available:

```bash
conda activate organoid_roi_incucyte_imaging
```

The script relies on:

- `imaris_tools.metadata.read_metadata` (already part of this repository)
- `h5py`, `numpy`, `tifffile`, `matplotlib`

---

## 3. Running the script

Typical workflow from a fresh terminal:

```bash
conda activate organoid_roi_incucyte_imaging
cd /Users/ecrespo/Documents/github_project_folder/organoid-roi-tool
python simple_projection_tool/simple_channel_projections.py \
    --source /Users/ecrespo/Desktop/nestin_dcx_pcdh19_kovswt
```

Another example for the archival dataset on the external drive:

```bash
conda activate organoid_roi_incucyte_imaging
cd /Users/ecrespo/Documents/github_project_folder/organoid-roi-tool
python simple_projection_tool/simple_channel_projections.py \
    --source /Volumes/Manny4TBUM/2025-10-15 \
    --recursive
```

Options:

- `--source`: Root folder containing `.ims` files (defaults to `/Volumes/Manny4TBUM/2025-10-15`).
- `--recursive`: Process files inside subdirectories as well.

You can invoke the script from any working directory as long as the repository is accessible and the Conda environment is active. The examples above show the exact commands tested in February 2025; replicate them verbatim to reproduce the results.

---

## 4. Output layout

```
simple_projections/
├── <ims_stem>/
│   ├── 16bit/
│   │   ├── <channel_name>_max.tif
│   │   ├── <channel_name>_mean.tif
│   │   └── <channel_name>_median.tif
│   ├── 8bit/
│   │   ├── <channel_name>_max.tif
│   │   ├── <channel_name>_mean.tif
│   │   └── <channel_name>_median.tif
│   └── figures/
│       ├── README.txt
│       ├── raw_min_max/
│       │   └── <channel_name>_<projection>.png
│       ├── percentile_95/
│       │   └── <channel_name>_<projection>.png
│       └── median_mad/
│           └── <channel_name>_<projection>.png
```

Channel names are pulled from the Imaris metadata, sanitised for filesystem safety. Each PNG title reports the scaling range (e.g. “min=0.0, max=1200.5”), and the colour bar is tied to the channel’s original colour.

---

## 5. Extending or modifying

- To change the scaling strategies or add new figure styles, edit `_save_colorbar_figure` and `_determine_scale` in `simple_channel_projections.py`.
- Additional TIFF outputs can be hooked into the loop where projections are computed (`projections = [...]`).
- Because the script sits in its own folder with an `imaris_tools` symlink, updating the main project’s utilities immediately affects this tool without duplicate code.

---

## 6. Reproducibility checklist

- Path to inputs recorded in the command you run.
- Outputs grouped by input filename.
- Scaling strategies documented in both the README and the per-folder manifest.
- No destructive operations: the script only reads `.ims` files and writes to `simple_projections/`.
