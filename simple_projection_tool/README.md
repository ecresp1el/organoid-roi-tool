# Simple Projection Export Tool

This utility creates a standardized set of projections and visualizations for each Imaris `.ims` file in a folder. It lives in its own subdirectory (`simple_projection_tool/`) so it can evolve independently while still relying on the shared `imaris_tools` package in this repository (exposed here via a symlink).

> ℹ️ **How this README relates to the main project guide**
>
> - The root [`README.md`](../README.md) covers the Organoid ROI Tool GUI, project organisation workflow, and an overview of the two Conda environments maintained in this repo.
> - This document focuses specifically on the projection / IHC analysis helpers under `simple_projection_tool/`.
> - When the instructions below mention activating `organoid_roi_incucyte_imaging` or `cellprofiler_env`, refer back to the root README for details on how those environments are created and when to use them.

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

Activate the main ROI environment so the shared dependencies (`imaris_tools`, TIFF stack, plotting libraries) are available:

```bash
conda activate organoid_roi_incucyte_imaging
```

The script relies on:

- `imaris_tools.metadata.read_metadata` (already part of this repository)
- `h5py`, `numpy`, `tifffile`, `matplotlib`

If you also plan to segment the exported projections in CellProfiler or Cellpose, keep the dedicated `cellprofiler_env` (defined in `environment.cellprofiler.yml`) on hand. The segmentation pipeline is described in [Section 8](#8-cellprofilercellpose-export-final-step).

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

> **Note:** The downstream `run_projection_analysis.py` helper automatically
> runs `simple_channel_projections.py` if it detects that
> `<base-path>/simple_projections/` is missing. You can still invoke the
> projection exporter manually (recommended when you want to reuse the exports),
> but it is no longer a hard prerequisite before launching an analysis.

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

---

## 7. Post-processing analyses (PCDH19/LHX6 example)


### 7. Post-processing analyses (PCDH19/LHX6 example)

The projection exports now feed into reproducible **analysis pipelines** stored
in [`ihc_analyses/`](./ihc_analyses/). Each pipeline is a conceptual “analysis
plan” tailored to one biological question and follows the same structure
(`import_data → process_data → run_statistics → generate_plots → save`). Because
the steps are standardised, non-programmers can trust that every analysis collects
the same metadata, produces the same folders, and prints clear status messages.
The first pipeline, `PCDHvsLHX6_WTvsKO_IHC`, shows how this template classifies
WT vs KO folders, computes per-image statistics from the 16-bit TIFFs, compares
the groups with parametric and non-parametric tests, and exports ready-to-interpret
tables and figures.

**Before you run an analysis**

1. Confirm the shared drive is mounted:
   ```bash
   ls /Volumes/Manny4TBUM
   ```
   If this command fails, mount the drive and try again.
2. Confirm the project folder exists. Example:
   ```bash
   ls /Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder
   ```
   The helper will skip channels if the `<base-path>` cannot be found.

To run the analysis from a terminal (make sure the external drive/volume is
mounted so the `<base-path>` exists):

```bash
conda activate organoid_roi_incucyte_imaging
cd /Users/ecrespo/Documents/github_project_folder/organoid-roi-tool
python simple_projection_tool/run_projection_analysis.py \
    PCDHvsLHX6_WTvsKO_IHC \
    --base-path /Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder
```

Key points for non-programmers:

- The command above can be copied verbatim; change `--base-path` if the data
  live somewhere else.
- The analysis defaults to the paired markers: LHX6 (Confocal - Green, 529 nm)
  and PCDH19 (Confocal - Red, 600 nm). Use `--channel` to narrow or extend the
  selection.
- Tables and plots only include the selected channel(s), so the exported figures
  match the specified markers.
- The LHX6 shortcut maps to the metadata label “Confocal - Green”; if your
  projections use different channel names, supply them explicitly with repeated
  `--channel` flags.
- `--channel PCDH19` targets the red channel (“Confocal - Red”, 600 nm) that pairs
  with LHX6 in the experiment.
- Every table carries `channel`, `channel_canonical`, `channel_marker`, and
  `channel_wavelength_nm` columns so downstream analyses can verify the exact
  imaging metadata alongside the measurements.
- All derived artefacts live in
  `<base-path>/analysis_results/PCDHvsLHX6_WTvsKO_IHC/analysis_pipeline/<channel-slug>/`:
  - `data/manifest.csv` - catalogue of every projection inspected for the selected channel(s).
  - `data/results.csv` - per-image pixel statistics (pixel count, mean, median,
    max, standard deviation, 95% confidence interval of the mean).
  - `data/group_summary.csv` - WT/KO summary table with `N`, mean, median, SEM,
    and confidence intervals per projection type.
  - `data/group_comparisons.csv` - Welch t-test and Mann-Whitney U outcomes with
    statistics and p-values for each projection type.
- All tables carry the original run folder name (`sample_id`) and an inferred
  `subject_label` token (`F`/`M`) so you can confirm which cohort each projection
  belongs to when comparing datasets.
- The console output now prints each processing stage (import, process,
  statistics, plotting, saving) and lists every CSV or figure written so you can
  verify the run immediately.
- If `<base-path>/simple_projections/` is missing, the CLI triggers
  `simple_channel_projections.py` automatically before analysing.
- `figures/` - paired boxplots plus mean+/-SEM charts (SVG and 300 dpi PNG)
  using Arial fonts so the annotations remain editable in Illustrator. Each plot
  overlays the per-image data points on top of the boxplot so the distribution is
  visible.
- `figures/per_image_summaries/` - one PNG/SVG per sample showing a 3×2 grid of
  max/mean/median projections (top row) and their statistics (bottom row), making
  it easy to correlate each TIFF with the CSV values without leaving the dataset.
- Statistical tests use `scipy`; install it in the same environment if it is not
  already available.

To process the Nestin/DCX dataset (remember to verify the base path first):

```bash
conda activate organoid_roi_incucyte_imaging
cd /Users/ecrespo/Documents/github_project_folder/organoid-roi-tool
python simple_projection_tool/run_projection_analysis.py \
    NestinvsDcx_WTvsKO_IHC \
    --base-path /Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt
```

- Defaults analyse Nestin (Confocal - Green, 529 nm), DCX (Confocal - Far red, 700 nm),
  and PCDH19 (Confocal - Red, 600 nm). Each channel runs independently and writes to
  its own subdirectory under
  `<base-path>/analysis_results/NestinvsDcx_WTvsKO_IHC/analysis_pipeline/<channel-slug>/`.
- Output tables include the same channel metadata columns and inferred `subject_label`
  so sex/cohort information stays attached to every measurement.
- Missing projections in `<base-path>/simple_projections/` are generated on the fly
  before any analysis starts.
- If a requested channel is absent in the dataset, the CLI prints a skip message
  and continues with the remaining markers.
- Per-image summary PNG/SVG files are written under
  `analysis_results/NestinvsDcx_WTvsKO_IHC/analysis_pipeline/<channel>/figures/per_image_summaries/`;
  each summarises max/mean/median projections for a sample in one figure.

- To build a new analysis, copy the template described in
  [`ihc_analyses/README.md`](./ihc_analyses/README.md) and register it in
  `ihc_analyses/__init__.py`.

### 8. CellProfiler/Cellpose export (final step)

After running the projections and analysis, gather 16-bit TIFFs for downstream
segmentation tools:

```bash
conda activate organoid_roi_incucyte_imaging
cd /Users/ecrespo/Documents/github_project_folder/organoid-roi-tool
python simple_projection_tool/prepare_for_cellprofiler_cellpose.py \
    --base-path /Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder \
    --analysis PCDHvsLHX6_WTvsKO_IHC
```

Repeat with ``NestinvsDcx_WTvsKO_IHC`` (or omit ``--analysis`` to export every
registered analysis). The script confirms that projections and analysis results
exist, then copies every 16-bit TIFF into
``<base-path>/cellprofilerandcellpose_folder/<analysis>/<channel>/<group>/``
and writes ``cellprofilerandcellpose_metadata.csv`` so each exported file can be
linked back to its source.

> **Launching CellProfiler / Cellpose**
>
> Use the separate environment described in the root README:
> ```bash
> conda activate cellprofiler_env
> cellprofiler          # CellProfiler GUI (legacy wxPython stack)
> cellpose              # Cellpose GUI (Qt/PyQt6)
> ```
> This environment is built from `environment.cellprofiler.yml` and pins the legacy imaging libraries that CellProfiler expects, while the main ROI / projection tooling stays on the modern `organoid_roi_incucyte_imaging` stack.

### Developing additional analyses

1. Duplicate one of the existing analysis modules in
   `simple_projection_tool/ihc_analyses/` (for example
   `pcdh_vs_lhx6.py`). Rename the class and update the module-level docstring to
   describe the new experiment.
2. Update ``CHANNEL_ALIASES`` and ``CHANNEL_METADATA`` so the CLI accepts the
   markers you care about while CSV/figure outputs retain canonical metadata.
3. Adapt helper functions such as ``_infer_group`` or ``_infer_projection_type``
   if your folder structure differs. The remainder of the pipeline (manifest,
   statistics, plotting) typically works unchanged.
4. Register the new analysis in `ihc_analyses/__init__.py` by adding an entry to
   ``ANALYSIS_REGISTRY``.
5. Execute the analysis via ``run_projection_analysis.py``. The helper verifies
   or regenerates projection exports, runs the analysis per channel, and prints
   the generated artefacts so you can validate the results immediately.
