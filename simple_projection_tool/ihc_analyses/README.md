# Immunohistochemistry Projection Analyses

This folder contains **analysis objects** that sit on top of the TIFF
projections created by `simple_channel_projections.py`. Each object represents
one biological question and is designed so that scientists can extend or adapt
it with minimal Python knowledge. The common structure comes from
[`base.py`](./base.py) and every analysis follows the same sequence of steps:

1. **Import data** - locate the projection TIFFs and collect them in a manifest.
2. **Process data** - read each TIFF, calculate statistics, and return a table.
3. **(Optional) Statistics** - perform comparisons or hypothesis tests.
4. **(Optional) Plotting** - create figures that summarise the findings.
5. **Save outputs** - write the manifest/results tables (and plots) to disk.
6. **Export** - push data somewhere else (shared drive, email, etc.).

The `ProjectionAnalysis` base class handles orchestration and saving, so each
new analysis only needs to focus on the biological logic. When a run begins the
base class checks for `<base-path>/simple_projections/` and automatically
invokes `simple_channel_projections.py` if the exports are missing, ensuring the
post-processing step always has source data to consume.

All derivative artefacts (CSV tables and figures) are saved in a dedicated
`analysis_pipeline/` folder inside each analysis output directory. This makes it
obvious that the content was produced by a downstream analysis rather than the
original projection exporter. The CLI prints a stage-by-stage log (importing,
processing, statistics, plotting, saving) and lists every CSV or figure written
so scientists can confirm the outputs immediately after the run.

---

## Current analyses

### `PCDHvsLHX6_WTvsKO_IHC`

*Location:* [`pcdh_vs_lhx6.py`](./pcdh_vs_lhx6.py)

This analysis studies the wild-type (WT) versus knockout (KO) groups for the
PCDH19 vs LHX6 immunostaining experiment. It demonstrates how to:

- Automatically classify runs into WT/KO based on folder names (`IGI` = WT,
  `IGIKO` = KO).
- Load the 16-bit projection TIFFs (max/mean/median) created by
  `simple_channel_projections.py`.
- Convert each 16-bit TIFF into a floating-point array without scaling and
  compute descriptive statistics for every image (pixel count, mean, median,
  standard deviation, maximum, and a 95 percent confidence interval for the
  mean).
- Restrict calculations to the requested channel(s). The default configuration
  processes both markers: LHX6 (Confocal - Green, 529 nm) and PCDH19 (Confocal -
  Red, 600 nm). Use `--channel` on the CLI to narrow or extend the selection.
  The `LHX6` shortcut maps to “Confocal - Green” and `PCDH19` maps to
  “Confocal - Red”.
- Summarise the WT vs KO comparison per projection type (max/mean/median) using
  both Welch t-tests and Mann-Whitney U tests so publication-ready statistics
  (`N`, group means/medians, SEM, confidence intervals, p-values) are saved for
  reporting.
- Generate matched figures - box plots plus mean+/-SEM bar charts with Arial
  fonts - saved as `.svg` (editable text) and `.png` (300 dpi) in
  `analysis_pipeline/<channel-slug>/figures/`. Each box plot overlays the raw
  per-image means so the distribution remains visible.
- Produce one PNG per TIFF under
  `analysis_pipeline/<channel-slug>/figures/per_image_summaries/` combining the
  projection image with the associated statistics.
- Save the manifest, per-image results, grouped summaries, and hypothesis tests
  as CSV tables in `analysis_pipeline/<channel-slug>/data/` inside
  `<project>/analysis_results/PCDHvsLHX6_WTvsKO_IHC/`.
- Preserve the original run folder name and inferred `subject_label` token
  (`F`/`M`), alongside `channel`, `channel_canonical`, `channel_marker`, and
  `channel_wavelength_nm` columns in every table so downstream comparisons
  between cohorts and markers remain traceable.
- Automatically launches `simple_channel_projections.py` when the required
  exports are absent so the analysis can run from a clean data directory.
- Statistical testing relies on `scipy` (Welch t-tests and Mann-Whitney U). If
  the module is missing, install it in the Conda environment used for the
  analysis.

Use this file as a template for future biological questions - copy it, update the
naming rules and statistics, and register the new class in `__init__.py`.

### `NestinvsDcx_WTvsKO_IHC`

*Location:* [`nestin_vs_dcx.py`](./nestin_vs_dcx.py)

This analysis mirrors the structure above but targets the Nestin/DCX cohort.

- Default channels: Nestin (Confocal - Green, 529 nm), DCX (Confocal - Far red,
  700 nm), and PCDH19 (Confocal - Red, 600 nm). Each channel runs independently
  and writes to its own `analysis_pipeline/<channel-slug>/` folder so figures
  and tables stay separated.
- Outputs include the same per-image statistics, WT vs KO comparisons, and
  statistical tests, annotated with marker metadata and inferred `subject_label`
  values.
- Automatically calls `simple_channel_projections.py` if the simple projections
  folder is absent before analysis begins.
- Recommended base path:
  `/Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt`.
- Saves per-image summary PNGs (projection + stats) under the channel-specific
  figures directory to keep visual context alongside the numbers.
- Channels that are not present in the dataset are skipped with a console
  message so you can confirm which markers were processed.

### Pixel handling cheat-sheet for imaging scientists

- Each projection TIFF is read exactly as exported (unsigned 16-bit). The
  analysis converts the array to `float64` solely to avoid rounding when
  calculating statistics; no normalisation or scaling is applied. The raw 2-D
  pixel matrix is flattened when computing means or medians so every pixel in
  the projection contributes to the statistics.
- Confidence intervals reported in the CSV files refer to the mean pixel
  intensity per image. Group-level summaries use the per-image means so the `N`
  value corresponds to the number of projection images contributing to a
  comparison.
- Figures display the distribution of per-image mean intensities; they do not
  alter the underlying TIFF values. The SVG output keeps text as live Arial
  objects for editing in Illustrator.

---

## How to add a new analysis

1. **Copy the template:** duplicate `pcdh_vs_lhx6.py` and rename the class to
   reflect the new experiment.
2. **Update the logic:** edit `_infer_group`, `_infer_projection_type`, and the
   statistic calculations to match the new data.
3. **Register the analysis:** open `__init__.py` in this folder and add the new
   class to `ANALYSIS_REGISTRY` so the CLI can discover it.
4. **Run the analysis:** from the repository root execute
   ```bash
   python simple_projection_tool/run_projection_analysis.py \
       NewAnalysisName \
       --base-path /path/to/experiment_folder
   ```

By following these steps every new biological question gains a reproducible
pipeline with consistent folder structure and saved outputs.

### Tips when creating new modules

- Keep ``CHANNEL_ALIASES`` and ``CHANNEL_METADATA`` up to date so the command
  line interface exposes friendly marker names while outputs retain canonical
  metadata (marker, wavelength, etc.).
- The base class already re-runs ``simple_channel_projections.py`` if the
  projections are missing, so new analyses do not need bespoke setup code.
- Reuse the per-channel execution pattern from the existing analyses when you
  want independent outputs per marker.
- Update docstrings at the top of each module to describe assumptions (dataset
  layout, markers, aliases). Future contributors can then duplicate the file
  and adjust it with minimal effort.
- Once an analysis is complete you can hand off to cell segmentation tools via
  ``prepare_for_cellprofiler_cellpose.py``, which consumes the manifest tables
  and exports the 16-bit TIFFs grouped by analysis/channel/group.
