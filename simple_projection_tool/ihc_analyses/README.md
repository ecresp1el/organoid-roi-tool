# Immunohistochemistry Projection Analyses

This folder contains **analysis objects** that sit on top of the TIFF
projections created by `simple_channel_projections.py`.  Each object represents
one biological question and is designed so that scientists can extend or adapt
it with minimal Python knowledge.  The common structure comes from
[`base.py`](./base.py) and every analysis follows the same sequence of steps:

1. **Import data** – locate the projection TIFFs and collect them in a manifest.
2. **Process data** – read each TIFF, calculate statistics, and return a table.
3. **(Optional) Statistics** – perform comparisons or hypothesis tests.
4. **(Optional) Plotting** – create figures that summarise the findings.
5. **Save outputs** – write the manifest/results tables (and plots) to disk.
6. **Export** – push data somewhere else (shared drive, email, etc.).

The `ProjectionAnalysis` base class handles orchestration and saving, so each
new analysis only needs to focus on the biological logic.

All derivative artefacts (CSV tables and figures) are saved in a dedicated
`analysis_pipeline/` folder inside each analysis output directory.  This makes it
obvious that the content was produced by a downstream analysis rather than the
original projection exporter.

---

## Current analyses

### `PCDHvsLHX6_WTvsKO_IHC`

*Location:* [`pcdh_vs_lhx6.py`](./pcdh_vs_lhx6.py)

This analysis studies the wild-type (WT) versus knockout (KO) groups for the
PCDH19 vs LHX6 immunostaining experiment.  It demonstrates how to:

- Automatically classify runs into WT/KO based on folder names (`IGI` = WT,
  `IGIKO` = KO).
- Load the 16-bit projection TIFFs (max/mean/median) created by
  `simple_channel_projections.py`.
- Convert each 16-bit TIFF into a floating-point array without scaling and
  compute descriptive statistics for every image (pixel count, mean, median,
  standard deviation, maximum, and a 95 % confidence interval for the mean).
- Summarise the WT vs KO comparison per projection type (max/mean/median) using
  both Welch t-tests and Mann–Whitney U tests so publication-ready statistics
  (`N`, group means/medians, SEM, confidence intervals, p-values) are saved for
  reporting.
- Generate matched figures—box plots plus mean±SEM bar charts with Arial fonts—
  saved as `.svg` (editable text) and `.png` (300 dpi) in
  `analysis_pipeline/figures/`.
- Save the manifest, per-image results, grouped summaries, and hypothesis tests
  as CSV tables in `analysis_pipeline/data/` inside
  `<project>/analysis_results/PCDHvsLHX6_WTvsKO_IHC/`.
- Statistical testing relies on `scipy` (Welch t-tests and Mann–Whitney U).  If
  the module is missing, install it in the Conda environment used for the
  analysis.

Use this file as a template for future biological questions—copy it, update the
naming rules and statistics, and register the new class in `__init__.py`.

### Pixel handling cheat-sheet for imaging scientists

- Each projection TIFF is read exactly as exported (unsigned 16-bit).  The
  analysis converts the array to `float64` solely to avoid rounding when
  calculating statistics—no normalisation or scaling is applied.
- Confidence intervals reported in the CSV files refer to the mean pixel
  intensity per image.  Group-level summaries use the per-image means so the
  `N` value corresponds to the number of projection images contributing to a
  comparison.
- Figures display the distribution of per-image mean intensities; they do **not**
  alter the underlying TIFF values.  The SVG output keeps text as live Arial
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
