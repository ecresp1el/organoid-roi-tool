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
- Compute descriptive statistics for each image, including a 95 % confidence
  interval for the mean pixel intensity.
- Save the manifest and the results to
  `<project>/analysis_results/PCDHvsLHX6_WTvsKO_IHC/`.

Use this file as a template for future biological questions—copy it, update the
naming rules and statistics, and register the new class in `__init__.py`.

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
