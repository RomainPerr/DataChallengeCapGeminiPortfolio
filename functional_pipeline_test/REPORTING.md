# Functional pipeline: statistics and PDF report

Install a LaTeX compiler once (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends latexmk texlive-latex-base texlive-latex-recommended
```

From the repository root, run the functional pipeline for an asset type:

```bash
.venv/bin/python -m functional_pipeline_test.pipeline --asset-type wind
# Or: --asset-type solar
```

The existing embedding and fusion-validation stages use Azure and their caches.
After classification, the new statistical/reporting stage runs entirely locally
with pandas, SciPy, Matplotlib and LaTeX; it makes no LLM calls. The notebook is
not needed. Calling `get_clusters_and_classified_incidents("wind")` directly also
generates the report, available through `result.report`.

Outputs are under `functional_pipeline_test/reports/wind/` (or `solar/`):

- `report.pdf`, `report.tex`, and `compile.log`.
- `classified_incidents.csv`, for subsequent report regeneration.
- `topic_summary.csv`: topic counts and severe proportion supporting the plots.
- `statistical_tests.csv`: statistics, effect sizes, p-values, BH-adjusted
  q-values, sample sizes and reasons for skipped tests.
- `plots/`: topic frequencies and severity composition, each as PDF/PNG.

Regenerate the report from saved classifications without running Azure or clustering:

```bash
.venv/bin/python -m functional_pipeline_test.statistical_report \
  functional_pipeline_test/reports/wind/classified_incidents.csv --output-dir functional_pipeline_test/reports/wind
```

The report contains exactly three analysis items per asset type:

1. Topic-frequency plot.
2. Severity-composition plot.
3. Topic–severity chi-square test (seeded Monte Carlo p-values for sparse tables).

Tests exclude noise (`topic=-1`); both plots retain it. Insufficient categories
are marked as skipped. Benjamini–Hochberg correction applies across asset types;
for a single asset type, the adjusted p-value equals the raw p-value.

These are exploratory associations: repeated equipment events may be dependent,
topics are learned from the same data, and the existing fusion selection can
use severity. The report documents these limitations and the supplied date range.

`generate_statistical_report()` calls `generate_report_content()` to create the
two plot images, test result and LaTeX source, then `compile_latex()` to build
the PDF. These stages are appended to `pipeline.drawio`; the original diagram
elements are preserved.

Edit `functional_pipeline_test/report_template.tex` to customize the report's
layout. Each run rewrites `report.tex` from that template and recomputes the
tables/plots, then recompiles the PDF. To recompile an existing source by hand:

```bash
cd functional_pipeline_test/reports/wind
latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape report.tex
```

Use `--no-compile` for source/plots only, `--output-dir` for a different destination,
or `report_output_dir=None` in the Python pipeline function to disable reporting.

The report uses the NumPy, pandas, Matplotlib and SciPy packages already installed
in the project environment. Sparse-table simulation requires SciPy >= 1.15.
