"""PDF export step — convert the filled markdown reports to PDF with pandoc.

The pipeline ends with two markdown reports in ``filled_templates/``
(``solar/report.md`` and ``wind/report.md``). This final brick converts each
of them to a PDF sitting next to the markdown file, by calling the ``pandoc``
command-line tool through :mod:`subprocess`.

Pandoc is an external dependency: install it from its GitHub releases
(https://github.com/jgm/pandoc/releases) and make sure the ``pandoc``
command is on your ``PATH``. Pandoc also needs a PDF engine — this module
uses ``xelatex`` (TeX Live), which handles the accented French text and the
Unicode symbols (``→``, ``·``) present in the reports.

Available as the last step of the top-level ``dc`` entry point:

- ``uv run dc`` — plotting + reporting + PDF export
"""

import argparse
import shutil
import subprocess
from pathlib import Path

# Project root: src/datachallengecapgemini/pdf_export.py -> 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# PDF engine used by pandoc. xelatex copes with the accented French text and
# Unicode symbols found in the reports, unlike pdflatex.
PDF_ENGINE = "xelatex"


def convert_markdown_to_pdf(markdown_path: Path, pdf_path: Path | None = None) -> Path:
    """Convert one markdown file to PDF with the ``pandoc`` command.

    Pandoc resolves relative image paths against the working directory, so
    the command runs with the markdown file's folder as ``cwd`` — the reports
    reference figures through relative paths like ``../../data/plots/...``.

    Returns the path of the generated PDF.

    Raises
    ------
    FileNotFoundError
        If the ``pandoc`` command is not installed.
    RuntimeError
        If pandoc fails to convert the file (missing PDF engine, bad
        markdown, …).
    """
    # Absolute paths: the command runs with the markdown file's folder as
    # cwd (so relative image paths resolve), which would break relative
    # input/output paths.
    markdown_path = markdown_path.resolve()
    if pdf_path is None:
        pdf_path = markdown_path.with_suffix(".pdf")
    else:
        pdf_path = pdf_path.resolve()

    if shutil.which("pandoc") is None:
        raise FileNotFoundError(
            "The 'pandoc' command was not found. Install it from "
            "https://github.com/jgm/pandoc/releases and make sure it is on "
            "your PATH."
        )

    result = subprocess.run(
        [
            "pandoc",
            "--from=markdown-implicit_figures",
            str(markdown_path),
            "--pdf-engine",
            PDF_ENGINE,
            "-V",
            "geometry:margin=2.5cm",
            "-o",
            str(pdf_path),
        ],
        cwd=markdown_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc failed to convert {markdown_path} "
            f"(exit code {result.returncode}):\n{result.stderr.strip()}"
        )
    return pdf_path


def run_pdf_export(reports_output_dir: Path) -> list[Path]:
    """Convert every markdown report under *reports_output_dir* to PDF.

    Looks for ``**/*.md`` files (e.g. ``solar/report.md`` and
    ``wind/report.md``) and writes a PDF with the same name next to each one.
    Returns the list of generated PDF paths.
    """
    markdown_reports = sorted(reports_output_dir.rglob("*.md"))
    if not markdown_reports:
        raise FileNotFoundError(
            f"No markdown reports found in {reports_output_dir} — run the "
            "reporting step first (uv run dc)."
        )

    pdf_paths = []
    for markdown_path in markdown_reports:
        pdf_path = convert_markdown_to_pdf(markdown_path)
        print(f"Saved {pdf_path}")
        pdf_paths.append(pdf_path)

    print(f"Done. PDF reports are in {reports_output_dir}")
    return pdf_paths


def add_pdf_arguments(parser: argparse.ArgumentParser) -> None:
    """Add PDF-export-specific arguments to *parser*."""
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip the final PDF export step (requires pandoc).",
    )
