"""DataChallengeCapGemini — incident analysis pipeline for PV and wind assets."""

import argparse

from datachallengecapgemini.plot_factory import (
    PlotFactory,
    SolarPlotFactory,
    WindPlotFactory,
)
from datachallengecapgemini.preprocessing import (
    ocr_directory,
    ocr_pdf,
    parse_ocr_directory,
    parse_ocr_json,
)
from datachallengecapgemini.report_factory import (
    ReportFactory,
    SolarReportFactory,
    WindReportFactory,
)

__all__ = [
    "PlotFactory",
    "ReportFactory",
    "SolarPlotFactory",
    "SolarReportFactory",
    "WindPlotFactory",
    "WindReportFactory",
    "ocr_directory",
    "ocr_pdf",
    "parse_ocr_directory",
    "parse_ocr_json",
]


def main() -> None:
    """CLI entry point — ``uv run dc``.

    Subcommands:

    - ``dc`` (no subcommand) — main pipeline: clustering (skipped by default,
      use ``--all`` to recompute) → plotting → reporting → PDF export.
    - ``dc preprocess`` — OCR scanned PDFs and parse them into a merged CSV.
    - ``dc equipment`` — per-equipment incident reports (solar or wind, ``--asset``).
    """
    from datachallengecapgemini.clustering import add_clustering_arguments
    from datachallengecapgemini.pdf_export import add_pdf_arguments
    from datachallengecapgemini.plot_factory import (
        add_equipment_plotting_arguments,
        add_plotting_arguments,
    )
    from datachallengecapgemini.preprocessing import add_preprocessing_arguments
    from datachallengecapgemini.report_factory import (
        add_equipment_report_arguments,
        add_report_arguments,
    )

    parser = argparse.ArgumentParser(
        prog="dc",
        description="Incident analysis pipeline for PV (solar) and wind assets.",
    )
    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    # --- Pipeline arguments (default when no subcommand is given) ---
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recompute every step, including clustering (CSV → CSV). "
        "By default, clustering is skipped and the existing clusterised "
        "CSVs are reused.",
    )
    add_clustering_arguments(parser)
    add_plotting_arguments(parser)
    add_report_arguments(parser)
    add_pdf_arguments(parser)

    # --- dc preprocess ---
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="OCR scanned PDFs and parse them into a merged incidents CSV.",
        description="OCR scanned PDFs with Surya, then parse the results into a CSV.",
    )
    add_preprocessing_arguments(preprocess_parser)

    # --- dc equipment ---
    equipment_parser = subparsers.add_parser(
        "equipment",
        help="Per-equipment incident reports (solar or wind, --asset), one Azure OpenAI agent per section.",
        description="Per-equipment incident reports (solar or wind, --asset), one Azure OpenAI agent per section.",
    )
    equipment_parser.add_argument(
        "--site", required=True, help="Site code (e.g. OUEST1) or full site name."
    )
    equipment_parser.add_argument(
        "--equipment",
        nargs="+",
        help="Report on these equipment only (default: the critical equipment of the site).",
    )
    equipment_parser.add_argument(
        "--all-equipment",
        action="store_true",
        help="Report on every equipment of the site, not only the critical ones.",
    )
    add_equipment_plotting_arguments(equipment_parser)
    add_equipment_report_arguments(equipment_parser)

    args = parser.parse_args()

    if args.command == "preprocess":
        _run_preprocess(args)
    elif args.command == "equipment":
        _run_equipment(args)
    else:
        _run_pipeline(args)


def _run_pipeline(args: argparse.Namespace) -> None:
    """Run the main pipeline: clustering (optional) → plotting → reporting → PDF."""
    from datachallengecapgemini.clustering import ClusteringArgs, run_clustering
    from datachallengecapgemini.pdf_export import run_pdf_export
    from datachallengecapgemini.plot_factory import run_plotting
    from datachallengecapgemini.report_factory import run_reporting

    if args.all:
        print()
        print("=" * 60)
        print("Clustering: topic modeling of solar and wind incidents")
        print("=" * 60)
        run_clustering(
            ClusteringArgs(
                input_csv=args.clustering_input_csv,
                solar_output_csv=args.solar_csv,
                wind_output_csv=args.wind_csv,
                min_cluster_size_start=args.min_cluster_size_start,
                max_cluster_size_fraction=args.max_cluster_size_fraction,
                min_cluster_size_step=args.min_cluster_size_step,
            )
        )
    else:
        print(
            "Skipping clustering (CSV → CSV step), assuming the clusterised "
            f"solar and wind CSVs already exist at {args.solar_csv} and "
            f"{args.wind_csv}. Use --all to recompute."
        )

    print()
    print("=" * 60)
    print("Plotting: generating separate solar and wind plots")
    print("=" * 60)
    run_plotting(args.solar_csv, args.wind_csv, args.plots_output_dir)

    print()
    print("=" * 60)
    print("Reporting: filling solar and wind markdown templates")
    print("=" * 60)
    run_reporting(
        args.solar_csv,
        args.wind_csv,
        args.plots_output_dir,
        args.templates_dir,
        args.reports_output_dir,
    )

    if args.skip_pdf:
        print("Skipping PDF export (--skip-pdf)...")
    else:
        print()
        print("=" * 60)
        print("PDF export: converting markdown reports to PDF with pandoc")
        print("=" * 60)
        run_pdf_export(args.reports_output_dir)


def _run_preprocess(args: argparse.Namespace) -> None:
    """Run the preprocessing pipeline (OCR + parse + merge)."""
    from datachallengecapgemini.preprocessing import (
        PreprocessingArgs,
        run_preprocessing,
    )

    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be at least 1")

    run_preprocessing(
        PreprocessingArgs(
            input_dir=args.input_dir,
            digital_input_dir=args.digital_input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            force=args.force,
            output_csv=args.output_csv,
            output_digital_csv=args.output_digital_csv,
            output_merged_csv=args.output_merged_csv,
            reference_csv=args.reference_csv,
        )
    )


def _run_equipment(args: argparse.Namespace) -> None:
    """Run the equipment-level reports for one site.

    plot_factory writes the figures and JSON of every selected equipment,
    then report_factory runs one Azure OpenAI agent per report section and
    assembles the report.
    """
    from datachallengecapgemini.equipment_stats import (
        compute_criticality,
        has_failure_modes,
        load_incidents,
        select_equipment,
        site_code,
    )
    from datachallengecapgemini.plot_factory import run_equipment_plotting
    from datachallengecapgemini.report_factory import run_equipment_reporting

    csv_path = args.wind_csv if args.asset == "wind" else args.solar_csv
    df = load_incidents(csv_path)
    criticality = compute_criticality(df)
    equipment_ids = select_equipment(
        criticality, args.site, args.equipment, args.all_equipment
    )
    skipped = [e for e in equipment_ids if not has_failure_modes(df, e)]
    equipment_ids = [e for e in equipment_ids if e not in skipped]
    if skipped:
        print(
            "Skipping (no moderate or severe incident with a failure mode): "
            + ", ".join(skipped)
        )
    if not equipment_ids:
        print("No equipment to report on.")
        return

    site_dir = args.equipment_output_dir / site_code(equipment_ids[0])
    site_dir.mkdir(parents=True, exist_ok=True)
    on_site = criticality[criticality["equipment_id"].map(site_code) == site_dir.name]
    on_site.to_csv(site_dir / "criticality.csv", index=False)
    print(f"{len(equipment_ids)} equipment: {', '.join(equipment_ids)}")

    print()
    print("=" * 60)
    print("Plotting: figures and JSON per equipment")
    print("=" * 60)
    run_equipment_plotting(df, criticality, equipment_ids, site_dir, args.asset)

    print()
    print("=" * 60)
    print("Reporting: one Azure OpenAI agent per report section")
    print("=" * 60)
    run_equipment_reporting(
        equipment_ids,
        site_dir,
        args.templates_dir,
        language=args.language,
        dry_run=args.dry_run,
        force=args.force,
        pdf=args.pdf,
    )
