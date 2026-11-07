"""Preprocessing utilities — OCR scanned PDFs with Surya, then parse the results.

Two steps:

1. **OCR** — ``ocr_directory`` runs Surya on scanned PDFs and writes one JSON
   file per PDF. Already-processed PDFs are skipped unless ``force=True``.
2. **Parse** — ``parse_ocr_directory`` reads the OCR JSONs, extracts the
   metadata table and incident description, and writes a CSV compatible with
   ``incidents_students.csv``. This step is cheap and always runs.

Both steps are importable and also available as a CLI command:

- ``uv run dc preprocess`` — full pipeline (OCR + parse)
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pdfplumber as pb
from surya.inference import SuryaInferenceManager
from surya.input.load import load_pdf
from surya.recognition import RecognitionPredictor
from surya.settings import settings

# Project root: src/datachallengecapgemini/preprocessing.py -> 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Parse an HTML ``<table>`` into a list of rows (each row = list of cell texts)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if (
            tag in ("td", "th")
            and self._current_row is not None
            and self._current_cell is not None
        ):
            self._current_row.append(self._current_cell.strip())
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell += data


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    return re.sub(r"<[^>]+>", "", html).strip()


# ---------------------------------------------------------------------------
# Field mapping: French table labels -> CSV column names
# ---------------------------------------------------------------------------

TABLE_FIELD_MAP: dict[str, str] = {
    "Site": "site",
    "Type d'installation": "asset_type",
    "Équipement": "equipment_id",
    "Date": "date",
    "Sévérité": "severity",
    "Durée d'arrêt (h)": "downtime_hours",
    "Coût estimé (€)": "estimated_cost_eur",
    "Conditions météo": "weather_condition",
    "Technicien": "technician_id",
}

ASSET_TYPE_MAP: dict[str, str] = {
    "Centrale photovoltaïque": "solar",
    "Parc éolien": "wind",
}

# Fields that must be present and non-empty for a parse to count as "success"
REQUIRED_FIELDS: list[str] = [
    "site",
    "asset_type",
    "equipment_id",
    "date",
    "severity",
    "downtime_hours",
    "estimated_cost_eur",
    "weather_condition",
    "technician_id",
    "text",
]

# CSV column order (matches incidents_students.csv)
CSV_COLUMNS: list[str] = [
    "incident_id",
    "site",
    "asset_type",
    "equipment_id",
    "date",
    "severity",
    "downtime_hours",
    "estimated_cost_eur",
    "weather_condition",
    "technician_id",
    "text",
    "document_format",
    "document_path",
]


### Tools to process flat pdf report


def digital_pdf_to_table_record(pdf_path: Path) -> dict | None:
    """Turn a digital pdf of an incident report into a record dict"""
    row_elements: list[list] = []
    with pb.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()  # list of list-of-lists
            for table in tables:
                for row in table:
                    row_elements.append(row.copy())
    records = {}
    num_elements = len(row_elements)
    for i, row in enumerate(row_elements):
        field_name: str | None = row[0] if len(row) > 0 else None

        field_value: str | None = row[1] if len(row) > 1 else None

        if field_name is not None:
            actual_name = TABLE_FIELD_MAP.get(field_name)

            if i == num_elements - 1:  # last element is the description
                actual_name = "text"
                field_value = row[0] if len(row) > 0 else None

            if actual_name is not None:
                # This mean the field was correctly identified
                if field_value is None:
                    print(
                        f"No value identified for field {field_name} for document {pdf_path.name}, skipping it ..."
                    )
                    return None
                records[actual_name] = field_value
    if set(records.keys()) == set(REQUIRED_FIELDS):
        records["document_format"] = "pdf_digital"
        records["document_path"] = f"data/pdf_digital/{pdf_path.stem}.pdf"
        try:
            records["incident_id"] = int(pdf_path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            return None
        return records
    else:
        print(f"Document {pdf_path.name} is missing required metadata, skipping it ...")
        return None


def digital_pdf_dir_to_csv(input_path: Path, output_csv_path: Path) -> tuple[int, int]:
    """turn every digital pdf from input_path to records, then turn save these records to a csv file and return the number of success and the total number"""
    pdf_files = input_path.glob("*.pdf")
    n_total = 0
    n_success = 0
    records: list[dict] = []
    for pdf_file in pdf_files:
        n_total += 1
        record = digital_pdf_to_table_record(pdf_file)
        if record is None:
            continue
        records.append(record)
        n_success += 1
        print(f"Processing {pdf_file.name}, n_processed = {n_total}")

    print(f"{n_success}/{n_total} digital pdf were successfully preprocessed !")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col, "") for col in CSV_COLUMNS})
    print(f"Wrote csv file to {output_csv_path}")
    return n_success, n_total


### Tools to process scanned documents


# ---------------------------------------------------------------------------
# Step 1 — OCR (Surya)
# ---------------------------------------------------------------------------


def ocr_pdf(pdf_path: Path, predictor: RecognitionPredictor) -> dict:
    """Run OCR on a single PDF and return the result as a JSON-serialisable dict."""
    images, _ = load_pdf(str(pdf_path))
    page_results = predictor(images, full_page=True)
    return {
        "source_pdf": pdf_path.name,
        "pages": [
            {"page_number": page_number, **page.model_dump(mode="json")}
            for page_number, page in enumerate(page_results, start=1)
        ],
    }


def ocr_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
) -> list[Path]:
    """OCR every PDF in *input_dir*, writing one JSON file per PDF into *output_dir*.

    Parameters
    ----------
    input_dir:
        Directory containing the scanned ``*.pdf`` files.
    output_dir:
        Directory where ``<stem>.json`` results are written.
    limit:
        Process at most this many PDFs (after sorting by incident number).
        ``None`` means no limit.
    force:
        When ``False`` (default), PDFs whose JSON output already exists are
        skipped. When ``True``, every PDF is re-processed.

    Returns
    -------
    list[Path]
        Paths of all JSON files in *output_dir* corresponding to the input PDFs
        (both newly written and previously existing).
    """
    pdf_files = sorted(
        input_dir.glob("*.pdf"),
        key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[-1]),
    )
    if limit is not None:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into already-done and to-do
    todo: list[tuple[Path, Path]] = []
    done: list[Path] = []
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.json"
        if output_path.exists() and not force:
            done.append(output_path)
        else:
            todo.append((pdf_path, output_path))

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    print(f"  Already processed (skipped): {len(done)}")
    print(f"  To process:                  {len(todo)}")

    if not todo:
        print("Nothing to do — all PDFs already have OCR output.")
        return done

    if not settings.SURYA_INFERENCE_URL:
        raise RuntimeError(
            "No external Surya inference server configured. Start the llama.cpp "
            "server and set SURYA_INFERENCE_URL (for example, "
            "http://127.0.0.1:8080/v1) before running preprocessing."
        )

    manager = SuryaInferenceManager(method="llamacpp")
    predictor = RecognitionPredictor(manager)
    print(f"Connecting to the Surya {manager.method} server...", flush=True)
    manager.start()
    print("Surya server is ready.", flush=True)

    try:
        for index, (pdf_path, output_path) in enumerate(todo, start=1):
            print(f"[{index}/{len(todo)}] Loading {pdf_path.name}...", flush=True)
            images, _ = load_pdf(str(pdf_path))
            print(
                f"[{index}/{len(todo)}] Running OCR on {len(images)} page(s)...",
                flush=True,
            )
            page_results = predictor(images, full_page=True)

            result = {
                "source_pdf": pdf_path.name,
                "pages": [
                    {"page_number": page_number, **page.model_dump(mode="json")}
                    for page_number, page in enumerate(page_results, start=1)
                ],
            }
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{index}/{len(todo)}] Saved {output_path}", flush=True)
            done.append(output_path)
    finally:
        manager.stop()

    print(f"Done. OCR results are in {output_dir}", flush=True)
    return done


# ---------------------------------------------------------------------------
# Step 2 — Parse OCR JSONs into structured rows
# ---------------------------------------------------------------------------


def parse_ocr_json(json_path: Path) -> dict | None:
    """Extract metadata from a single Surya OCR JSON file.

    Looks for:
    - A ``Table`` block (``reading_order == 1``) whose HTML contains the
      metadata fields (Site, Type d'installation, Équipement, Date, …).
    - A ``Text`` block (``reading_order == 3``) below the SectionHeader
      "Description de l'incident" — the incident description.

    Returns a dict keyed by CSV column names, or ``None`` if the file cannot
    be parsed (missing blocks, invalid JSON, …).
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Collect all blocks across pages, sorted by reading_order
    blocks: list[dict] = []
    for page in data.get("pages", []):
        blocks.extend(page.get("blocks", []))
    blocks.sort(key=lambda b: b.get("reading_order", 0))

    # --- Extract metadata table (label == "Table", reading_order == 1) ---
    table_block = next(
        (
            b
            for b in blocks
            if b.get("label") == "Table" and b.get("reading_order") == 1
        ),
        None,
    )
    if table_block is None:
        return None

    parser = _TableParser()
    parser.feed(table_block.get("html", ""))

    raw_fields: dict[str, str] = {}
    for row in parser.rows:
        if len(row) >= 2:
            raw_fields[row[0]] = row[1]

    # Map French labels to CSV columns
    record: dict = {}
    for fr_label, csv_col in TABLE_FIELD_MAP.items():
        value = raw_fields.get(fr_label, "")
        if csv_col == "asset_type":
            value = ASSET_TYPE_MAP.get(value, value)
        record[csv_col] = value

    # --- Extract incident description ---
    # Two possible layouts:
    #   A) Description is a separate Text block (reading_order == 3) below a
    #      SectionHeader "Description de l'incident" (reading_order == 2).
    #   B) The OCR model merged the description into the Table block itself,
    #      as extra rows after the metadata rows:
    #        <tr><td colspan="2"><b>Description de l'incident</b></td></tr>
    #        <tr><td colspan="2"><p>…description…</p></td></tr>
    description_text = ""

    # Try layout B first: look for a "Description de l'incident" row in the
    # parsed table, then take the following row's cell as the description.
    for i, row in enumerate(parser.rows):
        if (
            row
            and "description de l'incident" in row[0].lower()
            and i + 1 < len(parser.rows)
            and parser.rows[i + 1]
        ):
            description_text = parser.rows[i + 1][0]
            break

    if not description_text:
        # Layout A: separate Text block at reading_order == 3.
        # Verify the SectionHeader at reading_order == 2 mentions "Description"
        section_header = next(
            (b for b in blocks if b.get("reading_order") == 2),
            None,
        )
        if section_header is not None:
            header_text = _strip_html(section_header.get("html", ""))
            if "description" not in header_text.lower():
                print(
                    f"  Warning: SectionHeader at ro=2 is '{header_text}', "
                    "expected 'Description de l'incident'"
                )

        text_block = next(
            (b for b in blocks if b.get("reading_order") == 3),
            None,
        )
        description_text = _strip_html(text_block.get("html", "")) if text_block else ""

    record["text"] = description_text

    # --- incident_id from filename (e.g. incident_1.json -> 1) ---
    try:
        record["incident_id"] = int(json_path.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None

    record["document_format"] = "pdf_scan"
    record["document_path"] = f"data/pdf_scan/{json_path.stem}.pdf"

    return record


def parse_ocr_directory(
    ocr_json_dir: Path,
    output_csv_path: Path,
    *,
    reference_csv_path: Path | None = None,
) -> tuple[int, int]:
    """Parse all OCR JSONs in *ocr_json_dir* and write results to *output_csv_path*.

    Parameters
    ----------
    ocr_json_dir:
        Directory containing ``incident_*.json`` files produced by Surya.
    output_csv_path:
        Where to write the parsed CSV (same columns as ``incidents_students.csv``).
    reference_csv_path:
        Optional path to ``incidents_students.csv`` for cross-validation
        (compares the extracted ``site`` against the known value).

    Returns
    -------
    tuple[int, int]
        ``(n_success, n_total)`` — number of JSONs where all required fields
        were extracted vs. total JSONs found.
    """
    json_files = sorted(
        ocr_json_dir.glob("incident_*.json"),
        key=lambda p: int(p.stem.rsplit("_", maxsplit=1)[-1]),
    )
    if not json_files:
        print(f"No OCR JSON files found in {ocr_json_dir}")
        return 0, 0

    # Load reference sites for validation (incident_id -> site)
    reference_sites: dict[int, str] = {}
    if reference_csv_path is not None and reference_csv_path.exists():
        with open(reference_csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    reference_sites[int(row["incident_id"])] = row.get("site", "")
                except (ValueError, KeyError):
                    pass

    records: list[dict] = []
    n_success = 0

    for i, json_path in enumerate(json_files, start=1):
        record = parse_ocr_json(json_path)
        if record is None:
            print(f"[{i}/{len(json_files)}] {json_path.name}: FAILED (unparseable)")
            continue

        # Check all required fields are present and non-empty
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            print(
                f"[{i}/{len(json_files)}] {json_path.name}: "
                f"FAILED (missing: {', '.join(missing)})"
            )
            continue

        # Cross-validate site against reference CSV
        incident_id = record["incident_id"]
        expected_site = reference_sites.get(incident_id)
        if expected_site is not None and record["site"] != expected_site:
            print(
                f"[{i}/{len(json_files)}] {json_path.name}: "
                f"WARNING site mismatch — extracted '{record['site']}', "
                f"expected '{expected_site}'"
            )

        records.append(record)
        n_success += 1
        print(f"[{i}/{len(json_files)}] {json_path.name}: OK")

    # Write CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col, "") for col in CSV_COLUMNS})

    n_total = len(json_files)
    print()
    print(f"Parsing complete: {n_success}/{n_total} succeeded")
    print(f"Output written to {output_csv_path}")
    return n_success, n_total


def merge_csv(
    csv_dir_list: list[Path], output_dir: Path, sorting_col: str | None = None
) -> None:
    merged_df = None
    for csv_dir in csv_dir_list:
        df = pd.read_csv(csv_dir)
        df.dropna(subset = ["text"],inplace = True)

        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.concat([merged_df,df.copy()])
    if merged_df is None:
        print("Failed to merge CSV files !")
        return
    if sorting_col is not None:
        merged_df.sort_values(by=sorting_col)

    # renaming elements from the column 'asset_type'
    important_column = 'asset_type'

    rename_dict = {
        "Centrale photovoltaique" : "solar",
        "Centrale photovoltaïque" : "solar",
        "Parc éolien" : "wind",
    }
    merged_df[important_column] = merged_df[important_column].replace(rename_dict)

    merged_df.to_csv(output_dir, index =  False)
    print("Successfully merged CSV files !")


# ---------------------------------------------------------------------------
# CLI helpers (shared with the top-level datachallengecapgemini entry point)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingArgs:
    """Arguments for the preprocessing pipeline (OCR + parse + merge)."""

    input_dir: Path
    digital_input_dir: Path
    output_dir: Path
    limit: int | None
    force: bool
    output_csv: Path
    output_digital_csv: Path
    output_merged_csv: Path
    reference_csv: Path


def add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add preprocessing-specific arguments to *parser*."""
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pdf_scan",
        help="Directory containing scanned PDF files.",
    )
    parser.add_argument(
        "--digital-input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pdf_digital",
        help="Directory containing digital PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pdf_scan" / "ocr_json",
        help="Directory where one JSON file per PDF will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to OCR (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process PDFs even if their JSON output already exists.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_students_scanned.csv",
        help="Path for the parsed output CSV.",
    )
    parser.add_argument(
        "--output-merged-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_students_merged.csv",
        help="Path for the merged output CSV.",
    )
    parser.add_argument(
        "--output-digital-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_students_digital.csv",
        help="Path to the parsed from digital PDF output CSV.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_students.csv",
        help="Reference CSV for cross-validation (site comparison).",
    )


def run_preprocessing(args: PreprocessingArgs) -> None:
    """Run the full preprocessing pipeline: OCR → parse scanned → parse digital → merge."""
    # Step 1 — OCR
    ocr_directory(
        args.input_dir,
        args.output_dir,
        limit=args.limit,
        force=args.force,
    )

    # Step 2 — Parse (always runs, cheap)
    print()
    print("=" * 60)
    print("Step 2: Parsing OCR JSONs")
    print("=" * 60)
    parse_ocr_directory(
        args.output_dir,
        args.output_csv,
        reference_csv_path=args.reference_csv,
    )

    # Step 3 — Parse digital PDFs
    print()
    print("=" * 60)
    print("Step 3: Parsing digital PDFs")
    print("=" * 60)
    digital_pdf_dir_to_csv(
        args.digital_input_dir,
        args.output_digital_csv,
    )

    # Step 4 — Merge CSVs
    print("=" * 60)
    print("Step 4: Merging CSV")
    print("=" * 60)
    merge_csv(
        [args.output_digital_csv, args.output_csv, args.reference_csv],
        args.output_merged_csv,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — ``uv run python -m datachallengecapgemini.preprocessing``."""
    parser = argparse.ArgumentParser(
        description="OCR scanned PDFs with Surya, then parse the results into a CSV."
    )
    add_preprocessing_arguments(parser)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

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


if __name__ == "__main__":
    main()
