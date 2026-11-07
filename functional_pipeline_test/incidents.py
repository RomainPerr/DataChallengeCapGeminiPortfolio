"""Functional-pipeline entry points for preparing incident records."""

from pathlib import Path

import pandas as pd

from .cache_version import get_cache_version_of
from datachallengecapgemini.preprocessing import (
    digital_pdf_dir_to_csv,
    ocr_directory,
    parse_ocr_directory,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@get_cache_version_of
def get_pdf_digital_incidents_csv() -> pd.DataFrame:
    """Extract digital PDFs to a CSV and return their incident records."""
    output_csv = DATA_DIR / "incidents_students_digital.csv"
    digital_pdf_dir_to_csv(DATA_DIR / "pdf_digital", output_csv)
    return pd.read_csv(output_csv)


@get_cache_version_of
def get_pdf_scan_incidents_csv() -> pd.DataFrame:
    """OCR scanned PDFs, parse their OCR JSON, and return incident records."""
    pdf_dir = DATA_DIR / "pdf_scan"
    ocr_json_dir = pdf_dir / "ocr_json"
    output_csv = DATA_DIR / "incidents_students_scanned.csv"

    ocr_directory(pdf_dir, ocr_json_dir)
    parse_ocr_directory(
        ocr_json_dir,
        output_csv,
        reference_csv_path=DATA_DIR / "incidents_students.csv",
    )
    return pd.read_csv(output_csv)


@get_cache_version_of
def get_incidents_csv() -> pd.DataFrame:
    """Combine source reports and return one populated row per incident.

    The source CSV supplies text reports and reference rows. Extracted digital
    and scanned PDF rows are appended afterward, so a parsed report replaces
    any metadata-only reference row with the same ``incident_id``.
    """
    digital_incidents = get_pdf_digital_incidents_csv()
    scanned_incidents = get_pdf_scan_incidents_csv()
    source_incidents = pd.read_csv(DATA_DIR / "incidents_students.csv")

    incidents = pd.concat(
        [source_incidents, digital_incidents, scanned_incidents],
        ignore_index=True,
    )
    if "incident_id" in incidents.columns:
        incidents = incidents.drop_duplicates(subset="incident_id", keep="last")
    incidents["asset_type"] = incidents["asset_type"].replace(
        {
            "Centrale photovoltaique": "solar",
            "Centrale photovoltaïque": "solar",
            "Parc éolien": "wind",
        }
    )

    output_csv = DATA_DIR / "incidents_students_merged.csv"
    incidents.to_csv(output_csv, index=False)
    return incidents.reset_index(drop=True)
