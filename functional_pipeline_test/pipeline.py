"""High-level orchestration for the functional incident pipeline."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cache_version import get_cache_version_of
from .clustering_module import (
    ClusterFusion,
    ClusteringCandidate,
    ClusteringConfig,
    ClusteringScan,
    EmbeddingMatrix,
    detect_cluster_fusions,
    get_candidate,
    scan_cluster_sizes,
    select_change_points,
)
from .fusion_validation import (
    FusionValidationResult,
    validate_fusions_with_azure,
)
from .incidents import get_incidents_csv
from .statistical_report import (
    REPORTS_DIR,
    StatisticalReport,
    generate_statistical_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRENCH_STOPWORDS = (
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "de",
    "du",
    "et",
    "en",
    "sur",
    "au",
    "aux",
    "pour",
    "par",
    "est",
    "sont",
    "a",
    "été",
    "suite",
    "lors",
    "dans",
    "que",
    "qui",
    "ce",
    "cette",
    "ces",
    "afin",
    "plus",
    "après",
    "avant",
)

EmbeddingLoader = Callable[[list[str], str], EmbeddingMatrix]
FusionValidator = Callable[
    [
        str,
        pd.DataFrame,
        Sequence[ClusteringCandidate],
        Sequence[ClusterFusion],
    ],
    FusionValidationResult,
]


@dataclass
class ClustersAndClassifiedIncidents:
    """Validated clustering choice and its incident classifications."""

    scan: ClusteringScan
    clusters: list[ClusteringCandidate]
    selected_cluster: ClusteringCandidate
    classified_incidents: pd.DataFrame
    fusions: list[ClusterFusion]
    fusion_validation: FusionValidationResult
    report: StatisticalReport | None = None


def load_or_create_embeddings(
    texts: list[str],
    asset_type: str,
    batch_size: int = 16,
) -> EmbeddingMatrix:
    """Load a compatible embedding cache or populate it with Azure."""
    from dotenv import load_dotenv
    from openai import AzureOpenAI
    from tqdm import tqdm

    if not texts or any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError("Every incident must contain non-empty text.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    required_variables = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )
    missing_variables = [
        name for name in required_variables if not os.getenv(name)
    ]
    if missing_variables:
        raise RuntimeError(
            "Missing Azure configuration: " + ", ".join(missing_variables)
        )

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        max_retries=10,
    )
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    vectors = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Embedding {asset_type} incidents",
        unit="batch",
    ):
        response = client.embeddings.create(
            model=deployment,
            input=texts[start : start + batch_size],
        )
        vectors.extend(
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        )

    embeddings = np.asarray(vectors, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            "Azure did not return one embedding vector per incident."
        )
    return embeddings


# Azure calls are cached explicitly here so the orchestration function keeps
# the same dependency-injection interface while repeated visualizations reuse
# the saved embedding matrix.
load_or_create_embeddings = get_cache_version_of(load_or_create_embeddings)


def get_clusters_and_classified_incidents(
    asset_type: str,
    *,
    config: ClusteringConfig | None = None,
    stopwords: Sequence[str] = FRENCH_STOPWORDS,
    min_cluster_size_start: int = 2,
    max_cluster_ratio: float = 0.20,
    embedding_loader: EmbeddingLoader = load_or_create_embeddings,
    fusion_validator: FusionValidator = validate_fusions_with_azure,
    report_output_dir: str | Path | None = REPORTS_DIR,
    compile_pdf: bool = True,
) -> ClustersAndClassifiedIncidents:
    """Classify incidents, then generate local statistics, plots and a PDF.

    Reports go under report_output_dir/asset_type. Set report_output_dir=None
    to skip reporting, or compile_pdf=False to keep only sources and plots.
    The existing Azure clustering stage is unchanged; reporting makes no LLM calls.
    """
    if not isinstance(asset_type, str) or not asset_type.strip():
        raise ValueError("asset_type must be a non-empty string.")

    incidents = get_incidents_csv()
    if "asset_type" not in incidents or "text" not in incidents:
        raise ValueError(
            "Incident data must contain 'asset_type' and 'text' columns."
        )

    asset_mask = incidents["asset_type"] == asset_type
    asset_incidents = incidents.loc[asset_mask].copy()
    if asset_incidents.empty:
        raise ValueError(f"Unknown asset type: {asset_type!r}.")
    texts = asset_incidents["text"].tolist()
    embeddings = embedding_loader(texts, asset_type)

    scan = scan_cluster_sizes(
        texts,
        embeddings,
        config=config or ClusteringConfig(),
        stopwords=stopwords,
        min_cluster_size_start=min_cluster_size_start,
        max_cluster_ratio=max_cluster_ratio,
    )
    clusters = select_change_points(scan)
    fusions = [
        fusion
        for previous, current in itertools.pairwise(clusters)
        for fusion in detect_cluster_fusions(previous, current)
    ]
    fusion_validation = fusion_validator(
        asset_type,
        asset_incidents,
        clusters,
        fusions,
    )
    if not isinstance(fusion_validation, FusionValidationResult):
        raise TypeError(
            "fusion_validator must return FusionValidationResult."
        )
    selected_cluster = get_candidate(
        scan,
        fusion_validation.selected_min_cluster_size,
    )
    if all(selected_cluster is not candidate for candidate in clusters):
        raise ValueError(
            "Fusion validation selected a candidate that was not retained."
        )

    classified_incidents = asset_incidents.assign(
        min_cluster_size=selected_cluster.min_cluster_size,
        topic=selected_cluster.topics,
    )
    report = None
    if report_output_dir is not None:
        # Avoid treating an asset label as an arbitrary filesystem path.
        import re

        asset_directory = re.sub(r"[^a-zA-Z0-9_-]+", "_", asset_type)
        report = generate_statistical_report(
            classified_incidents,
            Path(report_output_dir) / asset_directory,
            compile_pdf=compile_pdf,
        )
    return ClustersAndClassifiedIncidents(
        scan=scan,
        clusters=clusters,
        selected_cluster=selected_cluster,
        classified_incidents=classified_incidents,
        fusions=fusions,
        fusion_validation=fusion_validation,
        report=report,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify incidents and compile their statistical report.")
    parser.add_argument("--asset-type", choices=["solar", "wind"], required=True)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    result = get_clusters_and_classified_incidents(
        args.asset_type, report_output_dir=args.output_dir, compile_pdf=not args.no_compile,
    )
    print(result.report.pdf_path or result.report.tex_path)


if __name__ == "__main__":
    main()
