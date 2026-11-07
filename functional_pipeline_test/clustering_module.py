"""Reusable BERTopic clustering operations for the functional pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


# =============================================================================
# Types
# =============================================================================

EmbeddingMatrix = NDArray[np.floating]
TopicAssignments = NDArray[np.integer]
ProbabilityMatrix = NDArray[np.floating]


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration shared by all BERTopic fits."""

    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True

    ngram_range: tuple[int, int] = (1, 2)
    token_pattern: str = r"(?u)\b[^\W\d_][^\W\d_]+\b"

    language: str = "french"
    calculate_probabilities: bool = True

    random_state: int = 42


@dataclass
class ClusteringCandidate:
    """Result of fitting one min_cluster_size value."""

    min_cluster_size: int
    n_topics: int
    topics: TopicAssignments
    probabilities: ProbabilityMatrix | None
    model: BERTopic


@dataclass
class ClusteringScan:
    """Complete scan over candidate min_cluster_size values."""

    candidates: list[ClusteringCandidate]


@dataclass(frozen=True)
class TopicOverlap:
    """Overlap from one topic into another between two clusterings."""

    source_topic_id: int
    destination_topic_id: int
    proportion: float
    shared_text_indices: tuple[int, ...]


@dataclass(frozen=True)
class ClusterFusion:
    """Multiple old topics converging into one new topic."""

    from_min_cluster_size: int
    to_min_cluster_size: int
    destination_topic_id: int
    source_topic_ids: tuple[int, ...]


# =============================================================================
# Public API
# =============================================================================


def build_topic_model(
    min_cluster_size: int,
    *,
    config: ClusteringConfig,
    stopwords: Sequence[str],
) -> BERTopic:
    """Build an unfitted BERTopic model."""
    _validate_min_cluster_size_value(min_cluster_size)
    _validate_stopwords(stopwords)

    return BERTopic(
        umap_model=_build_umap_model(config),
        hdbscan_model=_build_hdbscan_model(min_cluster_size, config),
        vectorizer_model=_build_vectorizer_model(stopwords, config),
        language=config.language,
        calculate_probabilities=config.calculate_probabilities,
        verbose=False,
    )


def fit_clustering(
    texts: Sequence[str],
    embeddings: EmbeddingMatrix,
    *,
    min_cluster_size: int,
    config: ClusteringConfig,
    stopwords: Sequence[str],
) -> ClusteringCandidate:
    """Fit BERTopic once for a specific min_cluster_size."""
    return _fit_one_cluster_size(
        texts,
        embeddings,
        min_cluster_size=min_cluster_size,
        config=config,
        stopwords=stopwords,
    )


def scan_cluster_sizes(
    texts: Sequence[str],
    embeddings: EmbeddingMatrix,
    *,
    config: ClusteringConfig,
    stopwords: Sequence[str],
    min_cluster_size_start: int = 2,
    max_cluster_ratio: float = 0.20,
) -> ClusteringScan:
    """
    Fit all min_cluster_size values in the requested range.

    This function scans candidates; it does not decide which candidate
    is semantically optimal.
    """
    _validate_clustering_inputs(texts, embeddings)
    _validate_scan_parameters(
        n_texts=len(texts),
        min_cluster_size_start=min_cluster_size_start,
        max_cluster_ratio=max_cluster_ratio,
    )
    _validate_stopwords(stopwords)

    cluster_sizes = _resolve_cluster_size_range(
        len(texts),
        min_cluster_size_start=min_cluster_size_start,
        max_cluster_ratio=max_cluster_ratio,
    )
    return ClusteringScan(
        candidates=[
            _fit_one_cluster_size(
                texts,
                embeddings,
                min_cluster_size=cluster_size,
                config=config,
                stopwords=stopwords,
            )
            for cluster_size in cluster_sizes
        ]
    )


def select_change_points(
    scan: ClusteringScan,
) -> list[ClusteringCandidate]:
    """
    Keep the first candidate and every candidate where the number
    of topics changes relative to the previous tested size.
    """
    if not scan.candidates:
        return []

    selected = [scan.candidates[0]]
    for previous, current in zip(scan.candidates, scan.candidates[1:]):
        if _is_change_point(previous, current):
            selected.append(current)
    return selected


def compute_topic_overlaps(
    previous_topics: TopicAssignments,
    current_topics: TopicAssignments,
) -> list[TopicOverlap]:
    """Compare text membership between two clustering assignments."""
    previous_array = np.asarray(previous_topics)
    current_array = np.asarray(current_topics)
    _validate_topic_assignments(previous_array)
    _validate_topic_assignments(
        current_array,
        expected_length=len(previous_array),
    )

    return [
        _compute_topic_overlap(
            previous_array,
            current_array,
            source_topic_id=source_topic_id,
            destination_topic_id=destination_topic_id,
        )
        for source_topic_id in _topic_ids(
            previous_array,
            include_noise=False,
        )
        for destination_topic_id in _topic_ids(
            current_array,
            include_noise=True,
        )
    ]


def detect_cluster_fusions(
    previous: ClusteringCandidate,
    current: ClusteringCandidate,
) -> list[ClusterFusion]:
    """
    Detect cases where multiple previous topics contribute to one
    topic in the next clustering.
    """
    previous_topics = np.asarray(previous.topics)
    current_topics = np.asarray(current.topics)
    _validate_topic_assignments(previous_topics)
    _validate_topic_assignments(
        current_topics,
        expected_length=len(previous_topics),
    )

    fusions = []
    for destination_topic_id in _topic_ids(
        current_topics,
        include_noise=False,
    ):
        source_topic_ids = _source_topics_for_destination(
            previous_topics,
            current_topics,
            destination_topic_id=destination_topic_id,
        )
        if len(source_topic_ids) >= 2:
            fusions.append(
                ClusterFusion(
                    from_min_cluster_size=previous.min_cluster_size,
                    to_min_cluster_size=current.min_cluster_size,
                    destination_topic_id=destination_topic_id,
                    source_topic_ids=source_topic_ids,
                )
            )
    return fusions


def get_candidate(
    scan: ClusteringScan,
    min_cluster_size: int,
) -> ClusteringCandidate:
    """Retrieve a fitted candidate by min_cluster_size."""
    _validate_min_cluster_size_value(min_cluster_size)
    matches = [
        candidate
        for candidate in scan.candidates
        if candidate.min_cluster_size == min_cluster_size
    ]
    if not matches:
        raise ValueError(
            f"No candidate found for min_cluster_size={min_cluster_size}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple candidates found for min_cluster_size={min_cluster_size}."
        )
    return matches[0]


# =============================================================================
# Internal model-construction helpers
# =============================================================================


def _build_umap_model(
    config: ClusteringConfig,
) -> UMAP:
    return UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
    )


def _build_hdbscan_model(
    min_cluster_size: int,
    config: ClusteringConfig,
) -> HDBSCAN:
    _validate_min_cluster_size_value(min_cluster_size)
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.cluster_selection_method,
        prediction_data=config.prediction_data,
    )


def _build_vectorizer_model(
    stopwords: Sequence[str],
    config: ClusteringConfig,
) -> CountVectorizer:
    _validate_stopwords(stopwords)
    return CountVectorizer(
        stop_words=list(stopwords),
        token_pattern=config.token_pattern,
        ngram_range=config.ngram_range,
    )


# =============================================================================
# Internal fitting / scan helpers
# =============================================================================


def _fit_one_cluster_size(
    texts: Sequence[str],
    embeddings: EmbeddingMatrix,
    *,
    min_cluster_size: int,
    config: ClusteringConfig,
    stopwords: Sequence[str],
) -> ClusteringCandidate:
    _validate_clustering_inputs(texts, embeddings)
    _validate_min_cluster_size(
        min_cluster_size,
        n_texts=len(texts),
    )
    model = build_topic_model(
        min_cluster_size,
        config=config,
        stopwords=stopwords,
    )
    topics, probabilities = model.fit_transform(
        list(texts),
        embeddings=embeddings,
    )

    topic_array = np.asarray(topics)
    _validate_topic_assignments(topic_array, expected_length=len(texts))
    probability_array = (
        None
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    if probability_array is not None and (
        probability_array.ndim not in {1, 2}
        or probability_array.shape[0] != len(texts)
    ):
        raise ValueError(
            "BERTopic probabilities must contain one row per text."
        )

    return ClusteringCandidate(
        min_cluster_size=min_cluster_size,
        n_topics=_count_topics(topic_array),
        topics=topic_array,
        probabilities=probability_array,
        model=model,
    )


def _resolve_cluster_size_range(
    n_texts: int,
    *,
    min_cluster_size_start: int,
    max_cluster_ratio: float,
) -> range:
    _validate_scan_parameters(
        n_texts=n_texts,
        min_cluster_size_start=min_cluster_size_start,
        max_cluster_ratio=max_cluster_ratio,
    )
    max_cluster_size = min(
        n_texts,
        max(
            min_cluster_size_start,
            int(n_texts * max_cluster_ratio),
        ),
    )
    return range(min_cluster_size_start, max_cluster_size + 1)


def _count_topics(
    topics: TopicAssignments,
) -> int:
    """Count real topics, excluding HDBSCAN noise topic -1."""
    topic_array = np.asarray(topics)
    _validate_topic_assignments(topic_array)
    return len(_topic_ids(topic_array, include_noise=False))


def _is_change_point(
    previous: ClusteringCandidate,
    current: ClusteringCandidate,
) -> bool:
    return previous.n_topics != current.n_topics


# =============================================================================
# Internal overlap / fusion helpers
# =============================================================================


def _topic_ids(
    topics: TopicAssignments,
    *,
    include_noise: bool,
) -> list[int]:
    topic_array = np.asarray(topics)
    _validate_topic_assignments(topic_array)
    return sorted(
        int(topic_id)
        for topic_id in np.unique(topic_array)
        if include_noise or int(topic_id) != -1
    )


def _indices_for_topic(
    topics: TopicAssignments,
    topic_id: int,
) -> NDArray[np.integer]:
    topic_array = np.asarray(topics)
    _validate_topic_assignments(topic_array)
    return np.flatnonzero(topic_array == topic_id)


def _compute_topic_overlap(
    previous_topics: TopicAssignments,
    current_topics: TopicAssignments,
    *,
    source_topic_id: int,
    destination_topic_id: int,
) -> TopicOverlap:
    previous_array = np.asarray(previous_topics)
    current_array = np.asarray(current_topics)
    _validate_topic_assignments(previous_array)
    _validate_topic_assignments(
        current_array,
        expected_length=len(previous_array),
    )

    source_indices = _indices_for_topic(previous_array, source_topic_id)
    if not len(source_indices):
        raise ValueError(f"Unknown source topic: {source_topic_id}.")
    shared_indices = np.flatnonzero(
        (previous_array == source_topic_id)
        & (current_array == destination_topic_id)
    )
    return TopicOverlap(
        source_topic_id=source_topic_id,
        destination_topic_id=destination_topic_id,
        proportion=float(len(shared_indices) / len(source_indices)),
        shared_text_indices=tuple(
            int(index) for index in shared_indices
        ),
    )


def _source_topics_for_destination(
    previous_topics: TopicAssignments,
    current_topics: TopicAssignments,
    *,
    destination_topic_id: int,
) -> tuple[int, ...]:
    previous_array = np.asarray(previous_topics)
    current_array = np.asarray(current_topics)
    _validate_topic_assignments(previous_array)
    _validate_topic_assignments(
        current_array,
        expected_length=len(previous_array),
    )

    destination_indices = _indices_for_topic(
        current_array,
        destination_topic_id,
    )
    if not len(destination_indices):
        raise ValueError(f"Unknown destination topic: {destination_topic_id}.")
    return tuple(
        topic_id
        for topic_id in _topic_ids(
            previous_array[destination_indices],
            include_noise=False,
        )
    )


# =============================================================================
# Internal validation
# =============================================================================


def _validate_clustering_inputs(
    texts: Sequence[str],
    embeddings: EmbeddingMatrix,
) -> None:
    _validate_texts(texts)
    _validate_embeddings(embeddings, expected_rows=len(texts))


def _validate_texts(
    texts: Sequence[str],
) -> None:
    if isinstance(texts, (str, bytes)) or not isinstance(
        texts,
        Sequence,
    ):
        raise TypeError("texts must be a sequence of strings.")
    if not texts:
        raise ValueError("texts must not be empty.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every text must be a string.")
    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not be empty.")


def _validate_embeddings(
    embeddings: EmbeddingMatrix,
    *,
    expected_rows: int,
) -> None:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix.")
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} embedding rows, got {embeddings.shape[0]}."
        )
    if embeddings.shape[1] == 0:
        raise ValueError("embeddings must contain at least one feature column.")
    if not np.issubdtype(embeddings.dtype, np.number) or np.issubdtype(
        embeddings.dtype,
        np.complexfloating,
    ):
        raise TypeError("embeddings must contain real numeric values.")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values.")


def _validate_min_cluster_size(
    min_cluster_size: int,
    *,
    n_texts: int,
) -> None:
    _validate_min_cluster_size_value(min_cluster_size)
    if min_cluster_size > n_texts:
        raise ValueError(
            "min_cluster_size cannot exceed the number of texts."
        )


def _validate_scan_parameters(
    *,
    n_texts: int,
    min_cluster_size_start: int,
    max_cluster_ratio: float,
) -> None:
    if not isinstance(n_texts, int) or isinstance(n_texts, bool):
        raise TypeError("n_texts must be an integer.")
    if n_texts < 2:
        raise ValueError("At least two texts are required for clustering.")
    _validate_min_cluster_size(
        min_cluster_size_start,
        n_texts=n_texts,
    )
    if not isinstance(max_cluster_ratio, Real) or isinstance(
        max_cluster_ratio,
        bool,
    ):
        raise TypeError("max_cluster_ratio must be a real number.")
    if not np.isfinite(max_cluster_ratio) or not 0 < max_cluster_ratio <= 1:
        raise ValueError("max_cluster_ratio must be in the interval (0, 1].")


def _validate_topic_assignments(
    topics: TopicAssignments,
    *,
    expected_length: int | None = None,
) -> None:
    if not isinstance(topics, np.ndarray):
        raise TypeError("topics must be a NumPy array.")
    if topics.ndim != 1:
        raise ValueError("topics must be a one-dimensional array.")
    if not np.issubdtype(topics.dtype, np.integer) or np.issubdtype(
        topics.dtype,
        np.bool_,
    ):
        raise TypeError("topics must contain integer identifiers.")
    if expected_length is not None and len(topics) != expected_length:
        raise ValueError(
            f"Expected {expected_length} topic assignments, got {len(topics)}."
        )
    if not len(topics):
        raise ValueError("topics must not be empty.")


def _validate_min_cluster_size_value(min_cluster_size: int) -> None:
    if not isinstance(min_cluster_size, int) or isinstance(
        min_cluster_size,
        bool,
    ):
        raise TypeError("min_cluster_size must be an integer.")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2.")


def _validate_stopwords(stopwords: Sequence[str]) -> None:
    if isinstance(stopwords, (str, bytes)) or not isinstance(
        stopwords,
        Sequence,
    ):
        raise TypeError("stopwords must be a sequence of strings.")
    if any(not isinstance(stopword, str) for stopword in stopwords):
        raise TypeError("Every stopword must be a string.")
