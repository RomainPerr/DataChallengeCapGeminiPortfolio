"""Compute solar site-profile indicators and compare sites statistically."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "incidents_organized_solar.csv"
DEFAULT_OUTPUT = DATA_DIR / "site_mapping_solar.json"
SIGNIFICANCE_LEVEL = 0.05

WEATHER_TRIGGERS = {
    "meteo_vent_tempete",
    "meteo_grele_pluie_neige",
    "foudre_surtension_atmospherique",
    "chaleur_canicule",
}
NON_WEATHER_TRIGGERS = {
    "corrosion_environnement_salin",
    "encrassement_poussieres",
    "nuisibles_animaux",
    "vieillissement_usure",
    "defaut_maintenance",
    "defaut_conception_qualite",
    "erreur_humaine_chantier",
    "surcharge_electrique_reseau",
}
PREVENTIVE_RESPONSES = {"remplacement_preventif"}
CORRECTIVE_RESPONSES = {"remplacement_correctif", "reparation_sur_place"}
DELAY_ISSUES = {
    "piece_en_attente",
    "acces_difficile",
    "delai_intervention",
}

# Expectations come from the written site profiles, not from observed data.
PROFILE_EXPECTATIONS: dict[str, dict[str, int | None]] = {
    "age_vieillissement": {"A": -1, "B": 0, "C": 1},
    "age_defaut_conception": {"A": 1, "B": 0, "C": -1},
    "environnement_corrosion_salin": {"A": -1, "B": 0, "C": 1},
    "environnement_poussieres": {"A": -1, "B": 0, "C": 1},
    "environnement_nuisibles": {"A": -1, "B": 0, "C": 1},
    "climat_vent_tempete": {"A": -1, "B": 0, "C": 1},
    "climat_grele_pluie_neige": {"A": -1, "B": 0, "C": 1},
    "climat_foudre": {"A": -1, "B": 0, "C": 1},
    "climat_chaleur": {"A": 1, "B": 0, "C": None},
    "maintenance_controle_periodique": {"A": 1, "B": 0, "C": -1},
    "maintenance_alerte_supervision": {"A": -1, "B": 0, "C": 1},
    "maintenance_ratio_preventif_correctif": {"A": 1, "B": 0, "C": -1},
    "maintenance_recidive_mentionnee": {"A": -1, "B": 0, "C": 1},
    "maintenance_defaut": {"A": -1, "B": 0, "C": 1},
    "maintenance_downtime_proxy_mttr_h": {"A": -1, "B": 0, "C": 1},
    "maintenance_delay_or_access_issue": {"A": -1, "B": 0, "C": 1},
}


def _parse_factors(value: Any, incident_id: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"contributing_factors illisible pour l'incident {incident_id}."
            ) from error
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"contributing_factors doit être une liste de chaînes "
            f"pour l'incident {incident_id}."
        )
    return value


def _parse_bool(value: Any, field: str, incident_id: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"{field} invalide pour l'incident {incident_id} : {value!r}.")


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def _contingency_test(
    counts: pd.DataFrame,
    test_id: str,
    *,
    category_axis: str,
) -> dict[str, Any]:
    table = counts.to_numpy(dtype=float)
    if category_axis == "rows":
        table = table[table.sum(axis=1) > 0, :]
    else:
        table = table[:, table.sum(axis=0) > 0]
    table = table[table.sum(axis=1) > 0, :]
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "test_id": test_id,
            "tested": False,
            "reason": "effectifs insuffisants ou une seule catégorie observée",
        }
    statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(table)
    min_expected = float(expected.min())
    eligible = min_expected >= 5
    return {
        "test_id": test_id,
        "tested": True,
        "chi2": float(statistic),
        "degrees_freedom": int(degrees_freedom),
        "p_value": float(p_value),
        "min_expected_count": min_expected,
        "eligible_for_discrimination": eligible,
        "reason": None if eligible else "effectif théorique inférieur à 5",
    }


def _rank_sites(values: dict[str, float | None]) -> dict[str, str | None]:
    present = {
        site: value
        for site, value in values.items()
        if value is not None and np.isfinite(value)
    }
    if not present:
        return {site: None for site in values}
    ranks = pd.Series(present, dtype=float).rank(method="average").to_dict()
    count = len(present)
    labels = {
        site: (
            "bas" if rank <= (count + 1) / 3
            else "haut" if rank >= 2 * (count + 1) / 3
            else "moyen"
        )
        for site, rank in ranks.items()
    }
    return {site: labels.get(site) for site in values}


def _category_flags(incidents: pd.DataFrame) -> dict[str, pd.Series]:
    primary = incidents["trigger_category"].fillna("").astype(str)
    return {
        category: pd.Series(
            [
                (main == category) or (category in factors)
                for main, factors in zip(primary, incidents["_factors"], strict=True)
            ],
            index=incidents.index,
            dtype=bool,
        )
        for category in sorted(NON_WEATHER_TRIGGERS | WEATHER_TRIGGERS)
    }


def _indicator(
    values: dict[str, float | None],
    test: dict[str, Any],
    *,
    unit: str,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "values_by_site": values,
        "relative_rank_by_site": _rank_sites(values),
        "chi_square": test,
    }


def _build_weather_cause_associations(
    incidents: pd.DataFrame,
    sites: list[str],
    weather_present: pd.Series,
    non_weather_present: pd.Series,
) -> dict[str, Any]:
    associations: dict[str, Any] = {}
    for site in sites:
        site_rows = incidents["site"].eq(site)
        table = pd.crosstab(
            weather_present.loc[site_rows],
            non_weather_present.loc[site_rows],
        ).reindex(index=[False, True], columns=[False, True], fill_value=0)
        test = _contingency_test(
            table,
            f"climat_meteo_vs_cause_non_meteo:{site}",
            category_axis="rows",
        )
        both_count = int(table.loc[True, True])
        site_total = int(site_rows.sum())
        rate = both_count / site_total if site_total else None
        expected_rate = (
            float(weather_present.loc[site_rows].mean())
            * float(non_weather_present.loc[site_rows].mean())
            if site_total
            else None
        )
        associations[site] = {
            "incidents_weather_and_non_weather_cause_pct": (
                100 * rate if rate is not None else None
            ),
            "expected_pct_if_independent": (
                100 * expected_rate if expected_rate is not None else None
            ),
            "overrepresentation_ratio": (
                rate / expected_rate
                if rate is not None and expected_rate not in (None, 0)
                else None
            ),
            "chi_square": test,
        }
    return associations


def _match_profiles(
    indicators: dict[str, dict[str, Any]],
    sites: list[str],
) -> dict[str, Any]:
    usable_indicators = [
        name for name, indicator in indicators.items()
        if name in PROFILE_EXPECTATIONS
        and indicator["chi_square"].get("significant_after_holm", False)
    ]
    if not usable_indicators:
        return {
            "status": "indeterminate",
            "reason": "aucun indicateur discriminant avec une direction de profil définie",
            "candidate_scores": [],
        }

    candidates = []
    for assigned_sites in itertools.permutations(sites):
        profile_to_site = dict(zip(("A", "B", "C"), assigned_sites, strict=True))
        agreements = 0
        comparisons = 0
        for indicator_name in usable_indicators:
            expectations = PROFILE_EXPECTATIONS[indicator_name]
            values = indicators[indicator_name]["values_by_site"]
            for left_profile, right_profile in itertools.combinations(("A", "B", "C"), 2):
                left_expected = expectations[left_profile]
                right_expected = expectations[right_profile]
                if left_expected is None or right_expected is None:
                    continue
                left_value = values[profile_to_site[left_profile]]
                right_value = values[profile_to_site[right_profile]]
                if left_value is None or right_value is None or left_value == right_value:
                    continue
                comparisons += 1
                expected_sign = np.sign(left_expected - right_expected)
                observed_sign = np.sign(left_value - right_value)
                agreements += int(expected_sign == observed_sign)
        score = agreements / comparisons if comparisons else None
        candidates.append(
            {
                "profile_to_site": profile_to_site,
                "directional_agreement": score,
                "comparisons": comparisons,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["directional_agreement"] is not None,
            candidate["directional_agreement"] or -1,
        ),
        reverse=True,
    )
    best_score = ranked[0]["directional_agreement"]
    best = [
        candidate for candidate in ranked
        if candidate["directional_agreement"] == best_score
    ]
    return {
        "status": "unique_best_fit" if len(best) == 1 else "ambiguous",
        "method": (
            "exhaustive assignment of profiles A/B/C to sites, maximizing "
            "directional agreement on Holm-significant indicators; profile "
            "expectations are predefined independently of the measured data"
        ),
        "indicators_used": usable_indicators,
        "best_candidates": best,
        "candidate_scores": ranked,
    }


def analyze_sites(
    csv_path: Path = DEFAULT_INPUT,
    *,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Calculate site indicators, chi-square tests, ranks, and profile candidates."""
    incidents = pd.read_csv(csv_path)
    required = {
        "incident_id",
        "site",
        "asset_type",
        "equipment_id",
        "date",
        "downtime_hours",
        "trigger_category",
        "contributing_factors",
        "detection_context",
        "response_type",
        "delay_or_access_issue",
        "recurrence_mentioned",
    }
    missing = sorted(required - set(incidents.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path} : {missing}")
    if incidents.empty:
        raise ValueError(f"Le fichier d'entrée est vide : {csv_path}")
    if incidents["incident_id"].isna().any() or incidents["incident_id"].duplicated().any():
        raise ValueError("incident_id doit être renseigné et unique.")
    if incidents["site"].isna().any():
        raise ValueError("Chaque incident doit avoir un site.")
    asset_types = set(incidents["asset_type"].dropna().astype(str))
    if asset_types != {"solar"}:
        raise ValueError(f"Ce module attend uniquement des incidents solar, trouvé : {asset_types}")

    incidents = incidents.copy()
    incidents["date"] = pd.to_datetime(incidents["date"], errors="coerce")
    if incidents["date"].isna().any():
        raise ValueError("Chaque incident doit avoir une date valide.")
    incidents["downtime_hours"] = pd.to_numeric(
        incidents["downtime_hours"], errors="coerce"
    )
    incidents["_factors"] = [
        _parse_factors(value, str(incident_id))
        for incident_id, value in zip(
            incidents["incident_id"], incidents["contributing_factors"], strict=True
        )
    ]
    incidents["_recurrence"] = [
        _parse_bool(value, "recurrence_mentioned", str(incident_id))
        for incident_id, value in zip(
            incidents["incident_id"], incidents["recurrence_mentioned"], strict=True
        )
    ]
    sites = sorted(incidents["site"].astype(str).unique().tolist())
    if len(sites) != 3:
        raise ValueError(f"Trois sites solar sont attendus, trouvé {len(sites)} : {sites}")

    site_counts = incidents["site"].value_counts().reindex(sites, fill_value=0)
    flags = _category_flags(incidents)
    weather_present = pd.Series(False, index=incidents.index)
    for category in WEATHER_TRIGGERS:
        weather_present |= flags[category]
    non_weather_present = pd.Series(False, index=incidents.index)
    for category in NON_WEATHER_TRIGGERS:
        non_weather_present |= flags[category]

    indicators: dict[str, dict[str, Any]] = {}
    tests: list[dict[str, Any]] = []

    def add_binary_indicator(name: str, mask: pd.Series, unit: str = "percent") -> None:
        table = pd.crosstab(mask, incidents["site"]).reindex(
            index=[False, True], columns=sites, fill_value=0
        )
        test = _contingency_test(table, name, category_axis="rows")
        test["indicator"] = name
        values: dict[str, float | None] = {
            site: 100.0 * float(mask.loc[incidents["site"].eq(site)].mean())
            for site in sites
        }
        indicators[name] = _indicator(values, test, unit=unit)
        tests.append(test)

    for category, name in (
        ("vieillissement_usure", "age_vieillissement"),
        ("defaut_conception_qualite", "age_defaut_conception"),
        ("corrosion_environnement_salin", "environnement_corrosion_salin"),
        ("encrassement_poussieres", "environnement_poussieres"),
        ("nuisibles_animaux", "environnement_nuisibles"),
        ("meteo_vent_tempete", "climat_vent_tempete"),
        ("meteo_grele_pluie_neige", "climat_grele_pluie_neige"),
        ("foudre_surtension_atmospherique", "climat_foudre"),
        ("chaleur_canicule", "climat_chaleur"),
        ("defaut_maintenance", "maintenance_defaut"),
    ):
        add_binary_indicator(name, flags[category])

    detection = incidents["detection_context"].fillna("").astype(str)
    add_binary_indicator(
        "maintenance_controle_periodique", detection.eq("controle_periodique")
    )
    add_binary_indicator(
        "maintenance_alerte_supervision", detection.eq("alerte_supervision")
    )
    add_binary_indicator(
        "maintenance_recidive_mentionnee", incidents["_recurrence"]
    )
    add_binary_indicator(
        "maintenance_delay_or_access_issue",
        ~incidents["delay_or_access_issue"].fillna("").astype(str).isin(
            {"", "aucun_mentionne", "non_precise"}
        ),
    )
    add_binary_indicator(
        "climat_meteo_et_cause_non_meteo",
        weather_present & non_weather_present,
    )

    responses = incidents["response_type"].fillna("").astype(str)
    preventive = responses.isin(PREVENTIVE_RESPONSES)
    corrective = responses.isin(CORRECTIVE_RESPONSES)
    response_eligible = preventive | corrective
    response_table = pd.crosstab(
        responses.loc[response_eligible],
        incidents.loc[response_eligible, "site"],
    ).reindex(
        index=sorted(PREVENTIVE_RESPONSES | CORRECTIVE_RESPONSES),
        columns=sites,
        fill_value=0,
    )
    response_test = _contingency_test(
        response_table, "maintenance_preventive_vs_corrective", category_axis="rows"
    )
    response_test["indicator"] = "maintenance_ratio_preventif_correctif"
    response_counts = {
        site: {
            "preventive": int((preventive & incidents["site"].eq(site)).sum()),
            "corrective": int((corrective & incidents["site"].eq(site)).sum()),
        }
        for site in sites
    }
    response_ratios = {
        site: (
            counts["preventive"] / counts["corrective"]
            if counts["corrective"]
            else None
        )
        for site, counts in response_counts.items()
    }
    indicators["maintenance_ratio_preventif_correctif"] = _indicator(
        response_ratios,
        response_test,
        unit="ratio",
    )
    indicators["maintenance_ratio_preventif_correctif"]["counts_by_site"] = response_counts
    tests.append(response_test)

    delay_categories = sorted(DELAY_ISSUES | {"aucun_mentionne", "non_precise"})
    delay_table = pd.crosstab(
        incidents["delay_or_access_issue"].fillna("non_precise").astype(str),
        incidents["site"],
    ).reindex(index=delay_categories, columns=sites, fill_value=0)
    delay_test = _contingency_test(
        delay_table, "maintenance_delay_categories", category_axis="rows"
    )
    delay_test["indicator"] = "maintenance_delay_categories"
    tests.append(delay_test)
    indicators["maintenance_delay_categories"] = {
        "unit": "percent",
        "categories_by_site": {
            site: {
                category: (
                    100 * int(delay_table.loc[category, site]) / int(site_counts[site])
                )
                for category in delay_categories
            }
            for site in sites
        },
        "chi_square": delay_test,
    }

    downtime = incidents["downtime_hours"]
    valid_downtime = downtime.notna()
    downtime_values = downtime.loc[valid_downtime]
    if len(downtime_values) < 3 or downtime_values.nunique() < 3:
        downtime_bins = pd.Series(pd.NA, index=incidents.index, dtype="string")
    else:
        quantile_edges = np.unique(
            downtime_values.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        )
        if len(quantile_edges) < 4:
            downtime_bins = pd.Series(pd.NA, index=incidents.index, dtype="string")
        else:
            downtime_bins = pd.cut(
                downtime,
                bins=quantile_edges,
                labels=["bas", "moyen", "haut"],
                include_lowest=True,
                duplicates="drop",
            ).astype("string")
    downtime_table = pd.crosstab(
        downtime_bins.loc[valid_downtime],
        incidents.loc[valid_downtime, "site"],
    ).reindex(index=["bas", "moyen", "haut"], columns=sites, fill_value=0)
    downtime_test = _contingency_test(
        downtime_table, "maintenance_downtime_classes", category_axis="rows"
    )
    downtime_test["indicator"] = "maintenance_downtime_proxy_mttr_h"
    tests.append(downtime_test)
    downtime_means = {
        site: (
            float(downtime.loc[incidents["site"].eq(site)].mean())
            if downtime.loc[incidents["site"].eq(site)].notna().any()
            else None
        )
        for site in sites
    }
    indicators["maintenance_downtime_proxy_mttr_h"] = _indicator(
        downtime_means,
        downtime_test,
        unit="mean_downtime_hours_proxy_not_true_mttr",
    )
    indicators["maintenance_downtime_proxy_mttr_h"]["categories_by_site"] = {
        site: {
            category: (
                100
                * int(downtime_table.loc[category, site])
                / int(downtime.loc[incidents["site"].eq(site)].notna().sum())
                if downtime.loc[incidents["site"].eq(site)].notna().any()
                else None
            )
            for category in ("bas", "moyen", "haut")
        }
        for site in sites
    }

    p_tests = [
        test for test in tests
        if test.get("tested") and test.get("eligible_for_discrimination")
    ]
    adjusted = _holm_adjust([test["p_value"] for test in p_tests])
    for test, p_adjusted in zip(p_tests, adjusted, strict=True):
        test["p_value_holm"] = p_adjusted
        test["significant_after_holm"] = p_adjusted < significance_level
    for test in tests:
        if "p_value_holm" not in test:
            test["p_value_holm"] = None
            test["significant_after_holm"] = False
    for indicator in indicators.values():
        test_id = indicator["chi_square"]["test_id"]
        indicator["chi_square"] = next(
            test for test in tests if test["test_id"] == test_id
        )

    weather_associations = _build_weather_cause_associations(
        incidents, sites, weather_present, non_weather_present
    )
    assoc_tests = [
        value["chi_square"]
        for value in weather_associations.values()
        if value["chi_square"].get("tested")
        and value["chi_square"].get("eligible_for_discrimination")
    ]
    assoc_adjusted = _holm_adjust([test["p_value"] for test in assoc_tests])
    for test, p_adjusted in zip(assoc_tests, assoc_adjusted, strict=True):
        test["p_value_holm_within_weather_tests"] = p_adjusted
        test["significant_after_holm"] = p_adjusted < significance_level

    profile_mapping = _match_profiles(indicators, sites)
    return {
        "source": str(csv_path),
        "asset_type": "solar",
        "site_incident_counts": {site: int(site_counts[site]) for site in sites},
        "metric_definitions": {
            "cause_parts": (
                "incident share where category occurs as primary trigger_category "
                "or in contributing_factors; each incident counted once"
            ),
            "recurrence": (
                "recurrence_mentioned text flag; this is not a measured 90-day "
                "recurrence rate"
            ),
            "preventive_corrective_ratio": (
                "preventive=remplacement_preventif; corrective="
                "remplacement_correctif or reparation_sur_place; other responses excluded"
            ),
            "mttr": (
                "mean downtime_hours used as a proxy; it is not an exact repair time"
            ),
            "downtime_test": (
                "chi-square across common global downtime terciles; the descriptive "
                "site value is mean downtime_hours"
            ),
            "significance": (
                f"chi-square tests with expected counts >=5; Holm correction across "
                f"{len(p_tests)} eligible site-indicator tests; alpha={significance_level}"
            ),
            "geotechnique": "omitted for solar as requested",
        },
        "indicators": indicators,
        "weather_cause_association_by_site": weather_associations,
        "site_indicator_tests": tests,
        "profile_mapping": profile_mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule les indicateurs solar et compare les profils des sites."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=SIGNIFICANCE_LEVEL,
        help="Seuil de significativité après correction de Holm (défaut : 0.05).",
    )
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha doit être strictement compris entre 0 et 1.")

    result = analyze_sites(args.input, significance_level=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Résultats enregistrés : {args.output}")
    print("Indicateurs significatifs après correction de Holm :")
    for test in result["site_indicator_tests"]:
        if test["significant_after_holm"]:
            print(f"- {test['indicator']} : p_Holm={test['p_value_holm']:.4g}")
    print(f"Correspondance de profils : {result['profile_mapping']['status']}")
    for candidate in result["profile_mapping"].get("best_candidates", []):
        print(
            f"- {candidate['profile_to_site']} "
            f"(accord directionnel : {candidate['directional_agreement']:.1%})"
        )


if __name__ == "__main__":
    main()