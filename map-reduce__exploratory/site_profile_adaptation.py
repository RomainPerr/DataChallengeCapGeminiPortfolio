"""Adapt solar remediation candidates to directly measured site indicators."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "fixing_plan_remediation_solar.json"
DEFAULT_SITE_MAPPING = DATA_DIR / "site_mapping_solar.json"
DEFAULT_ACTION_LIBRARY = DATA_DIR / "solar_action_library.json"
DEFAULT_OUTPUT = DATA_DIR / "fixing_plan_remediation_solar_adapted.json"

ATTRIBUTE_INDICATORS = {
    "age": ("age_vieillissement", "age_defaut_conception"),
    "environnement": (
        "environnement_corrosion_salin",
        "environnement_poussieres",
        "environnement_nuisibles",
    ),
    "climat": (
        "climat_vent_tempete",
        "climat_grele_pluie_neige",
        "climat_foudre",
        "climat_chaleur",
    ),
    "maintenance": (
        "maintenance_defaut",
        "maintenance_controle_periodique",
        "maintenance_alerte_supervision",
        "maintenance_recidive_mentionnee",
        "maintenance_delay_or_access_issue",
        "maintenance_ratio_preventif_correctif",
        "maintenance_downtime_proxy_mttr_h",
    ),
}


@dataclass(frozen=True)
class IndicatorCondition:
    indicator: str
    level: str | None = None
    value_gt: float | None = None
    value_eq: float | None = None


@dataclass(frozen=True)
class AdaptationRule:
    rule_id: str
    rule_type: str
    conditions: tuple[IndicatorCondition, ...]
    action_ids: tuple[str, ...]
    explanation: str
    add_if_missing: bool = False
    horizon: str | None = None


# Levels are the direct per-indicator ranks emitted by site_mapping_agent.py.
# The profile mapping is intentionally not consulted, even when it is confident.
RULES = (
    AdaptationRule(
        "age-fin-de-vie",
        "ajouter",
        (IndicatorCondition("age_vieillissement", "eleve", value_gt=0),),
        ("SOL-AGE-01",),
        "Le vieillissement mesuré justifie un diagnostic de fin de vie.",
    ),
    AdaptationRule(
        "poussieres-nettoyage",
        "ajouter",
        (IndicatorCondition("environnement_poussieres", "eleve", value_gt=0),),
        ("SOL-ENC-01",),
        "La part mesurée d'incidents liés aux poussières justifie un nettoyage piloté.",
    ),
    AdaptationRule(
        "recidive-alerte",
        "ajouter",
        (
            IndicatorCondition(
                "maintenance_recidive_mentionnee", "eleve", value_gt=0
            ),
        ),
        ("SOL-MAI-03",),
        "Les récidives sont directement mentionnées dans les incidents du site.",
    ),
    AdaptationRule(
        "acces-stock-pieces",
        "ajouter",
        (
            IndicatorCondition(
                "maintenance_delay_or_access_issue", "eleve", value_gt=0
            ),
        ),
        ("SOL-MAI-02",),
        "Les retards ou difficultés d'accès mesurés justifient un stock de pièces.",
    ),
    AdaptationRule(
        "corrosion-absente",
        "ecarter",
        (IndicatorCondition("environnement_corrosion_salin", "faible", value_eq=0),),
        ("SOL-COR-01",),
        "Aucune corrosion saline n'est mesurée pour ce site.",
    ),
    AdaptationRule(
        "maintenance-defaillante",
        "remonter",
        (IndicatorCondition("maintenance_defaut", "eleve", value_gt=0),),
        ("SOL-MAI-01", "SOL-INV-02", "SOL-MAI-03"),
        "Les défauts de maintenance mesurés placent les actions organisationnelles "
        "avant les remplacements techniques.",
    ),
    AdaptationRule(
        "controles-preventifs-faibles",
        "remonter",
        (IndicatorCondition("maintenance_controle_periodique", "faible"),),
        ("SOL-MAI-01", "SOL-INV-02", "SOL-MAI-03"),
        "Le faible niveau mesuré de contrôles périodiques priorise les actions "
        "organisationnelles.",
    ),
    AdaptationRule(
        "vents-forts",
        "remonter",
        (IndicatorCondition("climat_vent_tempete", "eleve", value_gt=0),),
        ("SOL-MET-02",),
        "L'exposition mesurée aux vents forts priorise la revue de structure.",
    ),
    AdaptationRule(
        "acces-phase-stock",
        "phaser",
        (
            IndicatorCondition(
                "maintenance_delay_or_access_issue", "eleve", value_gt=0
            ),
        ),
        ("SOL-MAI-02",),
        "Le stock de pièces est planifié à court terme avant les actions techniques.",
        add_if_missing=True,
        horizon="court_terme",
    ),
)

LEVEL_ALIASES = {
    "bas": "faible",
    "faible": "faible",
    "low": "faible",
    "moyen": "moyen",
    "medium": "moyen",
    "haut": "eleve",
    "élevé": "eleve",
    "eleve": "eleve",
    "high": "eleve",
}
TECHNICAL_NATURES = {"conception", "corrective", "preventive", "structurel"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Le fichier JSON doit contenir un objet : {path}.")
    return value


def _validate_inputs(
    plan: dict[str, Any],
    site_mapping: dict[str, Any],
    action_library: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results = plan.get("results")
    indicators = site_mapping.get("indicators")
    actions = action_library.get("actions")
    if not isinstance(results, list):
        raise TypeError("Le plan d'entrée doit contenir une liste 'results'.")
    if not isinstance(indicators, dict):
        raise TypeError("La correspondance de sites doit contenir 'indicators'.")
    if not isinstance(actions, list):
        raise TypeError("La bibliothèque doit contenir une liste 'actions'.")

    action_by_id: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("id"), str):
            raise TypeError("La bibliothèque contient une action sans identifiant.")
        action_id = action["id"]
        if action_id in action_by_id:
            raise ValueError(f"Identifiant d'action dupliqué : {action_id}.")
        action_by_id[action_id] = action

    missing_rule_actions = sorted(
        {action_id for rule in RULES for action_id in rule.action_ids}
        - set(action_by_id)
    )
    if missing_rule_actions:
        raise ValueError(
            "Actions de règles absentes de la bibliothèque : "
            f"{missing_rule_actions}."
        )

    for index, result in enumerate(results):
        if not isinstance(result, dict) or not isinstance(result.get("cell"), dict):
            raise TypeError(f"Résultat invalide à l'index {index}.")
        site = result["cell"].get("site")
        if not isinstance(site, str) or not site:
            raise ValueError(f"Nom de site manquant à l'index {index}.")
        if not isinstance(result.get("candidates"), list):
            raise TypeError(f"Liste 'candidates' invalide à l'index {index}.")
        for candidate in result["candidates"]:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("id"), str
            ):
                raise TypeError(f"Candidate invalide à l'index {index}.")

    return action_by_id


def _site_characteristics(
    site_mapping: dict[str, Any], site: str
) -> dict[str, dict[str, dict[str, Any]]]:
    indicators = site_mapping["indicators"]
    characteristics: dict[str, dict[str, dict[str, Any]]] = {}
    for attribute, indicator_ids in ATTRIBUTE_INDICATORS.items():
        measured: dict[str, dict[str, Any]] = {}
        for indicator_id in indicator_ids:
            indicator = indicators.get(indicator_id)
            if not isinstance(indicator, dict):
                measured[indicator_id] = {"level": None, "value": None, "unit": None}
                continue
            values = indicator.get("values_by_site")
            ranks = indicator.get("relative_rank_by_site")
            value = values.get(site) if isinstance(values, dict) else None
            rank = ranks.get(site) if isinstance(ranks, dict) else None
            measured[indicator_id] = {
                "level": LEVEL_ALIASES.get(str(rank).casefold()) if rank else None,
                "value": value,
                "unit": indicator.get("unit"),
            }
        characteristics[attribute] = measured
    return characteristics


def _condition_evidence(
    condition: IndicatorCondition,
    site: str,
    indicators: dict[str, Any],
) -> dict[str, Any] | None:
    indicator = indicators.get(condition.indicator)
    if not isinstance(indicator, dict):
        return None
    values = indicator.get("values_by_site")
    ranks = indicator.get("relative_rank_by_site")
    if not isinstance(values, dict) or not isinstance(ranks, dict):
        return None
    value = values.get(site)
    rank = ranks.get(site)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return None
    if condition.level is not None:
        normalized_rank = LEVEL_ALIASES.get(str(rank).casefold()) if rank else None
        if normalized_rank != condition.level:
            return None
    if condition.value_gt is not None and value <= condition.value_gt:
        return None
    if condition.value_eq is not None and value != condition.value_eq:
        return None
    return {
        "indicator": condition.indicator,
        "level": LEVEL_ALIASES.get(str(rank).casefold()) if rank else None,
        "value": value,
        "unit": indicator.get("unit"),
    }


def _matching_evidence(
    rule: AdaptationRule,
    site: str,
    indicators: dict[str, Any],
) -> list[dict[str, Any]] | None:
    evidence = [
        _condition_evidence(condition, site, indicators)
        for condition in rule.conditions
    ]
    if any(item is None for item in evidence):
        return None
    return [item for item in evidence if item is not None]


def _candidate_from_library(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: action.get(key, default)
        for key, default in (
            ("id", ""),
            ("title", ""),
            ("description", ""),
            ("nature", ""),
            ("effort", ""),
            ("horizon", ""),
            ("requires_investigation_first", False),
            ("prerequisites", []),
            ("caveat", ""),
            ("evidence_basis", ""),
        )
    } | {"matched_failure_modes": []}


def _apply_rules_to_result(
    result: dict[str, Any],
    site_mapping: dict[str, Any],
    action_by_id: dict[str, dict[str, Any]],
) -> None:
    site = result["cell"]["site"]
    indicators = site_mapping["indicators"]
    actions = deepcopy(result["candidates"])
    excluded_actions: list[dict[str, Any]] = []
    applied_rules: list[dict[str, Any]] = []

    matching_rules = [
        (rule, evidence)
        for rule in RULES
        if (evidence := _matching_evidence(rule, site, indicators)) is not None
    ]

    for rule, evidence in matching_rules:
        if rule.rule_type != "ecarter":
            continue
        removed = [item for item in actions if item["id"] in rule.action_ids]
        if not removed:
            continue
        actions = [item for item in actions if item["id"] not in rule.action_ids]
        excluded_actions.extend(
            {
                "action_id": item["id"],
                "title": item.get("title", ""),
                "rule_id": rule.rule_id,
                "reason": rule.explanation,
            }
            for item in removed
        )
        applied_rules.append(
            {
                "rule_id": rule.rule_id,
                "type": rule.rule_type,
                "reason": rule.explanation,
                "evidence": evidence,
            }
        )

    for rule, evidence in matching_rules:
        if rule.rule_type != "ajouter":
            continue
        added: list[str] = []
        for action_id in rule.action_ids:
            if any(item["id"] == action_id for item in actions):
                continue
            action = _candidate_from_library(action_by_id[action_id])
            action["added_by_rule"] = rule.rule_id
            actions.append(action)
            added.append(action_id)
        if added:
            applied_rules.append(
                {
                    "rule_id": rule.rule_id,
                    "type": rule.rule_type,
                    "reason": rule.explanation,
                    "evidence": evidence,
                    "added_action_ids": added,
                }
            )

    for rule, evidence in matching_rules:
        if rule.rule_type != "remonter":
            continue
        selected = [item for action_id in rule.action_ids for item in actions if item["id"] == action_id]
        if not selected:
            continue
        selected_ids = {item["id"] for item in selected}
        reordered = selected + [item for item in actions if item["id"] not in selected_ids]
        if [item["id"] for item in reordered] == [item["id"] for item in actions]:
            continue
        actions = reordered
        applied_rules.append(
            {
                "rule_id": rule.rule_id,
                "type": rule.rule_type,
                "reason": rule.explanation,
                "evidence": evidence,
                "prioritized_action_ids": [item["id"] for item in selected],
            }
        )

    for rule, evidence in matching_rules:
        if rule.rule_type != "phaser":
            continue
        target_id = rule.action_ids[0]
        target_index = next(
            (index for index, item in enumerate(actions) if item["id"] == target_id),
            None,
        )
        target = actions[target_index] if target_index is not None else None
        added = False
        if target is None and rule.add_if_missing:
            target = _candidate_from_library(action_by_id[target_id])
            target["added_by_rule"] = rule.rule_id
            actions.append(target)
            target_index = len(actions) - 1
            added = True
        if target is None:
            continue

        changed = (
            added
            or target.get("horizon") != rule.horizon
            or target.get("phase") != rule.horizon
            or target.get("phase_reason") != rule.explanation
        )
        target["horizon"] = rule.horizon
        target["phase"] = rule.horizon
        target["phase_reason"] = rule.explanation
        actions.remove(target)
        technical_index = next(
            (
                index
                for index, item in enumerate(actions)
                if item.get("nature") in TECHNICAL_NATURES
            ),
            None,
        )
        new_index = technical_index if technical_index is not None else target_index
        if new_index is None:
            raise RuntimeError(f"Index d'action perdu pendant le phasage : {target_id}.")
        changed = changed or new_index != target_index
        actions.insert(new_index, target)
        if changed:
            applied_rules.append(
                {
                    "rule_id": rule.rule_id,
                    "type": rule.rule_type,
                    "reason": rule.explanation,
                    "evidence": evidence,
                    "phased_action_id": target_id,
                    "horizon": rule.horizon,
                }
            )

    result["measured_site_characteristics"] = _site_characteristics(
        site_mapping, site
    )
    result["adapted_actions"] = actions
    result["applied_rules"] = applied_rules
    result["excluded_actions"] = excluded_actions


def adapt_plan(
    plan: dict[str, Any],
    site_mapping: dict[str, Any],
    action_library: dict[str, Any],
) -> dict[str, Any]:
    """Return an adapted copy of a remediation plan without changing its candidates."""
    adapted = deepcopy(plan)
    action_by_id = _validate_inputs(adapted, site_mapping, action_library)
    for result in adapted["results"]:
        _apply_rules_to_result(result, site_mapping, action_by_id)
    adapted["site_profile_adaptation"] = {
        "method": (
            "Règles déterministes basées uniquement sur les niveaux et valeurs "
            "des indicateurs mesurés par site ; aucun profil de site n'est utilisé."
        ),
        "rules": [
            {
                "id": rule.rule_id,
                "type": rule.rule_type,
                "conditions": [
                    {
                        "indicator": condition.indicator,
                        "level": condition.level,
                        "value_gt": condition.value_gt,
                        "value_eq": condition.value_eq,
                    }
                    for condition in rule.conditions
                ],
                "action_ids": list(rule.action_ids),
                "explanation": rule.explanation,
                "horizon": rule.horizon,
            }
            for rule in RULES
        ],
    }
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapte les candidates de remédiation aux indicateurs mesurés des sites."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--site-mapping", type=Path, default=DEFAULT_SITE_MAPPING)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTION_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    adapted = adapt_plan(
        _read_json(args.input),
        _read_json(args.site_mapping),
        _read_json(args.action_library),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    total_rules = sum(len(result["applied_rules"]) for result in adapted["results"])
    print(f"Résultats enregistrés : {args.output}")
    print(f"Cellules adaptées : {len(adapted['results'])}")
    print(f"Règles appliquées : {total_rules}")


if __name__ == "__main__":
    main()