"""Estimate hypothetical avoided costs for solar remediation actions."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "fixing_plan_remediation_solar_adapted.json"
DEFAULT_ACTION_LIBRARY = DATA_DIR / "solar_action_library.json"
DEFAULT_OUTPUT = DATA_DIR / "fixing_plan_remediation_solar_impact.json"
SCENARIOS = ("low", "central", "high")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Le fichier JSON doit contenir un objet : {path}.")
    return value


def _action_library(
    library: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    actions = library.get("actions")
    metadata = library.get("_meta")
    if not isinstance(actions, list):
        raise TypeError("La bibliothèque doit contenir une liste 'actions'.")
    if not isinstance(metadata, dict):
        raise TypeError("La bibliothèque doit contenir un objet '_meta'.")
    scenarios = metadata.get("effectiveness_scenarios")
    if not isinstance(scenarios, dict):
        raise TypeError(
            "La bibliothèque doit contenir '_meta.effectiveness_scenarios'."
        )

    reductions: dict[str, dict[str, float]] = {}
    for label, values in scenarios.items():
        if label == "note":
            continue
        if not isinstance(values, dict):
            raise TypeError(f"Scénario d'efficacité invalide : {label}.")
        parsed_values: dict[str, float] = {}
        for key in SCENARIOS:
            value = values.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Réduction invalide pour le scénario {label!r}, {key!r}."
                )
            parsed_values[key] = float(value)
        if not (
            parsed_values["low"]
            <= parsed_values["central"]
            <= parsed_values["high"]
        ):
            raise ValueError(
                f"Les réductions du scénario {label!r} ne sont pas ordonnées."
            )
        reductions[label] = parsed_values

    action_by_id: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("id"), str):
            raise TypeError("La bibliothèque contient une action sans identifiant.")
        action_id = action["id"]
        if action_id in action_by_id:
            raise ValueError(f"Identifiant d'action dupliqué : {action_id}.")
        action_by_id[action_id] = action
        label = action.get("expected_effectiveness")
        if not isinstance(label, str) or label not in reductions:
            raise ValueError(
                f"Scénario d'efficacité absent ou invalide pour {action_id} : {label!r}."
            )
        effort = action.get("effort")
        if effort not in {"S", "M", "L"}:
            raise ValueError(f"Effort invalide pour {action_id} : {effort!r}.")

    return action_by_id, reductions


def estimate_plan_impact(
    plan: dict[str, Any], library: dict[str, Any]
) -> dict[str, Any]:
    """Return an impact-enriched plan; gains are separate per-action hypotheses."""
    results = plan.get("results")
    if not isinstance(results, list):
        raise TypeError("Le plan d'entrée doit contenir une liste 'results'.")
    action_by_id, reductions = _action_library(library)
    estimated_plan = deepcopy(plan)

    for index, result in enumerate(estimated_plan["results"]):
        if not isinstance(result, dict):
            raise TypeError(f"Résultat invalide à l'index {index}.")
        stats = result.get("stats")
        if not isinstance(stats, dict):
            raise TypeError(f"Statistiques invalides à l'index {index}.")
        cell_cost = stats.get("cout_total_eur")
        if (
            isinstance(cell_cost, bool)
            or not isinstance(cell_cost, (int, float))
            or not math.isfinite(cell_cost)
            or cell_cost < 0
        ):
            raise ValueError(
                f"Coût de cellule invalide à l'index {index} : {cell_cost!r}."
            )

        actions = result.get("adapted_actions")
        if not isinstance(actions, list):
            raise TypeError(
                f"Liste 'adapted_actions' invalide à l'index {index}; "
                "exécuter d'abord site_profile_adaptation.py."
            )
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict) or not isinstance(
                action.get("id"), str
            ):
                raise TypeError(
                    f"Action invalide à l'index {action_index} de la cellule {index}."
                )
            action_id = action["id"]
            library_action = action_by_id.get(action_id)
            if library_action is None:
                raise ValueError(
                    f"Action absente de la bibliothèque : {action_id}."
                )
            label = library_action["expected_effectiveness"]
            assumed_reductions = reductions[label]
            gain = {
                scenario: round(cell_cost * assumed_reductions[scenario], 2)
                for scenario in SCENARIOS
            }
            action["effort"] = library_action["effort"]
            action["impact_estimation"] = {
                "is_hypothesis": True,
                "basis": "cout_total_eur de la cellule × réduction supposée",
                "cell_cost_eur": cell_cost,
                "effectiveness_scenario": label,
                "assumed_reduction": assumed_reductions.copy(),
                "expected_gain_eur": gain,
                "effort": library_action["effort"],
            }

    estimated_plan["impact_estimation"] = {
        "status": "hypotheses_non_garanties",
        "source": "solar_action_library.json > _meta.effectiveness_scenarios",
        "formula": "coût de la cellule × réduction supposée de la bibliothèque",
        "scenarios": list(SCENARIOS),
        "aggregation": (
            "Les gains sont fournis par action et ne sont pas additionnés, "
            "car les effets de plusieurs actions peuvent se recouvrir."
        ),
    }
    return estimated_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les gains de coûts hypothétiques par action à partir des "
            "réductions de la bibliothèque solaire."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTION_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    estimated_plan = estimate_plan_impact(
        _read_json(args.input),
        _read_json(args.action_library),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(estimated_plan, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    action_count = sum(
        len(result["adapted_actions"]) for result in estimated_plan["results"]
    )
    print(f"Résultats enregistrés : {args.output}")
    print(f"Cellules traitées : {len(estimated_plan['results'])}")
    print(f"Actions estimées : {action_count} (gains hypothétiques, non cumulés)")


if __name__ == "__main__":
    main()