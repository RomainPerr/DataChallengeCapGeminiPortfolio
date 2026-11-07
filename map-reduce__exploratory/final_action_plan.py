"""Build an actionable, trackable solar remediation plan from measured data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_IMPACT_PLAN = DATA_DIR / "fixing_plan_remediation_solar_impact.json"
DEFAULT_INCIDENTS = DATA_DIR / "incidents_organized_solar.csv"
DEFAULT_ACTION_LIBRARY = DATA_DIR / "solar_action_library.json"
DEFAULT_OUTPUT = DATA_DIR / "final_action_plan_solar.json"
DEFAULT_CSV_OUTPUT = DATA_DIR / "final_action_plan_solar.csv"

HORIZON_ORDER = {"immediat": 0, "court_terme": 1, "structurel": 2}
HORIZON_LABELS = {
    "immediat": "Immédiat (0 à 30 jours)",
    "court_terme": "Court terme (1 à 6 mois)",
    "structurel": "Structurel (6 à 24 mois)",
}
HORIZON_MONTHS = {"immediat": 1, "court_terme": 6, "structurel": 24}
EVIDENCE_LEVELS = ("explicite", "suggeree", "non_indiquee")


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Le fichier JSON doit contenir un objet : {path}.")
    return value


def _read_incidents(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "incident_id",
            "site",
            "cluster_code",
            "trigger_category",
            "evidence_level",
            "date",
            "equipment_id",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonnes manquantes dans {path} : {missing}.")
        return list(reader)


def _cell_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(
        _normalize(record.get(field))
        for field in ("site", "cluster_code", "trigger_category")
    )


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    month_days = (
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return date(year, month, min(value.day, month_days[month - 1]))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evidence_summary(incidents: list[dict[str, str]]) -> dict[str, Any]:
    counts = {level: 0 for level in EVIDENCE_LEVELS}
    for incident in incidents:
        level = _normalize(incident.get("evidence_level"))
        if level in counts:
            counts[level] += 1
    total = sum(counts.values())
    documented = counts["explicite"] + counts["suggeree"]
    return {
        "accord_extraction": {
            "value": round(documented / total, 4) if total else None,
            "numerator": documented,
            "denominator": total,
            "unit": "part",
            "evidence_level_counts": counts,
            "interpretation": (
                "Couverture des preuves explicites ou suggérées, et non accord "
                "inter-évaluateurs : aucune double extraction/comparaison "
                "d'extracteurs n'est disponible."
            ),
        }
    }


def _examples(incidents: list[dict[str, str]], planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {incident["incident_id"]: incident for incident in incidents}
    examples: list[dict[str, Any]] = []
    for planned_example in planned:
        source = by_id.get(str(planned_example.get("incident_id", "")))
        if source is None:
            examples.append(dict(planned_example))
            continue
        examples.append(
            {
                "incident_id": source["incident_id"],
                "date": source.get("date", ""),
                "equipment_id": source.get("equipment_id", ""),
                "evidence_level": source.get("evidence_level", ""),
                "failure_mode": source.get("failure_mode", ""),
                "component": source.get("component", ""),
                "trigger_quote": source.get("trigger_quote", ""),
            }
        )
    return examples


def _matches_indicator(
    indicator_name: str, incident: dict[str, str]
) -> bool | None:
    if indicator_name == "part_cause_documentee":
        return _normalize(incident.get("evidence_level")) in {"explicite", "suggeree"}
    if indicator_name == "part_detection_preventive":
        return _normalize(incident.get("detection_context")) in {
            "controle_periodique",
            "inspection_visuelle_ou_thermique",
        }
    if indicator_name == "part_detection_apres_meteo":
        return _normalize(incident.get("detection_context")) == "apres_evenement_meteo"
    if indicator_name == "part_alerte_supervision":
        return _normalize(incident.get("detection_context")) == "alerte_supervision"
    return None


def _repeat_within_90_days(
    target_incidents: list[dict[str, str]],
    site_incidents: list[dict[str, str]],
    data_end: date | None,
) -> dict[str, Any] | None:
    if data_end is None:
        return None
    by_equipment: dict[str, list[date]] = defaultdict(list)
    for incident in site_incidents:
        equipment = _normalize(incident.get("equipment_id"))
        incident_date = _parse_date(incident.get("date"))
        if equipment and incident_date is not None:
            by_equipment[equipment].append(incident_date)
    for dates in by_equipment.values():
        dates.sort()

    eligible = repeated = 0
    for incident in target_incidents:
        equipment = _normalize(incident.get("equipment_id"))
        incident_date = _parse_date(incident.get("date"))
        if not equipment or incident_date is None or data_end < incident_date + timedelta(days=90):
            continue
        eligible += 1
        if any(
            incident_date < next_date <= incident_date + timedelta(days=90)
            for next_date in by_equipment.get(equipment, [])
        ):
            repeated += 1
    return {
        "value": repeated / eligible if eligible else None,
        "numerator": repeated,
        "denominator": eligible,
        "unit": "part",
        "scope": "incidents de la cellule avec 90 jours complets de suivi",
    }


def _calculate_kpi(
    kpi: dict[str, Any],
    cell_incidents: list[dict[str, str]],
    site_incidents: list[dict[str, str]],
    all_incidents: list[dict[str, str]],
    *,
    window_start: date | None = None,
) -> dict[str, Any]:
    name = str(kpi.get("name", ""))
    data_dates = [
        parsed
        for row in all_incidents
        if (parsed := _parse_date(row.get("date"))) is not None
    ]
    data_end = max(data_dates, default=None)
    cell_dates = [
        parsed
        for row in cell_incidents
        if (parsed := _parse_date(row.get("date"))) is not None
    ]
    period_start = window_start or min(cell_dates, default=None)
    period_days = (
        (data_end - period_start).days + 1
        if data_end is not None and period_start is not None
        else 0
    )
    if name == "recidive_90j":
        measurement = _repeat_within_90_days(
            cell_incidents, site_incidents, data_end
        )
    elif name == "mttr":
        eligible_durations = [
            duration
            for row in cell_incidents
            if _normalize(row.get("delay_or_access_issue")) == "piece_en_attente"
            and (duration := _number(row.get("downtime_hours"))) is not None
        ]
        sorted_durations = sorted(eligible_durations)
        size = len(sorted_durations)
        median = (
            (sorted_durations[(size - 1) // 2] + sorted_durations[size // 2]) / 2
            if size
            else None
        )
        measurement = {
            "value": median,
            "numerator": size,
            "denominator": size,
            "unit": "heures",
        }
    elif name in {
        "part_cause_documentee",
        "part_detection_preventive",
        "part_detection_apres_meteo",
        "part_alerte_supervision",
    }:
        matched = [
            _matches_indicator(name, row) for row in cell_incidents
        ]
        measured = [value for value in matched if value is not None]
        measurement = {
            "value": sum(measured) / len(measured) if measured else None,
            "numerator": sum(measured),
            "denominator": len(measured),
            "unit": "part",
        }
    elif name in {"part_incidents_severes_analyses", "exercice_realise"}:
        measurement = None
    else:
        measurement = {
            "value": len(cell_incidents) / (period_days / 365.25)
            if period_days > 0
            else None,
            "observed_count": len(cell_incidents),
            "observation_days": period_days,
            "unit": "incidents/an",
        }

    if measurement is None or measurement.get("value") is None:
        return {
            "name": name,
            "definition": kpi.get("definition", ""),
            "direction": kpi.get("direction", ""),
            "baseline": None,
            "baseline_note": (
                "Non calculable à partir des incidents disponibles; aucun champ "
                "ne mesure cette action ou ce résultat."
            ),
        }
    measurement.update(
        {
            "name": name,
            "definition": kpi.get("definition", ""),
            "direction": kpi.get("direction", ""),
            "observation_start": period_start.isoformat() if period_start else None,
            "observation_end": data_end.isoformat() if data_end else None,
        }
    )
    if measurement["unit"] == "part":
        measurement["value"] = round(float(measurement["value"]), 4)
    else:
        measurement["value"] = round(float(measurement["value"]), 2)
    return {"name": name, "definition": kpi.get("definition", ""), "direction": kpi.get("direction", ""), "baseline": measurement}


def _kpi_target(kpi_measurement: dict[str, Any]) -> dict[str, Any]:
    baseline = kpi_measurement["baseline"]
    if baseline is None:
        if kpi_measurement["name"] == "part_incidents_severes_analyses":
            return {
                "proposed": 1.0,
                "unit": "part",
                "note": "Cible proposée : 100 % des incidents sévères analysés.",
            }
        if kpi_measurement["name"] == "exercice_realise":
            return {
                "proposed": 1,
                "unit": "exercice/an",
                "note": "Cible proposée : au moins un exercice documenté par an.",
            }
        return {
            "proposed": None,
            "note": "À fixer après mesure initiale et validation métier.",
        }
    value = baseline["value"]
    if kpi_measurement["direction"] == "hausse":
        target = min(1.0, value + 0.1) if baseline["unit"] == "part" else value * 1.1
    else:
        target = value * 0.9
    return {
        "proposed": round(target, 4 if baseline["unit"] == "part" else 2),
        "unit": baseline["unit"],
        "note": (
            "Cible indicative : amélioration relative de 10 % par rapport à la "
            "baseline (ou +10 points pour une part en hausse), à valider par le site."
        ),
    }


def _risk_if_inaction(result: dict[str, Any]) -> str:
    cell = result["cell"]
    stats = result["stats"]
    incidents = stats.get("n_incidents", 0)
    cost = stats.get("cout_total_eur")
    risk = (
        f"Risque de répétition du problème « {cell.get('cluster_name', '')} » : "
        f"{incidents} incident(s) recensé(s)"
    )
    if isinstance(cost, (int, float)):
        risk += f", pour {cost:,.2f} € de coûts historiques".replace(",", " ").replace(".", ",", 1)
    if _number(stats.get("score_safety_risk")) and float(stats["score_safety_risk"]) > 0:
        risk += "; un risque sécurité est signalé dans les données de la cellule"
    return risk + "."


def _confidence(
    evidence: dict[str, Any],
    action: dict[str, Any],
    previous: dict[str, Any] | None,
    kpi_target: dict[str, Any],
    latest_kpi: dict[str, Any],
) -> dict[str, Any]:
    previous_tracking = previous.get("suivi", {}) if previous else {}
    implementation_status = previous_tracking.get("statut_mise_en_oeuvre", "a_planifier")
    if implementation_status != "terminee":
        test_verdict = {
            "statut": "non_testé",
            "raison": (
                "Aucune mesure post-intervention marquée comme terminée; les "
                "données historiques ne démontrent pas l'effet de l'action."
            ),
        }
    elif latest_kpi.get("baseline") is None or kpi_target.get("proposed") is None:
        test_verdict = {
            "statut": "non_testable",
            "raison": "La baseline ou la cible KPI ne peut pas être calculée.",
        }
    else:
        value = latest_kpi["baseline"]["value"]
        target = kpi_target["proposed"]
        reached = (
            value >= target
            if latest_kpi["direction"] == "hausse"
            else value <= target
        )
        test_verdict = {
            "statut": "cible_atteinte" if reached else "cible_non_atteinte",
            "raison": (
                "Comparaison descriptive de la mesure rejouée à la cible; "
                "elle ne prouve pas à elle seule un lien causal avec l'action."
            ),
            "mesure": value,
            "cible": target,
        }
    return {
        **evidence,
        "verdict_test": test_verdict,
        "niveau": (
            "indéterminé"
            if evidence["accord_extraction"]["value"] is None
            else "à confirmer"
        ),
        "evidence_basis_action": action.get("evidence_basis", ""),
    }


def _stable_action_id(site: str, cluster_code: str, action_id: str) -> str:
    identity = f"{site}\x1f{cluster_code}\x1f{action_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"ACT-{digest.upper()}"


def _make_sheet(
    result: dict[str, Any],
    action: dict[str, Any],
    library_action: dict[str, Any],
    cell_incidents: list[dict[str, str]],
    site_incidents: list[dict[str, str]],
    all_incidents: list[dict[str, str]],
    plan_date: date,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    cell = result["cell"]
    stats = result["stats"]
    nature = _normalize(action.get("nature", library_action.get("nature")))
    cause_is_undocumented = (
        _normalize(cell.get("trigger_category")) == "cause_indeterminee"
        or "cause non document" in _normalize(cell.get("type_cellule"))
    )
    if nature == "securite" or (cause_is_undocumented and nature == "enquete"):
        horizon = "immediat"
    else:
        horizon = action.get("horizon") or library_action.get("horizon")
    if horizon not in HORIZON_ORDER:
        raise ValueError(
            f"Horizon invalide pour {action.get('id')}: {horizon!r}."
        )
    due = (
        plan_date + timedelta(days=30)
        if horizon == "immediat"
        else _add_months(plan_date, HORIZON_MONTHS[horizon])
    )
    previous_tracking = previous.get("suivi", {}) if previous else {}
    previous_status = previous_tracking.get("statut_mise_en_oeuvre", "a_planifier")
    completion_date = _parse_date(previous_tracking.get("date_mise_en_oeuvre"))
    followup_start = (
        completion_date
        if previous_status == "terminee" and completion_date is not None
        else None
    )
    followup_cell_incidents = (
        [
            incident
            for incident in cell_incidents
            if (incident_date := _parse_date(incident.get("date"))) is not None
            and incident_date >= followup_start
        ]
        if followup_start is not None
        else cell_incidents
    )
    followup_site_incidents = (
        [
            incident
            for incident in site_incidents
            if (incident_date := _parse_date(incident.get("date"))) is not None
            and incident_date >= followup_start
        ]
        if followup_start is not None
        else site_incidents
    )
    kpi = _calculate_kpi(
        library_action.get("kpi", {}),
        followup_cell_incidents,
        followup_site_incidents,
        all_incidents,
        window_start=followup_start,
    )
    target = _kpi_target(kpi)
    previous_baseline = previous_tracking.get("kpi_baseline")
    current_measurement = kpi.get("baseline")
    if previous_baseline is not None:
        reference_baseline = previous_baseline
        latest_measurement = current_measurement
        previous_target = previous_tracking.get("kpi_cible")
        if isinstance(previous_target, dict) and previous_target.get("proposed") is not None:
            target = previous_target
    else:
        reference_baseline = current_measurement
        latest_measurement = None

    evidence = _evidence_summary(cell_incidents)
    previous_examples = result.get("example_incidents", [])
    failure_modes = cell.get("failure_modes", [])
    cause = {
        "cluster_name": cell.get("cluster_name", ""),
        "trigger_category": cell.get("trigger_category", ""),
        "failure_modes": failure_modes,
        "components": sorted(
            {
                str(example.get("component"))
                for example in previous_examples
                if example.get("component")
            }
        ),
    }
    impact = action.get("impact_estimation", {})
    return {
        "identifiant": _stable_action_id(
            str(cell.get("site", "")),
            str(cell.get("cluster_code", "")),
            str(action.get("id", "")),
        ),
        "action_id": action.get("id", ""),
        "site": cell.get("site", ""),
        "cause_ciblee": cause,
        "preuve": {
            "nombre_incidents": stats.get("n_incidents"),
            "part_cout_pct": stats.get("part_cout_pct"),
            "cout_historique_cellule_eur": stats.get("cout_total_eur"),
            "exemples": _examples(cell_incidents, previous_examples),
        },
        "action": {
            "titre": action.get("title", library_action.get("title", "")),
            "description": action.get(
                "description", library_action.get("description", "")
            ),
            "responsable_role": library_action.get("owner_role", "À désigner"),
            "effort": action.get("effort", library_action.get("effort")),
        },
        "sequencement": {
            "horizon": horizon,
            "phase": HORIZON_LABELS[horizon],
            "date_plan": plan_date.isoformat(),
            "echeance": due.isoformat(),
            "ordre_phase": HORIZON_ORDER[horizon],
            "ordre_priorite": (
                0
                if nature == "securite"
                else 1
                if cause_is_undocumented and nature == "enquete"
                else 2
            ),
        },
        "kpi": kpi,
        "cible_kpi": target,
        "gain_attendu_eur": {
            **impact.get("expected_gain_eur", {}),
            "hypothese": bool(impact.get("is_hypothesis", False)),
            "scenario": impact.get("effectiveness_scenario"),
        },
        "risque_inaction": _risk_if_inaction(result),
        "confiance": _confidence(
            evidence,
            library_action,
            previous,
            target,
            {"baseline": latest_measurement, "direction": kpi.get("direction")},
        ),
        "suivi": {
            "statut_mise_en_oeuvre": previous_tracking.get(
                "statut_mise_en_oeuvre", "a_planifier"
            ),
            "date_mise_en_oeuvre": previous_tracking.get("date_mise_en_oeuvre"),
            "kpi_baseline": reference_baseline,
            "kpi_cible": target,
            "derniere_mesure": latest_measurement,
            "date_derniere_mesure": (
                plan_date.isoformat() if latest_measurement is not None else None
            ),
            "note": (
                "Pour le suivi, renseigner date_mise_en_oeuvre et passer le statut "
                "à 'terminee', puis relancer avec --previous-plan. La comparaison "
                "est descriptive et n'établit pas la causalité."
            ),
        },
    }


def build_final_action_plan(
    impact_plan: dict[str, Any],
    incidents: list[dict[str, str]],
    library: dict[str, Any],
    *,
    plan_date: date,
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = impact_plan.get("results")
    actions = library.get("actions")
    if not isinstance(results, list) or not isinstance(actions, list):
        raise TypeError("Le plan et la bibliothèque doivent contenir des listes.")
    action_by_id: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("id"), str):
            raise TypeError("La bibliothèque contient une action invalide.")
        action_by_id[action["id"]] = action
    previous_sheets = {
        sheet["identifiant"]: sheet
        for sheet in (previous_plan or {}).get("actions", [])
        if isinstance(sheet, dict) and isinstance(sheet.get("identifiant"), str)
    }
    incident_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    site_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for incident in incidents:
        incident_groups[_cell_key(incident)].append(incident)
        site_groups[_normalize(incident.get("site"))].append(incident)

    sheets = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("cell"), dict):
            raise TypeError("Résultat de cellule invalide dans le plan d'impact.")
        cell = result["cell"]
        cell_incidents = incident_groups.get(_cell_key(cell), [])
        if not cell_incidents:
            raise ValueError(
                "Aucun incident source ne correspond à la cellule "
                f"{cell.get('site')} / {cell.get('cluster_code')} / "
                f"{cell.get('trigger_category')}."
            )
        adapted_actions = result.get("adapted_actions")
        if not isinstance(adapted_actions, list):
            raise TypeError(
                "Une cellule n'a pas 'adapted_actions'; exécuter les étapes "
                "d'adaptation et d'estimation d'impact avant la fiche finale."
            )
        for action in adapted_actions:
            if not isinstance(action, dict) or not isinstance(action.get("id"), str):
                raise TypeError("Action adaptée invalide.")
            library_action = action_by_id.get(action["id"])
            if library_action is None:
                raise ValueError(f"Action absente de la bibliothèque : {action['id']}.")
            sheet_id = _stable_action_id(
                str(cell.get("site", "")),
                str(cell.get("cluster_code", "")),
                action["id"],
            )
            sheets.append(
                _make_sheet(
                    result,
                    action,
                    library_action,
                    cell_incidents,
                    site_groups[_normalize(cell.get("site"))],
                    incidents,
                    plan_date,
                    previous_sheets.get(sheet_id),
                )
            )
    sheets.sort(
        key=lambda sheet: (
            sheet["sequencement"]["ordre_phase"],
            sheet["sequencement"]["ordre_priorite"],
            sheet["site"],
            sheet["identifiant"],
        )
    )
    return {
        "asset_type": impact_plan.get("asset_type", "solar"),
        "generated_at": plan_date.isoformat(),
        "source_plan": DEFAULT_IMPACT_PLAN.name,
        "schedule": {
            "immediat": HORIZON_LABELS["immediat"],
            "court_terme": HORIZON_LABELS["court_terme"],
            "structurel": HORIZON_LABELS["structurel"],
        },
        "confidence_notice": (
            "Aucune donnée de double extraction ni de mesure post-intervention "
            "n'est disponible dans les fichiers sources; les estimations "
            "d'efficacité sont des hypothèses."
        ),
        "actions": sheets,
    }


CSV_FIELDS = (
    "identifiant",
    "action_id",
    "site",
    "cause_ciblee",
    "preuve",
    "action",
    "responsable",
    "effort",
    "phase",
    "echeance",
    "kpi",
    "cible_kpi",
    "gain_attendu_eur",
    "risque_inaction",
    "confiance",
    "statut_mise_en_oeuvre",
)


def write_action_csv(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sheet in plan["actions"]:
            writer.writerow(
                {
                    "identifiant": sheet["identifiant"],
                    "action_id": sheet["action_id"],
                    "site": sheet["site"],
                    "cause_ciblee": json.dumps(
                        sheet["cause_ciblee"], ensure_ascii=False
                    ),
                    "preuve": json.dumps(sheet["preuve"], ensure_ascii=False),
                    "action": sheet["action"]["titre"],
                    "responsable": sheet["action"]["responsable_role"],
                    "effort": sheet["action"]["effort"],
                    "phase": sheet["sequencement"]["phase"],
                    "echeance": sheet["sequencement"]["echeance"],
                    "kpi": json.dumps(sheet["kpi"], ensure_ascii=False),
                    "cible_kpi": json.dumps(sheet["cible_kpi"], ensure_ascii=False),
                    "gain_attendu_eur": json.dumps(
                        sheet["gain_attendu_eur"], ensure_ascii=False
                    ),
                    "risque_inaction": sheet["risque_inaction"],
                    "confiance": json.dumps(sheet["confiance"], ensure_ascii=False),
                    "statut_mise_en_oeuvre": sheet["suivi"][
                        "statut_mise_en_oeuvre"
                    ],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère les fiches d'action, échéances et KPI de suivi."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_IMPACT_PLAN)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTION_LIBRARY)
    parser.add_argument("--previous-plan", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--plan-date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    previous_plan = (
        _read_json(args.previous_plan) if args.previous_plan is not None else None
    )
    final_plan = build_final_action_plan(
        _read_json(args.input),
        _read_incidents(args.incidents),
        _read_json(args.action_library),
        plan_date=args.plan_date,
        previous_plan=previous_plan,
    )
    final_plan["source_plan"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(final_plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_action_csv(args.csv_output, final_plan)
    print(f"Fiches JSON enregistrées : {args.output}")
    print(f"Fiches CSV enregistrées : {args.csv_output}")
    print(f"Actions séquencées : {len(final_plan['actions'])}")


if __name__ == "__main__":
    main()