"""Select evidence-backed solar remediation actions for each problem cell."""

from __future__ import annotations

import argparse
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_STATS = DATA_DIR / "fixing_plan_solar.csv"
DEFAULT_INCIDENTS = DATA_DIR / "incidents_organized_solar.csv"
DEFAULT_ACTION_LIBRARY = DATA_DIR / "solar_action_library.json"
DEFAULT_OUTPUT = DATA_DIR / "fixing_plan_remediation_solar.json"

CELL_KEYS = ("site", "cluster_code", "trigger_category")
CELL_COLUMNS = (*CELL_KEYS, "cluster_name", "type_cellule", "motif_selection")
EXAMPLE_COUNT = 3
MAX_CANDIDATES = 6
CHAT_INPUT_PRICE_USD_PER_MILLION = 0.10
CHAT_CACHED_INPUT_PRICE_USD_PER_MILLION = 0.025
CHAT_OUTPUT_PRICE_USD_PER_MILLION = 0.40


def _normalize(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _cell_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_normalize(record[key]) for key in CELL_KEYS)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_action_library(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open(encoding="utf-8") as file:
        library = json.load(file)
    if not isinstance(library, dict) or not isinstance(library.get("actions"), list):
        raise TypeError(f"Bibliothèque d'actions invalide : {path}.")

    actions = library["actions"]
    seen_ids: set[str] = set()
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("La bibliothèque contient une action invalide.")
        action_id = action.get("id")
        addresses = action.get("addresses")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("Une action de la bibliothèque n'a pas d'identifiant.")
        if action_id in seen_ids:
            raise ValueError(f"Identifiant d'action dupliqué : {action_id}.")
        seen_ids.add(action_id)
        if not isinstance(addresses, dict) or any(
            not isinstance(addresses.get(field), list)
            for field in ("trigger_category", "failure_mode")
        ):
            raise ValueError(f"Champs addresses invalides pour l'action {action_id}.")

    notice = str(library.get("_meta", {}).get("status", "")).strip()
    return actions, notice


def _validate_inputs(stats: pd.DataFrame, incidents: list[dict[str, Any]]) -> None:
    required_stats = set(CELL_KEYS) | {"cluster_name"}
    missing_stats = sorted(required_stats - set(stats.columns))
    if missing_stats:
        raise ValueError(
            f"Colonnes manquantes dans les statistiques : {missing_stats}."
        )
    if stats.empty:
        raise ValueError("Le fichier de statistiques ne contient aucune cellule.")

    missing_incidents = (
        sorted({"incident_id", *CELL_KEYS, "failure_mode"} - set(incidents[0].keys()))
        if incidents
        else ["incident_id", *CELL_KEYS, "failure_mode"]
    )
    if missing_incidents:
        raise ValueError(
            f"Colonnes manquantes dans les incidents organisés : {missing_incidents}."
        )

    if stats.loc[:, CELL_KEYS].isna().any().any():
        raise ValueError("Une clé de cellule est vide dans les statistiques.")
    duplicate_cells = stats.duplicated(list(CELL_KEYS), keep=False)
    if duplicate_cells.any():
        raise ValueError("Les statistiques contiennent des cellules dupliquées.")

    for index, incident in enumerate(incidents):
        if any(
            not _normalize(incident.get(key)) for key in (*CELL_KEYS, "incident_id")
        ):
            raise ValueError(
                f"Une clé de cellule ou un incident_id est vide à la ligne "
                f"{index + 2} des incidents."
            )
    incident_ids = [_normalize(incident["incident_id"]) for incident in incidents]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("Les incidents contiennent des incident_id dupliqués.")


def _group_incidents(
    incidents: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for incident in incidents:
        grouped[_cell_key(incident)].append(incident)
    return grouped


def _action_matches(
    action: dict[str, Any], trigger_category: str, failure_modes: set[str]
) -> set[str]:
    addresses = action["addresses"]
    if _normalize(trigger_category) not in {
        _normalize(value) for value in addresses["trigger_category"]
    }:
        return set()
    return {
        mode
        for mode in failure_modes
        if mode and mode in {_normalize(value) for value in addresses["failure_mode"]}
    }


def _candidate_actions(
    actions: list[dict[str, Any]],
    trigger_category: str,
    failure_modes: set[str],
) -> tuple[list[dict[str, Any]], int]:
    matches: list[tuple[dict[str, Any], set[str]]] = []
    for action in actions:
        matched_modes = _action_matches(action, trigger_category, failure_modes)
        if matched_modes:
            matches.append((action, matched_modes))

    matches.sort(
        key=lambda item: (
            len(item[0]["addresses"]["trigger_category"]),
            len(item[0]["addresses"]["failure_mode"]),
            item[0]["id"],
        )
    )
    total_matches = len(matches)
    candidates = []
    for action, matched_modes in matches[:MAX_CANDIDATES]:
        candidates.append(
            {
                "id": action["id"],
                "title": action.get("title", ""),
                "description": action.get("description", ""),
                "nature": action.get("nature", ""),
                "effort": action.get("effort", ""),
                "horizon": action.get("horizon", ""),
                "requires_investigation_first": action.get(
                    "requires_investigation_first"
                ),
                "prerequisites": action.get("prerequisites", []),
                "caveat": action.get("caveat", ""),
                "evidence_basis": action.get("evidence_basis", ""),
                "matched_failure_modes": sorted(matched_modes),
            }
        )
    return candidates, total_matches


def _numeric_stats(row: pd.Series) -> dict[str, int | float]:
    stats: dict[str, int | float] = {}
    for key, value in row.items():
        if key in CELL_COLUMNS or pd.isna(value):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isfinite(number):
                stats[str(key)] = int(number) if number.is_integer() else number
    if not stats:
        raise ValueError("La cellule ne contient aucun chiffre exploitable.")
    return stats


def _sample_incidents(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(incidents, key=lambda item: str(item["incident_id"]))
    modes = sorted({_normalize(item.get("failure_mode")) for item in ordered} - {""})
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for mode in modes:
        match = next(
            item for item in ordered if _normalize(item.get("failure_mode")) == mode
        )
        selected.append(match)
        selected_ids.add(str(match["incident_id"]))
        if len(selected) >= EXAMPLE_COUNT:
            break
    for incident in ordered:
        if len(selected) >= EXAMPLE_COUNT:
            break
        if str(incident["incident_id"]) not in selected_ids:
            selected.append(incident)
            selected_ids.add(str(incident["incident_id"]))
    return selected


def _cell_payload(
    row: pd.Series,
    incidents: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    library_notice: str,
) -> tuple[dict[str, Any], list[str]]:
    failure_mode_counts = Counter(
        _normalize(incident.get("failure_mode")) for incident in incidents
    )
    failure_modes = sorted(mode for mode in failure_mode_counts if mode)
    candidates, total_matches = _candidate_actions(
        actions, str(row["trigger_category"]), set(failure_modes)
    )

    examples = _sample_incidents(incidents)
    example_ids = {str(incident["incident_id"]) for incident in examples}
    sample_records = []
    for incident in examples:
        sample_records.append(
            {
                "incident_id": str(incident["incident_id"]),
                "failure_mode": incident.get("failure_mode", ""),
                "component": incident.get("component", ""),
            }
        )

    cell = {
        key: _json_value(row[key])
        for key in CELL_COLUMNS
        if key in row.index and not pd.isna(row[key])
    }
    cell["failure_modes"] = failure_modes
    payload = {
        "cell": cell,
        "stats": _numeric_stats(row),
        "example_incidents": sample_records,
        "candidates": candidates,
        "action_library_notice": library_notice,
    }

    gaps = []
    if not failure_modes:
        gaps.append(
            "Le mode de défaillance manque dans les incidents de cette cellule."
        )
    if not candidates:
        gaps.append(
            "Aucune action ne correspond à la jointure exacte entre la cause et "
            "le mode de défaillance."
        )
    elif len(candidates) < 3:
        gaps.append(
            "La bibliothèque fournit moins de trois actions après la jointure exacte."
        )
    if total_matches > MAX_CANDIDATES:
        gaps.append(
            "Des actions correspondantes supplémentaires n'ont pas été transmises "
            "après le classement déterministe des candidates."
        )
    if len(example_ids) < 2:
        gaps.append("Moins de deux incident_id sont disponibles comme exemples.")
    return payload, gaps


def _response_schema(candidate_ids: list[str], stats_keys: list[str]) -> dict[str, Any]:
    action_id_schema: dict[str, Any] = {"type": "string"}
    if candidate_ids:
        action_id_schema["enum"] = candidate_ids

    stat_key_schema: dict[str, Any] = {"type": "string"}
    if stats_keys:
        stat_key_schema["enum"] = stats_keys

    return {
        "type": "object",
        "properties": {
            "selected_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action_id": action_id_schema,
                        "justification": {"type": "string"},
                        "supporting_stats": {
                            "type": "array",
                            "items": stat_key_schema,
                        },
                    },
                    "required": ["action_id", "justification", "supporting_stats"],
                    "additionalProperties": False,
                },
            },
            "overall_justification": {"type": "string"},
            "overall_supporting_stats": {
                "type": "array",
                "items": stat_key_schema,
            },
            "missing_information": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "selected_actions",
            "overall_justification",
            "overall_supporting_stats",
            "missing_information",
        ],
        "additionalProperties": False,
    }


def _validate_response(result: Any, payload: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "selected_actions",
        "overall_justification",
        "overall_supporting_stats",
        "missing_information",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ValueError("La réponse Azure ne respecte pas le schéma attendu.")

    candidate_ids = {item["id"] for item in payload["candidates"]}
    stats_keys = set(payload["stats"])
    selected = result["selected_actions"]
    if not isinstance(selected, list):
        raise TypeError("selected_actions doit être une liste.")
    selected_ids: set[str] = set()
    for action in selected:
        if not isinstance(action, dict) or set(action) != {
            "action_id",
            "justification",
            "supporting_stats",
        }:
            raise ValueError("Une action sélectionnée ne respecte pas le schéma.")
        action_id = action["action_id"]
        if action_id not in candidate_ids:
            raise ValueError(
                f"Azure a sélectionné une action hors candidates : {action_id!r}."
            )
        if action_id in selected_ids:
            raise ValueError(f"Azure a sélectionné deux fois l'action {action_id}.")
        selected_ids.add(action_id)
        if not isinstance(action["justification"], str):
            raise TypeError("La justification d'une action doit être une chaîne.")
        _validate_supporting_stats(
            action["supporting_stats"], stats_keys, "supporting_stats"
        )
        if not action["supporting_stats"]:
            raise ValueError(
                f"La justification de {action_id} ne cite aucune clé de statistiques."
            )

    overall_justification = result["overall_justification"]
    if not isinstance(overall_justification, str):
        raise TypeError("overall_justification doit être une chaîne.")
    _validate_supporting_stats(
        result["overall_supporting_stats"],
        stats_keys,
        "overall_supporting_stats",
    )
    if selected and not result["overall_supporting_stats"]:
        raise ValueError(
            "La justification générale ne cite aucune clé de statistiques."
        )

    missing = result["missing_information"]
    if not isinstance(missing, list) or any(
        not isinstance(item, str) for item in missing
    ):
        raise ValueError("missing_information doit être une liste de chaînes.")
    return result


def _estimate_chat_cost(response: Any) -> float:
    usage = response.usage
    if usage is None:
        raise ValueError("Azure n'a pas retourné l'usage des tokens.")

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
    token_counts = (input_tokens, output_tokens, cached_input_tokens)
    if (
        any(
            not isinstance(count, int) or isinstance(count, bool)
            for count in token_counts
        )
        or min(token_counts) < 0
        or cached_input_tokens > input_tokens
    ):
        raise ValueError("Azure a retourné un usage de tokens invalide.")

    uncached_input_tokens = input_tokens - cached_input_tokens
    input_cost = (
        uncached_input_tokens * CHAT_INPUT_PRICE_USD_PER_MILLION
        + cached_input_tokens * CHAT_CACHED_INPUT_PRICE_USD_PER_MILLION
    ) / 1_000_000
    output_cost = output_tokens * CHAT_OUTPUT_PRICE_USD_PER_MILLION / 1_000_000
    return input_cost + output_cost


def _validate_supporting_stats(
    fields: Any, stats_keys: set[str], field_name: str
) -> None:
    if (
        not isinstance(fields, list)
        or any(
            not isinstance(field, str) or field not in stats_keys for field in fields
        )
        or len(fields) != len(set(fields))
    ):
        raise ValueError(f"{field_name} contient une clé absente des statistiques.")


def _call_azure(
    client: AzureOpenAI,
    deployment: str,
    payload: dict[str, Any],
    on_cost: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    candidate_ids = [item["id"] for item in payload["candidates"]]
    stats_keys = list(payload["stats"])
    response_format: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {
            "name": "solar_remediation_selection",
            "strict": True,
            "schema": _response_schema(candidate_ids, stats_keys),
        },
    }
    system_prompt = (
        "Tu aides à sélectionner des actions de remédiation solaire. "
        "Tu dois travailler uniquement à partir du JSON fourni. Sélectionne "
        "uniquement des action_id présents dans candidates; si candidates est vide, "
        "ne sélectionne aucune action. Justifie chaque choix par les statistiques "
        "et les exemples, et renseigne supporting_stats avec les clés exactes "
        "utilisées. Signale les informations qui manquent ou empêchent une décision "
        "fiable. N'invente aucun fait. Dans les textes de justification et de manque, "
        "ne donne aucun chiffre et ne cite aucun titre ni identifiant d'action : "
        "les action_id sélectionnés sont fournis séparément. Ne recommande jamais "
        "une action hors de candidates. "
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
            },
        ],
        response_format=response_format,
        temperature=0,
    )
    request_cost = _estimate_chat_cost(response)
    if on_cost is not None:
        on_cost(request_cost)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Azure a retourné une réponse vide pour une cellule.")
    print(
        "Réponse brute Azure "
        f"({payload['cell'].get('site', '')} / "
        f"{payload['cell'].get('cluster_code', '')}) :\n{content}"
    )
    return _validate_response(json.loads(content), payload)


def _create_azure_client() -> tuple[AzureOpenAI, str]:
    load_dotenv(override=True)
    import truststore

    truststore.inject_into_ssl()

    def required_environment_value(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise OSError(f"Variable Azure manquante : {name}")
        return value

    client = AzureOpenAI(
        api_key=required_environment_value("AZURE_OPENAI_API_KEY"),
        azure_endpoint=required_environment_value("AZURE_OPENAI_ENDPOINT"),
        api_version=required_environment_value("AZURE_OPENAI_API_VERSION"),
    )
    return client, required_environment_value("AZURE_OPENAI_CHAT_DEPLOYMENT")


def build_remediation_results(
    stats: pd.DataFrame,
    incidents: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    client: AzureOpenAI,
    deployment: str,
    library_notice: str = "",
) -> list[dict[str, Any]]:
    requests = _prepare_remediation_requests(stats, incidents, actions, library_notice)
    return _run_remediation_requests(requests, client, deployment)


def _prepare_remediation_requests(
    stats: pd.DataFrame,
    incidents: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    library_notice: str,
) -> list[tuple[dict[str, Any], list[str]]]:
    _validate_inputs(stats, incidents)
    grouped_incidents = _group_incidents(incidents)
    ordered_stats = stats.sort_values(list(CELL_KEYS), kind="mergesort")
    requests = []
    for _, row in ordered_stats.iterrows():
        key = _cell_key(row.to_dict())
        matching_incidents = grouped_incidents.get(key, [])
        if not matching_incidents:
            raise ValueError(
                "Aucun incident ne correspond à la cellule "
                f"{tuple(row[column] for column in CELL_KEYS)!r}."
            )
        payload, deterministic_gaps = _cell_payload(
            row, matching_incidents, actions, library_notice
        )
        requests.append((payload, deterministic_gaps))
    return requests


def _run_remediation_requests(
    requests: list[tuple[dict[str, Any], list[str]]],
    client: AzureOpenAI,
    deployment: str,
) -> list[dict[str, Any]]:
    results = []
    cumulative_cost = 0.0
    total_calls = len(requests)
    for call_number, (payload, deterministic_gaps) in enumerate(requests, start=1):
        cell_label = (
            f"{payload['cell'].get('site', '')} / "
            f"{payload['cell'].get('cluster_code', '')}"
        )

        def report_cost(
            request_cost: float,
            *,
            call_number: int = call_number,
            cell_label: str = cell_label,
        ) -> None:
            nonlocal cumulative_cost
            cumulative_cost += request_cost
            print(
                f"Coût estimé après l'appel {call_number}/{total_calls} "
                f"({cell_label}) : "
                f"${request_cost:.8f} USD pour cet appel ; "
                f"${cumulative_cost:.8f} USD cumulés "
                "(tarif public GPT-4.1 nano)."
            )

        result = _call_azure(client, deployment, payload, on_cost=report_cost)
        missing_information = list(
            dict.fromkeys([*result["missing_information"], *deterministic_gaps])
        )
        results.append(
            {
                "cell": payload["cell"],
                "stats": payload["stats"],
                "example_incidents": payload["example_incidents"],
                "candidates": payload["candidates"],
                "selected_actions": result["selected_actions"],
                "overall_justification": result["overall_justification"],
                "overall_supporting_stats": result["overall_supporting_stats"],
                "missing_information": missing_information,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sélectionne des actions solaires par cellule à partir d'une jointure "
            "exacte cause / mode de défaillance et d'une validation Azure."
        )
    )
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTION_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stats = pd.read_csv(args.stats, encoding="utf-8-sig")
    incidents = pd.read_csv(
        args.incidents, encoding="utf-8-sig", dtype=str, keep_default_na=False
    ).to_dict(orient="records")
    actions, library_notice = _load_action_library(args.action_library)
    requests = _prepare_remediation_requests(stats, incidents, actions, library_notice)
    # print(f"Payloads préparés pour les {len(requests)} appels Azure :")
    # for call_number, (payload, _) in enumerate(requests, start=1):
    #     print(f"\n--- Payload {call_number}/{len(requests)} ---")
    #     print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    confirmation = (
        input(f"{len(requests)} appels Azure vont être lancés. Confirmer ? [y/N] ")
        .strip()
        .casefold()
    )
    if confirmation not in {"y", "yes"}:
        print("Traitement annulé ; aucun appel Azure n'a été lancé.")
        return

    client, deployment = _create_azure_client()
    results = _run_remediation_requests(requests, client, deployment)
    output = {
        "asset_type": "solar",
        "method": "jointure exacte normalisée sur trigger_category et failure_mode",
        "action_library_notice": library_notice,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Résultats enregistrés : {args.output}")
    print(f"Cellules traitées : {len(results)}")


if __name__ == "__main__":
    main()
