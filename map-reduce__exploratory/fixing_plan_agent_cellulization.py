"""Build a deterministic, site-level solar incident remediation plan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "incidents_organized_solar.csv"
DEFAULT_OUTPUT = DATA_DIR / "fixing_plan_solar.csv"
PARETO_COST_SHARE = 0.80

CELL_KEYS = ("site", "cluster_code", "trigger_category")
REQUIRED_COLUMNS = (
    *CELL_KEYS,
    "cluster_name",
    "severity",
    "estimated_cost_eur",
    "downtime_hours",
    "safety_risk",
    "recurrence_mentioned",
)
SEVERITIES = {"mineure", "modérée", "sévère"}
SAFETY_RISK_SCORES = {
    "aucun_mentionne": 0.0,
    "risque_incendie": 0.67,
    "risque_electrique": 0.67,
    "danger_personnes": 1.0,
}

ACTION_BY_TYPE = {
    "Rare et coûteux": (
        "Renforcer la prévention, revoir la conception et les dispositifs de sécurité."
    ),
    "Fréquent et peu coûteux": (
        "Standardiser et automatiser les contrôles ; simplifier la procédure d'intervention."
    ),
    "Systémique": (
        "Lancer une action flotte : revoir fournisseur et spécification, "
        "et sécuriser le stock de pièces."
    ),
    "Cause non documentée": (
        "Enquêter sur la cause (données terrain, expertise, contrôles) "
        "avant toute action corrective."
    ),
    "Non classée": (
        "Analyser le cas et compléter le diagnostic avant de définir une action."
    ),
}


def _parse_recurrence(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Valeur recurrence_mentioned invalide : {value!r}.")


def _min_max_scale(values: pd.Series) -> pd.Series:
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def _validate_and_prepare(incidents: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(incidents.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans les incidents : {missing_columns}.")
    if incidents.empty:
        raise ValueError("Le fichier d'incidents ne contient aucune ligne.")

    data = incidents.loc[:, REQUIRED_COLUMNS].copy()
    for column in (*CELL_KEYS, "cluster_name", "severity", "safety_risk"):
        if (
            data[column].isna().any()
            or data[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"La colonne {column} contient une valeur vide.")
        data[column] = data[column].astype(str).str.strip()

    data["severity"] = data["severity"].str.casefold()
    unexpected_severities = sorted(set(data["severity"]) - SEVERITIES)
    if unexpected_severities:
        raise ValueError(f"Niveaux de sévérité non reconnus : {unexpected_severities}.")

    data["safety_risk"] = data["safety_risk"].str.casefold()
    unexpected_risks = sorted(set(data["safety_risk"]) - SAFETY_RISK_SCORES.keys())
    if unexpected_risks:
        raise ValueError(f"Valeurs safety_risk non reconnues : {unexpected_risks}.")

    for column in ("estimated_cost_eur", "downtime_hours"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any() or not np.isfinite(data[column]).all():
            raise ValueError(
                f"La colonne {column} contient une valeur manquante ou non finie."
            )
        if data[column].lt(0).any():
            raise ValueError(
                f"La colonne {column} ne peut pas contenir de valeur négative."
            )

    data["recurrence_mentioned"] = data["recurrence_mentioned"].map(_parse_recurrence)
    data["safety_risk_score"] = data["safety_risk"].map(SAFETY_RISK_SCORES)
    return data


def _classify_cells(cells: pd.DataFrame) -> pd.Series:
    cost_share_high = cells["part_cout"].quantile(0.75)
    frequency_low = cells["part_incidents"].quantile(0.25)
    frequency_high = cells["part_incidents"].quantile(0.75)
    ratio_low = cells["ratio_cout_frequence"].quantile(0.25)
    sites_per_pattern = cells.groupby(
        ["cluster_code", "trigger_category"], observed=True
    )["site"].transform("nunique")

    labels = pd.Series("Non classée", index=cells.index, dtype=object)
    labels.loc[cells["trigger_category"].eq("cause_indeterminee")] = (
        "Cause non documentée"
    )
    labels.loc[labels.eq("Non classée") & sites_per_pattern.ge(2)] = "Systémique"
    labels.loc[
        labels.eq("Non classée")
        & cells["part_cout"].ge(cost_share_high)
        & cells["part_incidents"].le(frequency_low)
    ] = "Rare et coûteux"
    labels.loc[
        labels.eq("Non classée")
        & cells["part_incidents"].ge(frequency_high)
        & cells["ratio_cout_frequence"].le(ratio_low)
    ] = "Fréquent et peu coûteux"
    return labels


def build_fixing_plan(incidents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate and rank site/cluster/trigger cells, retaining Pareto and safety risks."""
    data = _validate_and_prepare(incidents)
    data["is_severe"] = data["severity"].eq("sévère")
    data["has_safety_risk"] = data["safety_risk_score"].gt(0)

    cells = (
        data.groupby(list(CELL_KEYS), observed=True, sort=True)
        .agg(
            cluster_name=("cluster_name", "first"),
            n_incidents=("trigger_category", "size"),
            cout_total_eur=("estimated_cost_eur", "sum"),
            arrets_total_h=("downtime_hours", "sum"),
            n_incidents_severes=("is_severe", "sum"),
            n_recidives=("recurrence_mentioned", "sum"),
            taux_recidive=("recurrence_mentioned", "mean"),
            n_incidents_a_risque=("has_safety_risk", "sum"),
            score_safety_risk=("safety_risk_score", "mean"),
        )
        .reset_index()
    )

    total_cost = float(data["estimated_cost_eur"].sum())
    total_downtime = float(data["downtime_hours"].sum())
    total_severe = int(data["is_severe"].sum())
    total_incidents = len(data)
    cells["part_cout"] = cells["cout_total_eur"] / total_cost if total_cost else 0.0
    cells["part_arrets"] = (
        cells["arrets_total_h"] / total_downtime if total_downtime else 0.0
    )
    cells["part_incidents"] = cells["n_incidents"] / total_incidents
    cells["part_incidents_severes"] = (
        cells["n_incidents_severes"] / total_severe if total_severe else 0.0
    )
    cells["ratio_cout_frequence"] = cells["part_cout"] / cells["part_incidents"]

    score_components = {
        "score_part_cout": "part_cout",
        "score_part_arrets": "part_arrets",
        "score_part_incidents_severes": "part_incidents_severes",
        "score_ratio_cout_frequence": "ratio_cout_frequence",
        "score_recidive": "taux_recidive",
        "score_safety_risk": "score_safety_risk",
    }
    for score_column, metric_column in score_components.items():
        cells[score_column] = _min_max_scale(cells[metric_column])
    cells["score_priorite"] = (cells[list(score_components)].mean(axis=1) * 100).round(
        2
    )

    cells["type_cellule"] = _classify_cells(cells)
    cells["action_recommandee"] = cells["type_cellule"].map(ACTION_BY_TYPE)

    cost_order = cells.sort_values(
        ["cout_total_eur", *CELL_KEYS],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).copy()
    if total_cost:
        cost_order["part_cout_cumulee"] = cost_order["part_cout"].cumsum()
        cost_order["selection_pareto"] = cost_order["cout_total_eur"].gt(
            0
        ) & cost_order["part_cout_cumulee"].shift(fill_value=0).lt(PARETO_COST_SHARE)
    else:
        cost_order["part_cout_cumulee"] = 0.0
        cost_order["selection_pareto"] = False
    pareto_selection = cost_order.set_index(list(CELL_KEYS))[
        ["part_cout_cumulee", "selection_pareto"]
    ]
    cells = cells.join(pareto_selection, on=list(CELL_KEYS))
    # Ne retenir pour sécurité que les cellules ayant au moins 2 incidents afin d'éviter le bruit
    # des incidents isolés uniques (ex. incident 703 avec n=1).
    cells["selection_securite"] = cells["n_incidents_a_risque"].gt(0) & cells["n_incidents"].ge(2)
    cells["motif_selection"] = np.select(
        [
            cells["selection_pareto"] & cells["selection_securite"],
            cells["selection_pareto"],
            cells["selection_securite"],
        ],
        ["pareto_80_cout;risque_securite", "pareto_80_cout", "risque_securite"],
        default="",
    )

    selected = cells.loc[cells["selection_pareto"] | cells["selection_securite"]].copy()
    selected["part_cout_pct"] = (selected["part_cout"] * 100).round(3)
    selected["part_arrets_pct"] = (selected["part_arrets"] * 100).round(3)
    selected["part_incidents_pct"] = (selected["part_incidents"] * 100).round(3)
    selected["part_incidents_severes_pct"] = (
        selected["part_incidents_severes"] * 100
    ).round(3)
    selected["part_cout_cumulee_pct"] = (selected["part_cout_cumulee"] * 100).round(3)

    output_columns = [
        "site",
        "cluster_code",
        "cluster_name",
        "trigger_category",
        "type_cellule",
        "action_recommandee",
        "score_priorite",
        "n_incidents",
        "cout_total_eur",
        "part_cout_pct",
        "arrets_total_h",
        "part_arrets_pct",
        "n_incidents_severes",
        "part_incidents_severes_pct",
        "ratio_cout_frequence",
        "n_recidives",
        "taux_recidive",
        "score_safety_risk",
        "n_incidents_a_risque",
        "part_incidents_pct",
        "part_cout_cumulee_pct",
        "motif_selection",
    ]
    return selected.sort_values(
        ["score_priorite", "cout_total_eur", *CELL_KEYS],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )[output_columns].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construit un plan de remédiation solaire par cellule problème."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    incidents = pd.read_csv(args.input, encoding="utf-8-sig")
    plan = build_fixing_plan(incidents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Plan enregistré : {args.output}")
    print(f"Cellules retenues : {len(plan)}")
    print(
        "Coût couvert : "
        f"{plan['part_cout_pct'].sum():.1f}% (incluant les cellules ajoutées pour sécurité)"
    )
    print(
        plan[
            [
                "site",
                "cluster_name",
                "trigger_category",
                "type_cellule",
                "score_priorite",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
