"""Render the final solar action plan as a readable French Markdown report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "final_action_plan_solar.json"
DEFAULT_OUTPUT = DATA_DIR / "final_action_report_solar.md"
DEFAULT_ABBREVIATED_OUTPUT = DATA_DIR / "final_action_report_solar_abbreviated.md"
DEFAULT_SITE_MAPPING = DATA_DIR / "site_mapping_solar.json"

PHASE_ORDER = ("immediat", "court_terme", "structurel")
PHASE_NAMES = {
    "immediat": "Immédiat · 0 à 30 jours",
    "court_terme": "Court terme · 1 à 6 mois",
    "structurel": "Structurel · 6 à 24 mois",
}
EFFORT_NAMES = {
    "S": "S · quelques jours-homme",
    "M": "M · quelques semaines",
    "L": "L · projet/investissement",
}
EVIDENCE_LABELS = {
    "explicite": "explicite",
    "suggeree": "suggérée",
    "non_indiquee": "non indiquée",
}
SITE_ATTRIBUTES = {
    "âge": ("age_vieillissement", "age_defaut_conception"),
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


def _read_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        raise TypeError(f"Plan final invalide : {path} (liste 'actions' attendue).")
    for index, action in enumerate(plan["actions"]):
        if not isinstance(action, dict):
            raise TypeError(f"Fiche invalide à l'index {index}.")
    return plan


def _markdown_text(value: Any) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text.replace("|", r"\|").replace("<", "&lt;").replace(">", "&gt;")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_number(value: Any, decimals: int = 0) -> str:
    number = _number(value)
    if number is None:
        return "—"
    formatted = f"{number:,.{decimals}f}"
    return formatted.replace(",", "\u202f").replace(".", ",")


def _format_eur(value: Any) -> str:
    return f"{_format_number(value, 2)} €" if _number(value) is not None else "—"


def _format_percent(value: Any, *, fraction: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if fraction:
        number *= 100
    return f"{_format_number(number, 1)} %"


def _format_kpi_value(value: Any, unit: str) -> str:
    number = _number(value)
    if number is None:
        return "Non mesurée"
    if unit == "part":
        return _format_percent(number, fraction=True)
    if unit == "incidents/an":
        return f"{_format_number(number, 2)} incident(s)/an"
    if unit == "heures":
        return f"{_format_number(number, 1)} h"
    return f"{_format_number(number, 2)} {_markdown_text(unit)}".strip()


def _phase(action: dict[str, Any]) -> str:
    phase = action.get("sequencement", {}).get("horizon")
    if phase not in PHASE_NAMES:
        raise ValueError(
            f"Horizon invalide dans la fiche {action.get('identifiant')}: {phase!r}."
        )
    return phase


def _render_summary(actions: list[dict[str, Any]], generated_at: str) -> list[str]:
    phases = Counter(_phase(action) for action in actions)
    efforts = Counter(
        action.get("action", {}).get("effort", "—") for action in actions
    )
    measured_kpis = sum(
        action.get("kpi", {}).get("baseline") is not None for action in actions
    )
    tested_actions = sum(
        action.get("confiance", {}).get("verdict_test", {}).get("statut")
        not in {None, "non_testé", "non_teste"}
        for action in actions
    )
    sites = {str(action.get("site", "")) for action in actions}
    rows = [
        "| Horizon | Actions |",
        "|:--|--:|",
    ]
    for phase in PHASE_ORDER:
        if phases[phase]:
            rows.append(f"| {_markdown_text(PHASE_NAMES[phase])} | {phases[phase]} |")

    return [
        "# Plan final de remédiation solaire",
        "",
        (
            f"> **Généré le {_markdown_text(generated_at)}** · "
            f"{len(actions)} fiches · {len(sites)} sites couverts"
        ),
        "",
        "## En bref",
        "",
        (
            f"- **Séquençage :** {phases['immediat']} actions immédiates, "
            f"{phases['court_terme']} à court terme et "
            f"{phases['structurel']} structurelles."
        ),
        "- **Effort estimé :** "
        + " · ".join(
            f"{efforts.get(level, 0)} × {level}"
            for level in ("S", "M", "L")
        )
        + ".",
        (
            f"- **Suivi :** {measured_kpis}/{len(actions)} KPI ont une baseline "
            f"calculable dans les données actuelles ; {tested_actions} verdict(s) "
            "post-intervention renseigné(s)."
        ),
        "",
        rows[0],
        rows[1],
        *rows[2:],
        "",
        "## À lire avant décision",
        "",
        (
            "- Les fourchettes de gains basse, centrale et haute sont des "
            "**hypothèses** issues de la bibliothèque d'actions, pas des économies "
            "garanties. Elles sont propres à chaque action et **ne doivent pas "
            "être additionnées** car les effets peuvent se recouvrir."
        ),
        (
            "- La part d'extractions explicites ou suggérées décrit la "
            "**couverture des preuves** ; ce n'est pas un accord entre extracteurs. "
            "Aucun contrôle inter-évaluateurs n'est disponible dans les données "
            "fournies."
        ),
        (
            "- Un verdict « non testé » signifie qu'aucune mesure post-intervention "
            "n'est encore disponible. Les comparaisons ultérieures restent "
            "descriptives et ne prouvent pas à elles seules la causalité."
        ),
        "",
        "---",
        "",
    ]


def _render_examples(evidence: dict[str, Any]) -> list[str]:
    examples = evidence.get("exemples", [])
    if not examples:
        return ["Aucun exemple d’incident n’est disponible.", ""]
    lines: list[str] = []
    for example in examples:
        incident_id = _markdown_text(example.get("incident_id", "—"))
        details = [
            f"équipement {_markdown_text(example.get('equipment_id') or 'non précisé')}",
            f"date {_markdown_text(example.get('date') or 'non précisée')}",
            "preuve "
            + EVIDENCE_LABELS.get(
                _markdown_text(example.get("evidence_level", "")),
                _markdown_text(example.get("evidence_level") or "non précisée"),
            ),
        ]
        quote = _markdown_text(example.get("trigger_quote", ""))
        if len(quote) > 320:
            quote = quote[:317].rstrip() + "..."
        lines.append(f"- **Incident {incident_id}** — " + " · ".join(details) + ".")
        if quote:
            lines.append(f"  > « {quote} »")
    lines.append("")
    return lines


def _render_action(action: dict[str, Any]) -> list[str]:
    identity = _markdown_text(action.get("identifiant", "Fiche sans identifiant"))
    action_id = _markdown_text(action.get("action_id", ""))
    action_data = action.get("action", {})
    cause = action.get("cause_ciblee", {})
    evidence = action.get("preuve", {})
    schedule = action.get("sequencement", {})
    kpi = action.get("kpi", {})
    baseline = kpi.get("baseline")
    target = action.get("cible_kpi", {})
    gains = action.get("gain_attendu_eur", {})
    confidence = action.get("confiance", {})
    extraction = confidence.get("accord_extraction", {})
    verdict = confidence.get("verdict_test", {})
    tracking = action.get("suivi", {})

    if baseline is None:
        baseline_label = _markdown_text(kpi.get("baseline_note") or "Non mesurée")
    else:
        baseline_label = _format_kpi_value(
            baseline.get("value"), baseline.get("unit", "")
        )
        if baseline.get("numerator") is not None and baseline.get("denominator") is not None:
            baseline_label += (
                f" ({baseline['numerator']}/{baseline['denominator']})"
            )
    target_label = _format_kpi_value(target.get("proposed"), target.get("unit", ""))
    target_note = target.get("note")
    if target_note:
        target_label += f" — {_markdown_text(target_note)}"
    gain_range = " · ".join(
        f"{label} : {_format_eur(gains.get(key))}"
        for label, key in (("basse", "low"), ("centrale", "central"), ("haute", "high"))
    )
    examples = evidence.get("exemples", [])
    components = cause.get("components", [])
    failure_modes = cause.get("failure_modes", [])
    extraction_coverage = _format_percent(extraction.get("value"), fraction=True)
    if extraction.get("numerator") is not None and extraction.get("denominator") is not None:
        extraction_coverage += (
            f" ({extraction['numerator']}/{extraction['denominator']})"
        )
    phase_name = _markdown_text(schedule.get("phase") or PHASE_NAMES[_phase(action)])
    due = _markdown_text(schedule.get("echeance") or "À planifier")
    effort = action_data.get("effort")
    effort_label = EFFORT_NAMES.get(effort, _markdown_text(effort or "À préciser"))

    lines = [
        f"#### {_markdown_text(action_data.get('titre') or action_id)}",
        "",
        f"`{identity}` · `{action_id}`",
        "",
        "| Champ | Détail |",
        "|:--|:--|",
        f"| **Site** | {_markdown_text(action.get('site'))} |",
        (
            f"| **Cause ciblée** | {_markdown_text(cause.get('cluster_name'))} "
            f"(`{_markdown_text(cause.get('trigger_category'))}`) |"
        ),
        (
            f"| **Mode(s) / composant(s)** | "
            f"{_markdown_text(', '.join(map(str, failure_modes)) or 'non précisé')} / "
            f"{_markdown_text(', '.join(map(str, components)) or 'non précisé')} |"
        ),
        (
            f"| **Preuve** | {evidence.get('nombre_incidents', '—')} incidents · "
            f"{_format_percent(evidence.get('part_cout_pct'))} du coût total · "
            f"{_format_eur(evidence.get('cout_historique_cellule_eur'))} de coût historique |"
        ),
        f"| **Responsable** | {_markdown_text(action_data.get('responsable_role') or 'À désigner')} |",
        f"| **Effort** | {effort_label} |",
        f"| **Échéance** | {phase_name} · **{due}** |",
        (
            f"| **KPI** | **{_markdown_text(kpi.get('name'))}** — "
            f"{_markdown_text(kpi.get('definition'))} |"
        ),
        f"| **Baseline → cible** | {baseline_label} → {target_label} |",
        (
            f"| **Gain attendu** | {gain_range} "
            f"({'hypothèse' if gains.get('hypothese') else 'non estimé'}"
            f"{', scénario ' + _markdown_text(gains.get('scenario')) if gains.get('scenario') else ''}) |"
        ),
        (
            f"| **Confiance extraction** | Couverture explicite/suggérée : "
            f"{extraction_coverage} — pas un accord inter-évaluateurs |"
        ),
        (
            f"| **Verdict du test** | **{_markdown_text(verdict.get('statut') or 'non renseigné')}**"
            f"{' — ' + _markdown_text(verdict.get('raison')) if verdict.get('raison') else ''} |"
        ),
        (
            f"| **Suivi** | {_markdown_text(tracking.get('statut_mise_en_oeuvre') or 'à planifier')}"
            f"{', mise en œuvre le ' + _markdown_text(tracking.get('date_mise_en_oeuvre')) if tracking.get('date_mise_en_oeuvre') else ''}"
            f"{', dernière mesure le ' + _markdown_text(tracking.get('date_derniere_mesure')) if tracking.get('date_derniere_mesure') else ''} |"
        ),
        "",
        _markdown_text(action_data.get("description")),
        "",
        f"> **Risque en cas d’inaction :** {_markdown_text(action.get('risque_inaction'))}",
        "",
        f"<details><summary>Voir les exemples d’incidents ({len(examples)})</summary>",
        "",
        *_render_examples(evidence),
        "</details>",
        "",
    ]
    return lines


def render_markdown(plan: dict[str, Any]) -> str:
    """Render a validated final action plan as a French Markdown report."""
    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise TypeError("Le plan doit contenir une liste 'actions'.")
    generated_at = str(plan.get("generated_at") or date.today().isoformat())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("Le plan contient une fiche d'action invalide.")
        grouped[_phase(action)].append(action)

    lines = _render_summary(actions, generated_at)
    for phase in PHASE_ORDER:
        phase_actions = grouped.get(phase, [])
        if not phase_actions:
            continue
        phase_actions.sort(
            key=lambda action: (
                action.get("site", ""),
                action.get("sequencement", {}).get("echeance", ""),
                action.get("identifiant", ""),
            )
        )
        lines.extend(
            [
                f"## {PHASE_NAMES[phase]} — {len(phase_actions)} actions",
                "",
            ]
        )
        current_site: str | None = None
        for action in phase_actions:
            site = str(action.get("site", "Site non renseigné"))
            if site != current_site:
                current_site = site
                lines.extend([f"### {_markdown_text(site)}", ""])
            lines.extend(_render_action(action))

    source = Path(str(plan.get("source_plan") or "plan final")).name
    lines.extend(
        [
            "## Méthode et sources",
            "",
            f"- **Plan source :** `{_markdown_text(source)}`.",
            (
                "- Les coûts, nombres d’incidents et exemples proviennent du plan "
                "final de remédiation et des incidents organisés."
            ),
            (
                "- Les rôles, KPI et fourchettes de gains sont ceux de la "
                "bibliothèque d’actions ; les gains restent des hypothèses non additives."
            ),
            (
                "- Les KPI non calculables sur les colonnes disponibles sont "
                "signalés explicitement plutôt que remplacés par une valeur supposée."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _read_site_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise TypeError(f"Mapping de sites invalide : {path}.")
    if not isinstance(mapping.get("profile_mapping"), dict):
        raise TypeError(f"Le mapping ne contient pas 'profile_mapping' : {path}.")
    if not isinstance(mapping.get("indicators"), dict):
        raise TypeError(f"Le mapping ne contient pas 'indicators' : {path}.")
    return mapping


def _site_profiles(
    site: str, profile_mapping: dict[str, Any]
) -> list[str]:
    profiles = []
    candidates = profile_mapping.get("best_candidates", [])
    if not isinstance(candidates, list):
        raise TypeError("'best_candidates' doit être une liste.")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        profile_to_site = candidate.get("profile_to_site")
        if not isinstance(profile_to_site, dict):
            continue
        profiles.extend(
            str(profile)
            for profile, mapped_site in profile_to_site.items()
            if mapped_site == site
        )
    return sorted(set(profiles))


def _site_attribute_summary(
    site: str, indicators: dict[str, Any]
) -> str:
    attributes: list[str] = []
    for attribute, indicator_ids in SITE_ATTRIBUTES.items():
        levels = set()
        for indicator_id in indicator_ids:
            indicator = indicators.get(indicator_id)
            if not isinstance(indicator, dict):
                continue
            ranks = indicator.get("relative_rank_by_site")
            if isinstance(ranks, dict) and ranks.get(site):
                levels.add(str(ranks[site]))
        label = ", ".join(sorted(levels)) if levels else "non mesuré"
        attributes.append(f"{attribute} : {label}")
    return " · ".join(attributes)


def _render_site_mapping(
    sites: set[str], site_mapping: dict[str, Any]
) -> list[str]:
    profile_mapping = site_mapping["profile_mapping"]
    indicators = site_mapping["indicators"]
    status = _markdown_text(profile_mapping.get("status") or "non renseigné")
    lines = [
        "## Mapping des sites",
        "",
        (
            f"**Statut de l’affectation aux profils : {status}.** "
            "Les profils ci-dessous sont les possibilités issues des meilleures "
            "affectations, et non une étiquette certaine."
        ),
        "",
        "| Site | Profil(s) candidat(s) | Niveaux mesurés par attribut |",
        "|:--|:--|:--|",
    ]
    for site in sorted(sites):
        profiles = _site_profiles(site, profile_mapping)
        profile_label = (
            " / ".join(f"`{profile}`" for profile in profiles)
            if profiles
            else "Aucun profil candidat"
        )
        attributes = _site_attribute_summary(site, indicators)
        lines.append(
            f"| {_markdown_text(site)} | {profile_label} | "
            f"{_markdown_text(attributes)} |"
        )
    lines.extend(
        [
            "",
            (
                "Les règles et recommandations restent fondées sur les "
                "indicateurs mesurés du site ; le mapping de profil ambigu ne "
                "sert pas de preuve."
            ),
            "",
        ]
    )
    return lines


def _render_abbreviated_action(action: dict[str, Any]) -> str:
    cause = action.get("cause_ciblee", {})
    action_data = action.get("action", {})
    gains = action.get("gain_attendu_eur", {})
    schedule = action.get("sequencement", {})
    gain_text = " · ".join(
        f"{label} {_format_eur(gains.get(key))}"
        for label, key in (
            ("basse", "low"),
            ("centrale", "central"),
            ("haute", "high"),
        )
    )
    if not gains.get("hypothese"):
        gain_text = "Non estimé"
    recommendation = _markdown_text(action_data.get("titre") or action.get("action_id"))
    description = _markdown_text(action_data.get("description"))
    if description:
        recommendation += f" — {description}"
    return (
        f"| {_markdown_text(schedule.get('echeance') or 'À planifier')} · "
        f"{_markdown_text(PHASE_NAMES[_phase(action)])} | "
        f"{_markdown_text(cause.get('cluster_name') or cause.get('trigger_category'))} | "
        f"{_markdown_text(action_data.get('effort') or '—')} | "
        f"{gain_text} | {recommendation} |"
    )


def render_abbreviated_markdown(
    plan: dict[str, Any], site_mapping: dict[str, Any]
) -> str:
    """Render a concise report with collapsible site-specific action lists."""
    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise TypeError("Le plan doit contenir une liste 'actions'.")
    sites = {str(action.get("site") or "Site non renseigné") for action in actions}
    generated_at = _markdown_text(
        plan.get("generated_at") or date.today().isoformat()
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("Le plan contient une fiche d'action invalide.")
        _phase(action)
        grouped[str(action.get("site") or "Site non renseigné")].append(action)

    lines = [
        "# Plan de remédiation solaire — synthèse",
        "",
        (
            f"> **Généré le {generated_at}** · {len(actions)} actions · "
            f"{len(sites)} sites"
        ),
        "",
        "## Introduction",
        "",
        (
            "Cette synthèse transforme les incidents et leurs coûts observés en "
            "recommandations priorisées par site. Elle présente le séquencement "
            "et les gains attendus sous forme de fourchettes hypothétiques ; "
            "ceux-ci ne constituent ni des économies garanties ni une preuve "
            "d’efficacité."
        ),
        "",
        (
            "Les sites sont repliables pour garder le rapport concis. Ouvrir un "
            "site pour consulter ses actions, triées par échéance. Le gain est "
            "présenté par action et ne doit pas être additionné, les effets "
            "pouvant se recouvrir."
        ),
        "",
        *_render_site_mapping(sites, site_mapping),
        "## Recommandations par site et échéance",
        "",
    ]
    for site in sorted(grouped):
        site_actions = sorted(
            grouped[site],
            key=lambda action: (
                action.get("sequencement", {}).get("echeance") or "9999-12-31",
                PHASE_ORDER.index(_phase(action)),
                action.get("identifiant", ""),
            ),
        )
        lines.extend(
            [
                "<details>",
                (
                    f"<summary><strong>{_markdown_text(site)}</strong> · "
                    f"{len(site_actions)} actions</summary>"
                ),
                "",
                "| Échéance | Cause ciblée | Effort | Gain attendu (hypothèse) | Recommandation |",
                "|:--|:--|:--:|:--|:--|",
            ]
        )
        lines.extend(_render_abbreviated_action(action) for action in site_actions)
        lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère une version complète ou abrégée du rapport final."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--abrege",
        action="store_true",
        help="Produit le rapport concis avec mapping et sections dépliables par site.",
    )
    parser.add_argument("--site-mapping", type=Path, default=DEFAULT_SITE_MAPPING)
    args = parser.parse_args()

    plan = _read_plan(args.input)
    if args.abrege:
        report = render_abbreviated_markdown(
            plan,
            _read_site_mapping(args.site_mapping),
        )
        output_path = args.output or DEFAULT_ABBREVIATED_OUTPUT
    else:
        report = render_markdown(plan)
        output_path = args.output or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Rapport Markdown enregistré : {output_path}")
    print(f"Fiches incluses : {len(plan['actions'])}")


if __name__ == "__main__":
    main()