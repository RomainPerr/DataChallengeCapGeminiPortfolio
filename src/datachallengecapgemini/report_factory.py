"""Report generation step — fill markdown report templates for solar and wind.

The pipeline ends with two markdown reports, one per asset type. Each report
is produced by filling a template stored in the ``templates/`` folder with:

- **Computed statistics** read from the clusterised incidents CSV (number of
  incidents, sites, total cost, downtime, severity breakdown, top clusters…).
- **Figures** — every ``{{ figure_<plot_name> }}`` placeholder is replaced by
  a markdown image pointing at ``data/plots/<label>/png/<plot_name>.png``
  (relative path, so the report stays portable inside the repository). If the
  PNG is missing but the interactive HTML version exists, a link to the HTML
  page is emitted instead; a figure missing from both folders is replaced by
  a placeholder note and a warning is printed.
- **Figure descriptions** — JSON descriptions from the ``json`` subfolder are
  passed to the prose agents as additional context.
- **Prose** — synopsis, methodology, per-cluster descriptions, recommendations
  and conclusion. These are generated in French by the LLM agents of
  :class:`ReportFactory` (Azure OpenAI, gpt-4.1-nano): each section has its
  own prompt template, the generated text is stored in a
  :class:`ReportProse` object so sections can depend on each other, and the
  total token usage is printed at the end of the step.

Placeholders use the ``{{ variable_name }}`` syntax. Filling is done with a
simple regex substitution (no templating dependency); unknown placeholders
raise an error so typos in templates are caught immediately.

Filled reports are written to ``filled_templates/solar/report.md`` and
``filled_templates/wind/report.md``.

Available as a step of the top-level ``dc`` entry point:

- ``uv run dc`` — plotting + reporting (+ PDF export)
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from datachallengecapgemini.equipment_stats import (
    ASSET_PROFILES,
    PRIORITY_RULES,
    RECURRENCE_DAYS,
    SAMPLE_THRESHOLD,
    SEQUENCE_DAYS,
    compute_criticality,
    format_eur,
)

# Project root: src/datachallengecapgemini/report_factory.py -> 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Placeholder syntax: {{ variable_name }}
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

# Prefix of the placeholders that embed a figure: {{ figure_<plot_name> }}
FIGURE_PLACEHOLDER_PREFIX = "figure_"

# Cluster code used for the unclassified / noise incidents
NOISE_CLUSTER_CODE = "C_NOISE"

# Number of clusters detailed in the "cluster_sections" variable
N_DETAILED_CLUSTERS = 5

# Severity levels in increasing order of seriousness
SEVERITY_LEVELS: tuple[str, ...] = ("mineure", "modérée", "sévère")


# ---------------------------------------------------------------------------
# LLM agents — generate the prose sections of the reports
# ---------------------------------------------------------------------------


class ReportState(TypedDict, total=False):
    """State shared by the nodes of the recommendations graph.

    The context fields (``label``, ``statistics``, ``clusters``) are filled
    by the factory before invocation; each node then writes its own output
    field (``cost_drivers``, ``draft_recommendations``, ``recommendations``).
    """

    # Context provided by the factory
    label: str
    statistics: str
    clusters: str
    plot_descriptions: str
    equipment_insights: str
    # Outputs of the graph nodes
    cost_drivers: str
    draft_recommendations: str
    recommendations: str


class _EquipmentCost(TypedDict):
    equipment_id: str
    total_cost_eur: float
    n_incidents: int
    n_severe: int


class _SiteEquipment(TypedDict):
    site_code: str
    equipment: list[_EquipmentCost]


def _new_state_graph() -> StateGraph[ReportState]:  # ty: ignore[invalid-type-arguments]
    """Build an empty ``StateGraph`` over :class:`ReportState`.

    The ``ty: ignore`` comments work around a ``ty`` limitation: it cannot
    match a TypedDict against langgraph's ``StateLike`` protocol bound
    (langgraph's own ``MessagesState`` fails the same check).
    """
    return StateGraph(ReportState)  # ty: ignore[invalid-argument-type]


class TokenUsageCallback(BaseCallbackHandler):
    """Accumulates the token usage of every LLM call made by a factory."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.n_calls += 1
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        return (
            f"{self.total_tokens:,} tokens "
            f"({self.prompt_tokens:,} prompt + {self.completion_tokens:,} "
            f"completion over {self.n_calls} LLM call(s))"
        )


SYNOPSIS_SYSTEM_PROMPT = (
    "Tu es le rédacteur technique d'un rapport d'analyse d'incidents "
    "sur des actifs industriels. Rédige un synopsis exécutif concis "
    "(un paragraphe de 100 à 200 mots) pour le parc {label}.\n\n"
    "Le synopsis doit présenter :\n"
    "- les principaux thèmes d'incidents et leur poids dans le coût "
    "estimé et le temps d'arrêt, à partir des statistiques et du résumé "
    "des clusters ;\n"
    "- les chiffres clés de la performance globale : nombre total "
    "d'incidents, coût total estimé, temps d'arrêt total, répartition "
    "par sévérité et coût moyen par incident ;\n"
    "- l'évolution annuelle, en mettant en évidence la variation du "
    "coût total et du temps d'arrêt entre les deux dernières années "
    "disponibles, si ces données sont fournies.\n\n"
    "Utilise uniquement les statistiques, le résumé des clusters et "
    "les descriptions des figures fournis. N'invente aucun chiffre et "
    "ne déduis pas de causalité non démontrée. Priorise les informations "
    "les plus significatives plutôt que d'énumérer toutes les statistiques. "
    "Rédige en français, avec un ton professionnel, analytique et concis."
)

METHODOLOGY_SYSTEM_PROMPT = (
    "Tu es le rédacteur technique d'un rapport d'analyse d'incidents sur "
    "des actifs industriels. Décris en un paragraphe (80 à 120 mots) la "
    "méthodologie utilisée pour analyser les incidents du parc {label} : "
    "consolidation des rapports d'incident en un jeu de données unique, "
    "plongement lexical (embedding) et clustering thématique des "
    "descriptions d'incidents, recoupement avec les métadonnées structurées "
    "(site, équipement, sévérité, temps d'arrêt, coût estimé, conditions "
    "météorologiques). Mentionne que les figures sont générées à partir du "
    "jeu de données clusterisé, que les coûts sont exprimés en euros et les "
    "temps d'arrêt en heures. Rédige en français, avec un ton de rapport "
    "professionnel."
)

CLUSTER_DESCRIPTION_SYSTEM_PROMPT = (
    "Tu es ingénieur fiabilité et tu rédiges un rapport d'analyse "
    "d'incidents sur des actifs industriels. Décris le cluster d'incidents "
    "'{cluster_name}' ({cluster_code}) du parc {label} en un paragraphe (60 "
    "à 100 mots) : la signature de défaillance commune, le site le plus "
    "touché, la sévérité dominante, l'exposition en coût et en temps "
    "d'arrêt, et une hypothèse de cause racine. Utilise uniquement les "
    "statistiques du cluster et les descriptions des figures fournies, "
    "n'invente aucun chiffre. Rédige en "
    "français, avec un ton de rapport professionnel."
)

COST_DRIVERS_SYSTEM_PROMPT = (
    "Tu es stratège de maintenance pour un parc d'actifs {label}. À partir "
    "des statistiques, du résumé des clusters, des descriptions des figures "
    "et des analyses d'équipements individuels fournis, liste les 3 à 5 "
    "principaux facteurs de coût estimé et de temps d'arrêt, un par ligne, "
    "chacun justifié en une phrase par les données. N'invente aucun "
    "chiffre. Rédige en français."
)

DRAFT_RECOMMENDATIONS_SYSTEM_PROMPT = (
    "Tu es stratège de maintenance pour un parc d'actifs {label}. Pour "
    "chaque facteur de coût et de temps d'arrêt listé ci-dessous, rédige "
    "une recommandation concrète et actionnable. Si des analyses d'équipements "
    "individuels sont fournies, inspire-toi de leurs diagnostics et actions "
    "spécifiques pour rendre tes recommandations plus ciblées. Réponds avec "
    "une puce markdown par facteur, à l'impératif, sans introduction. "
    "Rédige en français."
)

PRIORITISE_RECOMMENDATIONS_SYSTEM_PROMPT = (
    "Tu es directeur de maintenance et tu rédiges la section "
    "recommandations d'un rapport d'analyse d'incidents sur des actifs "
    "industriels pour le parc {label}. Réécris les recommandations "
    "provisoires ci-dessous comme la liste à puces markdown finale et "
    "priorisée du rapport : les plus impactantes en premier, au plus 6 "
    "puces, chacune commençant par '- '. Ne garde que des actions appuyées "
    "par les recommandations provisoires, les statistiques et les analyses "
    "d'équipements individuels, n'invente pas de nouvelles actions. "
    "Rédige en français."
)

CONCLUSION_SYSTEM_PROMPT = (
    "Tu es le rédacteur technique d'un rapport d'analyse d'incidents sur "
    "des actifs industriels. Rédige la conclusion du rapport du parc "
    "{label} en un paragraphe (60 à 100 mots) : résume les principaux "
    "constats et renvoie vers les recommandations. Utilise le synopsis, "
    "les statistiques, les descriptions des figures et les recommandations "
    "fournis, n'invente aucun "
    "chiffre. Rédige en français, avec un ton de rapport professionnel."
)

SYNTHESE_SYSTEM_PROMPT = """Tu es un expert en analyse de données et en fiabilité industrielle.

Tu reçois un rapport complet d'analyse des incidents d'un parc d'actifs industriels, déjà rédigé.

Ta mission est de produire une synthèse générale du rapport en français.

La synthèse doit :
- Identifier les 3 à 5 enseignements les plus importants du rapport.
- Mettre en évidence les tendances et les problèmes récurrents.
- Hiérarchiser les enjeux et les priorités d'action.
- Relier les constats aux recommandations du rapport.
- Mentionner les limites importantes de l'analyse, si elles sont explicitement documentées dans le rapport.

Contraintes :
- Lis le rapport dans son ensemble avant de rédiger.
- Ne répète pas simplement le synopsis ou les recommandations.
- Faire ressortir les principaux facteurs de coût et de temps d'arrêt.
- N'invente aucun chiffre, fait, cause racine ou recommandation.
- Distingue les constats établis des hypothèses.
- Signale les éventuelles contradictions ou limites du rapport.
- Évite les répétitions et les détails secondaires.
- Rédige une synthèse autonome, claire et professionnelle, de 150 à 250 mots, avec des paragraphes et des puces Markdown si nécessaire."""

FIGURE_INTERPRETATION_SYSTEM_PROMPT = (
    "Tu es un analyste de données spécialisé en fiabilité industrielle. "
    "On te fournit la description JSON d'une figure (titre, type de graphique, "
    "données clés) extraite d'un rapport d'analyse d'incidents sur un parc "
    "{label}. Rédige un commentaire d'analyse concis (2 à 4 phrases) de cette "
    "figure : identifie les tendances principales, les points remarquables, les "
    "anomalies ou les insights actionnables. Utilise uniquement les données "
    "fournies dans la description, n'invente aucun chiffre. Rédige en français, "
    "avec un ton de rapport professionnel."
)


@dataclass
class ReportProse:
    """Stores the prose sections generated for one report.

    Every section is ``None`` until it has been generated; the ``write_*``
    methods of :class:`ReportFactory` fill the matching attribute, so a
    section can reuse the sections it depends on (e.g. the conclusion
    reuses the synopsis and the recommendations).
    """

    synopsis: str | None = None
    methodology: str | None = None
    cluster_descriptions: list[str] | None = None
    recommendations: str | None = None
    conclusion: str | None = None


class ReportFactory:
    """Generates the prose sections of an incident report with LLM agents.

    Mirrors :class:`~datachallengecapgemini.plot_factory.PlotFactory`: the
    clusterised incidents dataframe is the main attribute (``df``), completed
    by the asset ``label``. The chat model (Azure OpenAI, gpt-4.1-nano
    deployment) is the ``llm`` attribute, and the context shared by the
    agents (``statistics``, ``clusters``, ``cluster_table`` and the JSON
    ``plot_descriptions``) is precomputed at construction.

    Private ``_*_prompt`` methods only build :class:`ChatPromptTemplate`
    instances. The public ``write_*`` methods handle the execution — a
    simple prompt chain for most sections, a small langgraph ``StateGraph``
    for the multi-step recommendations section — and store the generated
    text in the :class:`ReportProse` object passed as argument, so that
    dependent sections reuse already-generated text when available.
    :meth:`generate_prose` orchestrates the sections requested by the user.

    The token usage of every LLM call is accumulated in ``token_usage`` and
    printed with :meth:`print_token_usage`.
    """

    # Environment variables required to build the default Azure OpenAI LLM
    AZURE_ENV_VARS: ClassVar[tuple[str, ...]] = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    )

    # Maximum number of clusters rendered in the shared cluster summary
    N_SUMMARISED_CLUSTERS: ClassVar[int] = 10

    # Default prose paragraphs generated by generate_prose, in order
    DEFAULT_PARAGRAPHS: ClassVar[tuple[str, ...]] = (
        "synopsis",
        "methodology",
        "cluster_descriptions",
        "recommendations",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        label: str,
        llm: BaseChatModel | None = None,
        plot_descriptions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.df = df
        self.label = label
        self.plot_descriptions = plot_descriptions or {}
        self.plot_descriptions_context = json.dumps(
            self.plot_descriptions, ensure_ascii=False, indent=2
        )
        self.token_usage = TokenUsageCallback()
        self.llm = llm if llm is not None else self._default_llm()
        # Equipment-level insights, collected by collect_equipment_insights()
        # and injected into the recommendations prompt
        self.equipment_insights: str = ""
        # Context elements shared by every agent
        self.statistics = self._render_statistics()
        print("\n===== STATISTIQUES DU PARC =====")
        print(self.statistics)
        self.cluster_table = self._compute_cluster_table()
        self.clusters = self._render_clusters()

    @classmethod
    def _default_llm(cls) -> BaseChatModel:
        """Build the default LLM: Azure OpenAI, gpt-4.1-nano chat deployment."""
        load_dotenv()
        missing = [name for name in cls.AZURE_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"Missing Azure OpenAI environment variable(s): "
                f"{', '.join(missing)}. Set them in your environment or in "
                "the .env file at the project root."
            )
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.2,
        )

    # ------------------------------------------------------------------
    # Context rendering — shared by every agent
    # ------------------------------------------------------------------

    def _render_statistics(self) -> str:
        """Render global and yearly fleet statistics for the agents."""
        df = self.df.copy()

        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        # Global statistics
        total_incidents = len(df)
        total_cost = df["estimated_cost_eur"].sum()
        total_downtime = df["downtime_hours"].sum()

        severity_counts = df["severity"].value_counts()

        top_site = df.groupby("site")["estimated_cost_eur"].sum().idxmax()
        top_equipment = df.groupby("equipment_id")["estimated_cost_eur"].sum().idxmax()

        severity_breakdown = ", ".join(
            f"{level}: {int(severity_counts.get(level, 0))}"
            for level in SEVERITY_LEVELS
        )

        # Mean and median values
        mean_cost = df["estimated_cost_eur"].mean()
        median_cost = df["estimated_cost_eur"].median()

        mean_downtime = df["downtime_hours"].mean()
        median_downtime = df["downtime_hours"].median()

        # Cost per downtime hour
        cost_per_hour = total_cost / total_downtime if total_downtime > 0 else None

        # Yearly statistics
        yearly = (
            df.groupby("year")
            .agg(
                incidents=("date", "size"),
                cost=("estimated_cost_eur", "sum"),
                downtime=("downtime_hours", "sum"),
            )
            .sort_index()
        )

        lines = [
            "PERFORMANCE GLOBALE",
            f"- Nombre total d'incidents : {total_incidents}",
            f"- Sites couverts : {df['site'].nunique()}",
            f"- Équipements surveillés : {df['equipment_id'].nunique()}",
            (
                f"- Période analysée : {df['date'].min().date().isoformat()} "
                f"à {df['date'].max().date().isoformat()}"
            ),
            f"- Coût total estimé : {total_cost:,.2f} EUR",
            f"- Temps d'arrêt total : {total_downtime:,.1f} h",
            f"- Coût moyen par incident : {mean_cost:,.2f} EUR",
            f"- Coût médian par incident : {median_cost:,.2f} EUR",
            f"- Temps d'arrêt moyen par incident : {mean_downtime:,.2f} h",
            f"- Temps d'arrêt médian par incident : {median_downtime:,.2f} h",
            (
                f"- Coût moyen par heure d'arrêt : {cost_per_hour:,.2f} EUR/h"
                if cost_per_hour is not None
                else "- Coût par heure d'arrêt : non calculable"
            ),
            f"- Répartition par sévérité : {severity_breakdown}",
            f"- Site le plus coûteux : {top_site}",
            f"- Équipement le plus coûteux : {top_equipment}",
            "",
            "STATISTIQUES ANNUELLES",
        ]

        for year, row in yearly.iterrows():
            lines.append(
                f"- {year} : {int(row['incidents'])} incidents, "
                f"{row['downtime']:,.1f} h d'arrêt, "
                f"{row['cost']:,.2f} EUR de coût"
            )

        # Compare the last two available years
        lines.append("")
        lines.append("VARIATION ANNUELLE")

        if len(yearly) >= 2:
            last_year = yearly.index[-1]
            previous_year = yearly.index[-2]

            last = yearly.loc[last_year]
            previous = yearly.loc[previous_year]

            def percentage_change(current, prior):
                if prior == 0:
                    return "non calculable (valeur de référence nulle)"
                return f"{(current - prior) / prior * 100:+.2f} %"

            lines.append(
                f"- Temps d'arrêt ({last_year} vs {previous_year}) : "
                f"{percentage_change(last['downtime'], previous['downtime'])}"
            )
            lines.append(
                f"- Coût ({last_year} vs {previous_year}) : "
                f"{percentage_change(last['cost'], previous['cost'])}"
            )
        else:
            lines.append("- Comparaison impossible : moins de deux années disponibles.")

        return "\n".join(lines)

    def _compute_cluster_table(self) -> pd.DataFrame:
        """Per-cluster aggregates (count, cost, downtime), largest first."""
        return (
            self.df.groupby(["cluster_code", "cluster_name"])
            .agg(
                incident_count=("estimated_cost_eur", "size"),
                total_cost_eur=("estimated_cost_eur", "sum"),
                total_downtime_hours=("downtime_hours", "sum"),
            )
            .reset_index()
            .sort_values("incident_count", ascending=False)
            .reset_index(drop=True)
        )

    def _render_clusters(self) -> str:
        """Render the top thematic clusters as a text block for the agents."""
        n_incidents = len(self.df)
        table = self.cluster_table
        thematic = table[table["cluster_code"] != NOISE_CLUSTER_CODE]
        noise = table[table["cluster_code"] == NOISE_CLUSTER_CODE]

        lines = []
        for _, cluster in thematic.head(self.N_SUMMARISED_CLUSTERS).iterrows():
            cluster_df = self.df[self.df["cluster_code"] == cluster["cluster_code"]]
            top_site = cluster_df.groupby("site")["estimated_cost_eur"].sum().idxmax()
            dominant_severity = cluster_df["severity"].value_counts().idxmax()
            lines.append(
                f"- {cluster['cluster_code']} — {cluster['cluster_name']} : "
                f"{int(cluster['incident_count'])} incidents "
                f"({100 * cluster['incident_count'] / n_incidents:.1f} %), "
                f"{cluster['total_cost_eur']:,.2f} EUR, "
                f"{cluster['total_downtime_hours']:,.1f} h de temps d'arrêt, "
                f"site le plus touché : {top_site}, "
                f"sévérité dominante : {dominant_severity}"
            )
        if not noise.empty:
            count = int(noise.iloc[0]["incident_count"])
            lines.append(
                f"- Incidents non classés (bruit) : {count} "
                f"({100 * count / n_incidents:.1f} %)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt templates — one private helper per prose section
    # ------------------------------------------------------------------

    def _synopsis_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the executive synopsis section."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYNOPSIS_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Statistiques clés :\n{statistics}\n\n"
                        "Résumé des clusters :\n{clusters}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Rédige le synopsis du rapport."
                    ),
                ),
            ]
        )

    def _methodology_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the methodology section."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", METHODOLOGY_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Statistiques clés :\n{statistics}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Rédige la section méthodologie du rapport."
                    ),
                ),
            ]
        )

    def _cluster_description_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the description of a single cluster."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", CLUSTER_DESCRIPTION_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Cluster : {cluster_name} ({cluster_code})\n\n"
                        "Statistiques du cluster :\n{cluster_stats}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Rédige la description de ce cluster."
                    ),
                ),
            ]
        )

    def _cost_drivers_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the cost/downtime drivers identification."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", COST_DRIVERS_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Statistiques clés :\n{statistics}\n\n"
                        "Résumé des clusters :\n{clusters}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Analyses d'équipements individuels :\n{equipment_insights}\n\n"
                        "Liste les principaux facteurs de coût et de temps "
                        "d'arrêt."
                    ),
                ),
            ]
        )

    def _draft_recommendations_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the draft recommendations (one per driver)."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", DRAFT_RECOMMENDATIONS_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Facteurs de coût et de temps d'arrêt :\n"
                        "{cost_drivers}\n\n"
                        "Résumé des clusters :\n{clusters}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Analyses d'équipements individuels :\n{equipment_insights}\n\n"
                        "Rédige une recommandation par facteur."
                    ),
                ),
            ]
        )

    def _prioritise_recommendations_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the final prioritised recommendations list."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", PRIORITISE_RECOMMENDATIONS_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Statistiques clés :\n{statistics}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Analyses d'équipements individuels :\n{equipment_insights}\n\n"
                        "Recommandations provisoires :\n"
                        "{draft_recommendations}\n\n"
                        "Rédige la liste à puces finale et priorisée."
                    ),
                ),
            ]
        )

    def _conclusion_prompt(self) -> ChatPromptTemplate:
        """Prompt template of the conclusion section."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", CONCLUSION_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Statistiques clés :\n{statistics}\n\n"
                        "Descriptions des figures :\n{plot_descriptions}\n\n"
                        "Synopsis :\n{synopsis}\n\n"
                        "Recommandations :\n{recommendations}\n\n"
                        "Rédige la conclusion du rapport."
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Public API — write_* methods run the prompts and return the text
    # ------------------------------------------------------------------

    def write_synthese(self, report: str) -> str:
        """Generate a French synthesis from the complete rendered report."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYNTHESE_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Voici le rapport complet à analyser :\n\n{report}\n\n"
                        "Rédige la synthèse générale du rapport."
                    ),
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"report": report},
            config={"callbacks": [self.token_usage]},
        )

    def write_figure_interpretation(self, plot_name: str) -> str:
        """Generate a short analysis paragraph for one figure.

        Uses the figure's JSON description (loaded from the plots directory)
        as context for a small LLM agent that writes 2-4 sentences of analysis.
        """
        description = self.plot_descriptions.get(plot_name)
        if description is None:
            return ""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FIGURE_INTERPRETATION_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Type d'actif : {label}\n\n"
                        "Nom de la figure : {plot_name}\n\n"
                        "Description de la figure (JSON) :\n{description}\n\n"
                        "Statistiques clés du parc :\n{statistics}\n\n"
                        "Rédige le commentaire d'analyse de cette figure."
                    ),
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "label": self.label,
                "plot_name": plot_name,
                "description": json.dumps(description, ensure_ascii=False, indent=2),
                "statistics": self.statistics,
            },
            config={"callbacks": [self.token_usage]},
        )

    def write_figure_interpretations(
        self, plot_names: list[str] | None = None
    ) -> dict[str, str]:
        """Generate analysis paragraphs for selected figures.

        *plot_names* defaults to a curated list of the most insightful figures.
        Returns a mapping ``{plot_name: interpretation_text}``.
        """
        if plot_names is None:
            plot_names = [
                "plot_total_cost_by_site",
                "plot_incident_count_by_severity",
                "plot_total_cost_by_weather_condition",
                "plot_monthly_incident_count",
                "plot_cluster_frequency_cost",
            ]
        interpretations: dict[str, str] = {}
        for name in plot_names:
            if name in self.plot_descriptions:
                print(f"[{self.label}] Interpreting figure: {name}")
                interpretations[name] = self.write_figure_interpretation(name)
        return interpretations

    def collect_equipment_insights(
        self,
        top_n_per_site: int = 3,
        force: bool = False,
    ) -> str:
        """Run ``dc equipment`` for the top-N most critical equipment per site.

        Launches ``uv run dc equipment --site <SITE> --equipment <EQ1> ...``
        as a subprocess for each site, then reads the generated report sections
        (root_causes, recommendations, failure_modes, conclusion) and builds
        a structured context string stored in ``self.equipment_insights``.

        This context is later injected into the recommendations prompt so the
        LLM can enrich the fleet-level recommendations with equipment-specific
        diagnostics and actions.
        """
        import subprocess

        equipment_reports_dir = PROJECT_ROOT / "equipment_reports"
        solar_csv = PROJECT_ROOT / "data" / "incidents_clusterised_solar.csv"

        # Group by site, find the top N critical equipment per site: the
        # equipment criticality score of ``dc equipment`` (incidents, severe
        # incidents, cost and downtime vs the site and the fleet), P1 first,
        # then by score within the site; sites without critical equipment
        # are skipped
        criticality = compute_criticality(self.df)
        critical = criticality[criticality["is_critical"]].sort_values(
            ["equipment_priority", "score_site"], ascending=[True, False]
        )
        site_equipment: dict[Hashable, _SiteEquipment] = {}
        for site, site_critical in critical.groupby("site"):
            equip_cost = site_critical.head(top_n_per_site)
            site_code = str(equip_cost["equipment_id"].iloc[0]).split("-")[0]
            equipment_list: list[_EquipmentCost] = []
            for _, row in equip_cost.iterrows():
                equipment_list.append(
                    {
                        "equipment_id": str(row["equipment_id"]),
                        "total_cost_eur": float(row["total_cost"]),
                        "n_incidents": int(row["n_incidents"]),
                        "n_severe": int(row["n_severe"]),
                    }
                )
            site_equipment[site] = {
                "site_code": site_code,
                "equipment": equipment_list,
            }

        # Run dc equipment subprocess for each site
        for site, info in site_equipment.items():
            site_code = info["site_code"]
            equip_ids = [e["equipment_id"] for e in info["equipment"]]
            print(
                f"[{self.label}] Running dc equipment for site {site_code}: "
                f"{', '.join(equip_ids)}"
            )

            cmd = [
                "uv",
                "run",
                "dc",
                "equipment",
                "--site",
                site_code,
                "--equipment",
                *equip_ids,
                "--solar-csv",
                str(solar_csv),
            ]  # Pour un déploiment effectif, il faudrait bien sûr retirer 'uv run' avant 'dc equipment'
            if force:
                cmd.append("--force")

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(
                    f"[{self.label}] WARNING: dc equipment failed for "
                    f"{site_code} (exit {result.returncode}):\n"
                    f"{result.stderr[-500:]}"
                )
            else:
                print(f"[{self.label}] dc equipment done for {site_code}")

        # Read the generated sections and build the context
        context_parts: list[str] = []
        for site, info in site_equipment.items():
            site_code = info["site_code"]
            site_cost = sum(e["total_cost_eur"] for e in info["equipment"])
            parts = [
                (
                    f"Site : {site} (coût total des {len(info['equipment'])} "
                    f"équipements prioritaires : {site_cost:,.0f} EUR)"
                )
            ]
            for equip in info["equipment"]:
                equip_id = equip["equipment_id"]
                sections_dir = equipment_reports_dir / site_code / equip_id / "sections"
                parts.append(
                    f"\nÉquipement : {equip_id} "
                    f"(coût : {equip['total_cost_eur']:,.0f} EUR, "
                    f"{equip['n_incidents']} incidents dont "
                    f"{equip['n_severe']} sévères)"
                )
                section_found = False
                for section_name in (
                    "root_causes",
                    "recommendations",
                    "failure_modes",
                    "conclusion",
                ):
                    section_path = sections_dir / f"{section_name}.md"
                    if section_path.exists():
                        text = section_path.read_text(encoding="utf-8")
                        parts.append(f"\n--- {section_name} ---\n{text}")
                        section_found = True
                if not section_found:
                    parts.append("\n*Rapport d'équipement non disponible.*")
            context_parts.append("\n".join(parts))

        self.equipment_insights = "\n\n".join(context_parts)
        return self.equipment_insights

    def write_synopsis(self, prose: ReportProse | None = None) -> str:
        """Generate the executive synopsis; stored in *prose* if given."""
        chain = self._synopsis_prompt() | self.llm | StrOutputParser()
        text = chain.invoke(
            {
                "label": self.label,
                "statistics": self.statistics,
                "clusters": self.clusters,
                "plot_descriptions": self.plot_descriptions_context,
            },
            config={"callbacks": [self.token_usage]},
        )
        if prose is not None:
            prose.synopsis = text
        return text

    def write_methodology(self, prose: ReportProse | None = None) -> str:
        """Generate the methodology section; stored in *prose* if given."""
        chain = self._methodology_prompt() | self.llm | StrOutputParser()
        text = chain.invoke(
            {
                "label": self.label,
                "statistics": self.statistics,
                "plot_descriptions": self.plot_descriptions_context,
            },
            config={"callbacks": [self.token_usage]},
        )
        if prose is not None:
            prose.methodology = text
        return text

    def write_cluster_description(
        self, cluster_code: str, cluster_name: str, cluster_stats: str
    ) -> str:
        """Generate the description of a single cluster."""
        chain = self._cluster_description_prompt() | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "label": self.label,
                "cluster_code": cluster_code,
                "cluster_name": cluster_name,
                "cluster_stats": cluster_stats,
                "plot_descriptions": self.plot_descriptions_context,
            },
            config={"callbacks": [self.token_usage]},
        )

    def write_cluster_descriptions(
        self,
        n_clusters: int = N_DETAILED_CLUSTERS,
        prose: ReportProse | None = None,
    ) -> list[str]:
        """Generate one description per top thematic cluster (largest first)."""
        n_incidents = len(self.df)
        thematic = self.cluster_table[
            self.cluster_table["cluster_code"] != NOISE_CLUSTER_CODE
        ]
        descriptions = []
        for _, cluster in thematic.head(n_clusters).iterrows():
            code = cluster["cluster_code"]
            cluster_df = self.df[self.df["cluster_code"] == code]
            top_site = cluster_df.groupby("site")["estimated_cost_eur"].sum().idxmax()
            dominant_severity = cluster_df["severity"].value_counts().idxmax()
            cluster_stats = (
                f"{int(cluster['incident_count'])} incidents "
                f"({100 * cluster['incident_count'] / n_incidents:.1f} % de "
                f"tous les incidents), "
                f"{cluster['total_cost_eur']:,.2f} EUR de coût total estimé, "
                f"{cluster['total_downtime_hours']:,.1f} h de temps d'arrêt "
                f"total, site le plus touché : {top_site}, "
                f"sévérité dominante : {dominant_severity}"
            )
            descriptions.append(
                self.write_cluster_description(
                    code, cluster["cluster_name"], cluster_stats
                )
            )
        if prose is not None:
            prose.cluster_descriptions = descriptions
        return descriptions

    def write_recommendations(self, prose: ReportProse | None = None) -> str:
        """Generate the recommendations section; stored in *prose*.

        Recommendations are more complex than the other sections, so this
        writer merges three specialised prompt chains into a single langgraph
        graph:

        1. ``identify_drivers`` — extract the main cost/downtime drivers;
        2. ``draft_recommendations`` — draft one recommendation per driver;
        3. ``prioritise`` — merge, prioritise and format the final list.
        """
        if prose is None:
            prose = ReportProse()
        drivers_chain = self._cost_drivers_prompt() | self.llm | StrOutputParser()
        draft_chain = (
            self._draft_recommendations_prompt() | self.llm | StrOutputParser()
        )
        prioritise_chain = (
            self._prioritise_recommendations_prompt() | self.llm | StrOutputParser()
        )

        def identify_drivers(state: ReportState) -> ReportState:
            return {"cost_drivers": drivers_chain.invoke(dict(state))}

        def draft_recommendations(state: ReportState) -> ReportState:
            return {"draft_recommendations": draft_chain.invoke(dict(state))}

        def prioritise(state: ReportState) -> ReportState:
            return {"recommendations": prioritise_chain.invoke(dict(state))}

        graph = _new_state_graph()
        graph.add_node("identify_drivers", identify_drivers)
        graph.add_node("draft_recommendations", draft_recommendations)
        graph.add_node("prioritise", prioritise)
        graph.add_edge(START, "identify_drivers")
        graph.add_edge("identify_drivers", "draft_recommendations")
        graph.add_edge("draft_recommendations", "prioritise")
        graph.add_edge("prioritise", END)

        state: ReportState = {
            "label": self.label,
            "statistics": self.statistics,
            "clusters": self.clusters,
            "plot_descriptions": self.plot_descriptions_context,
            "equipment_insights": self.equipment_insights,
        }
        text = graph.compile().invoke(state, config={"callbacks": [self.token_usage]})[
            "recommendations"
        ]
        prose.recommendations = text
        return text

    def write_conclusion(self, prose: ReportProse | None = None) -> str:
        """Generate the conclusion; stored in *prose*.

        The conclusion depends on the synopsis and the recommendations: if
        *prose* already contains them they are reused, otherwise the
        corresponding ``write_*`` method is called to generate them first.
        """
        if prose is None:
            prose = ReportProse()
        if prose.synopsis is None:
            self.write_synopsis(prose)
        if prose.recommendations is None:
            self.write_recommendations(prose)
        chain = self._conclusion_prompt() | self.llm | StrOutputParser()
        text = chain.invoke(
            {
                "label": self.label,
                "statistics": self.statistics,
                "plot_descriptions": self.plot_descriptions_context,
                "synopsis": prose.synopsis,
                "recommendations": prose.recommendations,
            },
            config={"callbacks": [self.token_usage]},
        )
        prose.conclusion = text
        return text

    def generate_prose(
        self,
        paragraphs: Sequence[str] = DEFAULT_PARAGRAPHS,
        n_clusters: int = N_DETAILED_CLUSTERS,
    ) -> ReportProse:
        """Generate the requested prose paragraphs into a :class:`ReportProse`.

        Each name in *paragraphs* is mapped to the corresponding
        ``write_<name>`` method (e.g. ``"methodology"`` ->
        :meth:`write_methodology`) and called in order with the shared
        :class:`ReportProse` object, so a section can reuse — or generate on
        demand — the sections it depends on. A name without a matching
        method triggers a warning and is skipped.
        """
        print(f"[{self.label}] Generating report prose with LLM agents...")
        prose = ReportProse()
        for name in paragraphs:
            writer = getattr(self, f"write_{name}", None)
            if not callable(writer):
                print(
                    f"[{self.label}] WARNING: no write_{name} method for "
                    f"paragraph '{name}', skipping it."
                )
                continue
            if name == "cluster_descriptions":
                writer(n_clusters=n_clusters, prose=prose)
            else:
                writer(prose=prose)
        return prose

    def print_token_usage(self) -> None:
        """Print the token usage accumulated by this factory's agents."""
        print(f"Token usage ({self.label}): {self.token_usage.summary()}")


class SolarReportFactory(ReportFactory):
    """Generates report prose for the solar fleet (label ``solar``)."""

    def __init__(
        self,
        df: pd.DataFrame,
        llm: BaseChatModel | None = None,
        plot_descriptions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(df, "solar", llm=llm, plot_descriptions=plot_descriptions)

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        llm: BaseChatModel | None = None,
        plot_descriptions: dict[str, dict[str, Any]] | None = None,
    ) -> "SolarReportFactory":
        return cls(pd.read_csv(csv_path), llm=llm, plot_descriptions=plot_descriptions)


class WindReportFactory(ReportFactory):
    """Generates report prose for the wind fleet (label ``wind``)."""

    def __init__(
        self,
        df: pd.DataFrame,
        llm: BaseChatModel | None = None,
        plot_descriptions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(df, "wind", llm=llm, plot_descriptions=plot_descriptions)

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        llm: BaseChatModel | None = None,
        plot_descriptions: dict[str, dict[str, Any]] | None = None,
    ) -> "WindReportFactory":
        return cls(pd.read_csv(csv_path), llm=llm, plot_descriptions=plot_descriptions)


def fill_template(template: str, context: dict[str, str]) -> str:
    """Replace every ``{{ variable_name }}`` placeholder in *template*.

    Raises
    ------
    ValueError
        If a placeholder has no matching key in *context*.
    """
    unknown: list[str] = []

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in context:
            unknown.append(name)
            return match.group(0)
        return context[name]

    filled = PLACEHOLDER_PATTERN.sub(_replace, template)
    if unknown:
        raise ValueError(
            f"Unknown template placeholder(s): {', '.join(sorted(set(unknown)))}. "
            f"Available variables: {', '.join(sorted(context))}"
        )
    return filled


def _figure_markdown(plot_name: str, plots_dir: Path, report_dir: Path) -> str:
    """Markdown for one figure: a PNG image, or a link to the HTML version."""
    png_path = plots_dir / "png" / f"{plot_name}.png"
    if png_path.exists():
        relative = os.path.relpath(png_path, report_dir).replace(os.sep, "/")
        return f"![{plot_name}]({relative})"
    html_path = plots_dir / "html" / f"{plot_name}.html"
    if html_path.exists():
        relative = os.path.relpath(html_path, report_dir).replace(os.sep, "/")
        return f"[Interactive figure: {plot_name}]({relative})"
    raise FileNotFoundError(
        f"No figure found for plot '{plot_name}' in {plots_dir / 'png'} "
        f"or {plots_dir / 'html'} — run the plotting step first."
    )


def _load_plot_descriptions(plots_dir: Path) -> dict[str, dict[str, Any]]:
    """Load each plot's JSON description from *plots_dir/json*."""
    descriptions: dict[str, dict[str, Any]] = {}
    for path in sorted((plots_dir / "json").glob("plot_*.json")):
        description = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(description, dict):
            raise TypeError(f"Plot description must be a JSON object: {path}")
        descriptions[path.stem] = description
    return descriptions


def build_report_context(
    df: pd.DataFrame,
    label: str,
    plots_dir: Path,
    report_dir: Path,
    prose: ReportProse,
    figure_interpretations: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build every template variable for one report.

    Real statistics come from *df* (the clusterised incidents dataframe);
    figures are resolved from *plots_dir* (``png`` and ``html`` subfolders) as
    ``figure_<plot_name>`` variables pointing relatively to *report_dir*.
    Prose blocks come from the :class:`ReportProse` object generated by
    :meth:`ReportFactory.generate_prose`.
    """
    if (
        prose.synopsis is None
        or prose.methodology is None
        or prose.cluster_descriptions is None
        or prose.recommendations is None
    ):
        raise ValueError(
            "Missing prose section(s) in the ReportProse object — generate "
            "every section with ReportFactory.generate_prose before "
            "building the report context."
        )
    cluster_descriptions = prose.cluster_descriptions
    if not isinstance(cluster_descriptions, list):
        raise TypeError("The 'cluster_descriptions' prose value must be a list.")

    median_cost = df["estimated_cost_eur"].median()
    mean_downtime = df["downtime_hours"].mean()
    median_downtime = df["downtime_hours"].median()
    total_downtime = df["downtime_hours"].sum()

    n_incidents = len(df)
    severity_counts = df["severity"].value_counts()
    total_cost = df["estimated_cost_eur"].sum()
    dates = pd.to_datetime(df["date"])
    top_site = df.groupby("site")["estimated_cost_eur"].sum().idxmax()
    top_equipment = df.groupby("equipment_id")["estimated_cost_eur"].sum().idxmax()
    cost_per_hour = (
        total_cost / total_downtime if total_downtime > 0 else "non calculable"
    )

    # Clusters, largest first; the noise cluster is reported separately
    cluster_counts = (
        df.groupby(["cluster_code", "cluster_name"])
        .size()
        .to_frame("incident_count")
        .reset_index()
        .sort_values("incident_count", ascending=False)
    )
    thematic_counts = cluster_counts[
        cluster_counts["cluster_code"] != NOISE_CLUSTER_CODE
    ]

    rows = ["| Cluster | Incidents | Part |", "| --- | ---: | ---: |"]
    for _, cluster in cluster_counts.iterrows():
        code = cluster["cluster_code"]
        name = cluster["cluster_name"]
        count = cluster["incident_count"]
        cell = (
            f"{name} (non classé)" if code == NOISE_CLUSTER_CODE else f"{code} — {name}"
        )
        rows.append(f"| {cell} | {count} | {100 * count / n_incidents:.1f}% |")
    top_clusters_table = "\n".join(rows)

    sections = []
    detailed_clusters = thematic_counts.head(N_DETAILED_CLUSTERS)
    if len(cluster_descriptions) != len(detailed_clusters):
        raise ValueError(
            "The generated cluster descriptions must match the number of "
            f"reported clusters ({len(detailed_clusters)}); received "
            f"{len(cluster_descriptions)}."
        )
    for (_, cluster), description in zip(
        detailed_clusters.iterrows(), cluster_descriptions
    ):
        code = cluster["cluster_code"]
        name = cluster["cluster_name"]
        count = cluster["incident_count"]
        cluster_df = df[df["cluster_code"] == code]
        cluster_top_site = (
            cluster_df.groupby("site")["estimated_cost_eur"].sum().idxmax()
        )
        dominant_severity = cluster_df["severity"].value_counts().idxmax()
        sections.append(
            f"### {code} — {name}\n\n"
            f"- **Incidents :** {count} ({100 * count / n_incidents:.1f}% de tous les incidents)\n"
            f"- **Site le plus touché :** {cluster_top_site}\n"
            f"- **Sévérité dominante :** {dominant_severity}\n\n"
            f"{description}"
        )
    cluster_sections = "\n\n".join(sections)

    context: dict[str, str] = {
        "report_title": f"Rapport d'analyse des incidents — Parc {'solaire' if label == 'solar' else 'éolien'}",
        "asset_label": "Solaire" if label == "solar" else "Éolien",
        "asset_label_lower": label,
        "generation_date": date.today().isoformat(),
        "n_incidents": str(n_incidents),
        "n_sites": str(df["site"].nunique()),
        "n_equipment": str(df["equipment_id"].nunique()),
        "period_start": dates.min().date().isoformat(),
        "period_end": dates.max().date().isoformat(),
        "total_cost_eur": f"{total_cost:,.2f}",
        "total_downtime_hours": f"{total_downtime:,.1f}",
        "avg_cost_per_incident": f"{total_cost / n_incidents:,.2f}",
        "n_minor": str(int(severity_counts.get("mineure", 0))),
        "n_moderate": str(int(severity_counts.get("modérée", 0))),
        "n_severe": str(int(severity_counts.get("sévère", 0))),
        "top_site_by_cost": str(top_site),
        "top_equipment_by_cost": str(top_equipment),
        "n_clusters": str(len(thematic_counts)),
        "top_clusters_table": top_clusters_table,
        "synopsis": prose.synopsis,
        "methodology": prose.methodology,
        "cluster_sections": cluster_sections,
        "recommendations": prose.recommendations,
        "synthesis": "",
        "median_cost_per_incident": f"{median_cost:,.2f}",
        "avg_downtime_per_incident": f"{mean_downtime:,.2f}",
        "median_downtime_per_incident": f"{median_downtime:,.2f}",
        "avg_cost_per_downtime_hour": f"{cost_per_hour:,.2f}",
    }

    # Figure interpretation paragraphs (one per interpreted plot)
    # Figure interpretation paragraphs — always add keys for the plots
    # referenced in the solar template, so fill_template never fails on a
    # missing interpretation placeholder
    _INTERPRETED_PLOTS = (
        "plot_total_cost_by_site",
        "plot_incident_count_by_severity",
        "plot_total_cost_by_weather_condition",
        "plot_monthly_incident_count",
        "plot_cluster_frequency_cost",
    )
    for plot_name in _INTERPRETED_PLOTS:
        text = ""
        if figure_interpretations:
            text = figure_interpretations.get(plot_name, "")
        context[f"interpretation_{plot_name}"] = text

    # One figure_<plot_name> variable per available plot (PNG preferred,
    # interactive HTML as fallback)
    png_dir = plots_dir / "png"
    html_dir = plots_dir / "html"
    if not png_dir.is_dir() and not html_dir.is_dir():
        raise FileNotFoundError(
            f"No plots found in {plots_dir} — run the plotting step first (uv run dc)."
        )
    plot_names = sorted(
        {path.stem for path in png_dir.glob("plot_*.png")}
        | {path.stem for path in html_dir.glob("plot_*.html")}
    )
    for plot_name in plot_names:
        context[f"{FIGURE_PLACEHOLDER_PREFIX}{plot_name}"] = _figure_markdown(
            plot_name, plots_dir, report_dir
        )

    return context


def render_report(template: str, context: dict[str, str], plots_dir: Path) -> str:
    """Fill *template* with *context*, warning about missing figures.

    A ``figure_<plot_name>`` placeholder without a matching plot is replaced
    by a placeholder note and a warning is printed, so the rest of the
    report is still filled.
    """
    context = dict(context)
    missing_figures = sorted(
        {
            match.group(1)
            for match in PLACEHOLDER_PATTERN.finditer(template)
            if match.group(1).startswith(FIGURE_PLACEHOLDER_PREFIX)
            and match.group(1) not in context
        }
    )
    for name in missing_figures:
        plot_name = name[len(FIGURE_PLACEHOLDER_PREFIX) :]
        print(
            f"WARNING: figure '{plot_name}' is mentioned in the template "
            f"but missing from {plots_dir / 'png'} and "
            f"{plots_dir / 'html'} — run the plotting step first."
        )
        context[name] = f"*(Figure manquante : {plot_name})*"
    return fill_template(template, context)


def run_reporting(
    solar_csv_path: Path,
    wind_csv_path: Path,
    plots_output_dir: Path,
    templates_dir: Path,
    reports_output_dir: Path,
) -> None:
    """Fill the solar and wind markdown templates and save the reports.

    Reads ``<templates_dir>/<label>_report.md`` for each label, fills it with
    statistics from the clusterised CSV and figures from
    ``<plots_output_dir>/<label>/{png,html,json}``, and writes the result to
    ``<reports_output_dir>/<label>/report.md``.

    Prose is generated by the LLM agents of :class:`ReportFactory` (Azure
    OpenAI, gpt-4.1-nano), and token usage is printed per report.
    """
    for label, factory_type, csv_path in (
        ("solar", SolarReportFactory, solar_csv_path),
        ("wind", WindReportFactory, wind_csv_path),
    ):
        template_path = templates_dir / f"{label}_report.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Report template not found: {template_path}")

        df = pd.read_csv(csv_path)
        plots_dir = plots_output_dir / label
        report_dir = reports_output_dir / label
        report_dir.mkdir(parents=True, exist_ok=True)

        plot_descriptions = _load_plot_descriptions(plots_dir)
        factory = factory_type(df, plot_descriptions=plot_descriptions)

        # Solar-only: collect equipment insights (runs dc equipment subprocess)
        # before generating prose, so recommendations can use them
        if label == "solar":
            factory.collect_equipment_insights()

        prose = factory.generate_prose()

        # Solar-only: figure interpretations
        figure_interpretations: dict[str, str] | None = None
        if label == "solar":
            figure_interpretations = factory.write_figure_interpretations()

        template = template_path.read_text(encoding="utf-8")
        context = build_report_context(
            df,
            label,
            plots_dir,
            report_dir,
            prose=prose,
            figure_interpretations=figure_interpretations,
        )
        synthesis_input = render_report(template, context, plots_dir)
        context["synthesis"] = factory.write_synthese(synthesis_input).strip()
        report = render_report(template, context, plots_dir)

        report_path = report_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"Saved {report_path}")

        factory.print_token_usage()

    print(f"Done. Filled reports are in {reports_output_dir}")


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Add report-generation-specific arguments to *parser*."""
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=PROJECT_ROOT / "templates",
        help="Directory containing the markdown report templates (<label>_report.md).",
    )
    parser.add_argument(
        "--reports-output-dir",
        type=Path,
        default=PROJECT_ROOT / "filled_templates",
        help="Directory where filled reports are saved (<label>/report.md subfolders are created inside).",
    )


# ===========================================================================
# Equipment-level reports — one Azure OpenAI agent per report section
# ===========================================================================
#
# For every equipment of a site (see ``run_equipment_reporting``):
#   1. ``plot_factory.run_equipment_plotting`` has written figures/{png,html}/,
#      json/<figure>.json (the data of each figure) and json/stats.json;
#   2. each ``section_*`` function picks the statistics and figure data of
#      its section, asks its own specialised agent (system prompt = role +
#      shared rules; user prompt = section, questions, data) and renders the
#      section markdown, saved as sections/<name>.md;
#   3. the sections are assembled into report.md with
#      templates/equipment/equipment_report.md.
# The agents never calculate: every number comes from json/, and the
# priorities, risk level and confidence levels are decided in Python.

# Equipment templates live in their own subfolder of templates/
EQUIPMENT_TEMPLATE_NAME = "equipment/equipment_report.md"
INCIDENT_REF_PATTERN = re.compile(r"#(\d+)\b")
# Numbers written in text: "49 142", "207,762", "1.33", "2023" (signs ignored)
NUMBER_PATTERN = re.compile(r"\d{1,3}(?:[ ,  ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
DRY_RUN_TEXT = "*[dry run — text not generated]*"
# Confidence of a root-cause hypothesis, from its number of supporting incidents
CONFIDENCE_HIGH_MIN_INCIDENTS = 5
# Public pay-as-you-go GPT-4.1 nano rates (USD per 1M tokens), for the
# estimate printed at the end of a run — see README
USD_PER_M_INPUT_TOKENS = 0.10
USD_PER_M_OUTPUT_TOKENS = 0.40

# {asset_context} is replaced by the "shared_context" of the asset profile
SHARED_RULES = """You write one section of an equipment-level incident report for asset managers and maintenance planners of {asset_context}

Write every generated text field in French; the prompts may remain in English.

# Rules
1. NO CALCULATIONS. Do not add, subtract, average, compare numerically, compute percentages, ratios, durations or date differences. Every number you write must be copied from the input (rounding allowed). If a figure you would like to quote is not in the input, make the point without a number.
2. `facts` are verified, ready-made sentences: when you quote one of their figures, copy it exactly with the meaning given there (e.g. do not swap "most costly incident" and "most costly failure mode", or the ratio of one indicator for another).
3. Refer to incidents ONLY by their local number as written in the input (e.g. "#3"). Cite an incident for a failure mode only if it is listed among that failure mode's incidents.
4. Before linking a weather condition to a failure mode, check that failure mode's weather counts: if the weather is not listed, the failure mode never occurred under it. Never generalise a weather pattern from one failure mode to another. Weather co-occurrence is a correlation, not a proof.
5. When an entry is flagged "indicatif uniquement" (fewer than 3 incidents), say so briefly in the same sentence (e.g. "indicatif, 2 incidents"). Never write a separate paragraph about data limitations.
6. Priorities and risk levels are given: never change or re-derive them.
7. No mention of datasets, files, score formulas, algorithms, clustering or AI. Say "failure mode", not "cluster". Failure-mode and weather names are in French: keep them verbatim.
8. Answer only the questions of your section; other specialists write the other sections.

# Output
Return ONLY a JSON object following the provided schema. Text fields may use Markdown emphasis and bullet lists, but no headings and no tables. Be specific and concise."""

SECTION_PROMPT_TEMPLATE = """You are writing the section "{title}" of the incident report of equipment {equipment_id} (site: {site}).

Language of every text field: {language}.

Answer these questions:
{questions}

Statistics and figure data for this section (every value is already computed):
```json
{data}
```"""


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------


def get_azure_client():
    """Azure OpenAI client and chat deployment name, from the ``.env`` file."""
    try:
        # Corporate networks with SSL inspection (see the starter notebook)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    return client, os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]


def call_azure(
    client,
    deployment: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    schema_name: str,
    max_attempts: int = 6,
) -> tuple[dict, dict]:
    """One chat completion with structured JSON output; returns (answer, token usage).

    Uses ``json_schema`` (strict) and falls back to ``json_object`` if the
    deployment refuses it. Waits and retries on rate limits (HTTP 429).
    """
    from openai import BadRequestError, RateLimitError

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formats = [
        {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        {"type": "json_object"},
    ]
    for response_format in formats:
        if response_format["type"] == "json_object":
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "Return the JSON object following this schema:\n"
                    + json.dumps(schema),
                },
            ]
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.with_options(max_retries=0).chat.completions.create(
                    model=deployment,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.2,
                    max_tokens=4000,
                )
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                return json.loads(response.choices[0].message.content), usage
            except RateLimitError as error:
                if attempt == max_attempts:
                    raise
                match = re.search(r"retry after (\d+) seconds", str(error))
                wait = int(match.group(1)) + 1 if match else 30
                print(f"  rate limit reached (429), retrying in {wait} s")
                time.sleep(wait)
            except BadRequestError as error:
                if response_format["type"] != "json_schema":
                    raise
                print(
                    f"  json_schema refused ({error.message[:120]}), falling back to json_object"
                )
                break
    raise RuntimeError("No answer from Azure OpenAI")


# ---------------------------------------------------------------------------
# Output schemas and markdown helpers
# ---------------------------------------------------------------------------


def _text(description: str) -> dict:
    return {"type": "string", "description": description}


def _choice(values: list[str], description: str) -> dict:
    return {"type": "string", "enum": values, "description": description}


def _list(items: dict, description: str) -> dict:
    return {"type": "array", "items": items, "description": description}


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _placeholder(schema: dict):
    """Answer with the shape of *schema*, used in dry runs (no Azure call)."""
    if schema["type"] == "object":
        return {key: _placeholder(value) for key, value in schema["properties"].items()}
    if schema["type"] == "array":
        return []
    return schema["enum"][0] if "enum" in schema else DRY_RUN_TEXT


def _md_table(rows: list[dict], columns: dict[str, str]) -> str:
    """Markdown table of *rows*, with *columns* mapping row keys to headers."""
    lines = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = ("" if row.get(key) is None else str(row[key]) for key in columns)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _numbers_in_text(text: str) -> list[str]:
    """Numbers written in *text*; incident numbers (#k), priorities (P1),
    hypotheses (H2) and site / equipment codes (EST2, OUEST1-INV06) excluded."""
    text = re.sub(r"#\d+|\b[PpHh]\d\b|\b[A-Z]{2,}\d+(?:-[A-Z]{2,}\d+)?\b", " ", text)
    # French decimal comma ("142,8 h") -> "142.8"; "207,762" stays a thousands separator
    text = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", text)
    return [re.sub(r"[ ,  ]", "", n) for n in NUMBER_PATTERN.findall(text)]


def _numbers_in_data(data) -> set[float]:
    """Every number of a JSON-like value, including numbers written inside its strings."""
    if isinstance(data, dict):
        return set().union(
            *(_numbers_in_data(v) for v in (*data.keys(), *data.values()))
        )
    if isinstance(data, list):
        return set().union(*(_numbers_in_data(v) for v in data)) if data else set()
    if isinstance(data, bool) or data is None:
        return set()
    if isinstance(data, int | float):
        return {abs(float(data))}
    return {float(n) for n in _numbers_in_text(str(data))}


def _strings(answer) -> list[str]:
    if isinstance(answer, dict):
        return [s for v in answer.values() for s in _strings(v)]
    if isinstance(answer, list):
        return [s for v in answer for s in _strings(v)]
    return [answer] if isinstance(answer, str) else []


def _invented_numbers(answer: dict, data: dict) -> list[str]:
    """Numbers of *answer* that are not a rounding of any number of *data*."""
    known = _numbers_in_data(data)
    invented = []
    for text in _strings(answer):
        for number in _numbers_in_text(text):
            decimals = len(number.split(".")[1]) if "." in number else 0
            tolerance = 0.5 * 10**-decimals + 1e-9
            if not any(abs(float(number) - value) <= tolerance for value in known):
                invented.append(number)
    return sorted(set(invented), key=float)


def _wrong_refs(text: str, allowed: list[str]) -> list[str]:
    """Incident numbers cited in *text* that are not in *allowed*."""
    cited = {f"#{n}" for n in INCIDENT_REF_PATTERN.findall(text)}
    return sorted(cited - set(allowed), key=lambda ref: int(ref[1:]))


def _figure(name: str, alt: str) -> str:
    """Image link relative to sections/ (rewritten for report.md on assembly)."""
    return f"![{alt}](../figures/png/{name}.png)"


# ---------------------------------------------------------------------------
# Section context — what every section agent shares
# ---------------------------------------------------------------------------


@dataclass
class SectionContext:
    """Inputs of one equipment report and the answers collected so far.

    ``ask`` runs one section agent. Answers are cached in
    ``sections/<section>.llm.json`` together with a hash of the prompt: an
    agent is called again only if its prompt changed or if the section is in
    ``forced``.
    """

    equipment_id: str
    stats: dict
    figures: dict[str, dict]
    sections_dir: Path
    language: str = "French"
    client: object | None = None
    deployment: str | None = None
    dry_run: bool = False
    forced: set[str] = field(default_factory=set)
    answers: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )

    @property
    def site(self) -> str:
        return self.stats["equipment"]["site"]

    @property
    def profile(self) -> dict[str, str]:
        """Asset profile (solar or wind) of the equipment, see ``ASSET_PROFILES``."""
        return ASSET_PROFILES[self.stats["equipment"].get("asset", "solar")]

    @property
    def facts(self) -> dict[str, list[str]]:
        return self.stats["facts"]

    @property
    def failure_modes(self) -> list[dict]:
        return self.stats["failure_modes"]

    @property
    def refs(self) -> list[str]:
        return [incident["ref"] for incident in self.stats["register"]]

    def ask(
        self,
        section: str,
        title: str,
        role: str,
        questions: list[str],
        data: dict,
        schema: dict,
        check: Callable[[dict], list[str]] | None = None,
    ) -> dict:
        """Run the agent of *section* and return its JSON answer.

        The answer is checked for numbers absent from *data* and with
        *check* (section-specific problems, e.g. a P1 failure mode without
        action). If there is a problem, the agent gets one correction round;
        problems left afterwards are recorded in ``warnings``.
        """

        def problems_of(answer: dict) -> list[str]:
            problems = list(check(answer)) if check and not self.dry_run else []
            unknown = _wrong_refs(" ".join(_strings(answer)), self.refs)
            if unknown:
                problems.append(
                    f"incident numbers that do not exist on this equipment: {', '.join(unknown)}"
                )
            invented = _invented_numbers(answer, data)
            if invented:
                problems.append(
                    f"numbers not found in the input: {', '.join(invented)} — copy every "
                    "number exactly from the input, or make the point without a number"
                )
            return problems

        shared_rules = SHARED_RULES.replace(
            "{asset_context}", self.profile["shared_context"]
        )
        system_prompt = f"You are {role}\n\n{shared_rules}"
        user_prompt = SECTION_PROMPT_TEMPLATE.format(
            title=title,
            equipment_id=self.equipment_id,
            site=self.site,
            language=self.language,
            questions="\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)),
            data=json.dumps(data, ensure_ascii=False, indent=1),
        )
        (self.sections_dir / f"{section}.prompt.md").write_text(
            f"# System\n\n{system_prompt}\n\n# User\n\n{user_prompt}\n",
            encoding="utf-8",
        )
        prompt_hash = hashlib.sha256(
            (system_prompt + user_prompt + json.dumps(schema)).encode("utf-8")
        ).hexdigest()

        cache_path = self.sections_dir / f"{section}.llm.json"
        cached = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.exists()
            else None
        )
        if (
            cached
            and cached["prompt_sha256"] == prompt_hash
            and section not in self.forced
        ):
            answer = cached["answer"]
        elif self.dry_run:
            answer = _placeholder(schema)
        else:
            print(f"  {section}: asking Azure OpenAI")
            # client and deployment are always set outside dry runs
            assert self.deployment is not None
            answer, usage = call_azure(
                self.client,
                self.deployment,
                system_prompt,
                user_prompt,
                schema,
                section,
            )
            problems = problems_of(answer)
            if problems:
                # One correction round, telling the agent what was wrong
                print(f"  {section}: {'; '.join(problems)} — asking again")
                correction = (
                    f"{user_prompt}\n\nA previous answer to this request had these problems, "
                    "fix them:\n" + "\n".join(f"- {p}" for p in problems)
                )
                answer, retry_usage = call_azure(
                    self.client,
                    self.deployment,
                    system_prompt,
                    correction,
                    schema,
                    section,
                )
                usage = {key: usage[key] + retry_usage[key] for key in usage}
            for key in self.usage:
                self.usage[key] += usage[key]
            cache_path.write_text(
                json.dumps(
                    {"prompt_sha256": prompt_hash, "usage": usage, "answer": answer},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.warnings.extend(f"{section}: {problem}" for problem in problems_of(answer))
        self.answers[section] = answer
        return answer


# ---------------------------------------------------------------------------
# One function per report section
# ---------------------------------------------------------------------------


def section_key_figures(ctx: SectionContext) -> str:
    """1. Key figures — tables computed in Python + a benchmarking agent's reading."""
    equipment = ctx.stats["equipment"]
    comparison = ctx.stats["comparison_with_site_median"]
    key = ctx.stats["key_figures"]
    answer = ctx.ask(
        "key_figures",
        "1. Chiffres clés",
        role=f"un analyste de performance des actifs qui compare les {ctx.profile['equipment_plural']} aux autres {ctx.profile['equipment_plural']} de leur site.",
        questions=[
            (
                "Sur quels indicateurs cet équipement se distingue-t-il de l'équipement médian de son "
                "site, et sur lesquels est-il dans la moyenne ? Cite les ratios tels qu'écrits dans `facts`."
            ),
            (
                "Pourquoi cet équipement a-t-il été sélectionné (`why_selected`) et que signifie sa "
                "priorité pour le lecteur ?"
            ),
            "Dans quelle mesure le coût est-il concentré sur les incidents sévères ?",
        ],
        data={
            "equipment": equipment,
            "comparison_with_site_median_equipment": comparison,
            "key_figures_moderate_and_severe": key,
            "facts": ctx.facts["benchmark"] + ctx.facts["cost"][:1],
        },
        schema=_object({"reading": _text("2-4 phrases répondant aux questions.")}),
    )

    def row(label: str, block: dict, unit: str = "", money: bool = False) -> dict:
        fmt = format_eur if money else str
        return {
            "label": label,
            "this": f"{fmt(block['this_equipment'])}{unit}",
            "median": f"{fmt(block['site_median_equipment'])}{unit}",
            "ratio": f"×{block['ratio_to_site_median']}",
        }

    benchmark = _md_table(
        [
            row("Incidents (toutes sévérités)", comparison["incidents_all_severities"]),
            row("Incidents sévères", comparison["severe_incidents"]),
            row(
                "Coût total estimé (toutes sévérités)",
                comparison["cost_all_severities_eur"],
                " EUR",
                True,
            ),
            row(
                "Temps d'arrêt total (toutes sévérités)",
                comparison["downtime_all_severities_h"],
                " h",
            ),
        ],
        {
            "label": "Indicateur",
            "this": "Cet équipement",
            "median": "Équipement médian du site",
            "ratio": "Ratio",
        },
    )
    top_mode = ctx.failure_modes[0]
    top_weather = ctx.figures["downtime_weather"]["rows"][0]
    period = ctx.stats["period"]
    scope = _md_table(
        [
            {
                "label": "Incidents analysés",
                "value": f"{key['n_incidents']} ({key['n_moderate']} modérés · {key['n_severe']} sévères)",
            },
            {
                "label": "Période couverte",
                "value": f"{period['first_record']} → {period['last_record']}",
            },
            {
                "label": "Coût total estimé",
                "value": f"{format_eur(key['cost_total_eur'])} EUR "
                f"({key['severe_share_of_cost_pct']} % provenant des incidents sévères)",
            },
            {
                "label": "Coût moyen par incident",
                "value": f"{format_eur(key['cost_mean_eur'])} EUR",
            },
            {"label": "Temps d'arrêt total", "value": f"{key['downtime_total_h']} h"},
            {
                "label": "Intervalle moyen entre incidents",
                "value": f"{key['mean_days_between_incidents']} jours",
            },
            {
                "label": "Mode de défaillance le plus coûteux",
                "value": f"{top_mode['failure_mode']} ({format_eur(top_mode['cost_total_eur'])} EUR)",
            },
            {
                "label": "Météo avec le plus de temps d'arrêt",
                "value": f"{top_weather['weather_condition']} ({top_weather['downtime_total_h']} h)",
            },
        ],
        {
            "label": "Incidents modérés et sévères (périmètre de ce rapport)",
            "value": "Valeur",
        },
    )
    return "\n\n".join(["## 1. Chiffres clés", benchmark, scope, answer["reading"]])


def section_methodology(ctx: SectionContext) -> str:
    """2. Comment ce rapport a été construit — texte fixe, sans agent.

    Les règles décrites ici (sélection, priorités, confiance) doivent correspondre
    exactement au code, cette section n'est donc pas réécrite par un modèle.
    """
    equipment = ctx.stats["equipment"]
    period = ctx.stats["period"]
    priority = equipment["equipment_priority"]
    outcome = (
        f"**{priority}**"
        if priority in ("P1", "P2")
        else "non signalé comme critique (inclus sur demande)"
    )
    paragraphs = [
        "## 2. Comment ce rapport a été construit",
        (
            f"**Périmètre.** Ce rapport couvre les incidents de maintenance enregistrés sur "
            f"{equipment['equipment_id']} entre le {period['first_record']} et le {period['last_record']}. "
            "Chaque rapport d'incident a été lu et assigné à un *mode de défaillance* en fonction de sa "
            "description. L'analyse détaillée porte sur les incidents modérés et sévères ; les incidents "
            "mineurs n'apparaissent que dans la comparaison avec le site. Les incidents sont numérotés "
            "#1, #2, … dans l'ordre chronologique pour cet équipement (voir l'annexe)."
            + (
                " Les incidents dont le rapport n'a pas pu être assigné à un mode de défaillance sont "
                "marqués *non classés* : ils comptent dans tous les totaux mais dans aucun mode de "
                "défaillance."
                if ctx.stats["unclassified_incidents"]["n_incidents"]
                else ""
            )
        ),
        (
            "**Pourquoi cet équipement.** Chaque équipement reçoit un score de criticité de 0 à "
            "100 qui combine quatre indicateurs : nombre d'incidents, nombre d'incidents sévères "
            "(comptés double), coût total de réparation et temps d'arrêt total. Le score est calculé "
            "deux fois : par rapport aux autres équipements du même site et par rapport à tous les "
            f"équipements du {ctx.profile['fleet']}. 50 correspond à un équipement typique, 75 ou plus à un "
            "équipement clairement au-dessus des autres. Un équipement atteignant 75 ou plus sur les "
            "deux comparaisons est prioritaire **P1**, sur l'une d'elles prioritaire **P2**. Ici : "
            f"{equipment['why_selected']} → {outcome}."
        ),
        "**Priorité des recommandations.** Chaque recommandation adresse un mode de défaillance et "
        "hérite de sa priorité :\n"
        + "\n".join(f"- **{p}** — {rule}" for p, rule in PRIORITY_RULES.items()),
        (
            "**Confiance des causes racines.** La confiance d'une hypothèse de cause racine reflète le "
            f"nombre d'incidents qui la soutiennent : élevée à partir de {CONFIDENCE_HIGH_MIN_INCIDENTS} "
            f"incidents, moyenne à partir de {SAMPLE_THRESHOLD}, faible en dessous."
        ),
        (
            "**Analyse.** Toutes les figures, tableaux et graphiques sont calculés directement à partir "
            f"des enregistrements d'incidents (une *séquence rapprochée* est une série d'incidents à moins "
            f"de {SEQUENCE_DAYS} jours d'intervalle ; une *récurrence* est le même mode de défaillance "
            f"qui revient à moins de {RECURRENCE_DAYS} jours). Le commentaire de chaque section a été "
            "rédigé avec l'aide d'un assistant IA spécialisé dans ce sujet, à partir des seules figures "
            "de la section."
        ),
        f"*Les patterns basés sur moins de {SAMPLE_THRESHOLD} incidents sont signalés comme indicatifs dans le texte.*",
    ]
    return "\n\n".join(paragraphs)


def section_cost_timeline(ctx: SectionContext) -> str:
    """3. Chronologie des coûts — tendance, incidents dominants, séquences et saisonnalité."""
    stats = ctx.stats
    answer = ctx.ask(
        "cost_timeline",
        "3. Chronologie des coûts",
        role=f"un ingénieur fiabilité qui analyse l'historique des coûts et la récurrence des incidents sur des {ctx.profile['equipment_plural']}.",
        questions=[
            (
                "Comment le nombre d'incidents et leur coût ont-ils évolué année par année ? Commente "
                "chaque année, y compris le changement de la dernière année, en utilisant les variations "
                "annuelles fournies."
            ),
            "Quels incidents dominent le coût (`most_costly_incidents`) et quels sont leurs modes de défaillance ?",
            (
                "Y a-t-il des séquences rapprochées d'incidents ou des récurrences rapides du même mode "
                "de défaillance ? Nomme les incidents et modes de défaillance impliqués."
            ),
            "Y a-t-il un pattern saisonnier ? Signale les saisons avec peu d'incidents comme indicatives.",
        ],
        data={
            "cost_timeline_figure": ctx.figures["cost_timeline"],
            "yearly_trend": stats["yearly_trend"],
            "seasonality": stats["seasonality"],
            "most_costly_incidents": stats["most_costly_incidents"],
            f"close_sequences_within_{SEQUENCE_DAYS}_days": stats[
                f"close_sequences_within_{SEQUENCE_DAYS}_days"
            ],
            "facts": ctx.facts["trend"] + ctx.facts["cost"][1:],
        },
        schema=_object(
            {
                "trend": _text("Réponse à la question 1 (2-4 phrases)."),
                "dominant_incidents": _text("Réponse à la question 2 (1-3 phrases)."),
                "sequences_and_recurrences": _text(
                    "Réponse à la question 3 (1-3 phrases)."
                ),
                "seasonality": _text("Réponse à la question 4 (1-2 phrases)."),
            }
        ),
    )

    years = [
        {
            **year,
            "cost": format_eur(year["cost_eur"]),
            "change": ""
            if year["cost_change_vs_previous_year_pct"] is None
            else f"{year['cost_change_vs_previous_year_pct']:+} %",
        }
        for year in stats["yearly_trend"]["by_year"]
    ]
    yearly_table = _md_table(
        years,
        {
            "year": "Année",
            "n_incidents": "Incidents",
            "n_severe": "Sévères",
            "cost": "Coût (EUR)",
            "change": "Évolution vs année précédente",
            "downtime_h": "Temps d'arrêt (h)",
        },
    )
    return "\n\n".join(
        [
            "## 3. Chronologie des coûts",
            _figure("cost_timeline", "Chronologie des coûts"),
            (
                "*Coût estimé de chaque incident modéré (creux) et sévère (plein). Couleur = famille de "
                "modes de défaillance, forme du marqueur = mode de défaillance, croix grise = non classé ; "
                "les trois incidents les plus coûteux sont étiquetés.*"
            ),
            yearly_table,
            f"**Tendance.** {answer['trend']}",
            f"**Incidents dominants.** {answer['dominant_incidents']}",
            f"**Séquences et récurrences.** {answer['sequences_and_recurrences']}",
            f"**Saisonnalité.** {answer['seasonality']}",
        ]
    )


def section_weather(ctx: SectionContext) -> str:
    """4. Exposition météorologique — temps d'arrêt vs météo et modes de défaillance vs météo."""
    downtime = ctx.figures["downtime_weather"]
    answer = ctx.ask(
        "weather",
        "4. Exposition météorologique",
        role=f"un ingénieur spécialisé dans l'exposition météorologique et environnementale des {ctx.profile['plants']}.",
        questions=[
            (
                "Sous quelles conditions météorologiques cet équipement perd-il le plus d'heures ? Ces "
                "conditions sont-elles sur- ou sous-représentées par rapport au reste du site "
                "(`representation`) ?"
            ),
            (
                "Quels modes de défaillance se produisent sous quelles conditions météorologiques ? "
                "Lis chaque ligne de la matrice : associe une condition météorologique à un mode de "
                "défaillance uniquement si elle apparaît dans cette ligne. Quelles implications pour "
                "l'exploitation (ex : inspections après un événement météo donné) ?"
            ),
        ],
        data={
            "downtime_vs_weather_figure": downtime,
            "failure_modes_vs_weather_figure": ctx.figures["cluster_weather"],
            "facts": ctx.facts["weather"],
        },
        schema=_object(
            {
                "downtime_vs_weather": _text("Réponse à la question 1 (2-4 phrases)."),
                "failure_modes_vs_weather": _text(
                    "Réponse à la question 2 (2-4 phrases)."
                ),
            }
        ),
    )

    table = _md_table(
        downtime["rows"],
        {
            "weather_condition": "Météo",
            "n_incidents": "Incidents",
            "n_severe": "Sévères",
            "downtime_total_h": "Temps d'arrêt (h)",
            "downtime_median_h": "Médiane (h)",
            "share_of_equipment_downtime_pct": "% du temps d'arrêt",
            "representation": "Vs site",
        },
    )
    return "\n\n".join(
        [
            "## 4. Exposition météorologique",
            "### 4.1 Temps d'arrêt vs condition météorologique",
            _figure("downtime_weather", "Temps d'arrêt vs météo"),
            "*Temps d'arrêt de chaque incident par condition météorologique ; le trait noir est la médiane.*",
            table,
            answer["downtime_vs_weather"],
            "### 4.2 Modes de défaillance vs condition météorologique",
            _figure("cluster_weather", "Modes de défaillance vs météo"),
            "*Nombre d'incidents par mode de défaillance et condition météorologique.*",
            answer["failure_modes_vs_weather"],
        ]
    )


def section_failure_modes(ctx: SectionContext) -> str:
    """5. Modes de défaillance — tableau récapitulatif + un paragraphe par mode de défaillance."""
    modes = ctx.failure_modes
    names = [mode["failure_mode"] for mode in modes]
    refs_by_mode = {
        mode["failure_mode"]: [i["ref"] for i in mode["incidents"]] for mode in modes
    }

    def check(answer: dict) -> list[str]:
        problems = []
        missing = [
            n
            for n in names
            if n not in {e["failure_mode"] for e in answer["failure_modes"]}
        ]
        if missing:
            problems.append(
                f"pas d'entrée pour le(s) mode(s) de défaillance : {'; '.join(missing)}"
            )
        for entry in answer["failure_modes"]:
            text = entry["mechanism"] + " " + entry["evidence_and_impact"]
            wrong = _wrong_refs(text, refs_by_mode[entry["failure_mode"]])
            if wrong:
                problems.append(
                    f"{', '.join(wrong)} cités pour {entry['failure_mode']}, auxquels ils n'appartiennent pas"
                )
        return problems

    answer = ctx.ask(
        "failure_modes",
        "5. Modes de défaillance",
        role=f"un expert en analyse de {ctx.profile['failure_expertise']}.",
        questions=[
            "Pour chaque mode de défaillance, dans cet ordre : "
            + "; ".join(f'"{n}"' for n in names)
            + ", donne "
            f"(a) son mécanisme physique probable sur {ctx.profile['mechanism_scope']}, et "
            "(b) les preuves sur cet équipement — incidents (uniquement ceux de ce mode de "
            "défaillance), dates, sévérité, coût, récurrences — et son impact.",
        ],
        data={"failure_modes": modes, "facts": ctx.facts["priorities"]},
        schema=_object(
            {
                "failure_modes": _list(
                    _object(
                        {
                            "failure_mode": _choice(
                                names, "Nom du mode de défaillance, verbatim."
                            ),
                            "mechanism": _text(
                                "Mécanisme physique probable (1-2 phrases)."
                            ),
                            "evidence_and_impact": _text(
                                "Preuves sur cet équipement et impact (1-3 phrases)."
                            ),
                        }
                    ),
                    "Exactement une entrée par mode de défaillance.",
                ),
            }
        ),
        check=check,
    )
    by_name = {entry["failure_mode"]: entry for entry in answer["failure_modes"]}

    table = _md_table(
        [
            {
                **mode,
                "cost": format_eur(mode["cost_total_eur"]),
                "refs": ", ".join(i["ref"] for i in mode["incidents"]),
            }
            for mode in modes
        ],
        {
            "failure_mode": "Mode de défaillance",
            "recommendation_priority": "Priorité",
            "n_incidents": "Incidents",
            "n_severe": "Sévères",
            "cost": "Coût (EUR)",
            "cost_share_pct": "% du coût",
            f"recurrences_within_{RECURRENCE_DAYS}_days": f"Récurrences < {RECURRENCE_DAYS} j",
            "refs": "Incidents #",
        },
    )
    parts = ["## 5. Modes de défaillance", table]
    for mode in modes:
        name = mode["failure_mode"]
        parts.append(f"### {name} · {mode['recommendation_priority']}")
        entry = by_name.get(name)
        if entry is None:
            parts.append("*Aucun commentaire généré.*")
            continue
        parts.append(f"{entry['mechanism']} {entry['evidence_and_impact']}")
    return "\n\n".join(parts)


def section_root_causes(ctx: SectionContext) -> str:
    """6. Hypothèses de causes racines — confiance calculée à partir des incidents à l'appui."""
    modes = ctx.failure_modes
    names = [mode["failure_mode"] for mode in modes]
    refs_by_mode = {
        mode["failure_mode"]: [i["ref"] for i in mode["incidents"]] for mode in modes
    }

    def allowed_refs(hypothesis: dict) -> set[str]:
        return {
            ref for name in hypothesis["failure_modes"] for ref in refs_by_mode[name]
        }

    def check(answer: dict) -> list[str]:
        problems = []
        for i, hypothesis in enumerate(answer["hypotheses"], 1):
            text = " ".join(
                [*hypothesis["supporting_incidents"], hypothesis["reasoning"]]
            )
            wrong = _wrong_refs(text, sorted(allowed_refs(hypothesis)))
            if wrong:
                problems.append(
                    f"l'hypothèse {i} cite {', '.join(wrong)}, qui n'appartiennent pas à ses modes de défaillance"
                )
        return problems

    answer = ctx.ask(
        "root_causes",
        "6. Hypothèses de causes racines",
        role=f"un ingénieur en analyse de causes racines (RCA) pour les {ctx.profile['plants']}.",
        questions=[
            (
                "Propose 3 à 5 hypothèses de causes racines expliquant les incidents de cet équipement. "
                "Une hypothèse peut couvrir un mode de défaillance ou une cause commune derrière plusieurs "
                "(ex : stress thermique, mauvaises connexions, exposition à la foudre)."
            ),
            (
                "Pour chaque hypothèse, liste les modes de défaillance qu'elle explique et les incidents "
                "qui la soutiennent — uniquement les incidents appartenant à ces modes de défaillance."
            ),
            "Explique le raisonnement en 1-3 phrases (mécanisme, patterns météo ou temporels).",
        ],
        data={
            "failure_modes": [
                {
                    key: mode[key]
                    for key in (
                        "failure_mode",
                        "n_incidents",
                        "n_severe",
                        "sample",
                        "weather_counts",
                        "dominant_weather",
                        "median_days_between_recurrences",
                        f"recurrences_within_{RECURRENCE_DAYS}_days",
                        "incidents",
                    )
                }
                for mode in modes
            ],
            f"close_sequences_within_{SEQUENCE_DAYS}_days": ctx.stats[
                f"close_sequences_within_{SEQUENCE_DAYS}_days"
            ],
            "seasonality": ctx.stats["seasonality"],
            "facts": ctx.facts["weather"],
        },
        schema=_object(
            {
                "hypotheses": _list(
                    _object(
                        {
                            "hypothesis": _text(
                                "L'hypothèse de cause racine, en une phrase."
                            ),
                            "failure_modes": _list(
                                _choice(names, "Nom du mode de défaillance."),
                                "Modes de défaillance expliqués.",
                            ),
                            "supporting_incidents": _list(
                                _choice(ctx.refs, "Numéro d'incident."),
                                "Incidents soutenant l'hypothèse.",
                            ),
                            "reasoning": _text("Raisonnement (1-3 phrases)."),
                        }
                    ),
                    "3 à 5 hypothèses.",
                ),
            }
        ),
        check=check,
    )

    items = []
    for i, hypothesis in enumerate(answer["hypotheses"], 1):
        # Les incidents hors des modes de défaillance de l'hypothèse ne comptent jamais comme soutien
        support = sorted(
            set(hypothesis["supporting_incidents"]) & allowed_refs(hypothesis),
            key=lambda ref: int(ref[1:]),
        )
        if len(support) >= CONFIDENCE_HIGH_MIN_INCIDENTS:
            confidence = "élevée"
        elif len(support) >= SAMPLE_THRESHOLD:
            confidence = "moyenne"
        else:
            confidence = "faible"
        items.append(
            f"- **H{i} — {hypothesis['hypothesis']}** — *confiance : {confidence} "
            f"({len(support)} incident{'s' if len(support) != 1 else ''} à l'appui"
            f"{': ' + ', '.join(support) if support else ''})*  \n"
            f"   Modes de défaillance : {', '.join(hypothesis['failure_modes'])}.  \n"
            f"   {hypothesis['reasoning']}"
        )
    return "\n\n".join(
        ["## 6. Hypothèses de causes racines", "\n".join(items) or DRY_RUN_TEXT]
    )


def section_recommendations(ctx: SectionContext) -> str:
    """7. Recommandations — chaque action hérite de la priorité de son mode de défaillance."""
    modes = ctx.failure_modes
    names = [mode["failure_mode"] for mode in modes]
    priority_of = {
        mode["failure_mode"]: mode["recommendation_priority"] for mode in modes
    }
    p1 = [name for name in names if priority_of[name] == "P1"]
    coverage = (
        "Adresse chaque mode de défaillance de priorité P1 ("
        + "; ".join(f'"{n}"' for n in p1)
        + ") avec au moins une action."
        if p1
        else "Commence par les modes de défaillance de priorité P2."
    )
    hypotheses = ctx.answers.get("root_causes", {}).get("hypotheses", [])
    # H1, H2… as numbered in the root-cause section
    hypothesis_modes = {
        f"H{i}": set(h["failure_modes"]) for i, h in enumerate(hypotheses, 1)
    }

    def check(answer: dict) -> list[str]:
        problems = []
        missing = sorted(
            set(p1) - {a["target_failure_mode"] for a in answer["actions"]}
        )
        if missing:
            problems.append(
                f"pas d'action pour le(s) mode(s) de défaillance P1 : {'; '.join(missing)}"
            )
        for action in answer["actions"]:
            if action["target_failure_mode"] not in hypothesis_modes.get(
                action["root_cause"], set()
            ):
                problems.append(
                    f"l'action « {action['action'][:60]} » cible "
                    f"{action['target_failure_mode']}, que l'hypothèse "
                    f"{action['root_cause']} n'explique pas : choisis une hypothèse "
                    "qui couvre ce mode de défaillance"
                )
        return problems

    answer = ctx.ask(
        "recommendations",
        "7. Recommandations",
        role=f"un planificateur de maintenance O&M pour des {ctx.profile['plants']} de taille industrielle.",
        questions=[
            (
                "Propose 4 à 7 actions concrètes pour l'équipe de maintenance. Chaque action "
                "découle d'une hypothèse de cause racine fournie (H1, H2…) et adresse un mode "
                "de défaillance que cette hypothèse explique."
            ),
            coverage,
            (
                "Chaque action doit viser le mécanisme décrit par son hypothèse sur cet "
                "équipement : précise quoi vérifier ou modifier, où, et pourquoi. Évite les "
                "actions génériques qui s'appliqueraient à n'importe quel équipement, et ne "
                "propose pas la même action pour des modes de défaillance différents."
            ),
            (
                "Pour chaque action, indique l'impact attendu qualitativement, en ne citant "
                "que des chiffres de l'entrée."
            ),
        ],
        data={
            "failure_modes": [
                {
                    key: mode[key]
                    for key in (
                        "failure_mode",
                        "recommendation_priority",
                        "priority_reason",
                        "n_incidents",
                        "n_severe",
                        "cost_share_pct",
                        f"recurrences_within_{RECURRENCE_DAYS}_days",
                        "dominant_weather",
                    )
                }
                for mode in modes
            ],
            "root_cause_hypotheses": [
                {
                    "id": f"H{i}",
                    "hypothesis": h["hypothesis"],
                    "failure_modes": h["failure_modes"],
                    "supporting_incidents": h["supporting_incidents"],
                    "reasoning": h["reasoning"],
                }
                for i, h in enumerate(hypotheses, 1)
            ],
        },
        schema=_object(
            {
                "actions": _list(
                    _object(
                        {
                            "action": _text(
                                "Action de maintenance ou technique concrète."
                            ),
                            "type": _choice(
                                [
                                    "préventive",
                                    "corrective",
                                    "surveillance",
                                    "modification",
                                    "pièces de rechange",
                                ],
                                "Type d'action.",
                            ),
                            "root_cause": _choice(
                                list(hypothesis_modes) or ["H1"],
                                "Hypothèse de cause racine (H1, H2…) dont découle l'action.",
                            ),
                            "target_failure_mode": _choice(
                                names,
                                "Mode de défaillance adressé (définit la priorité).",
                            ),
                            "expected_impact": _text("Impact attendu (1 phrase)."),
                        }
                    ),
                    "4 à 7 actions.",
                ),
            }
        ),
        check=check,
    )
    actions = sorted(
        (
            {**a, "priority": priority_of[a["target_failure_mode"]]}
            for a in answer["actions"]
        ),
        key=lambda a: a["priority"],
    )
    ctx.answers["prioritised_actions"] = {"actions": actions}

    table = _md_table(
        actions,
        {
            "priority": "Priorité",
            "action": "Action",
            "type": "Type",
            "target_failure_mode": "Mode de défaillance adressé",
            "root_cause": "Cause visée",
            "expected_impact": "Impact attendu",
        },
    )
    return "\n\n".join(["## 7. Recommandations", table if actions else DRY_RUN_TEXT])


def _first_actions(ctx: SectionContext) -> list[dict]:
    """Actions to do first: the P1 ones, else the three highest-priority ones."""
    actions = ctx.answers.get("prioritised_actions", {}).get("actions", [])
    first = [a for a in actions if a["priority"] == "P1"] or actions[:3]
    return [
        {
            "priority": a["priority"],
            "action": a["action"],
            "failure_mode": a["target_failure_mode"],
        }
        for a in first
    ]


def section_conclusion(ctx: SectionContext) -> str:
    """8. Conclusion — rédigée après les recommandations, qu'elle référence."""
    equipment = ctx.stats["equipment"]
    answer = ctx.ask(
        "conclusion",
        "8. Conclusion",
        role="le responsable fiabilité qui signe les rapports d'équipement.",
        questions=[
            (
                "En 2-3 phrases : le problème principal de cet équipement, quoi faire en premier (les "
                "actions indiquées) et le bénéfice attendu."
            ),
        ],
        data={
            "equipment": {
                k: equipment[k]
                for k in ("equipment_id", "equipment_priority", "risk_level")
            },
            "facts": ctx.facts["cost"] + ctx.facts["benchmark"][:2],
            "failure_mode_priorities": ctx.facts["priorities"],
            "first_actions": _first_actions(ctx),
        },
        schema=_object({"conclusion": _text("Conclusion en 2-3 phrases.")}),
    )
    return "\n\n".join(["## 8. Conclusion", answer["conclusion"]])


def section_summary(ctx: SectionContext) -> str:
    """En-tête du rapport — titre et résumé exécutif, rédigé une fois toutes les autres sections connues."""
    equipment = ctx.stats["equipment"]
    answer = ctx.ask(
        "summary",
        "Titre et résumé exécutif",
        role=f"un rédacteur de résumés exécutifs pour les gestionnaires d'actifs de {ctx.profile['plants']}.",
        questions=[
            "Donne au rapport un titre court nommant l'identifiant de l'équipement et ses principaux modes de défaillance.",
            (
                "Rédige un résumé exécutif de 3-4 phrases : ce qui tombe en panne, combien cela coûte, "
                "comment l'équipement se compare à son site et quoi faire en premier."
            ),
        ],
        data={
            "equipment": equipment,
            "facts": ctx.facts["benchmark"][:2]
            + ctx.facts["cost"]
            + ctx.facts["trend"][-1:],
            "failure_mode_priorities": ctx.facts["priorities"],
            "first_actions": _first_actions(ctx),
        },
        schema=_object(
            {
                "title": _text("Titre court du rapport nommant l'équipement."),
                "synopsis": _text("Résumé exécutif de 3-4 phrases."),
            }
        ),
    )
    title = (
        answer["title"]
        if answer["title"] != DRY_RUN_TEXT
        else f"Rapport d'incidents — {ctx.equipment_id}"
    )
    priority = equipment["equipment_priority"]
    why = equipment["why_selected"]
    return "\n\n".join(
        [
            f"# {title}",
            (
                f"**Équipement :** {ctx.equipment_id} · **Site :** {ctx.site} · **Type d'actif :** {ctx.profile['equipment']} · "
                f"**Généré le :** {date.today().isoformat()}"
            ),
            f"> {answer['synopsis']}",
            (
                f"**Priorité équipement : {priority} · Niveau de risque : {equipment['risk_level'].upper()}** — "
                f"{why[0].upper()}{why[1:]}."
            ),
        ]
    )


def section_incident_register(ctx: SectionContext) -> str:
    """Annexe — registre des incidents, sans agent."""
    rows = [{**r, "cost": format_eur(r["cost_eur"])} for r in ctx.stats["register"]]
    table = _md_table(
        rows,
        {
            "ref": "#",
            "date": "Date",
            "severity": "Sévérité",
            "failure_mode": "Mode de défaillance",
            "cost": "Coût (EUR)",
            "downtime_h": "Temps d'arrêt (h)",
            "weather_condition": "Météo",
        },
    )
    return f"### Annexe — Registre des incidents\n\n{table}"


# Generation order: the conclusion and the summary come after the sections
# they summarise. The order in the report is set by the template.
EQUIPMENT_SECTIONS: dict[str, Callable[[SectionContext], str]] = {
    "key_figures": section_key_figures,
    "methodology": section_methodology,
    "cost_timeline": section_cost_timeline,
    "weather": section_weather,
    "failure_modes": section_failure_modes,
    "root_causes": section_root_causes,
    "recommendations": section_recommendations,
    "conclusion": section_conclusion,
    "summary": section_summary,
    "incident_register": section_incident_register,
}


def run_equipment_reporting(
    equipment_ids: list[str],
    site_dir: Path,
    templates_dir: Path,
    language: str = "French",
    dry_run: bool = False,
    force: list[str] | None = None,
    pdf: bool = False,
) -> None:
    """Write the section markdowns and the assembled report of every equipment.

    Reads ``<site_dir>/<equipment_id>/json/`` (written by
    ``plot_factory.run_equipment_plotting``) and writes
    ``sections/<section>.md`` (plus the prompt and the cached answer of each
    agent) and ``report.md`` next to it.

    *force*: ``None`` reuses cached answers whose prompt did not change, an
    empty list regenerates every section, a list of names only those. With
    *dry_run*, no Azure call is made: the prompts are written and the text
    fields of the sections are placeholders.
    """
    template = (templates_dir / EQUIPMENT_TEMPLATE_NAME).read_text(encoding="utf-8")
    if force is None:
        forced: set[str] = set()
    else:
        unknown = sorted(set(force) - set(EQUIPMENT_SECTIONS))
        if unknown:
            raise ValueError(
                f"Unknown section(s) {', '.join(unknown)}; choose from {', '.join(EQUIPMENT_SECTIONS)}"
            )
        forced = set(force) or set(EQUIPMENT_SECTIONS)

    client, deployment = (None, None) if dry_run else get_azure_client()
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    for equipment_id in equipment_ids:
        equipment_dir = site_dir / equipment_id
        json_dir = equipment_dir / "json"
        stats_path = json_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found — run the equipment plotting step first."
            )
        print(f"- {equipment_id}")
        figures = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(json_dir.glob("*.json"))
            if path.stem != "stats"
        }
        sections_dir = equipment_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        ctx = SectionContext(
            equipment_id=equipment_id,
            stats=json.loads(stats_path.read_text(encoding="utf-8")),
            figures=figures,
            sections_dir=sections_dir,
            language=language,
            client=client,
            deployment=deployment,
            dry_run=dry_run,
            forced=forced,
        )

        context: dict[str, str] = {}
        for name, build_section in EQUIPMENT_SECTIONS.items():
            markdown = build_section(ctx)
            (sections_dir / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
            context[f"section_{name}"] = markdown.replace("](../figures/", "](figures/")

        report_path = equipment_dir / "report.md"
        report_path.write_text(fill_template(template, context), encoding="utf-8")
        print(f"  Saved {report_path}")
        (sections_dir / "checks.txt").write_text(
            "\n".join(ctx.warnings) + "\n", encoding="utf-8"
        )
        for warning in ctx.warnings:
            print(f"  check: {warning}")
        for key in total:
            total[key] += ctx.usage[key]

        if pdf:
            from datachallengecapgemini.pdf_export import convert_markdown_to_pdf

            print(f"  Saved {convert_markdown_to_pdf(report_path)}")

    cost = (
        total["prompt_tokens"] * USD_PER_M_INPUT_TOKENS
        + total["completion_tokens"] * USD_PER_M_OUTPUT_TOKENS
    ) / 1e6
    print(
        f"Done. {total['prompt_tokens']} input + {total['completion_tokens']} output tokens "
        f"(estimated {cost:.4f} USD at GPT-4.1 nano rates). Reports are in {site_dir}"
    )


def add_equipment_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Add equipment-report-specific arguments to *parser*."""
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=PROJECT_ROOT / "templates",
        help=f"Directory containing the {EQUIPMENT_TEMPLATE_NAME} template.",
    )
    parser.add_argument(
        "--language",
        default="French",
        help="Language of the generated text (headings and tables stay in English).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write figures, JSON, prompts and a report skeleton without calling Azure OpenAI.",
    )
    parser.add_argument(
        "--force",
        nargs="*",
        metavar="SECTION",
        choices=list(EQUIPMENT_SECTIONS),
        help="Call the agents again even if their prompt did not change: every section, "
        "or only the sections listed.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also convert each report to PDF with pandoc (see README).",
    )
