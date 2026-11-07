"""Plotting step — build plotly figures from the solar and wind incidents CSVs.

``SolarPlotFactory`` and ``WindPlotFactory`` each wrap one clusterised
incidents dataframe and expose one ``plot_*`` method per figure. Every method
returns a ``plotly.graph_objects.Figure`` and a dictionary describing the
figure, built from the dataframe held in the ``df`` attribute.

The pipeline iterates over every ``plot_*`` method (see
``PlotFactory.plot_method_names``) and saves each figure as a PNG in the
``png`` subfolder (requires the ``kaleido`` package) and an interactive HTML
page in the ``html`` subfolder. Each method's description is saved as JSON in
the ``json`` subfolder.

Available as a CLI step through the top-level ``dc`` entry point:

- ``uv run dc`` — plotting (clustering is skipped by default; use ``--all`` to recompute)
"""

import argparse
import inspect
import json
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datachallengecapgemini.equipment_stats import (
    SEVERE,
    UNCLASSIFIED,
    build_equipment_stats,
    classified,
    failure_mode_weather_matrix,
    format_eur,
    number_incidents,
    ordered_weather,
    site_focus,
    weather_table,
)

# Project root: src/datachallengecapgemini/plot_factory.py -> 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Prefix marking the graph-building methods of PlotFactory. The pipeline
# iterates over every method with this prefix, so adding a new ``plot_*``
# method is enough to add a new figure to the pipeline output.
PLOT_METHOD_PREFIX = "plot_"

# Severity levels in increasing order of seriousness (used for axis ordering)
SEVERITY_ORDER: list[str] = ["mineure", "modérée", "sévère"]

# Colour scheme: blue for counts/frequency, red for cost
COLOR_COUNT = "#2a78d6"
COLOR_COST = "#e34948"
COLOR_SITE = "#1baf7a"


class PlotFactory:
    """Builds plotly figures from an incidents dataframe.

    The dataframe is the main attribute (``df``). Use ``PlotFactory.from_csv``
    to load and validate it from a CSV, or pass a dataframe directly to the
    constructor.
    """

    # Columns the plots rely on. ``text``, ``document_format`` and
    # ``document_path`` are intentionally excluded.
    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "site",
        "asset_type",
        "equipment_id",
        "date",
        "severity",
        "downtime_hours",
        "estimated_cost_eur",
        "weather_condition",
        "technician_id",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def from_csv(cls, csv_path: Path) -> "PlotFactory":
        """Load the merged incidents CSV from *csv_path* and validate it.

        Raises
        ------
        ValueError
            If a required column is missing, or if a required column contains
            NaN values.
        """
        df = pd.read_csv(csv_path)

        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing)}. "
                f"Expected columns: {', '.join(cls.REQUIRED_COLUMNS)}"
            )

        nan_columns = [col for col in cls.REQUIRED_COLUMNS if df[col].isna().any()]
        if nan_columns:
            raise ValueError(
                f"{csv_path} has NaN values in required columns: "
                f"{', '.join(nan_columns)}"
            )

        return cls(df)

    @classmethod
    def plot_method_names(cls) -> list[str]:
        """Names of all graph-building methods (``plot_*``), sorted alphabetically."""
        return sorted(
            name
            for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith(PLOT_METHOD_PREFIX)
        )

    # ------------------------------------------------------------------
    # Graph-building methods — each returns a plotly Figure and its description
    # ------------------------------------------------------------------

    def plot_daily_cost_for_equipment(
        self, equipment_id: str | None = None
    ) -> tuple[go.Figure, dict[str, object]]:
        """Daily estimated cost over time for a single equipment.

        Inspired by ``notebooks/view_csv_data.py``: costs are summed per day
        and missing days are filled with 0 over the full date range. When
        *equipment_id* is ``None``, the equipment with the highest total
        estimated cost is used.
        """
        if equipment_id is None:
            equipment_id = (
                self.df.groupby("equipment_id")["estimated_cost_eur"].sum().idxmax()
            )
        elif equipment_id not in set(self.df["equipment_id"]):
            raise ValueError(f"Unknown equipment_id: {equipment_id}")

        df_equipment = self.df[self.df["equipment_id"] == equipment_id].copy()
        df_equipment["date"] = pd.to_datetime(df_equipment["date"])

        df_daily_cost = (
            df_equipment.groupby("date")["estimated_cost_eur"].sum().reset_index()
        )
        full_date_range = pd.date_range(
            start=df_daily_cost["date"].min(),
            end=df_daily_cost["date"].max(),
            freq="D",
        )
        df_daily_cost = (
            df_daily_cost.set_index("date")
            .reindex(full_date_range, fill_value=0)
            .reset_index()
            .rename(columns={"index": "date"})
        )

        fig = px.line(
            df_daily_cost,
            x="date",
            y="estimated_cost_eur",
            title=f"Coût estimé quotidien — Équipement {equipment_id}",
            labels={
                "date": "Date",
                "estimated_cost_eur": "Coût estimé (EUR)",
            },
        )
        fig.update_traces(mode="lines+markers", line_color=COLOR_COST)
        fig.update_layout(showlegend=False)
        description = {
            "title": f"Coût estimé quotidien — Équipement {equipment_id}",
            "chart_type": "courbe avec marqueurs quotidiens",
            "main_features": (
                "Coût estimé quotidien pour un équipement ; les jours sans "
                "incident sont représentés par une valeur nulle."
            ),
            "equipment_id": str(equipment_id),
            "date_range": [
                df_daily_cost["date"].min().date().isoformat(),
                df_daily_cost["date"].max().date().isoformat(),
            ],
            "total_estimated_cost_eur": float(df_equipment["estimated_cost_eur"].sum()),
            "days_shown": len(df_daily_cost),
        }
        return fig, description

    def plot_cost_distribution_per_equipment_by_site(
        self,
    ) -> tuple[go.Figure, dict[str, object]]:
        """Box plot of the total estimated cost per equipment, grouped by site.

        Inspired by ``notebooks/view_csv_data.py``.
        """
        cost_by_site_equipment = (
            self.df.groupby(["site", "equipment_id"])["estimated_cost_eur"]
            .sum()
            .reset_index()
        )
        fig = px.box(
            cost_by_site_equipment,
            x="site",
            y="estimated_cost_eur",
            title="Distribution du coût total par équipement et par site",
            labels={
                "site": "Site",
                "estimated_cost_eur": "Coût total estimé par équipement (EUR)",
            },
            points="all",
            color="site",
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(
            showgrid=True,
            zeroline=False,
            title_text="Coût total estimé par équipement (EUR)",
        )
        description = {
            "title": "Distribution du coût total par équipement et par site",
            "chart_type": "boîte à moustaches avec les points de chaque équipement",
            "main_features": (
                "Compare la distribution du coût total estimé par équipement "
                "entre les sites ; chaque point représente un équipement."
            ),
            "sites": sorted(cost_by_site_equipment["site"].astype(str).unique()),
            "equipment_count": len(cost_by_site_equipment),
            "cost_metric": "coût total estimé par équipement (EUR)",
        }
        return fig, description

    def plot_daily_cost_top_equipment_at_top_site(
        self, top_n: int = 3
    ) -> tuple[go.Figure, dict[str, object]]:
        """Daily estimated cost over time for the top-N equipment at the highest-cost site.

        Inspired by ``notebooks/view_csv_data.py``: the site with the highest
        total estimated cost is selected, then its top-N equipment by total
        cost. Markers only (no connecting lines) to avoid interpolating
        between days without incidents.
        """
        cost_by_site = self.df.groupby("site")["estimated_cost_eur"].sum()
        selected_site = cost_by_site.idxmax()

        df_site = self.df[self.df["site"] == selected_site].copy()
        df_site["date"] = pd.to_datetime(df_site["date"])

        cost_by_equipment_site = (
            df_site.groupby("equipment_id")["estimated_cost_eur"]
            .sum()
            .reset_index()
            .sort_values("estimated_cost_eur", ascending=False)
        )
        top_equipment = cost_by_equipment_site.head(top_n)["equipment_id"].tolist()

        full_date_range = pd.date_range(
            start=df_site["date"].min(), end=df_site["date"].max(), freq="D"
        )
        frames = []
        for equip_id in top_equipment:
            df_equip_daily = (
                df_site[df_site["equipment_id"] == equip_id]
                .groupby("date")["estimated_cost_eur"]
                .sum()
                .reset_index()
                .set_index("date")
                .reindex(full_date_range, fill_value=0)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            df_equip_daily["equipment_id"] = str(equip_id)
            frames.append(df_equip_daily)
        df_plot = pd.concat(frames, ignore_index=True)

        fig = px.line(
            df_plot,
            x="date",
            y="estimated_cost_eur",
            color="equipment_id",
            title=f"Coût estimé quotidien — Top {top_n} équipements du site {selected_site}",
            labels={
                "date": "Date",
                "estimated_cost_eur": "Coût estimé (EUR)",
                "equipment_id": "Équipement",
            },
        )
        fig.update_traces(mode="markers", marker={"size": 10})
        fig.update_layout(
            legend={"title": {"text": "Équipement"}, "orientation": "h", "y": -0.15},
            yaxis_title="Coût estimé (EUR)",
        )
        description = {
            "title": (
                f"Coût estimé quotidien — {top_n} équipements principaux "
                f"du site {selected_site}"
            ),
            "chart_type": "nuage de points quotidien avec une série par équipement",
            "main_features": (
                "Coûts quotidiens des équipements les plus coûteux du site dont "
                "le coût estimé total est le plus élevé. Les marqueurs ne sont "
                "pas reliés par des lignes."
            ),
            "selected_site": str(selected_site),
            "site_total_estimated_cost_eur": float(cost_by_site.loc[selected_site]),
            "top_equipment": [
                {
                    "equipment_id": str(row["equipment_id"]),
                    "total_estimated_cost_eur": float(row["estimated_cost_eur"]),
                }
                for _, row in cost_by_equipment_site.head(top_n).iterrows()
            ],
        }
        return fig, description

    def plot_incident_count_by_severity(
        self,
    ) -> tuple[go.Figure, dict[str, object]]:
        """Number of incidents (blue bars) and total cost (red line) per severity level.

        Combo chart: left axis = incident count (blue bars),
        right axis = total estimated cost in EUR (red line with markers).
        """
        counts = (
            self.df.groupby("severity")
            .agg(
                incident_count=("severity", "size"),
                total_cost_eur=("estimated_cost_eur", "sum"),
            )
            .reindex(SEVERITY_ORDER)
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=counts["severity"],
                y=counts["incident_count"],
                name="Nombre d'incidents",
                marker_color=COLOR_COUNT,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=counts["severity"],
                y=counts["total_cost_eur"],
                name="Coût total estimé (EUR)",
                mode="lines+markers",
                line={"color": COLOR_COST, "width": 2.5},
                marker={"size": 8},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Nombre d'incidents et coût total par niveau de sévérité",
            xaxis_title="Niveau de sévérité",
            yaxis={"title": "Nombre d'incidents", "side": "left"},
            yaxis2={
                "title": "Coût total estimé (EUR)",
                "side": "right",
                "overlaying": "y",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.2},
        )
        description = {
            "title": "Nombre d'incidents et coût total par niveau de sévérité",
            "chart_type": "diagramme combiné barres + courbe",
            "main_features": (
                "Pour chaque niveau de sévérité, le nombre d'incidents est "
                "représenté par des barres bleues (axe de gauche) et le coût "
                "total estimé par une courbe rouge avec marqueurs (axe de droite)."
            ),
            "incident_counts": [
                {
                    "severity": str(row["severity"]),
                    "incident_count": int(row["incident_count"]),
                    "total_cost_eur": float(row["total_cost_eur"]),
                }
                for _, row in counts.iterrows()
            ],
        }
        return fig, description

    def plot_total_cost_by_weather_condition(
        self,
    ) -> tuple[go.Figure, dict[str, object]]:
        """Total estimated cost (red bars) and incident count (blue line) per weather condition.

        Combo chart: left axis = total estimated cost in EUR (red bars),
        right axis = incident count (blue line with markers).
        """
        totals = (
            self.df.groupby("weather_condition")
            .agg(
                total_cost_eur=("estimated_cost_eur", "sum"),
                incident_count=("weather_condition", "size"),
            )
            .reset_index()
            .sort_values("total_cost_eur", ascending=False)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=totals["weather_condition"],
                y=totals["incident_count"],
                name="Nombre d'incidents",
                marker_color=COLOR_COUNT,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=totals["weather_condition"],
                y=totals["total_cost_eur"],
                name="Coût total estimé (EUR)",
                mode="lines+markers",
                line={"color": COLOR_COST, "width": 2.5},
                marker={"size": 8},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Coût total estimé et nombre d'incidents par condition météorologique",
            xaxis_title="Condition météorologique",
            yaxis={"title": "Coût total estimé (EUR)", "side": "left"},
            yaxis2={
                "title": "Nombre d'incidents",
                "side": "right",
                "overlaying": "y",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.2},
        )
        fig.update_xaxes(tickangle=-45)
        description = {
            "title": "Coût total estimé et nombre d'incidents par condition météorologique",
            "chart_type": "diagramme combiné barres + courbe",
            "main_features": (
                "Pour chaque condition météorologique, le coût total estimé "
                "est représenté par des barres rouges (axe de gauche) et le "
                "nombre d'incidents par une courbe bleue avec marqueurs "
                "(axe de droite)."
            ),
            "cost_totals_eur": [
                {
                    "weather_condition": str(row["weather_condition"]),
                    "total_estimated_cost_eur": float(row["total_cost_eur"]),
                    "incident_count": int(row["incident_count"]),
                }
                for _, row in totals.iterrows()
            ],
        }
        return fig, description

    def plot_downtime_vs_cost(self) -> tuple[go.Figure, dict[str, object]]:
        """Scatter plot of downtime duration vs estimated cost, coloured by severity."""
        fig = px.scatter(
            self.df,
            x="downtime_hours",
            y="estimated_cost_eur",
            color="severity",
            opacity=0.6,
            title="Coût estimé en fonction du temps d'arrêt",
            labels={
                "downtime_hours": "Temps d'arrêt (heures)",
                "estimated_cost_eur": "Coût estimé (EUR)",
                "severity": "Sévérité",
            },
            category_orders={"severity": SEVERITY_ORDER},
        )
        description = {
            "title": "Coût estimé en fonction du temps d'arrêt",
            "chart_type": "nuage de points",
            "main_features": (
                "Chaque point représente un incident ; le temps d'arrêt est en "
                "abscisse, le coût estimé en ordonnée, la couleur indique la "
                "sévérité."
            ),
            "incident_count": len(self.df),
            "incident_counts_by_severity": {
                str(key): int(value)
                for key, value in self.df["severity"].value_counts().items()
            },
        }
        return fig, description

    def plot_monthly_incident_count(
        self,
    ) -> tuple[go.Figure, dict[str, object]]:
        """Monthly incident count (blue bars) and total cost (red line) over time.

        Combo chart: left axis = incident count (blue bars),
        right axis = total estimated cost in EUR (red line with markers).
        """
        df_plot = self.df.copy()
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        df_plot["month"] = df_plot["date"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            df_plot.groupby("month")
            .agg(
                incident_count=("month", "size"),
                total_cost_eur=("estimated_cost_eur", "sum"),
            )
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["incident_count"],
                name="Nombre d'incidents",
                marker_color=COLOR_COUNT,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["total_cost_eur"],
                name="Coût total estimé (EUR)",
                mode="lines+markers",
                line={"color": COLOR_COST, "width": 2},
                marker={"size": 5},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Nombre d'incidents et coût total par mois",
            xaxis_title="Mois",
            yaxis={"title": "Nombre d'incidents", "side": "left"},
            yaxis2={
                "title": "Coût total estimé (EUR)",
                "side": "right",
                "overlaying": "y",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.2},
        )
        fig.update_xaxes(tickformat="%Y-%m")
        description = {
            "title": "Nombre d'incidents et coût total par mois",
            "chart_type": "diagramme combiné barres + courbe mensuel",
            "main_features": (
                "Pour chaque mois, le nombre d'incidents est représenté par "
                "des barres bleues (axe de gauche) et le coût total estimé "
                "par une courbe rouge avec marqueurs (axe de droite)."
            ),
            "monthly_data": [
                {
                    "month": row["month"].date().isoformat(),
                    "incident_count": int(row["incident_count"]),
                    "total_cost_eur": float(row["total_cost_eur"]),
                }
                for _, row in monthly.iterrows()
            ],
        }
        return fig, description

    def plot_total_cost_by_site(self) -> tuple[go.Figure, dict[str, object]]:
        """Total estimated cost per site (red bars) and incident count (blue line).

        Combo chart: left axis = total estimated cost in EUR (red bars),
        right axis = incident count (blue line with markers).
        """
        totals = (
            self.df.groupby("site")
            .agg(
                total_cost_eur=("estimated_cost_eur", "sum"),
                incident_count=("site", "size"),
            )
            .reset_index()
            .sort_values("total_cost_eur", ascending=False)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=totals["site"],
                y=totals["incident_count"],
                name="Nombre d'incidents",
                marker_color=COLOR_COUNT,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=totals["site"],
                y=totals["total_cost_eur"],
                name="Coût total estimé (EUR)",
                mode="lines+markers",
                line={"color": COLOR_COST, "width": 2.5},
                marker={"size": 8},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Coût total estimé et nombre d'incidents par site",
            xaxis_title="Site",
            yaxis={"title": "Coût total estimé (EUR)", "side": "left"},
            yaxis2={
                "title": "Nombre d'incidents",
                "side": "right",
                "overlaying": "y",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.2},
        )
        description = {
            "title": "Coût total estimé et nombre d'incidents par site",
            "chart_type": "diagramme combiné barres + courbe",
            "main_features": (
                "Pour chaque site, le coût total estimé est représenté par "
                "des barres rouges (axe de gauche) et le nombre d'incidents "
                "par une courbe bleue avec marqueurs (axe de droite)."
            ),
            "cost_totals_eur": [
                {
                    "site": str(row["site"]),
                    "total_estimated_cost_eur": float(row["total_cost_eur"]),
                    "incident_count": int(row["incident_count"]),
                }
                for _, row in totals.iterrows()
            ],
        }
        return fig, description

    def plot_cluster_frequency_cost(
        self,
    ) -> tuple[go.Figure, dict[str, object]]:
        """Incident count (blue bars) and total cost (red line) per thematic cluster.

        Combo chart: left axis = incident count (blue bars),
        right axis = total estimated cost in EUR (red line with markers).
        The noise cluster (C_NOISE) is included and labelled.
        """
        cluster_agg = (
            self.df.groupby(["cluster_code", "cluster_name"])
            .agg(
                incident_count=("cluster_code", "size"),
                total_cost_eur=("estimated_cost_eur", "sum"),
            )
            .reset_index()
            .sort_values("incident_count", ascending=False)
        )
        # Build display labels: "C_0 — Name" or "Non classés" for noise
        labels = []
        for _, row in cluster_agg.iterrows():
            if row["cluster_code"] == "C_NOISE":
                labels.append("Non classés (bruit)")
            else:
                labels.append(f"{row['cluster_code']} — {row['cluster_name']}")
        cluster_agg["label"] = labels

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cluster_agg["label"],
                y=cluster_agg["incident_count"],
                name="Nombre d'incidents",
                marker_color=COLOR_COUNT,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cluster_agg["label"],
                y=cluster_agg["total_cost_eur"],
                name="Coût total estimé (EUR)",
                mode="lines+markers",
                line={"color": COLOR_COST, "width": 2.5},
                marker={"size": 8},
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Fréquence et coût total estimé par cluster thématique",
            xaxis_title="Cluster",
            yaxis={"title": "Nombre d'incidents", "side": "left"},
            yaxis2={
                "title": "Coût total estimé (EUR)",
                "side": "right",
                "overlaying": "y",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.25},
        )
        fig.update_xaxes(tickangle=-30)
        description = {
            "title": "Fréquence et coût total estimé par cluster thématique",
            "chart_type": "diagramme combiné barres + courbe",
            "main_features": (
                "Pour chaque cluster thématique, le nombre d'incidents est "
                "représenté par des barres bleues (axe de gauche) et le coût "
                "total estimé par une courbe rouge avec marqueurs (axe de "
                "droite). Le cluster de bruit (incidents non classés) est inclus."
            ),
            "clusters": [
                {
                    "cluster_code": str(row["cluster_code"]),
                    "cluster_name": str(row["cluster_name"]),
                    "incident_count": int(row["incident_count"]),
                    "total_cost_eur": float(row["total_cost_eur"]),
                }
                for _, row in cluster_agg.iterrows()
            ],
        }
        return fig, description


    def plot_yearly_cost(self) -> tuple[go.Figure, dict[str, object]]:
        """Annual total estimated cost across the fleet."""
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        yearly = (
            df.groupby("year")["estimated_cost_eur"]
            .sum()
            .rename("total_cost_eur")
            .reset_index()
        )

        fig = px.bar(
            yearly,
            x="year",
            y="total_cost_eur",
            text=yearly["total_cost_eur"].map(lambda value: f"{value:,.0f} €"),
            title="Évolution annuelle du coût estimé",
            labels={"year": "Année", "total_cost_eur": "Coût estimé (EUR)"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.update_xaxes(type="category")
        fig.update_yaxes(rangemode="tozero", tickformat=",.0f", gridcolor="#e4e3df")

        description = {
            "title": "Évolution annuelle du coût estimé",
            "chart_type": "diagramme à barres annuel",
            "main_features": (
                "Somme des coûts estimés des incidents pour chaque année."
            ),
            "yearly_costs": [
                {"year": int(row["year"]), "total_cost_eur": round(float(row["total_cost_eur"]), 2)}
                for _, row in yearly.iterrows()
            ],
        }
        return fig, description

    def plot_yearly_downtime(self) -> tuple[go.Figure, dict[str, object]]:
        """Annual total downtime across the fleet."""
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        yearly = (
            df.groupby("year")["downtime_hours"]
            .sum()
            .rename("total_downtime_hours")
            .reset_index()
        )

        fig = px.bar(
            yearly,
            x="year",
            y="total_downtime_hours",
            text=yearly["total_downtime_hours"].map(lambda value: f"{value:,.1f} h"),
            title="Évolution annuelle du temps d'arrêt",
            labels={"year": "Année", "total_downtime_hours": "Temps d'arrêt (h)"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.update_xaxes(type="category")
        fig.update_yaxes(rangemode="tozero", gridcolor="#e4e3df")

        description = {
            "title": "Évolution annuelle du temps d'arrêt",
            "chart_type": "diagramme à barres annuel",
            "main_features": (
                "Somme des heures d'arrêt recensées pour chaque année."
            ),
            "yearly_downtime": [
                {"year": int(row["year"]), "total_downtime_hours": round(float(row["total_downtime_hours"]), 1)}
                for _, row in yearly.iterrows()
            ],
        }
        return fig, description

    def plot_yearly_incidents(self) -> tuple[go.Figure, dict[str, object]]:
        """Annual incident count (blue bars) and estimated cost (red line)."""
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["year"] = df["date"].dt.year

        aggregations = {"incident_count": ("date", "size")}
        if "estimated_cost_eur" not in df.columns:
            raise KeyError("La colonne 'estimated_cost_eur' est nécessaire pour le graphique annuel combiné.")

        yearly = (
            df.groupby("year")
            .agg(
                incident_count=("date", "size"),
                total_cost_eur=("estimated_cost_eur", "sum"),
            )
            .reset_index()
            .sort_values("year")
        )
        yearly["total_cost_eur"] = pd.to_numeric(yearly["total_cost_eur"], errors="coerce").fillna(0)
        years = yearly["year"].astype(str)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=years,
                y=yearly["incident_count"],
                name="Nombre d'incidents",
                marker_color="#4472C4",
                text=yearly["incident_count"],
                textposition="outside",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=yearly["total_cost_eur"],
                name="Coût total estimé",
                mode="lines+markers+text",
                line={"color": "#D62728", "width": 3},
                marker={"size": 9, "color": "#D62728"},
                text=[f"{value:,.0f} €".replace(",", " ") for value in yearly["total_cost_eur"]],
                # Décaler la dernière étiquette pour éviter le chevauchement
                # avec le nombre d'incidents affiché au-dessus de la barre.
                textposition=["top center", "top center", "middle left"]
                if len(yearly) >= 3 else "top center",
                yaxis="y2",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title="Évolution annuelle des incidents et du coût estimé",
            xaxis={"title": "Année", "type": "category"},
            yaxis={
                "title": "Nombre d'incidents",
                "rangemode": "tozero",
                "showgrid": True,
                "gridcolor": "#e4e3df",
            },
            yaxis2={
                "title": "Coût total estimé (€)",
                "overlaying": "y",
                "side": "right",
                "rangemode": "tozero",
                "showgrid": False,
                "tickformat": ",.0f",
            },
            barmode="group",
            legend={"orientation": "h", "y": 1.12, "x": 0},
            margin={"t": 120, "r": 110, "b": 60},
        )

        description = {
            "title": "Évolution annuelle des incidents et du coût estimé",
            "chart_type": "barres bleues pour le nombre d'incidents et courbe rouge pour le coût total, avec deux axes Y",
            "main_features": "Compare le nombre annuel d'incidents au coût estimé total correspondant.",
            "yearly_incidents_and_cost": [
                {
                    "year": int(row["year"]),
                    "incident_count": int(row["incident_count"]),
                    "total_cost_eur": float(row["total_cost_eur"]),
                }
                for _, row in yearly.iterrows()
            ],
        }
        return fig, description

    def plot_yearly_variation(self) -> tuple[go.Figure, dict[str, object]]:
        """Percentage change in cost and downtime between the two latest years."""
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        yearly = (
            df.groupby("year")
            .agg(
                cost=("estimated_cost_eur", "sum"),
                downtime=("downtime_hours", "sum"),
            )
            .sort_index()
        )

        if len(yearly) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Deux années au minimum sont nécessaires pour calculer la variation.",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            )
            fig.update_layout(
                template="plotly_white",
                title="Variation annuelle du coût et du temps d'arrêt",
            )
            return fig, {
                "title": "Variation annuelle du coût et du temps d'arrêt",
                "chart_type": "diagramme à barres de variation en pourcentage",
                "main_features": "Variation non calculable : moins de deux années disponibles.",
                "variation": None,
            }

        previous_year, last_year = int(yearly.index[-2]), int(yearly.index[-1])
        previous, current = yearly.iloc[-2], yearly.iloc[-1]

        def pct_change(current_value: float, previous_value: float) -> float | None:
            if previous_value == 0:
                return None
            return round((current_value - previous_value) / previous_value * 100, 2)

        cost_change = pct_change(float(current["cost"]), float(previous["cost"]))
        downtime_change = pct_change(
            float(current["downtime"]), float(previous["downtime"])
        )

        labels, values = [], []
        if cost_change is not None:
            labels.append("Coût estimé")
            values.append(cost_change)
        if downtime_change is not None:
            labels.append("Temps d'arrêt")
            values.append(downtime_change)

        colors = ["#2a78d6" if value >= 0 else "#d95f59" for value in values]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{value:+.2f} %" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>Variation : %{y:+.2f} %<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_color="#777777", line_width=1)
        fig.update_layout(
            template="plotly_white",
            title=f"Variation annuelle : {last_year} vs {previous_year}",
            yaxis_title="Variation (%)",
            xaxis_title="Indicateur",
            showlegend=False,
        )
        fig.update_yaxes(ticksuffix=" %", gridcolor="#e4e3df")

        description = {
            "title": f"Variation annuelle : {last_year} vs {previous_year}",
            "chart_type": "diagramme à barres de variation en pourcentage",
            "main_features": (
                "Compare les variations du coût estimé et du temps d'arrêt "
                "entre les deux dernières années disponibles."
            ),
            "comparison_years": {"previous": previous_year, "current": last_year},
            "variation": {
                "cost_pct": cost_change,
                "downtime_pct": downtime_change,
            },
        }
        return fig, description


class SolarPlotFactory(PlotFactory):
    """Build plots from the clusterised solar incidents dataframe."""


class WindPlotFactory(PlotFactory):
    """Build plots from the clusterised wind incidents dataframe."""


# ---------------------------------------------------------------------------
# Pipeline step — build every plot and save it as PNG + HTML + JSON
# ---------------------------------------------------------------------------


def run_plotting(
    solar_csv_path: Path, wind_csv_path: Path, plots_output_dir: Path
) -> None:
    """Build solar and wind plots under separate directories in *plots_output_dir*.

    Each figure and its description are saved as ``<name>.png`` in the
    ``png`` subfolder, ``<name>.html`` in the ``html`` subfolder, and
    ``<name>.json`` in the ``json`` subfolder, where ``<name>`` is a
    ``plot_*`` method name.
    """
    for label, factory_type, csv_path in (
        ("solar", SolarPlotFactory, solar_csv_path),
        ("wind", WindPlotFactory, wind_csv_path),
    ):
        factory = factory_type.from_csv(csv_path)
        output_dir = plots_output_dir / label
        png_dir = output_dir / "png"
        html_dir = output_dir / "html"
        json_dir = output_dir / "json"
        png_dir.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        for name in factory_type.plot_method_names():
            fig, description = getattr(factory, name)()
            png_path = png_dir / f"{name}.png"
            html_path = html_dir / f"{name}.html"
            json_path = json_dir / f"{name}.json"
            fig.write_image(png_path)
            fig.write_html(html_path)
            json_path.write_text(
                json.dumps(description, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved {png_path}")
            print(f"Saved {html_path}")
            print(f"Saved {json_path}")

        print(
            f"Done. {len(factory_type.plot_method_names())} {label} plots are in {output_dir}"
        )


def add_plotting_arguments(parser: argparse.ArgumentParser) -> None:
    """Add plotting-specific arguments to *parser*."""
    parser.add_argument(
        "--solar-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_clusterised_solar.csv",
        help="Path to the clusterised solar incidents CSV.",
    )
    parser.add_argument(
        "--wind-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_clusterised_wind.csv",
        help="Path to the clusterised wind incidents CSV.",
    )
    parser.add_argument(
        "--plots-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "plots",
        help="Directory where plots are saved (png/, html/ and json/ subfolders are created inside).",
    )


# ===========================================================================
# Equipment-level figures — one plotly figure + its data per plot_* method
# ===========================================================================
#
# The equipment reports are built per equipment of a site. Like PlotFactory,
# each plot_* method returns a plotly figure and a dictionary; here the
# dictionary holds the exact data drawn, which ``report_factory`` injects
# into the prompt of the section that comments the figure.

# Failure-mode families per asset type: colour = family, marker shape =
# failure mode inside the family. Cluster names are regenerated by the LLM at
# each clustering run, so a failure mode is assigned to its family by keywords
# (lowercase, accents removed). Families are tried in this order: the first
# match wins (e.g. "Salissure et nettoyage modules PV" is soiling, not
# modules; "Rupture de pale et effondrement" is a blade failure).
FAILURE_MODE_FAMILIES: dict[str, dict[str, list[str]]] = {
    "solar": {
        "Fire": ["incendie", "feu", "fumee"],
        "Soiling": ["salissure", "nettoyage", "encrassement", "poussiere"],
        "Overheating": ["surchauffe", "temperature"],
        "Junction box / connectors": ["jonction", "connecteur", "chaine"],
        "Cabling": ["cablage", "cable", "isolement", "court-circuit"],
        "Overvoltage": ["surtension", "foudre", "parafoudre"],
        "Inverter": ["onduleur"],
        "Modules": ["module", "panneau"],
    },
    "wind": {
        "Blades": ["pale"],
        "Overheating": ["surchauffe", "temperature", "incendie"],
        "Electrical": ["electrique", "disjoncteur", "armoire", "convertisseur"],
        "High-wind safety stops": ["vent excessif", "survitesse", "arret automatique"],
        "Structure / foundation": ["effondrement", "fondation", "fissure", "mat "],
        "Mechanical / drivetrain": [
            "mecanique",
            "frein",
            "rotor",
            "usure",
            "vibration",
            "maintenance",
            "piece",
        ],
    },
}
# Reference categorical palette, fixed colour per family (CVD-validated on
# adjacent pairs); failure modes matching no family are grey
FAMILY_COLORS: dict[str, dict[str, str]] = {
    "solar": {
        "Fire": "#2a78d6",
        "Junction box / connectors": "#eb6834",
        "Cabling": "#1baf7a",
        "Overvoltage": "#eda100",
        "Modules": "#e87ba4",
        "Overheating": "#008300",
        "Soiling": "#6250d6",
        "Inverter": "#e34948",
        "Other": "#8a8984",
    },
    "wind": {
        "Blades": "#2a78d6",
        "Mechanical / drivetrain": "#eb6834",
        "Electrical": "#1baf7a",
        "High-wind safety stops": "#eda100",
        "Structure / foundation": "#e87ba4",
        "Overheating": "#008300",
        "Other": "#8a8984",
    },
}
# Plotly marker symbols; the "-open" variant marks moderate incidents
FAMILY_MARKERS: list[str] = [
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "triangle-down",
    "cross",
]
# Unclassified incidents (noise cluster): grey cross
DEFAULT_STYLE = ("#8a8984", "x")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SEQUENTIAL_BLUES = ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]


def _plain(text: str) -> str:
    """Lowercase *text* without accents, for keyword matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def failure_mode_family(mode: str, asset: str = "solar") -> str:
    """Family of a failure mode name (``"Other"`` when no keyword matches)."""
    name = _plain(mode)
    for family, keywords in FAILURE_MODE_FAMILIES[asset].items():
        if any(keyword in name for keyword in keywords):
            return family
    return "Other"


def failure_mode_styles(
    modes: Iterable[str], asset: str = "solar"
) -> dict[str, tuple[str, str]]:
    """``{failure_mode: (colour, marker symbol)}`` for every failure mode of *modes*.

    Within a family, markers follow the alphabetical order of the failure
    modes, so pass every failure mode of the dataset (not only those of one
    equipment) to keep the same style across all equipment.
    """
    styles = {UNCLASSIFIED: DEFAULT_STYLE}
    by_family: dict[str, list[str]] = {}
    for mode in sorted(set(modes) - {UNCLASSIFIED}):
        by_family.setdefault(failure_mode_family(mode, asset), []).append(mode)
    for family, members in by_family.items():
        for i, mode in enumerate(members):
            styles[mode] = (
                FAMILY_COLORS[asset][family],
                FAMILY_MARKERS[i % len(FAMILY_MARKERS)],
            )
    return styles


def _equipment_layout(fig: go.Figure, title: str, width: int, height: int) -> None:
    """Shared look of the equipment figures (light surface, recessive grid)."""
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        font={"family": "Arial, sans-serif", "size": 12, "color": TEXT_PRIMARY},
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
        hoverlabel={"bgcolor": "white"},
        # Thousands separated by a space, as in the report ("49 142")
        separators=". ",
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont={"color": TEXT_SECONDARY}
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont={"color": TEXT_SECONDARY}
    )


class EquipmentPlotFactory:
    """Builds the figures of one equipment report from the solar incidents dataframe.

    The whole dataframe is kept (``df``) because some figures compare the
    equipment with the rest of its site. Like ``PlotFactory``, each ``plot_*``
    method returns a plotly figure and a dictionary — here the data drawn by
    the figure; adding a method is enough to add a figure (and its JSON) to
    the pipeline output.
    """

    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        *PlotFactory.REQUIRED_COLUMNS,
        "incident_id",
        "cluster_name",
    ]

    def __init__(
        self, df: pd.DataFrame, equipment_id: str, asset: str = "solar"
    ) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        self.df = df
        self.equipment_id = equipment_id
        self.asset = asset
        self.styles = failure_mode_styles(df["cluster_name"].unique(), asset)
        self.incidents = number_incidents(df, equipment_id)
        if self.incidents.empty:
            raise ValueError(
                f"{equipment_id} has no moderate or severe incident to plot"
            )
        self.site = self.incidents["site"].iloc[0]

    @classmethod
    def plot_method_names(cls) -> list[str]:
        """Names of all graph-building methods (``plot_*``), sorted alphabetically."""
        return sorted(
            name
            for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith(PLOT_METHOD_PREFIX)
        )

    def plot_cost_timeline(self) -> tuple[go.Figure, dict]:
        """Coût de chaque incident modéré/sévère dans le temps, par mode de défaillance.

        Couleur = famille de modes de défaillance, forme du marqueur = mode de
        défaillance, plein = sévère, creux = modéré. Les trois incidents les
        plus coûteux sont étiquetés.
        """
        d = self.incidents
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["estimated_cost_eur"],
                mode="lines",
                line={"color": GRID_COLOR, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        for mode, g in d.groupby("cluster_name"):
            color, symbol = self.styles[str(mode)]
            # Legend entry: the filled marker of the failure mode
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=str(mode),
                    legendgroup=str(mode),
                    marker={"symbol": symbol, "color": color, "size": 11},
                )
            )
            for severe, g_sev in g.groupby(g["severity"] == SEVERE):
                fig.add_trace(
                    go.Scatter(
                        x=g_sev["date"],
                        y=g_sev["estimated_cost_eur"],
                        mode="markers",
                        legendgroup=str(mode),
                        showlegend=False,
                        marker={
                            "symbol": symbol if severe else f"{symbol}-open",
                            "color": color,
                            "size": 12 if severe else 11,
                            "line": {
                                "color": "white" if severe else color,
                                "width": 1.5 if severe else 2.5,
                            },
                        },
                        customdata=g_sev[
                            ["ref", "cluster_name", "severity", "downtime_hours"]
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b> · %{x|%Y-%m-%d}<br>"
                            "%{customdata[1]}<br>%{customdata[2]} · "
                            "%{y:,.0f} € · %{customdata[3]} h<extra></extra>"
                        ),
                    )
                )
        for name, symbol in (
            ("plein = sévère", "circle"),
            ("creux = modéré", "circle-open"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=name,
                    marker={
                        "symbol": symbol,
                        "color": TEXT_SECONDARY,
                        "size": 11,
                        "line": {"width": 2.5},
                    },
                )
            )

        top = d.nlargest(3, "estimated_cost_eur")
        for _, r in top.iterrows():
            fig.add_annotation(
                x=r["date"],
                y=r["estimated_cost_eur"],
                text=f"{r['ref']} · {format_eur(r['estimated_cost_eur'])} €",
                showarrow=False,
                xanchor="left",
                xshift=9,
                yshift=10,
                font={"size": 12, "color": TEXT_SECONDARY},
            )

        pad = pd.Timedelta(days=20)
        _equipment_layout(
            fig,
            f"{self.equipment_id} — Coût des incidents modérés et sévères par mode de défaillance",
            width=1150,
            height=500,
        )
        fig.update_layout(
            legend={"x": 1.02, "y": 1, "font": {"size": 11}},
            yaxis_title="Coût estimé (€)",
        )
        fig.update_xaxes(
            range=[self.df["date"].min() - pad, self.df["date"].max() + pad],
            tickformat="%Y-%m",
        )
        fig.update_yaxes(tickformat=",.0f", rangemode="tozero")

        points = [
            {
                "ref": r["ref"],
                "date": f"{r['date']:%Y-%m-%d}",
                "severity": r["severity"],
                "failure_mode": r["cluster_name"],
                "cost_eur": round(float(r["estimated_cost_eur"])),
                "downtime_h": round(float(r["downtime_hours"]), 1),
                "weather_condition": r["weather_condition"],
                "days_since_previous_incident": None
                if pd.isna(r["days_since_previous"])
                else int(r["days_since_previous"]),
                "days_since_previous_same_failure_mode": None
                if pd.isna(r["days_since_previous_same_mode"])
                else int(r["days_since_previous_same_mode"]),
            }
            for _, r in d.iterrows()
        ]
        data = {
            "figure": "Chronologie des coûts des incidents modérés et sévères",
            "how_to_read": "Un point par incident, dans l'ordre chronologique ; les trois "
            "incidents les plus coûteux sont étiquetés sur la figure.",
            "labelled_incidents": top["ref"].tolist(),
            "points": points,
        }
        return fig, data

    def plot_downtime_weather(self) -> tuple[go.Figure, dict]:
        """Temps d'arrêt de chaque incident par condition météorologique (nuage de points + médiane)."""
        d = self.incidents
        conditions = ordered_weather(d["weather_condition"])
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for i, weather in enumerate(conditions):
            g = d[d["weather_condition"] == weather]
            fig.add_trace(
                go.Scatter(
                    x=g["downtime_hours"],
                    y=i + rng.uniform(-0.15, 0.15, len(g)),
                    mode="markers",
                    marker={
                        "color": "#2a78d6",
                        "size": 10,
                        "line": {"color": "white", "width": 1},
                    },
                    customdata=g[["ref", "cluster_name"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · %{x} h<br>%{customdata[1]}"
                        f"<br>{weather}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[g["downtime_hours"].median()],
                    y=[i],
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 24,
                        "line": {"color": TEXT_PRIMARY, "width": 3},
                    },
                    hovertemplate="médiane : %{x} h<br>" + weather + "<extra></extra>",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                xref="paper",
                x=1.01,
                y=i,
                text=f"n={len(g)} · {g['downtime_hours'].sum():.0f} h",
                showarrow=False,
                xanchor="left",
                font={"size": 11, "color": TEXT_SECONDARY},
            )
        _equipment_layout(
            fig,
            f"{self.equipment_id} — Temps d'arrêt vs météo (modérés + sévères)",
            width=950,
            height=55 * len(conditions) + 170,
        )
        fig.update_layout(
            margin={"r": 120},
            xaxis_title="Temps d'arrêt par incident en heures (trait noir = médiane)",
        )
        fig.update_xaxes(rangemode="tozero")
        fig.update_yaxes(
            tickvals=list(range(len(conditions))),
            ticktext=conditions,
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        )

        data = {
            "figure": "Temps d'arrêt vs condition météorologique",
            "how_to_read": "Lignes triées par temps d'arrêt total. 'representation' compare la part "
            "des incidents de l'équipement sous cette météo avec la part sur l'ensemble du site.",
            "rows": weather_table(d, site_focus(self.df, self.site)),
        }
        return fig, data

    def plot_cluster_weather(self) -> tuple[go.Figure, dict]:
        """Nombre d'incidents par mode de défaillance et condition météorologique (heatmap séquentielle).

        Les incidents non classés sont exclus : ils n'appartiennent à aucun mode de défaillance.
        """
        d = classified(self.incidents)
        ct = pd.crosstab(d["cluster_name"], d["weather_condition"])
        ct = ct[ordered_weather(ct.columns)]
        steps = len(SEQUENTIAL_BLUES) - 1
        fig = go.Figure(
            go.Heatmap(
                z=ct.values,
                x=list(ct.columns),
                y=list(ct.index),
                zmin=0,
                zmax=max(int(ct.values.max()), 1),
                colorscale=[[i / steps, c] for i, c in enumerate(SEQUENTIAL_BLUES)],
                text=[[str(n) if n else "" for n in row] for row in ct.values],
                texttemplate="%{text}",
                textfont={"size": 13},
                xgap=3,
                ygap=3,
                showscale=False,
                hovertemplate="%{y}<br>%{x}: %{z} incident(s)<extra></extra>",
            )
        )
        _equipment_layout(
            fig,
            f"{self.equipment_id} — Nombre d'incidents : mode de défaillance × météo",
            width=130 * ct.shape[1] + 420,
            height=48 * ct.shape[0] + 200,
        )
        fig.update_xaxes(showgrid=False, tickangle=-30)
        fig.update_yaxes(showgrid=False, autorange="reversed")

        data = {
            "figure": "Modes de défaillance vs condition météorologique",
            "how_to_read": "Nombre d'incidents par mode de défaillance (lignes) et condition "
            "météorologique. Une condition météorologique absente de la ligne d'un mode de "
            "défaillance ne s'est jamais produite avec ce mode.",
            "matrix": failure_mode_weather_matrix(d),
        }
        return fig, data


def run_equipment_plotting(
    df: pd.DataFrame,
    criticality: pd.DataFrame,
    equipment_ids: list[str],
    site_dir: Path,
    asset: str = "solar",
) -> None:
    """Write the figures and JSON files of every equipment report of one site.

    *asset* (``solar`` or ``wind``) selects the failure-mode families of the
    figures and the asset profile stored in ``stats.json``.

    For each equipment, in ``<site_dir>/<equipment_id>/`` (same layout as
    ``run_plotting``):

    - ``figures/png/<name>.png`` and ``figures/html/<name>.html`` — one figure
      per ``EquipmentPlotFactory.plot_*`` method (PNG export needs kaleido);
    - ``json/<name>.json`` — the data shown by that figure;
    - ``json/stats.json`` — every statistic of the report
      (``equipment_stats.build_equipment_stats``).
    """
    for equipment_id in equipment_ids:
        png_dir = site_dir / equipment_id / "figures" / "png"
        html_dir = site_dir / equipment_id / "figures" / "html"
        json_dir = site_dir / equipment_id / "json"
        for directory in (png_dir, html_dir, json_dir):
            directory.mkdir(parents=True, exist_ok=True)

        factory = EquipmentPlotFactory(df, equipment_id, asset)
        for method in EquipmentPlotFactory.plot_method_names():
            fig, data = getattr(factory, method)()
            name = method.removeprefix(PLOT_METHOD_PREFIX)
            fig.write_image(png_dir / f"{name}.png", scale=2)
            fig.write_html(html_dir / f"{name}.html", include_plotlyjs="cdn")
            (json_dir / f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        stats = build_equipment_stats(df, criticality, equipment_id, asset)
        (json_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved figures and JSON of {equipment_id} in {site_dir / equipment_id}")


def add_equipment_plotting_arguments(parser: argparse.ArgumentParser) -> None:
    """Add equipment-plotting-specific arguments to *parser*."""
    parser.add_argument(
        "--asset",
        choices=["solar", "wind"],
        default="solar",
        help="Asset type of the site: solar (PV inverters) or wind (wind turbines).",
    )
    parser.add_argument(
        "--solar-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_clusterised_solar.csv",
        help="Path to the clusterised solar incidents CSV (used with --asset solar).",
    )
    parser.add_argument(
        "--wind-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents_clusterised_wind.csv",
        help="Path to the clusterised wind incidents CSV (used with --asset wind).",
    )
    parser.add_argument(
        "--equipment-output-dir",
        type=Path,
        default=PROJECT_ROOT / "equipment_reports",
        help="Directory where <site>/<equipment>/{figures,json,sections,report.md} are written.",
    )
