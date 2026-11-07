"""Stats : calculs 100 % déterministes (aucun LLM).

Entrée  : DataFrame clusterisé (colonnes de incidents_clusterised_*.csv)
Sortie  : ``compute_stats_and_graphs`` renvoie les statistiques JSON et les figures
          Plotly séparément. Chaque métrique porte un identifiant stable (ex.
          ``site.Est-2.cost_total``) que l'agent critique utilise pour vérifier
          les chiffres cités.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

type Graph = go.Figure | None
type GraphCollection = dict[str, go.Figure]

data_path = "data/incidents_clusterised_solar.csv"

SEV_ORDER = ["mineure", "modérée", "sévère"]
SEASONS = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps",
           5: "printemps", 6: "été", 7: "été", 8: "été", 9: "automne",
           10: "automne", 11: "automne"}
NOISE = "C_NOISE"
MIN_N = 5  # en dessous, on n'affiche pas de test (trop peu de données)


# ───────────────────────── préparation ─────────────────────────
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["season"] = d["month"].map(SEASONS)
    d["severity"] = pd.Categorical(d["severity"], SEV_ORDER, ordered=True)
    for c in ("downtime_hours", "estimated_cost_eur"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_noise"] = d["cluster_code"].eq(NOISE)
    return d


def data_quality(d: pd.DataFrame) -> tuple[dict[str, Any], Graph]:
    return {
        "n_incidents": len(d),
        "date_min": str(d["date"].min().date()),
        "date_max": str(d["date"].max().date()),
        "missing_by_column": {c: int(n) for c, n in d.isna().sum().items() if n},
        "duplicate_ids": int(d["incident_id"].duplicated().sum()),
        "noise_share_pct": round(float(100 * d["is_noise"].mean()), 1),
        "sites": sorted(d["site"].unique().tolist()),
    }, None


# ───────────────────────── briques ─────────────────────────
def _num_summary(s: pd.Series) -> dict[str, float]:
    """Les coûts sont à queue lourde : médiane + quantiles, pas que la moyenne."""
    return {
        "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "p90": round(float(s.quantile(0.9)), 2), "max": round(float(s.max()), 2),
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(x[x >= 0])
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def overview(d: pd.DataFrame, title: str) -> tuple[dict[str, Any], Graph]:
    stats = {
        "n": len(d),
        "cost": _num_summary(d["estimated_cost_eur"]),
        "downtime_h": _num_summary(d["downtime_hours"]),
        "severity_counts": d["severity"].value_counts().reindex(SEV_ORDER, fill_value=0).to_dict(),
        # part du coût portée par les 10 % d'incidents les plus chers
        "top10pct_cost_share_pct": round(
            100 * d["estimated_cost_eur"].nlargest(max(1, len(d) // 10)).sum()
            / d["estimated_cost_eur"].sum(), 1),
        "cost_gini": round(_gini(d["estimated_cost_eur"].to_numpy()), 3),
    }
    counts = stats["severity_counts"]
    figure = go.Figure(
        go.Bar(x=list(counts), y=list(counts.values()), marker_color="#3973ac")
    )
    figure.update_layout(
        title=f"{title} — incidents par niveau de gravité",
        xaxis_title="Gravité",
        yaxis_title="Nombre d'incidents",
    )
    return stats, figure


def by_cluster(
    d: pd.DataFrame, title: str
) -> tuple[list[dict[str, Any]], GraphCollection]:
    """Table de Pareto : fréquence vs coût vs arrêt, triée par coût."""
    tot_n, tot_c = len(d), d["estimated_cost_eur"].sum()
    tot_h = d["downtime_hours"].sum()
    g = d.groupby(["cluster_code", "cluster_name"], observed=True)
    rows = []
    for (code, name), x in g:
        rows.append({
            "cluster_code": code, "cluster_name": name, "n": len(x),
            "incident_share_pct": round(100 * len(x) / tot_n, 1),
            "cost_total": round(float(x["estimated_cost_eur"].sum()), 2),
            "cost_share_pct": round(100 * x["estimated_cost_eur"].sum() / tot_c, 1),
            "cost_median": round(float(x["estimated_cost_eur"].median()), 2),
            "downtime_total_h": round(float(x["downtime_hours"].sum()), 1),
            "downtime_share_pct": round(100 * x["downtime_hours"].sum() / tot_h, 1),
            "severe_share_pct": round(100 * (x["severity"] == "sévère").mean(), 1),
            # >1 : le cluster pèse plus en coût qu'en fréquence (à traiter en priorité)
            "cost_to_freq_ratio": round(
                (x["estimated_cost_eur"].sum() / tot_c) / (len(x) / tot_n), 2),
            "top_incident_ids": x.nlargest(3, "estimated_cost_eur")["incident_id"].tolist(),
        })
    rows.sort(key=lambda r: r["cost_total"], reverse=True)
    cum = 0.0
    for r in rows:  # courbe de Pareto cumulée
        cum += r["cost_share_pct"]
        r["cost_cum_share_pct"] = round(cum, 1)
    graphs: GraphCollection = {}
    if rows:
        labels = [cluster["cluster_name"] for cluster in rows]
        pareto = go.Figure()
        pareto.add_trace(go.Bar(
            x=labels, y=[cluster["cost_total"] for cluster in rows],
            name="Coût total (€)", marker_color="#3973ac",
        ))
        pareto.add_trace(go.Scatter(
            x=labels, y=[cluster["cost_cum_share_pct"] for cluster in rows],
            name="Part cumulée des coûts (%)", mode="lines+markers",
            line={"color": "#e07a2d"}, yaxis="y2",
        ))
        pareto.update_layout(
            title=f"{title} — Pareto des coûts par cluster",
            xaxis_title="Cluster",
            yaxis={"title": "Coût total (€)"},
            yaxis2={"title": "Part cumulée (%)", "range": [0, 100], "overlaying": "y",
                    "side": "right"},
        )
        shares = go.Figure()
        shares.add_trace(go.Bar(
            x=labels, y=[cluster["incident_share_pct"] for cluster in rows],
            name="Part des incidents (%)",
        ))
        shares.add_trace(go.Bar(
            x=labels, y=[cluster["cost_share_pct"] for cluster in rows],
            name="Part des coûts (%)",
        ))
        shares.update_layout(
            title=f"{title} — fréquence et coût par cluster",
            barmode="group", xaxis_title="Cluster", yaxis_title="Part (%)",
        )
        graphs = {"cluster_pareto": pareto, "cluster_shares": shares}
    return rows, graphs


def time_series(d: pd.DataFrame, title: str) -> tuple[dict[str, Any], GraphCollection]:
    m = d.groupby(d["date"].dt.to_period("M")).agg(
        n=("incident_id", "size"), cost=("estimated_cost_eur", "sum"))
    full = pd.period_range(m.index.min(), m.index.max(), freq="M")
    m = m.reindex(full, fill_value=0)
    x = np.arange(len(m))
    slope = sps.linregress(x, m["n"]).slope if len(m) > 3 else float("nan")
    # tendance : Mann-Kendall simplifié via Kendall tau
    tau, p = sps.kendalltau(x, m["n"]) if len(m) > 3 else (float("nan"), float("nan"))
    by_season = d.groupby("season").agg(
        n=("incident_id", "size"), cost=("estimated_cost_eur", "sum")).round(2)
    by_year = d.groupby("year").agg(
        n=("incident_id", "size"), cost=("estimated_cost_eur", "sum")).round(2)
    # pics : mois dont le nombre d'incidents dépasse médiane + 2 MAD
    med = m["n"].median()
    mad = (m["n"] - med).abs().median() or 1
    peaks = [str(i) for i, v in m["n"].items() if v > med + 2 * 1.4826 * mad]
    result = {
        "monthly": {str(i): {"n": int(r.n), "cost": round(float(r.cost), 2)} for i, r in m.iterrows()},
        "trend_incidents_per_month": {"slope": round(float(slope), 3),
                                      "kendall_tau": round(float(tau), 3),
                                      "p_value": round(float(p), 4)},
        "by_season": by_season.to_dict("index"),
        "by_year": {int(k): v for k, v in by_year.to_dict("index").items()},
        "peak_months": peaks,
    }
    graphs: GraphCollection = {}
    if result["monthly"]:
        months = sorted(result["monthly"])
        monthly_figure = go.Figure()
        monthly_figure.add_trace(go.Bar(
            x=months, y=[result["monthly"][month]["n"] for month in months],
            name="Nombre d'incidents",
        ))
        monthly_figure.add_trace(go.Scatter(
            x=months, y=[result["monthly"][month]["cost"] for month in months],
            name="Coût (€)", mode="lines+markers", line={"color": "#e07a2d"},
            yaxis="y2",
        ))
        monthly_figure.update_layout(
            title=f"{title} — évolution mensuelle", xaxis_title="Mois",
            yaxis={"title": "Nombre d'incidents"},
            yaxis2={"title": "Coût (€)", "overlaying": "y", "side": "right"},
        )
        graphs["monthly"] = monthly_figure
    for key, values, group_name in (
        ("seasonal", result["by_season"], "saison"),
        ("yearly", {str(year): data for year, data in result["by_year"].items()}, "année"),
    ):
        if not values:
            continue
        groups = list(values)
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=groups, y=[values[group]["n"] for group in groups],
            name="Nombre d'incidents",
        ))
        figure.add_trace(go.Scatter(
            x=groups, y=[values[group]["cost"] for group in groups],
            name="Coût (€)", mode="lines+markers", line={"color": "#e07a2d"},
            yaxis="y2",
        ))
        figure.update_layout(
            title=f"{title} — incidents et coûts par {group_name}",
            xaxis_title=group_name.capitalize(),
            yaxis={"title": "Nombre d'incidents"},
            yaxis2={"title": "Coût (€)", "overlaying": "y", "side": "right"},
        )
        graphs[key] = figure
    return result, graphs


def association(
    d: pd.DataFrame, cat: str, with_: str = "cluster_code"
) -> tuple[dict[str, Any], Graph]:
    """Chi² + V de Cramér + résidus standardisés (quelles cases sont sur-représentées).
    À utiliser pour météo × cluster, saison × cluster, équipement × cluster, etc."""
    x = d.loc[~d["is_noise"], [cat, with_]].dropna()
    ct = pd.crosstab(x[cat], x[with_])
    if ct.shape[0] < 2 or ct.shape[1] < 2 or ct.values.sum() < 30:
        return {"variable": cat, "tested": False, "reason": "effectifs insuffisants"}, None
    chi2, p, _, exp = sps.chi2_contingency(ct)
    n = ct.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    resid = (ct - exp) / np.sqrt(exp)  # résidus de Pearson
    over = [{cat: r, with_: c, "observed": int(ct.loc[r, c]),
             "expected": round(float(exp[ct.index.get_loc(r), ct.columns.get_loc(c)]), 1),
             "std_resid": round(float(resid.loc[r, c]), 2)}
            for r in ct.index for c in ct.columns
            if resid.loc[r, c] > 2 and ct.loc[r, c] >= MIN_N]
    over.sort(key=lambda o: -o["std_resid"])
    result = {
        "variable": cat, "tested": True, "chi2": round(float(chi2), 2),
        "p_value": round(float(p), 4), "cramers_v": round(v, 3),
        "low_expected_cells_pct": round(100 * float((exp < 5).mean()), 1),  # fiabilité du test
        "over_represented": over[:10],
    }
    if not over:
        return result, None
    labels = [
        f"{cat}={item[cat]} / {with_}={item[with_]}"
        for item in result["over_represented"]
    ]
    figure = go.Figure(go.Bar(
        x=[item["std_resid"] for item in result["over_represented"]],
        y=labels, orientation="h", marker_color="#3973ac",
    ))
    figure.update_layout(
        title=f"{cat} — associations sur-représentées",
        xaxis_title="Résidu standardisé",
        yaxis_title="Modalité / cluster",
        yaxis={"autorange": "reversed"},
    )
    return result, figure


def cost_by_group_test(d: pd.DataFrame, group: str) -> tuple[dict[str, Any], Graph]:
    """Kruskal-Wallis (coûts non gaussiens) : le coût diffère-t-il selon `group` ?"""
    groups = [g["estimated_cost_eur"].to_numpy() for _, g in d.groupby(group, observed=True)
              if len(g) >= MIN_N]
    if len(groups) < 2:
        return {"variable": group, "tested": False}, None
    h, p = sps.kruskal(*groups)
    return {
        "variable": group, "tested": True, "H": round(float(h), 2),
        "p_value": round(float(p), 4),
    }, None


def equipment_and_people(
    d: pd.DataFrame, top_n: int = 15
) -> tuple[dict[str, Any], Graph]:
    eq = d.groupby("equipment_id").agg(
        n=("incident_id", "size"), cost=("estimated_cost_eur", "sum"),
        downtime=("downtime_hours", "sum")).sort_values("cost", ascending=False)
    eq["cost_share_pct"] = (100 * eq["cost"] / eq["cost"].sum()).round(1)
    # récidive : intervalle médian entre deux incidents sur le même équipement
    gaps = (d.sort_values("date").groupby("equipment_id")["date"].diff().dt.days.dropna())
    tech = d.groupby("technician_id").agg(
        n=("incident_id", "size"), cost_median=("estimated_cost_eur", "median"),
        severe_share_pct=("severity", lambda s: round(100 * (s == "sévère").mean(), 1)))
    technician_cost_test, _ = cost_by_group_test(d, "technician_id")
    result = {
        "equipment": eq.round(2).reset_index().to_dict("records"),
        "median_days_between_incidents_same_equipment": float(gaps.median()) if len(gaps) else None,
        "technicians": tech.round(2).reset_index().to_dict("records"),
        "technician_cost_test": technician_cost_test,
        "note": "Comparaison de techniciens indicative : affectation non aléatoire, "
                "ne pas l'utiliser pour évaluer des personnes.",
    }
    equipment = result["equipment"]
    if not equipment:
        return result, None
    top_equipment = equipment[:top_n]
    figure = go.Figure(go.Bar(
        x=[item["cost"] for item in reversed(top_equipment)],
        y=[str(item["equipment_id"]) for item in reversed(top_equipment)],
        orientation="h", marker_color="#3973ac",
    ))
    figure.update_layout(
        title=f"Top {len(top_equipment)} des équipements par coût",
        xaxis_title="Coût total (€)", yaxis_title="Équipement",
    )
    return result, figure


def cluster_overlap_hints(
    d: pd.DataFrame, jaccard_min: float = 0.5
) -> tuple[list[dict], Graph]:
    """Repère des clusters quasi redondants via le co-usage d'équipements/mois (indice, pas preuve)."""
    codes = [c for c in d["cluster_code"].unique() if c != NOISE]
    out = []
    key = d["equipment_id"].astype(str) + "|" + d["date"].dt.to_period("M").astype(str)
    sets = {c: set(key[d["cluster_code"] == c]) for c in codes}
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            j = len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
            if j >= jaccard_min:
                out.append({"clusters": [a, b], "jaccard_equipment_month": round(j, 2)})
    return out, None


# ───────────────────────── orchestration ─────────────────────────
def site_comparison(d: pd.DataFrame) -> tuple[dict[str, Any], GraphCollection]:
    cluster_x_site, association_figure = association(d, "site")
    result = {
        "cluster_x_site": cluster_x_site,
        "cost_test": cost_by_group_test(d, "site")[0],
        "cluster_share_pct_by_site": (
            pd.crosstab(d["site"], d["cluster_code"], normalize="index")
            .mul(100).round(1).to_dict("index")),
    }
    shares = result["cluster_share_pct_by_site"]
    sites = list(shares)
    clusters = list(dict.fromkeys(cluster for site in sites for cluster in shares[site]))
    figure = go.Figure(go.Heatmap(
        x=clusters, y=sites,
        z=[[shares[site].get(cluster, 0) for cluster in clusters] for site in sites],
        colorbar={"title": "Part (%)"}, colorscale="Blues",
    ))
    figure.update_layout(
        title="Répartition des clusters par site",
        xaxis_title="Cluster", yaxis_title="Site",
    )
    figures: GraphCollection = {"cluster_shares": figure}
    if association_figure is not None:
        figures["association_site"] = association_figure
    return result, figures


def compute_stats_and_graphs(
    df: pd.DataFrame, cross_vars: tuple[str, ...] = (
        "weather_condition", "season", "equipment_id")
) -> tuple[dict[str, Any], dict[str, Any]]:
    d = prepare(df)
    quality, _ = data_quality(d)
    global_overview, overview_figure = overview(d, "Vue globale")
    global_clusters, cluster_figures = by_cluster(d, "Vue globale")
    global_time, time_figures = time_series(d, "Vue globale")
    severity_test, _ = cost_by_group_test(d, "severity")
    global_associations = []
    graphs: dict[str, Any] = {
        "global": {},
        "sites": {},
        "site_comparison": {},
    }
    graphs["global"]["severity"] = overview_figure
    graphs["global"].update(cluster_figures)
    graphs["global"].update(time_figures)
    for variable in cross_vars:
        result, figure = association(d, variable)
        global_associations.append(result)
        if figure is not None:
            graphs["global"][f"association_{variable}"] = figure
    equipment_and_people_stats, equipment_figure = equipment_and_people(d)
    if equipment_figure is not None:
        graphs["global"]["equipment_costs"] = equipment_figure
    overlap_hints, _ = cluster_overlap_hints(d)
    out: dict[str, Any] = {
        "data_quality": quality,
        "global": {
            "overview": global_overview,
            "by_cluster": global_clusters,
            "time": global_time,
            "cost_by_severity_test": severity_test,
            "associations": global_associations,
            "equipment_and_people": equipment_and_people_stats,
            "cluster_overlap_hints": overlap_hints,
        },
    }
    sites = out["data_quality"]["sites"]
    out["sites"] = {}
    for site in sites:  # mêmes métriques, site par site
        ds = d[d["site"] == site]
        title = site
        site_overview, site_overview_figure = overview(ds, title)
        site_clusters, site_cluster_figures = by_cluster(ds, title)
        site_time, site_time_figures = time_series(ds, title)
        site_associations = []
        site_graphs: GraphCollection = {"severity": site_overview_figure}
        site_graphs.update(site_cluster_figures)
        site_graphs.update(site_time_figures)
        for variable in cross_vars:
            result, figure = association(ds, variable)
            site_associations.append(result)
            if figure is not None:
                site_graphs[f"association_{variable}"] = figure
        out["sites"][site] = {
            "overview": site_overview,
            "by_cluster": site_clusters,
            "time": site_time,
            "associations": site_associations,
        }
        graphs["sites"][site] = site_graphs
    if len(sites) > 1:  # comparaison inter-sites (utile pour le mappage de profils)
        comparison, comparison_graphs = site_comparison(d)
        out["site_comparison"] = comparison
        graphs["site_comparison"].update(comparison_graphs)
    return out, graphs


def open_graphs_in_browser(graphs: dict[str, Any]) -> None:
    """Open every figure returned by ``compute_stats_and_graphs`` in the browser."""
    figure_groups = [graphs.get("global", {}), graphs.get("site_comparison", {})]
    figure_groups.extend(graphs.get("sites", {}).values())

    for group in figure_groups:
        for figure in group.values():
            if isinstance(figure, go.Figure):
                figure.show(renderer="browser")


if __name__ == "__main__":
    df = pd.read_csv(data_path)
    stats, graphs = compute_stats_and_graphs(df)
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str)[:6000])
    open_graphs_in_browser(graphs)