import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Equipment-level incident reports — pipeline walkthrough

    This notebook shows, with runnable cells, the per-equipment report pipeline
    (command `uv run dc equipment`). It has four parts:

    1. **The pipeline** — what each step does and why it is built this way.
    2. **Step by step on one equipment** — `OUEST1-INV10`: every step of the pipeline runs in its own cell.
    3. **Extrapolating to a site-level analysis** — a working prototype of the same recipe with a *site* as the unit.
    4. **Running it on another equipment of another site** — `EST1-INV07`, in one call.

    Every output goes to a temporary folder: running this notebook never modifies `equipment_reports/`.
    The written documentation is in `PIPELINE_EQUIPMENT.md`.

    **Requirements:** `uv sync`, the clusterised solar CSV in `data/incidents_clusterised_solar.csv`
    (written by `uv run dc --all`), then `uv run marimo edit equipment_analysis/equipment_pipeline_walkthrough.py`.
    Azure calls need the `.env` file (see `../.env.example`); without it, the agents run in *dry-run* mode
    (prompts are written, text fields are placeholders).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Setup
    """)
    return


@app.cell
def _():
    import json
    import shutil
    import tempfile
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    # Project root = first parent folder containing pyproject.toml
    PROJECT_ROOT = next(
        p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
    )
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    CSV_PATH = PROJECT_ROOT / "data" / "incidents_clusterised_solar.csv"
    mo.stop(
        not CSV_PATH.exists(),
        mo.md(
            f"**Missing data:** `{CSV_PATH.relative_to(PROJECT_ROOT)}` — run the clustering step first."
        ),
    )

    # Every output of this notebook goes to a temporary folder
    DEMO_DIR = Path(tempfile.gettempdir()) / "equipment_pipeline_demo"
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    DEMO_DIR.mkdir(parents=True)

    # Azure OpenAI is used only if the .env file exists; otherwise dry-run mode
    USE_AZURE = (PROJECT_ROOT / ".env").exists()
    mo.md(
        f"Data: `{CSV_PATH.relative_to(PROJECT_ROOT)}` · Azure calls enabled: **{USE_AZURE}**"
    )
    return (
        CSV_PATH,
        DEMO_DIR,
        PROJECT_ROOT,
        Path,
        TEMPLATES_DIR,
        USE_AZURE,
        json,
        mo,
        pd,
        shutil,
    )


@app.cell
def _(json, mo):
    def show_markdown(path, max_chars=None):
        """Render a generated markdown file (image lines removed: figures are shown separately)."""
        text = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("![")
        )
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n\n*[…truncated]*"
        return mo.md(text)

    def show_json(data, max_chars=2500):
        text = json.dumps(data, ensure_ascii=False, indent=1)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…"
        return mo.md(f"```json\n{text}\n```")

    return show_json, show_markdown


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 1 — The pipeline

    ```
    data/incidents_clusterised_solar.csv  (1 row = 1 incident, with its failure mode = cluster)
            │
            ▼
    [1] SELECTION        equipment_stats.compute_criticality / select_equipment
            │            0-100 score vs site and vs fleet → critical equipment P1 / P2
            ▼  for each critical equipment
    [2] STATISTICS       equipment_stats.build_equipment_stats
            │            local numbering #1..#n, figures, trends, priorities + "facts"
            ▼
    [3] FIGURES + JSON   plot_factory.EquipmentPlotFactory / run_equipment_plotting
            │            figures/{png,html}/<name> + json/<name>.json (data drawn) + json/stats.json
            ▼  report_factory only reads json/
    [4] AGENTS           report_factory.section_*  — one Azure OpenAI agent per section
            │            role + shared rules + questions + only the data of the section
            │            answers checked in Python (invented numbers, wrong #k) + 1 correction round
            ▼            sections/<section>.md
    [5] ASSEMBLY         templates/equipment/equipment_report.md → report.md
    ```

    **Three principles**

    | Principle | How it is enforced |
    | --- | --- |
    | The LLM computes nothing | every number, ratio, change, priority and risk level comes from Python; the agents' answers are scanned for numbers absent from their input |
    | One specialised agent per section | each `section_*` function has its own role, questions, data subset and JSON schema |
    | A figure always comes with its data | each `plot_*` method returns the plotly figure **and** the data it draws; the agent commenting the figure receives that data |

    **Sections and who writes them**

    | Section | Written by | Data received |
    | --- | --- | --- |
    | Title + executive summary | agent *executive-summary writer* (generated last) | key facts, priorities, first actions |
    | 1. Key figures | tables in Python + agent *benchmarking analyst* | comparison with the site median, key figures |
    | 2. How this report was built | fixed text (no agent) | selection and priority rules |
    | 3. Cost timeline | agent *reliability engineer* | `cost_timeline.json`, yearly trend, seasonality, top incidents, close sequences |
    | 4. Weather exposure | agent *weather-exposure engineer* | `downtime_weather.json`, `cluster_weather.json` |
    | 5. Failure modes | agent *PV failure-analysis expert* | failure modes with their incidents and priorities |
    | 6. Root-cause hypotheses | agent *RCA engineer* (confidence set by Python) | failure modes, weather, close sequences |
    | 7. Recommendations | agent *O&M planner* (priority set by Python) | failure-mode priorities + the root-cause hypotheses |
    | 8. Conclusion | agent *reliability manager* | key facts + prioritised actions |
    | Annex — incident register | fixed table (no agent) | register |

    Incidents of the clustering's noise cluster are labelled *Unclassified*: they count in totals,
    timeline and register, but form no failure mode.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 2 — Step by step on one equipment: `OUEST1-INV10`

    Each cell below runs one step of the pipeline. Together they do exactly what
    `uv run dc equipment --site OUEST1 --equipment OUEST1-INV10` does.
    """)
    return


@app.cell
def _():
    SITE = "OUEST1"
    EQUIPMENT = "OUEST1-INV10"
    return EQUIPMENT, SITE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Load the data and select the critical equipment
    """)
    return


@app.cell
def _(CSV_PATH, mo):
    from datachallengecapgemini.equipment_stats import (
        build_equipment_stats,
        compute_criticality,
        load_solar_incidents,
        number_incidents,
        select_equipment,
    )

    df = load_solar_incidents(CSV_PATH)
    mo.vstack(
        [
            mo.md(
                f"{len(df)} solar incidents, {df['site'].nunique()} sites, "
                f"{df['equipment_id'].nunique()} equipment, {df['cluster_name'].nunique()} failure modes"
            ),
            df[
                [
                    "incident_id",
                    "site",
                    "equipment_id",
                    "date",
                    "severity",
                    "estimated_cost_eur",
                    "downtime_hours",
                    "weather_condition",
                    "cluster_name",
                ]
            ].head(3),
        ]
    )
    return (
        build_equipment_stats,
        compute_criticality,
        df,
        number_incidents,
        select_equipment,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The criticality score combines 4 indicators (incidents ×1, severe incidents ×2, cost ×1, downtime ×1)
    as robust leave-one-out z-scores, mapped to 0-100 (50 = typical, ≥ 75 = clear outlier). It is computed
    against the site (`score_site`) and against the fleet (`score_fleet`). Critical = one of them ≥ 75;
    P1 if both, P2 if one.
    """)
    return


@app.cell
def _(SITE, compute_criticality, df, mo, select_equipment):
    criticality = compute_criticality(df)
    _on_site = criticality[criticality["equipment_id"].str.startswith(SITE)]
    mo.vstack(
        [
            _on_site[
                [
                    "equipment_id",
                    "n_incidents",
                    "n_severe",
                    "total_cost",
                    "total_downtime",
                    "score_site",
                    "score_fleet",
                    "equipment_priority",
                    "risk_level",
                ]
            ].round(1),
            mo.md(
                f"Critical equipment of {SITE} → `{select_equipment(criticality, SITE)}`"
            ),
        ]
    )
    return (criticality,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Statistics and facts

    Moderate and severe incidents are numbered `#1..#n` by date for this equipment (the global
    `incident_id` never reaches the report).
    """)
    return


@app.cell
def _(EQUIPMENT, df, number_incidents):
    incidents = number_incidents(df, EQUIPMENT)
    incidents[
        [
            "ref",
            "date",
            "severity",
            "cluster_name",
            "estimated_cost_eur",
            "downtime_hours",
            "weather_condition",
            "days_since_previous",
        ]
    ].head(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `build_equipment_stats` computes every number of the report. The `facts` are ready-made
    sentences for the figures the model used to misread (most costly incident vs most costly failure
    mode, ratios, year-on-year changes).
    """)
    return


@app.cell
def _(EQUIPMENT, build_equipment_stats, criticality, df, mo):
    stats = build_equipment_stats(df, criticality, EQUIPMENT)
    _lines = [f"**Blocks:** {', '.join(stats)}"]
    for _topic, _sentences in stats["facts"].items():
        _lines.append(f"\n**[{_topic}]**\n")
        _lines += [f"- {_sentence}" for _sentence in _sentences]
    mo.md("\n".join(_lines))
    return (stats,)


@app.cell
def _(pd, stats):
    # Priority of each failure mode (inherited by the recommendations that target it)
    pd.DataFrame(stats["failure_modes"])[
        [
            "failure_mode",
            "recommendation_priority",
            "priority_reason",
            "n_incidents",
            "n_severe",
            "cost_share_pct",
        ]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 — Figures + JSON

    Like the fleet `PlotFactory`, each `plot_*` method of `EquipmentPlotFactory` returns a plotly
    figure and a dictionary — here the exact data the figure draws, which is what the agent of the
    section receives. The figures below are interactive (hover the points).
    """)
    return


@app.cell
def _(DEMO_DIR, EQUIPMENT, SITE, df, json, stats):
    from datachallengecapgemini.plot_factory import (
        PLOT_METHOD_PREFIX,
        EquipmentPlotFactory,
    )

    equipment_dir = DEMO_DIR / SITE / EQUIPMENT
    for _sub in ("figures/png", "figures/html", "json"):
        (equipment_dir / _sub).mkdir(parents=True, exist_ok=True)

    _factory = EquipmentPlotFactory(df, EQUIPMENT)
    figures = {}
    for _method in EquipmentPlotFactory.plot_method_names():
        _name = _method.removeprefix(PLOT_METHOD_PREFIX)
        figures[_name] = getattr(_factory, _method)()
        _fig, _data = figures[_name]
        _fig.write_image(equipment_dir / "figures" / "png" / f"{_name}.png", scale=2)
        _fig.write_html(
            equipment_dir / "figures" / "html" / f"{_name}.html", include_plotlyjs="cdn"
        )
        (equipment_dir / "json" / f"{_name}.json").write_text(
            json.dumps(_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (equipment_dir / "json" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sorted(p.relative_to(equipment_dir).as_posix() for p in equipment_dir.rglob("*.*"))
    return equipment_dir, figures


@app.cell
def _(figures, mo, show_json):
    _fig, _data = figures["cost_timeline"]
    mo.vstack(
        [
            _fig,
            mo.md(
                f"`labelled_incidents`: {_data['labelled_incidents']} — first points of the JSON:"
            ),
            show_json(_data["points"][:3]),
        ]
    )
    return


@app.cell
def _(figures, mo, show_json):
    _fig, _data = figures["cluster_weather"]
    mo.vstack([_fig, show_json(_data)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4 — One agent per section

    `report_factory` only reads `json/`. A `SectionContext` holds the inputs of the report and the
    answers collected so far; each `section_*` function calls `ctx.ask(...)` with its own role,
    questions, data subset and output schema.

    **Cache:** each answer is stored in `sections/<section>.llm.json` with the hash of its prompt. If
    `uv run dc equipment --site OUEST1` was already run locally, its answers are copied into the demo
    folder: the prompts are identical, so they are reused and no Azure call is made.
    """)
    return


@app.cell
def _(EQUIPMENT, PROJECT_ROOT, SITE, USE_AZURE, equipment_dir, figures, json, shutil):
    from datachallengecapgemini.report_factory import (
        EQUIPMENT_SECTIONS,
        SectionContext,
        get_azure_client,
    )

    sections_dir = equipment_dir / "sections"
    sections_dir.mkdir(exist_ok=True)
    _local_run = PROJECT_ROOT / "equipment_reports" / SITE / EQUIPMENT / "sections"
    for _cached in _local_run.glob("*.llm.json"):
        shutil.copy(_cached, sections_dir / _cached.name)

    client, deployment = get_azure_client() if USE_AZURE else (None, None)
    ctx = SectionContext(
        equipment_id=EQUIPMENT,
        stats=json.loads(
            (equipment_dir / "json" / "stats.json").read_text(encoding="utf-8")
        ),
        figures={name: data for name, (_fig, data) in figures.items()},
        sections_dir=sections_dir,
        client=client,
        deployment=deployment,
        dry_run=not USE_AZURE,
    )
    f"{len(list(sections_dir.glob('*.llm.json')))} cached agent answers available"
    return EQUIPMENT_SECTIONS, SectionContext, client, ctx, deployment, sections_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us run **one** section first, the cost timeline, and look at what its agent receives.
    The prompt is saved next to the answer (`sections/cost_timeline.prompt.md`).
    """)
    return


@app.cell
def _(ctx, mo, sections_dir):
    from datachallengecapgemini.report_factory import section_cost_timeline

    cost_timeline_md = section_cost_timeline(ctx)
    _system, _user = (
        (sections_dir / "cost_timeline.prompt.md")
        .read_text(encoding="utf-8")
        .split("# User", 1)
    )
    mo.md(f"```\n{_system[:1200]}\n…\n\n# User{_user[:1500]}\n…\n```")
    return (cost_timeline_md,)


@app.cell
def _(cost_timeline_md, ctx, mo, sections_dir, show_json, show_markdown):
    # The answer: one field per question (JSON schema), then rendered with the Python tables
    (sections_dir / "cost_timeline.md").write_text(
        cost_timeline_md + "\n", encoding="utf-8"
    )
    mo.vstack(
        [
            show_json(ctx.answers["cost_timeline"]),
            show_markdown(sections_dir / "cost_timeline.md"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now every section, in generation order: the conclusion and the executive summary come last
    because they receive the prioritised actions. Python checks each answer; problems left after the
    correction round would be listed in `checks.txt`.
    """)
    return


@app.cell
def _(EQUIPMENT_SECTIONS, cost_timeline_md, ctx, mo, sections_dir):
    assert cost_timeline_md  # run after the single-section demo above
    section_context = {}
    _rows = []
    for _name, _build_section in EQUIPMENT_SECTIONS.items():
        _markdown = _build_section(ctx)
        (sections_dir / f"{_name}.md").write_text(_markdown + "\n", encoding="utf-8")
        section_context[f"section_{_name}"] = _markdown.replace(
            "](../figures/", "](figures/"
        )
        _agent = "agent" if _name in ctx.answers else "Python only"
        _rows.append(f"| {_name} | {_agent} | {len(_markdown)} |")
    mo.md(
        "| Section | Written by | Characters |\n| --- | --- | ---: |\n"
        + "\n".join(_rows)
        + f"\n\nChecks left after correction: **{ctx.warnings or 'none'}**"
    )
    return (section_context,)


@app.cell
def _(section_context, sections_dir, show_markdown):
    assert section_context
    show_markdown(sections_dir / "recommendations.md")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5 — Assembly
    """)
    return


@app.cell
def _(TEMPLATES_DIR, equipment_dir, mo, section_context, show_markdown):
    from datachallengecapgemini.report_factory import (
        EQUIPMENT_TEMPLATE_NAME,
        fill_template,
    )

    _template = (TEMPLATES_DIR / EQUIPMENT_TEMPLATE_NAME).read_text(encoding="utf-8")
    report_path = equipment_dir / "report.md"
    report_path.write_text(fill_template(_template, section_context), encoding="utf-8")
    mo.vstack(
        [
            mo.md(
                f"Template `templates/{EQUIPMENT_TEMPLATE_NAME}`:\n```\n{_template}\n```"
            ),
            show_markdown(report_path, max_chars=2500),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Same result in one call** (what `uv run dc equipment` runs):

    ```python
    from datachallengecapgemini.plot_factory import run_equipment_plotting
    from datachallengecapgemini.report_factory import run_equipment_reporting

    run_equipment_plotting(df, criticality, ["OUEST1-INV10"], site_dir)
    run_equipment_reporting(["OUEST1-INV10"], site_dir, TEMPLATES_DIR)
    ```

    Part 4 uses exactly these two calls on another equipment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 3 — Extrapolating the pipeline to a site-level analysis

    The pipeline does not depend on the unit being a piece of equipment. To change the unit, define:

    | | What to define | Equipment report | Site report (this prototype) |
    | --- | --- | --- | --- |
    | 1 | the **unit** | one inverter | one site |
    | 2 | its **peers** | the other equipment of the site / the fleet | the other sites of the fleet |
    | 3 | the **statistics + facts** | `build_equipment_stats` | `build_site_stats` below |
    | 4 | the **figures + data** | `EquipmentPlotFactory` | `plot_equipment_failure_modes` below |
    | 5 | the **sections** | `section_*` in `report_factory` | `section_systemic_vs_local` below |

    What is reused unchanged: `SectionContext.ask` (prompt, cache, checks, correction round),
    `call_azure`, `_md_table`, `fill_template`. The only coupling to generalise in a real implementation:
    `SectionContext` reads the unit's site from `stats["equipment"]["site"]`, and `SHARED_RULES` describes
    the unit as "a PV inverter" — both would become parameters.

    The prototype below builds one site-level section: **are the failure modes of the site systemic
    (present on many equipment) or local (concentrated on one)?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Site statistics + facts (all computed in Python)
    """)
    return


@app.cell
def _(pd):
    from datachallengecapgemini.equipment_stats import (
        FOCUS_SEVERITIES,
        SEVERE,
        classified,
        format_eur,
        site_code,
    )

    def build_site_stats(
        df: pd.DataFrame, criticality: pd.DataFrame, site: str
    ) -> dict:
        """Statistics of one site: comparison with the median site, failure-mode spread, ranking."""
        df = df.assign(site_code=df["equipment_id"].map(site_code))
        per_site = df.groupby("site_code").agg(
            n_incidents=("incident_id", "count"),
            n_severe=("severity", lambda s: int((s == SEVERE).sum())),
            total_cost=("estimated_cost_eur", "sum"),
            total_downtime=("downtime_hours", "sum"),
        )
        median = per_site.median()
        digits = {"total_downtime": 1}
        comparison = {
            indicator: {
                "this_site": round(
                    float(per_site.loc[site, indicator]), digits.get(indicator)
                ),
                "median_site": round(float(median[indicator]), digits.get(indicator)),
                "ratio_to_median_site": round(
                    float(per_site.loc[site, indicator] / median[indicator]), 2
                ),
            }
            for indicator in per_site.columns
        }

        on_site = df[df["site_code"] == site]
        n_equipment = on_site["equipment_id"].nunique()
        focus = classified(on_site[on_site["severity"].isin(FOCUS_SEVERITIES)])
        modes = []
        for mode, g in focus.groupby("cluster_name"):
            affected = g["equipment_id"].nunique()
            if affected >= max(3, n_equipment / 2):
                spread = "systemic"
            elif affected == 1:
                spread = "local"
            else:
                spread = "partial"
            modes.append(
                {
                    "failure_mode": mode,
                    "n_incidents": len(g),
                    "n_severe": int((g["severity"] == SEVERE).sum()),
                    "cost_total_eur": round(float(g["estimated_cost_eur"].sum())),
                    "n_equipment_affected": affected,
                    "spread": spread,
                    "equipment_affected": sorted(g["equipment_id"].unique()),
                }
            )
        modes.sort(key=lambda m: m["cost_total_eur"], reverse=True)

        labels = ["Incidents", "Severe incidents", "Total cost (EUR)", "Downtime (h)"]
        formats = [str, str, format_eur, str]
        facts = [
            f"{label}: {fmt(c['this_site'])} on this site vs {fmt(c['median_site'])} for the "
            f"median site (ratio ×{c['ratio_to_median_site']})."
            for label, fmt, c in zip(labels, formats, comparison.values())
        ] + [
            f"{m['failure_mode']}: {m['n_incidents']} incident(s) on {m['n_equipment_affected']} "
            f"of the {n_equipment} equipment, {format_eur(m['cost_total_eur'])} EUR ({m['spread']})."
            for m in modes
        ]
        ranking = criticality[criticality["equipment_id"].map(site_code) == site]
        return {
            # SectionContext reads the unit's site here (the coupling to generalise)
            "equipment": {"equipment_id": site, "site": on_site["site"].iloc[0]},
            "register": [],
            "n_equipment": n_equipment,
            "comparison_with_median_site": comparison,
            "failure_modes": modes,
            "equipment_ranking": ranking[
                ["equipment_id", "score_site", "equipment_priority"]
            ]
            .round(1)
            .to_dict("records"),
            "facts": {"site": facts},
        }

    return FOCUS_SEVERITIES, build_site_stats, classified, site_code


@app.cell
def _(SITE, build_site_stats, criticality, df, mo):
    site_stats = build_site_stats(df, criticality, SITE)
    mo.md("\n".join(f"- {_sentence}" for _sentence in site_stats["facts"]["site"]))
    return (site_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 A site figure and its data (same `(figure, data)` contract as `plot_*`)
    """)
    return


@app.cell
def _(FOCUS_SEVERITIES, classified, pd, site_code):
    import plotly.graph_objects as go

    from datachallengecapgemini.plot_factory import SEQUENTIAL_BLUES, _equipment_layout

    def plot_equipment_failure_modes(
        df: pd.DataFrame, site: str
    ) -> tuple[go.Figure, dict]:
        """Heatmap: moderate/severe incidents per failure mode (rows) and equipment (columns)."""
        d = classified(
            df[
                (df["equipment_id"].map(site_code) == site)
                & df["severity"].isin(FOCUS_SEVERITIES)
            ]
        )
        ct = pd.crosstab(d["cluster_name"], d["equipment_id"])
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
        steps = len(SEQUENTIAL_BLUES) - 1
        fig = go.Figure(
            go.Heatmap(
                z=ct.values,
                x=[c.split("-")[1] for c in ct.columns],
                y=list(ct.index),
                zmin=0,
                colorscale=[[i / steps, c] for i, c in enumerate(SEQUENTIAL_BLUES)],
                text=[[str(n) if n else "" for n in row] for row in ct.values],
                texttemplate="%{text}",
                xgap=3,
                ygap=3,
                showscale=False,
                hovertemplate="%{y}<br>%{x}: %{z} incident(s)<extra></extra>",
            )
        )
        _equipment_layout(
            fig,
            f"{site} — moderate and severe incidents: failure mode × equipment",
            width=70 * ct.shape[1] + 480,
            height=40 * ct.shape[0] + 180,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        data = {
            "figure": "Failure modes vs equipment of the site",
            "matrix": {
                mode: {eq: int(n) for eq, n in row.items() if n}
                for mode, row in ct.iterrows()
            },
        }
        return fig, data

    return (plot_equipment_failure_modes,)


@app.cell
def _(DEMO_DIR, SITE, df, plot_equipment_failure_modes):
    site_dir = DEMO_DIR / f"site_{SITE}"
    (site_dir / "sections").mkdir(parents=True, exist_ok=True)
    site_fig, site_fig_data = plot_equipment_failure_modes(df, SITE)
    site_fig
    return site_dir, site_fig_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 A site-level section with its own agent

    Same pattern as the equipment sections: a role, questions, the data of the section, a strict
    schema and a Python check (here: every failure mode named must exist on the site).
    """)
    return


@app.cell
def _(SectionContext):
    from datachallengecapgemini.report_factory import _list, _md_table, _object, _text

    def section_systemic_vs_local(ctx: SectionContext) -> str:
        """Site section — are the failure modes systemic (many equipment) or local (one)?"""
        modes = ctx.stats["failure_modes"]
        names = {m["failure_mode"] for m in modes}

        def check(answer: dict) -> list[str]:
            unknown = [
                m
                for m in answer["systemic_issues"] + answer["local_issues"]
                if m not in names
            ]
            return [f"unknown failure modes: {'; '.join(unknown)}"] if unknown else []

        answer = ctx.ask(
            "systemic_vs_local",
            "3. Systemic vs local failure modes",
            role="a reliability engineer who compares PV inverters within a site to separate "
            "site-wide issues from equipment-specific ones.",
            questions=[
                (
                    "Which failure modes are systemic (spread over many equipment of the site) "
                    "and which are local? Use the `spread` field as given."
                ),
                (
                    "For the most costly systemic failure modes, what does a site-wide cause "
                    "suggest (design, installation, environment) compared with a local cause "
                    "(one faulty component)?"
                ),
                "Which equipment should be inspected first, according to the given ranking and priorities?",
            ],
            data={
                "failure_modes": modes,
                "equipment_ranking": ctx.stats["equipment_ranking"],
                "failure_modes_vs_equipment_figure": ctx.figures[
                    "equipment_failure_modes"
                ],
                "facts": ctx.facts["site"],
            },
            schema=_object(
                {
                    "systemic_issues": _list(
                        _text("Failure mode name, verbatim."), "Systemic failure modes."
                    ),
                    "local_issues": _list(
                        _text("Failure mode name, verbatim."), "Local failure modes."
                    ),
                    "analysis": _text("Answer to questions 1-2 (3-5 sentences)."),
                    "inspection_order": _text("Answer to question 3 (1-2 sentences)."),
                }
            ),
            check=check,
        )
        table = _md_table(
            modes,
            {
                "failure_mode": "Failure mode",
                "n_incidents": "Incidents",
                "n_equipment_affected": "Equipment affected",
                "spread": "Spread",
                "cost_total_eur": "Cost (EUR)",
            },
        )
        return "\n\n".join(
            [
                "## 3. Systemic vs local failure modes",
                table,
                answer["analysis"],
                f"**Inspection order.** {answer['inspection_order']}",
            ]
        )

    return (section_systemic_vs_local,)


@app.cell
def _(
    SITE,
    SectionContext,
    USE_AZURE,
    client,
    deployment,
    mo,
    section_systemic_vs_local,
    show_markdown,
    site_dir,
    site_fig_data,
    site_stats,
):
    _site_ctx = SectionContext(
        equipment_id=SITE,
        stats=site_stats,
        figures={"equipment_failure_modes": site_fig_data},
        sections_dir=site_dir / "sections",
        client=client,
        deployment=deployment,
        dry_run=not USE_AZURE,
    )
    _path = site_dir / "sections" / "systemic_vs_local.md"
    _path.write_text(section_systemic_vs_local(_site_ctx) + "\n", encoding="utf-8")
    mo.vstack(
        [mo.md(f"Checks: **{_site_ctx.warnings or 'none'}**"), show_markdown(_path)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **To turn this prototype into a full site pipeline**

    1. move `build_site_stats` to `src/datachallengecapgemini/` (e.g. `site_stats.py`) and add the other
       blocks (yearly trend, weather, `#k` numbering per site, more facts);
    2. add a `SitePlotFactory` in `plot_factory.py` with `plot_*` methods returning `(figure, data)`;
    3. add `section_site_*` functions and a `SITE_SECTIONS` dict in `report_factory.py`, plus
       `templates/site/site_report.md` with `{{ section_<x> }}` markers;
    4. add a `dc-site` command modelled on `main_equipment`. A site report can also receive, as the data of
       one section, the `sections/summary.llm.json` already generated for its critical equipment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 4 — Running the pipeline on another equipment of another site

    To analyse any equipment, only two variables change. Here: `EST1-INV07`, the most critical
    equipment of the site Est-1. From a terminal, the equivalent is:

    ```bash
    uv run dc equipment --site EST1 --equipment EST1-INV07
    ```
    """)
    return


@app.cell
def _(criticality, mo, select_equipment):
    SITE_2 = "EST1"
    EQUIPMENT_2 = "EST1-INV07"
    mo.md(f"Critical equipment of {SITE_2} → `{select_equipment(criticality, SITE_2)}`")
    return EQUIPMENT_2, SITE_2


@app.cell
def _(
    DEMO_DIR,
    EQUIPMENT_2,
    Path,
    SITE_2,
    TEMPLATES_DIR,
    USE_AZURE,
    criticality,
    df,
    mo,
):
    from contextlib import chdir

    from datachallengecapgemini.plot_factory import run_equipment_plotting
    from datachallengecapgemini.report_factory import run_equipment_reporting

    # Run from the demo folder with a relative site folder, so that printed paths stay short
    with chdir(DEMO_DIR):
        run_equipment_plotting(df, criticality, [EQUIPMENT_2], Path(SITE_2))
        run_equipment_reporting(
            [EQUIPMENT_2], Path(SITE_2), TEMPLATES_DIR, dry_run=not USE_AZURE
        )
    equipment_dir_2 = DEMO_DIR / SITE_2 / EQUIPMENT_2
    mo.md(
        "Files written:\n\n"
        + "\n".join(
            f"- `{_p.relative_to(DEMO_DIR / SITE_2).as_posix()}`"
            for _p in sorted(equipment_dir_2.rglob("*"))
            if _p.is_file()
        )
    )
    return (equipment_dir_2,)


@app.cell
def _(equipment_dir_2, mo, show_markdown):
    mo.vstack(
        [
            mo.image(src=equipment_dir_2 / "figures" / "png" / "cost_timeline.png"),
            show_markdown(equipment_dir_2 / "report.md", max_chars=4000),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To generate the reports of a whole site, run the command without a temporary folder
    (`uv run dc equipment --site EST1`): the outputs go to `equipment_reports/EST1/` (not versioned).
    """)
    return


if __name__ == "__main__":
    app.run()
