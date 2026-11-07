# Equipment-level report pipeline

Data Challenge Capgemini x Mines Paris 2026, PV incidents.

A runnable walkthrough of everything below is in
[`equipment_pipeline_walkthrough.py`](equipment_pipeline_walkthrough.py)
(marimo: `uv run marimo edit equipment_analysis/equipment_pipeline_walkthrough.py`).

## Contents

1. [Goal](#1-goal)
2. [Pipeline overview](#2-pipeline-overview)
3. [Step by step](#3-step-by-step)
   - 3.1 Selection of critical equipment
   - 3.2 Statistics + facts (`equipment_stats.py`)
   - 3.3 Figures + JSON (`plot_factory.py`)
   - 3.4 One Azure agent per section (`report_factory.py`)
   - 3.5 Report assembly (`templates/equipment/equipment_report.md`)
4. [Full example: OUEST1-INV10](#4-full-example-ouest1-inv10)
5. [Transposing the pipeline to other scenarios](#5-transposing-the-pipeline-to-other-scenarios)
6. [Repository layout](#6-repository-layout)
7. [Useful commands](#7-useful-commands)

---

## 1. Goal

Generate, for each critical piece of equipment of a site (a PV inverter and the
DC field upstream of it), a well-structured markdown report: key figures, cost
history, weather exposure, failure modes, root-cause hypotheses, prioritised
recommendations and conclusion.

Three principles drive the design:

- **a) The LLM computes nothing.** Azure OpenAI (gpt-4.1-nano) only writes.
  Every figure, ratio, change, priority and risk level is computed in Python and
  injected into the prompt.
- **b) One specialised agent per section.** Each section of the report has its
  own agent (its own role, questions and data). An agent only sees the data of
  its section.
- **c) Every figure comes with its JSON.** The agent that comments a chart
  receives exactly the data the reader sees.

## 2. Pipeline overview

```
  data/incidents_clusterised_solar.csv  (1 row = 1 incident, with its
            |                            cluster = "failure mode")
            v
  +--------------------------------------------------------------+
  | [1] SELECTION          equipment_stats.compute_criticality   |
  |     0-100 score vs site and vs fleet -> P1 / P2 equipment    |
  +--------------------------------------------------------------+
            |   for each critical equipment of the site
            v
  +--------------------------------------------------------------+
  | [2] STATISTICS         equipment_stats.build_equipment_stats |
  |     local numbering #1..#n, figures, trends, priorities      |
  |     + "facts": ready-made, verified sentences                |
  +--------------------------------------------------------------+
            |
            v
  +--------------------------------------------------------------+
  | [3] FIGURES + JSON     plot_factory.EquipmentPlotFactory     |
  |     plot_cost_timeline   -> figures/{png,html}/cost_timeline |
  |                             json/cost_timeline.json          |
  |     plot_downtime_weather-> figures/.../downtime_weather     |
  |                             json/downtime_weather.json       |
  |     plot_cluster_weather -> figures/.../cluster_weather      |
  |                             json/cluster_weather.json        |
  |     + json/stats.json  (every statistic of step 2)           |
  +--------------------------------------------------------------+
            |   report_factory only reads json/ (never the CSV)
            v
  +--------------------------------------------------------------+
  | [4] ONE AGENT PER SECTION   report_factory.section_*         |
  |                                                              |
  |   key_figures ----> "benchmarking" agent          --+        |
  |   methodology ----> (fixed text, no agent)          |        |
  |   cost_timeline --> "cost & recurrence" agent       |        |
  |   weather --------> "weather exposure" agent        |        |
  |   failure_modes --> "PV failure analysis" agent     |  each  |
  |   root_causes ----> "RCA" agent  -------+           |  one   |
  |   recommendations > "O&M planner" agent <-----------+  saves |
  |   conclusion -----> "reliability manager" agent     |  its   |
  |   summary --------> "executive summary" agent       |  .md   |
  |   incident_register (fixed table, no agent)       --+        |
  |                                                              |
  |   Every answer is checked in Python and, if there are        |
  |   problems, the agent gets one correction round.             |
  +--------------------------------------------------------------+
            |   sections/<section>.md
            v
  +--------------------------------------------------------------+
  | [5] ASSEMBLY  templates/equipment/equipment_report.md        |
  |               -> report.md                                   |
  |     (optional: --pdf with pandoc, via pdf_export.py)         |
  +--------------------------------------------------------------+
```

Outputs (generated files, ignored by git — run the command to get them):

| Path | Content |
| --- | --- |
| `equipment_reports/<SITE>/criticality.csv` | scores of every equipment of the site |
| `equipment_reports/<SITE>/<EQUIPMENT>/report.md` | final report |
| `.../figures/png/*.png`, `figures/html/*.html` | the 3 plotly figures (static and interactive) |
| `.../json/<figure>.json`, `json/stats.json` | data of each figure + every statistic |
| `.../sections/<section>.md` | each section on its own |
| `.../sections/<section>.prompt.md`, `.llm.json` | exact prompt and cached Azure answer of each agent |
| `.../sections/checks.txt` | problems left after the checks (empty = clean) |

## 3. Step by step

### 3.1 Selection of critical equipment (`equipment_stats.compute_criticality`)

For each equipment, 4 indicators are computed over **all** its incidents:

| Indicator | Weight |
| --- | ---: |
| number of incidents | 1 |
| severe incidents | 2 |
| total cost | 1 |
| downtime hours | 1 |

Each indicator becomes a leave-one-out robust z-score (median and MAD of the
*other* equipment, clipped to ±3). The weighted mean is mapped to 0-100 with the
normal cumulative distribution: **50 = typical equipment, ≥ 75 = clearly above
the others**.

The score is computed twice:

- `score_site`: against the other equipment of the same site;
- `score_fleet`: against all solar equipment of the fleet.

Criterion (the "criterion 2" we chose): **critical if `score_site` ≥ 75 or
`score_fleet` ≥ 75**; P1 if both hold, P2 if only one does.

Risk level (deterministic): P1 → critical, P2 → high; otherwise medium when
max(score) ≥ 50, low otherwise.

### 3.2 Statistics + facts (`equipment_stats.build_equipment_stats`)

Only moderate and severe incidents enter the detailed analysis. They are
numbered `#1`, `#2`, … in chronological order **per equipment** (the global
`incident_id` never reaches the report; the `#k → incident_id` mapping is kept
in `stats.json`, under `incident_mapping`).

Incidents of the clustering's noise cluster (`C_NOISE`, or without a cluster)
are relabelled **"Unclassified (no failure mode)"**: they count in every total,
the timeline and the register, but form no failure mode (no priority, not in
the weather matrix nor in the root causes). `stats.json` summarises them under
`unclassified_incidents`, with a matching sentence in the cost facts.

`stats.json` contains, among other blocks:

| Block | Content |
| --- | --- |
| `equipment` | priority, risk, scores, why it was selected |
| `comparison_with_site_median` | equipment vs median equipment of the site (+ ratio) |
| `key_figures` | totals, means, % of cost from severe incidents |
| `yearly_trend` / `seasonality` | figures per year (change already computed) and per season |
| `most_costly_incidents` | top 3 with their % of the cost |
| `close_sequences_within_30_days` | incidents chained less than 30 days apart |
| `failure_modes` | per failure mode: incidents #k, cost, %, recurrences < 90 days, weather, priority |
| `register` | incident register (annex) |
| `facts` | ready-made sentences by topic: benchmark / cost / trend / weather / priorities |

The **facts** exist because the small model used to mix up figures (for
example "most costly incident" vs "most costly failure mode", or the ratio of
severe incidents vs the ratio of all incidents). It now receives the exact
sentence.

Priority of a failure mode (and of every recommendation targeting it):

| Priority | Deadline | Rule |
| --- | --- | --- |
| P1 | 1 month | ≥ 1 severe incident **and** (≥ 20 % of the cost **or** a recurrence within 90 days) |
| P2 | 6 months | ≥ 1 severe incident **or** ≥ 10 % of the cost |
| P3 | yearly plan | everything else |

### 3.3 Figures + JSON (`plot_factory.EquipmentPlotFactory`)

Like the fleet `PlotFactory`, each `plot_*` method returns a plotly figure
**and** a dictionary — here the exact data the figure draws. The chart and the JSON given
to the agent therefore come from the same computation and cannot diverge.
Adding a figure only takes a new `plot_*` method.

| Method | Figure |
| --- | --- |
| `plot_cost_timeline` | cost of each incident over time; colour = failure-mode family, shape = failure mode, filled = severe, hollow = moderate, top 3 labelled |
| `plot_downtime_weather` | downtime hours per incident and weather condition (black tick = median) |
| `plot_cluster_weather` | heatmap: number of incidents per failure mode × weather |

Colours do not depend on exact cluster names (the clustering step renames the
clusters at each run): `failure_mode_family` assigns each failure mode to a
family by keywords (accents and case ignored, e.g. *incendie* → Fire,
*nettoyage* → Soiling, *jonction* / *connecteur* → Junction box), and
`failure_mode_styles` gives the markers within a family in alphabetical order
over the whole dataset, so a failure mode has the same style on every
equipment. Unmatched names are grey; unclassified incidents are grey crosses.

`run_equipment_plotting` writes, for each equipment, `figures/png/` (PNG
export through kaleido, like `run_plotting`), `figures/html/` (interactive),
`json/<figure>.json` and `json/stats.json`.

### 3.4 One Azure agent per section (`report_factory.py`)

Every section goes through `SectionContext.ask()`, which builds:

- **System** = `"You are <agent role>"` + `SHARED_RULES` (shared rules: never
  calculate, copy figures from `facts`, cite only existing `#k` from the right
  failure mode, never generalise weather patterns across failure modes, flag
  samples < 3 as "indicative", never change priorities, never mention datasets /
  clustering / AI).
- **User** = `You are writing the section "<title>" of the report of
  <equipment> (site ...)` + numbered questions + JSON with **only** the data of
  that section.
- **Output** = JSON with a strict schema (`json_schema`), one field per question.

After each answer, Python checks:

- numbers that do not appear in the input data (invented figure);
- `#k` that do not exist or do not belong to the cited failure mode;
- section-specific checks (e.g. a P1 failure mode with no action, or a
  recommendation whose failure mode is not explained by the root-cause
  hypothesis H1, H2… it claims to address: every action must come from one
  of the equipment's hypotheses, there is no generic action list in the prompt).

If there are problems, the prompt is sent again with the list of problems (one
correction round). Anything left is written to `sections/checks.txt`.

**Cache:** the answer is stored in `sections/<section>.llm.json` with the hash
of the prompt. Azure is called again only if the prompt changes or with
`--force <section>`.

**Generation order** (not the order of the report):
`key_figures → methodology → cost_timeline → weather → failure_modes →
root_causes → recommendations → conclusion → summary → incident_register`.

**Chaining:** `recommendations` receives the hypotheses of `root_causes`;
`conclusion` and `summary` receive the already prioritised actions. That is why
the executive summary is written last although it appears first.

What Python decides and what the agent writes:

| Python | Agent |
| --- | --- |
| every table, priorities, risk level | the analysis paragraphs |
| confidence of each root cause (high ≥ 5 supporting incidents, medium ≥ 3, low < 3) | the mechanism of each failure mode |
| order of the recommendations | the hypotheses and the concrete actions |
| methodology, annex | the title and the executive summary |

### 3.5 Assembly

`templates/equipment/equipment_report.md` only contains `{{ section_<x> }}`
markers.
`fill_template` (the same function used by the teammates' fleet pipeline)
replaces each marker with the section's markdown and fixes the image paths
(`../figures/` → `figures/`).

## 4. Full example: OUEST1-INV10

*Real figures from the generated report.* Command: `uv run dc equipment --site OUEST1`

**[1] Selection.** OUEST1 has 10 equipment; 6 are critical (INV06, INV03 = P1;
INV10, INV08, INV04, INV02 = P2). INV10: score vs site 64.3 (< 75), score vs
fleet 86.2 (≥ 75) → **P2**, risk "high". Rank 3 of 10 on its site.

**[2] Statistics.** 29 incidents in total; 15 moderate/severe → numbered
#1..#15 (9 moderate, 6 severe), 206 762 EUR, 414 h of downtime. Generated facts
(excerpt):

> "Severe incidents: 6 on this equipment vs 4.5 for the median equipment of the
> site (ratio ×1.33, ABOVE the site median)."
>
> "The most costly single incident is #3 (Dégradation et court-circuit câblage,
> sévère, 2023-11-21): 49 142 EUR, 23.8% …"
>
> "The most costly failure mode is Incendie bâtiment support PV: 51 131 EUR over
> 2 incident(s), 24.7% of the cost."
>
> "2025: 4 incident(s) (0 severe), 19 821 EUR (-79.5% vs previous year)."

Priorities: *Incendie bâtiment support PV* → P1 (2 severe + 24.7 % of the cost
+ 1 recurrence < 90 d); *Défaillance chaîne et boîtier de jonction* → P2;
*Dommages superficiels modules PV* → P3; …

**[3] Figures + JSON.** `figures/png/cost_timeline.png` + `json/cost_timeline.json`,
where each point is, for example:

```json
{"ref": "#3", "date": "2023-11-21", "severity": "sévère",
 "failure_mode": "Dégradation et court-circuit câblage",
 "cost_eur": 49142, "downtime_h": 82.4,
 "weather_condition": "temps calme",
 "days_since_previous_incident": 187, "...": "..."}
```

and `"labelled_incidents": ["#3", "#10", "#4"]` (the 3 labelled on the figure).

**[4] Agent of the section "3. Cost timeline".**

- *System:* "You are a reliability engineer who analyses the cost history and
  recurrence of incidents on PV inverters." + `SHARED_RULES`
- *User:* 4 questions — (1) yearly evolution (comment on every year, the last
  too), (2) which incidents dominate the cost, (3) close sequences /
  recurrences, (4) seasonality — + JSON: `cost_timeline.json`, `yearly_trend`,
  `seasonality`, `most_costly_incidents`, `close_sequences`, trend and cost
  facts (about 3 800 input tokens, 250 output tokens).
- *Answer* (excerpt of `sections/cost_timeline.llm.json`):
  > "The most costly incident was #3, involving degradation and
  > short-circuiting of wiring, accounting for nearly a quarter of the total
  > incident costs. The failure mode that dominates costs is incendie bâtiment
  > support PV …"
- *Python check:* no invented figure, no wrong `#k` → `sections/checks.txt` is
  empty.
- → `sections/cost_timeline.md` = title + figure + yearly table (Python) + 4
  paragraphs from the agent.

In v2 (a single prompt for the whole report) the model attributed the fire to
the most costly incident and ignored the 2025 drop; with the facts and a
dedicated agent, both errors are gone.

**[5] Assembly.** `equipment_reports/OUEST1/OUEST1-INV10/report.md` = summary +
key_figures + methodology + cost_timeline + weather + failure_modes +
root_causes + recommendations + conclusion + annex. Cost of the 6 OUEST1
reports: about 0.02 USD (~116 000 input tokens and ~13 500 output tokens with
gpt-4.1-nano, 8 agents per equipment + correction rounds; the total is printed
at the end of each run).

## 5. Transposing the pipeline to other scenarios

The pipeline does not depend on the unit being a "piece of equipment". Each
scenario only needs five things to be defined:

| | Definition | Examples |
| --- | --- | --- |
| 1 | the **unit** of analysis | equipment, site, technology, technician |
| 2 | its **peers** for comparison | compared with what (median of what) |
| 3 | the **statistics + facts** | which figures each section needs |
| 4 | the **figures + their JSON** | what is drawn and from which data |
| 5 | the **sections** | role, questions, data, schema, checks |

**Reused as is** (`report_factory.py`): `get_azure_client`, `call_azure`
(strict `json_schema`, 429 retries), `SectionContext.ask` (prompt, cache by
hash, checks on figures and `#k`, correction round), `_md_table`,
`fill_template`, `pdf_export`.

**To generalise first:** `SHARED_RULES` mentions "PV inverter and the DC
field": turn that first sentence into a parameter (description of the unit and
of the reader).

### Example A — site-level analysis (instead of equipment-level)

1. **Unit:** a site, e.g. "Centrale PV Ouest-1".
2. **Peers:** the other sites of the fleet (median site), and within the site,
   its equipment (to point out the worst ones).
3. **Stats:** `site_stats.py` (or a `build_site_stats` function) with:
   - site vs median site comparison (ratios + facts);
   - ranking of the site's equipment = `criticality.csv`, which already exists
     (`compute_criticality` filtered by site);
   - failure modes of the site and on how many equipment they appear
     ("systemic issue" when on many equipment, "local issue" when concentrated
     on one);
   - yearly trend, seasonality, weather of the site;
   - `#k` numbering per site.
4. **Figures:** a `SitePlotFactory` with `plot_*` methods returning
   (figure, data), e.g. total cost per equipment of the site (bars), heatmap
   equipment × failure mode, cost timeline of the site by failure mode,
   downtime vs weather on the site.
5. **Sections:** `section_site_key_figures`, `section_equipment_ranking`
   (agent: "which equipment concentrates the risk"),
   `section_systemic_vs_local` (site-level RCA agent), `section_site_weather`,
   `section_site_recommendations` (site actions: O&M contract, site-wide
   thermography campaigns, spare parts…), conclusion, summary. Methodology and
   annex remain fixed text.
   - plus `templates/site_report.md` with `{{ section_<x> }}` markers;
   - plus a `dc site --site OUEST1` subcommand modelled on `dc equipment`.

Natural chaining: the site report can receive, as data for one section, the
summaries already generated for its critical equipment
(`sections/summary.llm.json` of each equipment), without recomputing anything.

### Example B — wind (implemented)

`uv run dc equipment --asset wind --site SUD1` runs the same pipeline on the
wind turbines of `data/incidents_clusterised_wind.csv`. Only two things depend
on the asset type: the asset profile (`ASSET_PROFILES` in `equipment_stats.py`:
how the equipment, plants and fleet are named in the reports and in the
agents' roles and shared rules) and the failure-mode families of the figures
(`FAILURE_MODE_FAMILIES["wind"]` in `plot_factory.py`: blades, mechanical /
drivetrain, electrical, overheating, high-wind safety stops, structure /
foundation). Statistics, criticality, checks and assembly are unchanged.

### Example C — fleet, technology or technician

Same recipe: the unit is the whole fleet (peers = sites), a failure mode
(peers = other failure modes) or a technician (peers = other technicians;
careful with the reading: it measures workload, not blame).

> **Golden rule when transposing:** if a figure appears in the report, it must
> come from Python (stats or figure JSON). If the agent needs a comparison, give
> it already computed, ideally as a sentence in `facts`.

## 6. Repository layout

```
DataChallengeCapGemini/
|
|-- README.md                    project overview and setup
|-- pyproject.toml / uv.lock     dependencies (uv); command `dc`
|-- .python-version              Python version pin
|-- .env.example                 Azure credentials template
|-- equipment_analysis/          this documentation and its walkthrough
|-- functional_pipeline_test/    functional clustering and statistics pipeline
|-- map-reduce__exploratory/      recommendation workflow and site statistics
|-- topic_modeling/              BERTopic notebook and noise exploration
|-- tests/                        automated tests
|-- .gitignore
|
|-- src/datachallengecapgemini/  Python package and `dc` CLI
|
|-- templates/
|   |-- solar_report.md
|   |-- wind_report.md
|   `-- equipment/equipment_report.md
|
|-- data/                        local inputs and generated outputs (not tracked)
|-- filled_templates/            generated fleet reports (not tracked)
`-- equipment_reports/           generated equipment reports (not tracked)
```

The fleet/equipment CLI implementation lives in `src/datachallengecapgemini/`.
The topic-modeling and map-reduce directories contain separate experiments.
Local data, credentials and generated outputs are excluded from Git.

## 7. Useful commands

| Command | Effect |
| --- | --- |
| `uv sync` | install / register commands |
| `uv run dc equipment --site OUEST1` | critical equipment of the site |
| `uv run dc equipment --site OUEST1 --dry-run` | figures, JSON and prompts, no Azure call |
| `uv run dc equipment --site OUEST1 --equipment OUEST1-INV10` | a single equipment |
| `uv run dc equipment --site OUEST1 --all-equipment` | every equipment of the site |
| `uv run dc equipment --site OUEST1 --force root_causes recommendations` | regenerate only those sections |
| `uv run dc equipment --site OUEST1 --language French` | text in French |
| `uv run dc equipment --site OUEST1 --pdf` | also a PDF with pandoc |

Requirement: `.env` with `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_VERSION` and `AZURE_OPENAI_CHAT_DEPLOYMENT` (see
`../.env.example`).
