"""Local statistics, static plots and a compiled LaTeX report; no model calls.

Run from the project root:
    python -m functional_pipeline_test.statistical_report classified.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
SEVERITIES = ["mineure", "modérée", "sévère"]
REQUIRED = {"incident_id", "asset_type", "topic", "severity", "date"}


@dataclass(frozen=True)
class StatisticalReport:
    output_dir: Path
    tex_path: Path
    pdf_path: Path | None
    summary_path: Path
    tests_path: Path
    classified_path: Path


def prepare_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous inputs rather than silently losing or duplicating events."""
    missing = REQUIRED - set(incidents.columns)
    if missing:
        raise ValueError(f"Missing report columns: {', '.join(sorted(missing))}")
    if incidents.empty:
        raise ValueError("Cannot report on an empty incident table.")
    df = incidents.copy()
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required report columns must not contain missing values.")
    if df.incident_id.duplicated().any():
        raise ValueError("incident_id must be unique; deduplicate events first.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any():
        raise ValueError("Invalid incident dates.")
    for column in ["topic"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite numbers.")
    if (df.topic < -1).any() or (df.topic % 1 != 0).any():
        raise ValueError("topic must be an integer >= -1 (-1 means noise).")
    df["topic"] = df.topic.astype(int)
    if not df.severity.isin(SEVERITIES).all():
        raise ValueError(f"severity must use {SEVERITIES}")
    df["asset_type"] = df.asset_type.astype(str)
    if df.asset_type.str.strip().eq("").any():
        raise ValueError("asset_type must not be blank.")
    return df


def topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["asset_type", "topic"], sort=True)
        .agg(
            incidents=("incident_id", "size"),
            severe_share=("severity", lambda s: s.eq("sévère").mean()),
        )
        .reset_index()
    )


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Only topic versus severity, separately per asset; noise is excluded."""
    rows = []
    for asset, frame in df.groupby("asset_type", sort=True):
        classified = frame.loc[frame.topic.ne(-1)]
        table = pd.crosstab(classified.topic, classified.severity)
        row = {
            "asset_type": asset,
            "comparison": "Topic vs severity",
            "test": "Chi-square",
            "n": len(classified),
            "statistic": np.nan,
            "p_value": np.nan,
            "effect": np.nan,
            "effect_name": "Cramer's V",
            "status": "skipped",
            "note": "Needs at least two topics and two severity categories.",
        }
        if min(table.shape) >= 2:
            result = stats.chi2_contingency(table, correction=False)
            sparse = bool(
                (result.expected_freq < 5).any() or (table.to_numpy() < 5).any()
            )
            if sparse:
                result = stats.chi2_contingency(
                    table,
                    correction=False,
                    method=stats.MonteCarloMethod(
                        n_resamples=9999, rng=np.random.default_rng(42)
                    ),
                )
            row.update(
                test="Chi-square (Monte Carlo)" if sparse else "Chi-square",
                statistic=result.statistic,
                p_value=result.pvalue,
                effect=np.sqrt(
                    result.statistic / (len(classified) * (min(table.shape) - 1))
                ),
                status="ok",
                note="Sparse table: 9,999 conditional simulations, seed 42."
                if sparse
                else "Asymptotic p-value.",
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    valid = result.p_value.notna()
    result["q_value"] = np.nan
    if valid.any():
        result.loc[valid, "q_value"] = stats.false_discovery_control(
            result.loc[valid, "p_value"], method="bh"
        )
    result["significant_fdr_05"] = result.q_value.lt(0.05)
    return result


def generate_plots(df: pd.DataFrame, output_dir: Path) -> list[tuple[str, str]]:
    """Save figures as PDF and PNG without a display server or browser."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for asset_index, (asset, frame) in enumerate(df.groupby("asset_type", sort=True)):
        topics = sorted(frame.topic.unique())
        labels = ["Noise (-1)" if t == -1 else str(t) for t in topics]

        def make(title, height=4.5, asset=asset):
            fig = Figure(figsize=(9, height), layout="constrained")
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.set_title(f"{asset}: {title}")
            return fig, ax

        def save(fig, name, caption, asset=asset, asset_index=asset_index):
            stem = f"asset_{asset_index}_{name}"
            for extension in ["pdf", "png"]:
                fig.savefig(plot_dir / f"{stem}.{extension}", dpi=160)
            figures.append((f"plots/{stem}.pdf", f"{asset}: {caption}"))
            fig.clear()

        counts = frame.groupby("topic").size().reindex(topics)
        fig, ax = make("Incidents by topic", max(4.5, len(topics) * 0.25))
        ax.barh(labels, counts)
        ax.set_xlabel("Incident count")
        save(
            fig,
            "frequency",
            "Incident frequency. Topic -1 contains unclassified noise.",
        )

        table = pd.crosstab(frame.topic, frame.severity).reindex(
            index=topics, columns=SEVERITIES, fill_value=0
        )
        percentages = table.div(table.sum(axis=1), axis=0) * 100
        fig, ax = make("Severity composition")
        bottom = np.zeros(len(topics))
        for severity, color in zip(SEVERITIES, ["#4c9f70", "#e3a53b", "#c95454"]):
            ax.bar(
                labels,
                percentages[severity],
                bottom=bottom,
                label=severity,
                color=color,
            )
            bottom += percentages[severity].to_numpy()
        ax.set(ylabel="Percent within topic", xlabel="Topic", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, "severity", "Severity proportions normalized within each topic.")

    return figures


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in str(value))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    def line(row):
        return " & ".join(latex_escape(v) for v in row) + r" \\" + "\n"

    return (
        r"\begin{longtable}{"
        + "l" * len(headers)
        + "}\n\\toprule\n"
        + line(headers)
        + "\\midrule\n\\endhead\n"
        + "".join(line(row) for row in rows)
        + "\\bottomrule\n\\end{longtable}\n"
    )


def write_latex(df, tests, figures, output_dir: Path, title: str) -> Path:
    sections = [
        r"\section{Dataset}",
        latex_escape(
            f"{len(df):,} unique incidents, {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}. "
            f"Unclassified noise: {df.topic.eq(-1).sum():,} events. "
            "Topic identifiers are local to each asset type; all tests are stratified by asset type."
        ),
    ]
    sections += [
        r"\section{Topic versus severity test}",
        (
            "The report contains three analysis items per asset: topic frequency, "
            "severity composition, and a topic--severity independence test. "
            "The test excludes noise (-1); the two plots retain it. "
            "Benjamini--Hochberg q-values adjust p-values across asset types "
            "(with one asset, q equals p). A q-value below 0.05 is an exploratory "
            "signal, not evidence of causation. Insufficient-data tests are skipped."
        ),
    ]

    def number(v, *, probability=False):
        if pd.isna(v):
            return "--"
        return "<1e-300" if probability and v == 0 else f"{v:.4g}"

    for r in tests.itertuples():
        sections.append(
            r"\subsection{" + latex_escape(f"{r.asset_type}: {r.comparison}") + "}"
        )
        sections.append(latex_escape(f"{r.test}; n={r.n}; status={r.status}. {r.note}"))
        if r.status == "ok":
            sections.append(
                _table(
                    ["Statistic", "p", "q (BH)", r.effect_name, "q < 0.05"],
                    [
                        [
                            number(r.statistic),
                            number(r.p_value, probability=True),
                            number(r.q_value, probability=True),
                            f"{r.effect:.4g}",
                            "yes" if r.significant_fdr_05 else "no",
                        ]
                    ],
                )
            )
    sections += [
        r"\section{Interpretation limits}",
        (
            "Incidents from the same equipment or site can be dependent; the basic tests assume independent observations. "
            "Clusters were learned from the same incident texts and may have been selected using severity information, "
            "so these results are exploratory and do not independently validate the classifier. "
            "This report describes only the supplied rows, which may be a sample. "
            "The statistics and report generation use deterministic local code and make no LLM calls."
        ),
        r"\section{Plots}",
    ]
    for path, caption in figures:
        sections.append(
            r"\begin{figure}[p]\centering"
            + "\n"
            + r"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{"
            + path
            + "}\n"
            + r"\caption{"
            + latex_escape(caption)
            + "}\n\\end{figure}\n\\clearpage"
        )
    template = (
        Path(__file__).with_name("report_template.tex").read_text(encoding="utf-8")
    )
    source = template.replace("%%TITLE%%", latex_escape(title)).replace(
        "%%BODY%%", "\n\n".join(sections)
    )
    tex = output_dir / "report.tex"
    tex.write_text(source, encoding="utf-8")
    return tex


def compile_latex(tex_path: Path) -> Path:
    """Recompile with shell escape disabled and retain diagnostics on failure."""
    tex_path = tex_path.resolve()
    if shutil.which("latexmk"):
        commands = [
            [
                "latexmk",
                "-norc",
                "-pdf",
                "-g",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                tex_path.name,
            ]
        ]
    elif shutil.which("pdflatex"):
        commands = [
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                tex_path.name,
            ]
        ] * 2
    else:
        raise RuntimeError(
            "No LaTeX compiler found. Install latexmk texlive-latex-base texlive-latex-recommended, then rerun."
        )
    log = tex_path.with_name("compile.log")
    with log.open("w", encoding="utf-8") as stream:
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=tex_path.parent,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(
                    f"LaTeX compilation failed; see {log}. Sources and plots were retained."
                ) from error
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError(f"Compiler produced no PDF; see {log}")
    return pdf


def generate_report_content(
    incidents: pd.DataFrame,
    output_dir: str | Path,
    *,
    title: str = "Classified incident statistical report",
) -> StatisticalReport:
    """Generate the two plots, test results and LaTeX source, without compiling."""
    df = prepare_incidents(incidents)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, tests = topic_summary(df), statistical_tests(df)
    summary_path, tests_path = (
        output_dir / "topic_summary.csv",
        output_dir / "statistical_tests.csv",
    )
    classified_path = output_dir / "classified_incidents.csv"
    df.to_csv(classified_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    figures = generate_plots(df, output_dir)
    tex = write_latex(df, tests, figures, output_dir, title)
    return StatisticalReport(
        output_dir, tex, None, summary_path, tests_path, classified_path
    )


def generate_statistical_report(
    incidents: pd.DataFrame,
    output_dir: str | Path,
    *,
    compile_pdf: bool = True,
    title: str = "Classified incident statistical report",
) -> StatisticalReport:
    """Generate report content first, then optionally compile its PDF."""
    report = generate_report_content(incidents, output_dir, title=title)
    if compile_pdf:
        report = replace(report, pdf_path=compile_latex(report.tex_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classified_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "classified")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Generate source, tables and plots only.",
    )
    args = parser.parse_args()
    report = generate_statistical_report(
        pd.read_csv(args.classified_csv),
        args.output_dir,
        compile_pdf=not args.no_compile,
    )
    print(report.pdf_path or report.tex_path)


if __name__ == "__main__":
    main()
