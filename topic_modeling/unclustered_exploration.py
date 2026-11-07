import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return Path, mo, pd


@app.cell
def _(Path, pd):
    asset_type = "wind"
    csv_path = Path(__file__).resolve().parent.parent / "data" / f"incidents_clusterised_{asset_type}.csv"
    df = pd.read_csv(csv_path)
    df_unclustered = df[df["cluster_code"] == "C_NOISE"].copy()

    print(f"Number of unclustered incidents: {len(df_unclustered)}")
    print(f"Number of clustered incidents: {len(df) - len(df_unclustered)}")
    return df, df_unclustered


@app.cell
def _(df):
    df["cluster_name"].unique()
    return


@app.cell
def _():
    # mo.vstack(
    # 	[
    # 		mo.md(
    # 			f"## Unclustered solar incident reports\n"
    # 			f"Showing all {len(df_unclustered)} reports marked `C_NOISE`."
    # 		),
    # 		mo.ui.table(
    # 			df_unclustered,
    # 			selection=None,
    # 			page_size=max(1, len(df_unclustered)),
    # 		),
    # 	]
    # )
    return


@app.cell
def _(df_unclustered):
    print(df_unclustered["site"].value_counts())
    return


@app.cell
def _(df_unclustered):
    import plotly.express as px

    fig_unclustered_costs = px.histogram(
        df_unclustered,
        x="estimated_cost_eur",
        nbins=30,
        title="Répartition des coûts des incidents non clusterisés",
        labels={
            "estimated_cost_eur": "Coût estimé (€)",
            "count": "Nombre d'incidents",
        },
    )
    fig_unclustered_costs.show()
    return


@app.cell
def _():
    # _incidents_with_cost = df_unclustered.copy()
    # _incidents_with_cost["_cost"] = pd.to_numeric(
    #     _incidents_with_cost["estimated_cost_eur"], errors="coerce"
    # )
    # _incidents_with_cost = _incidents_with_cost.dropna(subset=["_cost"])
    # if _incidents_with_cost.empty:
    #     raise ValueError("Aucun coût valide dans les incidents non clusterisés.")

    # _target_costs = [
    #     _incidents_with_cost["_cost"].min()
    #     + index
    #     * (
    #         _incidents_with_cost["_cost"].max()
    #         - _incidents_with_cost["_cost"].min()
    #     )
    #     / 4
    #     for index in range(5)
    # ]
    # _example_sections = []
    # for _target_cost in _target_costs:
    #     _closest_incidents = (
    #         _incidents_with_cost.assign(
    #             _distance=lambda frame: (frame["_cost"] - _target_cost).abs()
    #         )
    #         .nsmallest(2, "_distance")
    #         .loc[:, ["incident_id", "_cost", "text"]]
    #         .rename(columns={"_cost": "estimated_cost_eur"})
    #     )
    #     _example_sections.extend(
    #         [
    #             mo.md(f"### Coût cible : {_target_cost:,.2f} €"),
    #             mo.ui.table(_closest_incidents, selection=None, page_size=2),
    #         ]
    #     )

    # mo.vstack(_example_sections)
    return


@app.cell
def _():
    # import numpy as np
    # import plotly.graph_objects as cost_go
    # from sklearn.mixture import GaussianMixture

    # df_unclustered_by_cost = df_unclustered.copy()
    # _costs = pd.to_numeric(
    #     df_unclustered_by_cost["estimated_cost_eur"], errors="coerce"
    # )
    # _valid_cost_mask = np.isfinite(_costs)
    # _valid_costs = _costs.loc[_valid_cost_mask].to_numpy().reshape(-1, 1)
    # if len(_valid_costs) < 2:
    #     raise ValueError("Il faut au moins deux coûts valides pour la clusterisation.")

    # _candidate_models = []
    # for _n_components in range(1, min(6, len(_valid_costs)) + 1):
    #     _model = GaussianMixture(
    #         n_components=_n_components,
    #         random_state=42,
    #         n_init=5,
    #     ).fit(_valid_costs)
    #     _candidate_models.append((_model.bic(_valid_costs), _model))
    # _best_bic, _best_model = min(_candidate_models, key=lambda _candidate: _candidate[0])

    # _raw_labels = _best_model.predict(_valid_costs)
    # _component_order = np.argsort(_best_model.means_.ravel())
    # _label_by_component = {
    #     _component_id: f"Groupe coût {rank + 1}"
    #     for rank, _component_id in enumerate(_component_order)
    # }
    # _labels = [_label_by_component[_label] for _label in _raw_labels]
    # df_unclustered_by_cost["cost_cluster"] = pd.NA
    # df_unclustered_by_cost.loc[_valid_cost_mask, "cost_cluster"] = _labels

    # _cluster_summary = (
    #     df_unclustered_by_cost.loc[_valid_cost_mask]
    #     .groupby("cost_cluster")["estimated_cost_eur"]
    #     .agg(incident_count="count", average_cost_eur="mean")
    #     .reset_index()
    # )
    # print(
    #     f"Nombre de groupes retenu par BIC : {_best_model.n_components} "
    #     f"(BIC = {_best_bic:.2f})."
    # )
    # if (~_valid_cost_mask).any():
    #     print(f"Incidents sans coût valide : {(~_valid_cost_mask).sum()}.")
    # print(_cluster_summary.to_string(index=False))

    # _means = _best_model.means_.ravel()
    # _stds = np.sqrt(_best_model.covariances_.reshape(-1))
    # _weights = _best_model.weights_
    # _x = np.linspace(
    #     min(_valid_costs.min(), max(0, (_means - 4 * _stds).min())),
    #     max(_valid_costs.max(), (_means + 4 * _stds).max()),
    #     500,
    # )
    # _hist_counts, _bin_edges = np.histogram(_valid_costs.ravel(), bins=30)
    # _bin_width = _bin_edges[1] - _bin_edges[0]
    # _component_densities = []
    # _fig_cost_mixture = cost_go.Figure()
    # _fig_cost_mixture.add_trace(
    #     cost_go.Bar(
    #         x=(_bin_edges[:-1] + _bin_edges[1:]) / 2,
    #         y=_hist_counts,
    #         width=_bin_width,
    #         marker_color="lightgray",
    #         name="Incidents observés",
    #         opacity=0.75,
    #     )
    # )
    # for _component_id in _component_order:
    #     _density = (
    #         _weights[_component_id]
    #         / (_stds[_component_id] * np.sqrt(2 * np.pi))
    #         * np.exp(
    #             -0.5
    #             * ((_x - _means[_component_id]) / _stds[_component_id]) ** 2
    #         )
    #     )
    #     _expected_counts = _density * len(_valid_costs) * _bin_width
    #     _component_densities.append(_expected_counts)
    #     _fig_cost_mixture.add_trace(
    #         cost_go.Scatter(
    #             x=_x,
    #             y=_expected_counts,
    #             mode="lines",
    #             name=(
    #                 f"{_label_by_component[_component_id]} "
    #                 f"(moyenne : {_means[_component_id]:,.0f} €)"
    #             ),
    #         )
    #     )
    # _fig_cost_mixture.add_trace(
    #     cost_go.Scatter(
    #         x=_x,
    #         y=np.sum(_component_densities, axis=0),
    #         mode="lines",
    #         line={"color": "black", "width": 3},
    #         name="Mélange total",
    #     )
    # )
    # _fig_cost_mixture.update_layout(
    #     title="Clusterisation des incidents C_NOISE par coût (mélange gaussien)",
    #     xaxis_title="Coût estimé (€)",
    #     yaxis_title="Nombre d'incidents (effectif par classe)",
    #     barmode="overlay",
    # )
    # _fig_cost_mixture.show()
    return


@app.cell
def _(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_cluster_costs = (
        df.groupby(["cluster_code", "cluster_name"], dropna=False)
        .agg(
            incident_count=("estimated_cost_eur", "count"),
            average_cost_eur=("estimated_cost_eur", "mean"),
            total_cost_eur=("estimated_cost_eur", "sum"),
        )
        .reset_index()
    )
    df_cluster_costs["_is_noise"] = df_cluster_costs["cluster_code"] == "C_NOISE"
    df_cluster_costs = (
        df_cluster_costs.sort_values(["_is_noise", "cluster_code"])
        .drop(columns="_is_noise")
        .reset_index(drop=True)
    )

    fig_cluster_costs = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Coût total par cluster", "Coût moyen par incident"),
    )
    for row, metric, color in [
        (1, "total_cost_eur", "indianred"),
        (2, "average_cost_eur", "steelblue"),
    ]:
        fig_cluster_costs.add_trace(
            go.Bar(
                x=df_cluster_costs["cluster_code"],
                y=df_cluster_costs[metric],
                marker_color=color,
                customdata=df_cluster_costs[["cluster_name", "incident_count"]],
                hovertemplate=(
                    "%{customdata[0]} (%{x})<br>"
                    "Coût : %{y:,.2f} €<br>"
                    "Incidents : %{customdata[1]}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        fig_cluster_costs.update_yaxes(title_text="Euros (€)", row=row, col=1)

    fig_cluster_costs.update_xaxes(title_text="Cluster", row=2, col=1)
    fig_cluster_costs.update_layout(
        title_text="Coûts des incidents par cluster, bruit inclus",
        showlegend=False,
    )
    fig_cluster_costs.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mini analyse Azure
    """)
    return


@app.cell
def _(mo):
    unclustered_analysis_button = mo.ui.run_button(
        label="Demander à Azure d'analyser les incidents non clusterisés"
    )
    unclustered_analysis_button
    return (unclustered_analysis_button,)


@app.cell
def _(df, mo, unclustered_analysis_button):
    import json
    import os

    import numpy as sample_np

    if not unclustered_analysis_button.value:
        unclustered_analysis_output = mo.md(
            "Cliquez sur le bouton ci-dessus pour lancer l'analyse Azure."
        )
    else:
        import truststore
        from dotenv import load_dotenv
        from openai import AzureOpenAI

        truststore.inject_into_ssl()
        load_dotenv(override=True)

        _noise = df.loc[df["cluster_code"] == "C_NOISE"].copy()
        if _noise.empty:
            raise ValueError("Aucun incident C_NOISE à analyser.")
        if _noise["text"].isna().any() or not _noise["text"].map(
            lambda value: isinstance(value, str)
        ).all():
            raise ValueError("Chaque incident C_NOISE doit avoir un texte valide.")

        _sample_size = max(1, round(len(_noise) * 0.1))
        _sample_positions = sample_np.linspace(
            0, len(_noise) - 1, num=_sample_size
        ).round().astype(int)
        _sample = _noise.iloc[_sample_positions]
        _cluster_names = sorted(
            df.loc[df["cluster_code"] != "C_NOISE", "cluster_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not _cluster_names:
            raise ValueError("Aucun nom de cluster existant à fournir à Azure.")

        _harness = {
            "asset_type": str(_noise["asset_type"].iloc[0]),
            "cluster_names": _cluster_names,
            "unclustered_incidents": [
                {
                    "incident_id": str(_row.incident_id),
                    "text": _row.text,
                }
                for _row in _sample.itertuples(index=False)
            ],
        }
        _system_prompt = """Tu analyses des rapports d'incidents techniques non clusterisés.
    À partir des textes fournis et des noms des clusters existants, explique en français en quelques
    phrases les principaux types d'incidents présents et pourquoi ils ne semblent pas correspondre aux
    six clusters existants. Distingue les observations des hypothèses : les textes seuls ne permettent
    pas de connaître avec certitude la raison algorithmique de leur exclusion. N'invente aucun détail.
    Retourne uniquement un objet JSON valide de la forme : {"analysis":"..."}."""
        _user_prompt = (
            "Voici les noms des clusters existants et un échantillon uniformément réparti "
            "sur les incidents non clusterisés :\n"
            + json.dumps(_harness, ensure_ascii=False)
        )

        _api_key = os.environ["AZURE_OPENAI_API_KEY"]
        _endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        _api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        _deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        _client = AzureOpenAI(
            api_key=_api_key,
            azure_endpoint=_endpoint,
            api_version=_api_version,
        )
        _response = _client.chat.completions.create(
            model=_deployment,
            messages=[
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": _user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        _usage = _response.usage
        if _usage is None:
            raise ValueError("Azure n'a pas retourné l'usage des tokens.")
        _input_tokens = _usage.prompt_tokens
        _output_tokens = _usage.completion_tokens
        _prompt_details = getattr(_usage, "prompt_tokens_details", None)
        _cached_tokens = getattr(_prompt_details, "cached_tokens", 0) or 0
        if (
            not isinstance(_input_tokens, int)
            or not isinstance(_output_tokens, int)
            or not isinstance(_cached_tokens, int)
            or min(_input_tokens, _output_tokens, _cached_tokens) < 0
            or _cached_tokens > _input_tokens
        ):
            raise ValueError("Azure a retourné un usage de tokens invalide.")
        _input_cost = (
            (_input_tokens - _cached_tokens) * 0.10
            + _cached_tokens * 0.025
        ) / 1_000_000
        _output_cost = _output_tokens * 0.40 / 1_000_000
        print(
            f"Usage Azure — analyse des incidents non clusterisés : "
            f"{_input_tokens:,} tokens en entrée "
            f"(dont {_cached_tokens:,} en cache), {_output_tokens:,} en sortie."
        )
        print(
            f"Coût estimé (GPT-4.1 nano, tarif public Azure à l'usage) : "
            f"${_input_cost + _output_cost:.8f} USD "
            f"(entrée ${_input_cost:.8f} + sortie ${_output_cost:.8f})."
        )

        _content = _response.choices[0].message.content
        if not _content:
            raise ValueError("Azure a renvoyé une réponse vide pour l'analyse.")
        _result = json.loads(_content)
        if not isinstance(_result, dict) or set(_result) != {"analysis"}:
            raise ValueError("La réponse Azure doit contenir uniquement la clé 'analysis'.")
        _analysis = _result["analysis"]
        if not isinstance(_analysis, str) or not _analysis.strip():
            raise ValueError("Azure n'a pas renvoyé d'analyse textuelle valide.")
        unclustered_analysis_output = mo.md(_analysis.strip())

    unclustered_analysis_output
    return


if __name__ == "__main__":
    app.run()
