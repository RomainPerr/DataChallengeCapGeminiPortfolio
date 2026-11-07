import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Starter kit — Hackathon Mines Paris 2026 — Analyse d'incidents PV / éolien

    ## Objectif du hackathon

    Construire un pipeline agentique qui, à partir des rapports d'incidents PV et éolien `data/incidents_students.csv`, réalise :
    - du **topic modeling** (regroupement des incidents par thème),
    - une **analyse de causes**,
    - des **statistiques globales** (par site, sévérité, saisonalité, durée d'arrêt, coût...),
    - des **propositions de remédiation** (maintenance, évolutions techniques).

    Ce notebook couvre la première brique, le topic modeling, et sert de base pour construire le reste
    du pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Configuration
    """)
    return


@app.cell
def _():
    import os

    # Réseau d'entreprise avec inspection SSL : utiliser le magasin de certificats de l'OS
    # (doit être appelé avant la création de tout client HTTP).
    import truststore

    truststore.inject_into_ssl()

    from dotenv import load_dotenv

    # override=True : le .env prime sur d'éventuelles variables déjà présentes dans le kernel
    load_dotenv(override=True)

    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
    EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    LLM_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    # Vérification de la config lue (la clé n'est jamais affichée)
    print("endpoint   :", repr(AZURE_OPENAI_ENDPOINT))
    print("api_version:", repr(AZURE_OPENAI_API_VERSION))
    print("embedding  :", repr(EMBEDDING_DEPLOYMENT))
    print("chat       :", repr(LLM_DEPLOYMENT))
    if "/api/projects" in AZURE_OPENAI_ENDPOINT:
        print(
            "⚠️ L'endpoint contient /api/projects : AzureOpenAI attend l'endpoint de la ressource, "
            "ex. https://<ressource>.cognitiveservices.azure.com/"
        )
    return (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_ENDPOINT,
        EMBEDDING_DEPLOYMENT,
        os,
    )


@app.cell
def _(
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    EMBEDDING_DEPLOYMENT,
):
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Test de connexion : isole un problème de config (endpoint / version / déploiement)
    # avant de lancer le calcul des embeddings sur tout le jeu de données.
    _probe = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=["test"])
    print("Connexion OK, dimension des embeddings :", len(_probe.data[0].embedding))
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Charger les données

    Les rapports sont fournis dans 3 formats (`document_format`) :
    - `text` : texte déjà disponible dans la colonne `text` du CSV.
    - `pdf_digital` : PDF avec texte extractible (`document_path` pointe vers `data/pdf_digital/`).
    - `pdf_scan` : PDF scanné (image), nécessitant de l'OCR (`document_path` pointe vers
      `data/pdf_scan/`).

    Pour ces deux derniers formats, seules les colonnes `site`, `document_format` et `document_path`
    sont renseignées dans le CSV : à vous d'extraire le texte (et le reste des métadonnées) des PDF.

    Ce starter kit se limite volontairement au sous-ensemble `text`, déjà prêt à l'emploi, pour
    démarrer le topic modeling.
    """)
    return


@app.cell
def _():
    import pandas as pd

    df : pd.DataFrame = pd.read_csv("data/incidents_students.csv")
    print(df["document_format"].value_counts())

    df : pd.DataFrame = df[df["document_format"] == "text"].reset_index(drop=True)
    print(df.shape)
    df.head()
    return (df,)


@app.cell
def _(df: "pd.DataFrame"):
    df["asset_type"].value_counts(), df["severity"].value_counts()
    return


@app.cell
def _(df: "pd.DataFrame"):
    for text in df["text"].sample(3, random_state=0):
        print(text)
        print("-" * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Calculer les embeddings du texte

    Un embedding représente un texte par un vecteur tel que deux textes similaires ont des vecteurs
    proches. Calcul via Azure OpenAI, par lots, avec mise en cache locale (`.npy`) pour éviter de
    refaire les appels à chaque exécution.
    """)
    return


@app.cell
def _(EMBEDDING_DEPLOYMENT, client, df: "pd.DataFrame", os):
    import math
    import re
    import time

    import numpy as np
    from openai import RateLimitError
    from tqdm import tqdm


    def wait_with_progress(seconds: int, desc: str) -> None:
        """Attente avec barre de progression (décompte en secondes)."""
        for _ in tqdm(range(seconds), desc=desc, unit="s", leave=False):
            time.sleep(1)


    def embed_batch_with_retry(batch: list[str], max_attempts: int = 10):
        """Appelle l'API d'embeddings ; sur erreur 429, attend le délai indiqué par Azure puis réessaie.

        `max_retries=0` : on désactive les tentatives silencieuses du SDK pour que l'attente
        soit gérée ici, avec une barre de progression visible.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                return client.with_options(max_retries=0).embeddings.create(
                    model=EMBEDDING_DEPLOYMENT, input=batch
                )
            except RateLimitError as e:
                if attempt == max_attempts:
                    raise
                match = re.search(r"retry after (\d+) seconds", str(e))
                wait = int(match.group(1)) + 1 if match else 30
                wait_with_progress(wait, f"Quota embeddings atteint (429), tentative {attempt}/{max_attempts}")


    def azure_embed_documents(
        docs: list[str], batch_size: int = 16, partial_path: str = "embeddings_students.partial.npy"
    ) -> np.ndarray:
        """Calcule les embeddings par lots ; la progression est sauvegardée pour reprendre après interruption."""
        all_embeddings = list(np.load(partial_path)) if os.path.exists(partial_path) else []
        start = len(all_embeddings) // batch_size  # on ne garde que des lots complets
        all_embeddings = all_embeddings[: start * batch_size]
        n_batches = math.ceil(len(docs) / batch_size)

        for i in tqdm(range(start, n_batches), desc="Embeddings", unit="lot", initial=start, total=n_batches):
            batch = docs[i * batch_size : (i + 1) * batch_size]
            response = embed_batch_with_retry(batch)
            all_embeddings.extend(item.embedding for item in response.data)
            if (i + 1) % 10 == 0:
                np.save(partial_path, np.array(all_embeddings))

        if os.path.exists(partial_path):
            os.remove(partial_path)
        return np.array(all_embeddings)


    embeddings_path = "embeddings_students.npy"
    embeddings = None
    if os.path.exists(embeddings_path):
        cached = np.load(embeddings_path)
        if cached.ndim == 2 and cached.shape[0] == len(df):
            embeddings = cached
        else:
            # Cache incohérent (calcul partiel, autre jeu de données...) : on le recalcule
            print(f"Cache ignoré : forme {cached.shape}, attendu ({len(df)}, dim)")

    if embeddings is None:
        embeddings = azure_embed_documents(df["text"].tolist())
        np.save(embeddings_path, embeddings)

    assert embeddings.shape[0] == len(df), f"{embeddings.shape[0]} embeddings pour {len(df)} documents"
    embeddings.shape
    return (embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Topic modeling avec BERTopic

    - `CountVectorizer` : extrait les mots-clés de chaque topic une fois les clusters formés
      (stopwords français, tokens numériques exclus).
    - `UMAP` : réduit la dimension des embeddings avant clustering.
    - `HDBSCAN` : forme les clusters ; les documents isolés sont marqués `-1` (non classés).
    - `min_cluster_size` : taille minimale d'un topic.
    """)
    return


@app.cell
def _(df: "pd.DataFrame", embeddings):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    FRENCH_STOPWORDS = [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "et",
        "en",
        "sur",
        "au",
        "aux",
        "pour",
        "par",
        "est",
        "sont",
        "a",
        "été",
        "suite",
        "lors",
        "dans",
        "que",
        "qui",
        "ce",
        "cette",
        "ces",
        "afin",
        "plus",
        "après",
        "avant",
    ]

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=FRENCH_STOPWORDS,
        token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="french",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings=embeddings)
    df["topic"] = topics
    print(f"{len(set(topics)) - (1 if -1 in topics else 0)} topics trouvés")
    return (topic_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Explorer les résultats
    """)
    return


@app.cell
def _(topic_model):
    topic_model.get_topic_info()
    return


@app.cell
def _(topic_model):
    topic_model.visualize_barchart(top_n_topics=12)
    return


@app.cell
def _(topic_model):
    topic_model.visualize_hierarchy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les mots-clés seuls suffisent rarement à comprendre un topic : regarde aussi
    quelques documents représentatifs.
    """)
    return


@app.cell
def _(df: "pd.DataFrame", topic_model):
    def show_topic_examples(topic_id: int, n: int = 3) -> None:
        print("Mots-clés :", [word for word, _ in topic_model.get_topic(topic_id)])
        examples = df.loc[df["topic"] == topic_id, "text"].head(n)
        for text in examples:
            print("-", text)


    show_topic_examples(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pistes pour aller plus loin

    0. **Sauvegarder vos générations.** Pensez à sauvegarder vos générations pour éviter de perdre votre
       travail et consommer des ressources LLM supplémentaires.
    1. **Rajouter les docs PDF.** Lire et inclure les documents au format PDF digitaux et scans.
    2. **Caractériser les topics.** Nommer les différents topics identifier et les analyser au mieux
        pour en faire ressortir les informations associées.
    3. **Plan de remédiation.** Rédiger un plan de remédiation associé avec un agent
       ou un système d'agents en charge de differents aspects.
    4. **Traiter les rapports `pdf_digital` et `pdf_scan`.** Ce starter kit ne couvre que le
       sous-ensemble `text` (600/1800 rapports).
    5. **Séparer PV et éolien avant de clusteriser.** Le vocabulaire des deux domaines diffère
       fortement ; les mélanger risque de faire ressortir cette séparation avant les vraies causes.
    6. **Ajuster `min_cluster_size`.** Plus petit → plus de topics, plus fins ; plus grand → moins de
       topics, plus larges.
    7. **Ajuster `ngram_range`** pour capter les expressions à plusieurs mots.
    8. **Fusionner les topics redondants** avec `topic_model.reduce_topics(...)`, guidé par
       `visualize_hierarchy()`.
    9. **Croiser les topics avec les métadonnées** (`site`, `date`, `severity`, `downtime_hours`,
       `estimated_cost_eur`, `weather_condition`) pour enrichir l'interprétation.
    10. **Approfondir avec une pipeline agentique.** Les deux appels LLM ci-dessus (nom + analyse)
       sont figés en un seul passage par topic. Un agent outillé (accès au dataframe, aux fonctions
       ci-dessus, à une recherche web) peut itérer pour approfondir l'analyse et croiser plusieurs
       topics entre eux, vérifier une hypothèse sur les métadonnées, ou creuser un signal faible avant
       de conclure, plutôt qu'une synthèse en un seul coup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
