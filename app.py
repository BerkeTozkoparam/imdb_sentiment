import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
@st.cache_resource
def _nltk_setup():
    for pkg in ["punkt", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)

_nltk_setup()

# ── Text processing ───────────────────────────────────────────────────────────
_stemmer   = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_PUNCT_RE   = re.compile(r"[" + re.escape(string.punctuation) + r"\d]+")

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text.lower())).strip()

def _tokens(text: str):
    return re.findall(r"[a-z]+", text)

def _wn_pos(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(
        tag[0] if tag else "", wordnet.NOUN
    )

def stem_text(text: str) -> str:
    return " ".join(_stemmer.stem(t) for t in _tokens(_clean(text)))

def lemma_text(text: str) -> str:
    tokens = _tokens(_clean(text))
    return " ".join(
        _lemmatizer.lemmatize(w, _wn_pos(p)) for w, p in nltk.pos_tag(tokens)
    )

PREPROCESS = {"Stemming": stem_text, "Lemmatization": lemma_text}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(n: int) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train+test", trust_remote_code=False)
    df = pd.DataFrame({
        "review":    ds["text"],
        "sentiment": ["positive" if l == 1 else "negative" for l in ds["label"]],
    })
    return df.iloc[:n] if n else df

# ── Model pipeline ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_pipeline(n: int, prep: str):
    df        = load_df(n)
    docs      = df["review"].astype(str)
    y         = (df["sentiment"] == "positive").astype(int).values
    processed = docs.apply(PREPROCESS[prep])

    vec = TfidfVectorizer(stop_words="english", max_features=20_000, ngram_range=(1, 2))
    X   = vec.fit_transform(processed)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Linear SVM":          LinearSVC(max_iter=2000, C=1.0),
    }

    results = {}
    for name, clf in clfs.items():
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        results[name] = {
            "model":    clf,
            "accuracy": accuracy_score(y_te, preds),
            "cm":       confusion_matrix(y_te, preds),
        }

    return vec, results, processed.values, y

# ── Inference helpers ─────────────────────────────────────────────────────────
def infer(text: str, prep: str, vec, model):
    X    = vec.transform([PREPROCESS[prep](text)])
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        pos_p = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        s     = float(model.decision_function(X)[0])
        pos_p = float(1 / (1 + np.exp(-s)))
    else:
        pos_p = float(pred)

    return pred, pos_p


def top_features(vec, model, n: int = 15) -> pd.DataFrame | None:
    if not hasattr(model, "coef_"):
        return None
    feat = vec.get_feature_names_out()
    coef = model.coef_.ravel()
    rows = (
        [{"word": feat[i], "weight": float(coef[i]), "type": "Positive"}
         for i in coef.argsort()[-n:][::-1]]
        + [{"word": feat[i], "weight": float(coef[i]), "type": "Negative"}
           for i in coef.argsort()[:n]]
    )
    return pd.DataFrame(rows)


def metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        tn, fp, fn, tp = r["cm"].ravel()
        prec = tp / (tp + fp) if tp + fp else 0
        rec  = tp / (tp + fn) if tp + fn else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rows.append({
            "Model":     name,
            "Accuracy":  r["accuracy"],
            "Precision": prec,
            "Recall":    rec,
            "F1":        f1,
        })
    return pd.DataFrame(rows)

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 IMDB Sentiment Analyzer")
    st.caption("TF-IDF · Stemming / Lemmatization · Logistic Regression · Naive Bayes · SVM")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        n_docs = st.select_slider(
            "Dataset size",
            options=[1_000, 2_000, 5_000, 10_000, 25_000, 50_000],
            value=5_000,
            help="More data → better accuracy, slower training",
        )

        prep = st.radio(
            "Preprocessing",
            ["Stemming", "Lemmatization"],
            help="Lemmatization is linguistically richer but slower (POS tagging per token)",
        )

        if prep == "Lemmatization" and n_docs > 5_000:
            st.warning("Lemmatization + large dataset can be slow. Consider ≤ 5 000 docs.")

        model_name = st.selectbox(
            "Classifier",
            ["Logistic Regression", "Naive Bayes", "Linear SVM"],
        )

        st.divider()
        st.info(f"**{n_docs:,} reviews**\n\n{prep} · {model_name}")

    # ── Training ──────────────────────────────────────────────────────────────
    with st.spinner(f"Loading & training on {n_docs:,} reviews ({prep})…"):
        vec, results, processed, y = build_pipeline(n_docs, prep)

    model = results[model_name]["model"]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(
        ["🔍 Live Prediction", "📊 Model Performance", "📝 Word Analysis"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 · Live Prediction
    # ─────────────────────────────────────────────────────────────────────────
    with t1:
        EXAMPLES = {
            "—": "",
            "Positive": (
                "This film was an absolute masterpiece! The performances were outstanding, "
                "the screenplay witty and moving, and the direction flawless. "
                "I was completely absorbed from start to finish. Highly recommended!"
            ),
            "Negative": (
                "What a dreadful waste of time. The story was incoherent, "
                "the acting wooden, and the special effects looked cheap. "
                "I walked out after 40 minutes and never looked back. Avoid!"
            ),
        }

        col_txt, col_ex = st.columns([3, 1])
        with col_ex:
            choice = st.selectbox("Load example", list(EXAMPLES.keys()))
        with col_txt:
            review = st.text_area(
                "Movie review:",
                value=EXAMPLES[choice],
                height=160,
                placeholder="Write or paste a review here…",
            )

        if review.strip():
            pred, pos_p = infer(review, prep, vec, model)

            label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
            color = "green" if pred == 1 else "red"

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"### Sentiment\n:{color}[**{label}**]")
            m2.metric("Positive probability", f"{pos_p:.1%}")
            m3.metric("Negative probability", f"{1 - pos_p:.1%}")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pos_p * 100,
                    number={"suffix": "%", "font": {"size": 42}},
                    title={"text": "Positivity Score"},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar":  {"color": "#2ecc71" if pos_p > 0.5 else "#e74c3c"},
                        "steps": [
                            {"range": [0,  50], "color": "#fde8e8"},
                            {"range": [50, 100], "color": "#e8fde8"},
                        ],
                        "threshold": {
                            "line":      {"color": "gray", "width": 3},
                            "thickness": 0.75,
                            "value":     50,
                        },
                    },
                )
            )
            fig.update_layout(height=300, margin={"t": 60, "b": 10, "l": 40, "r": 40})
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Type or paste a movie review above to get a sentiment prediction.")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 · Model Performance
    # ─────────────────────────────────────────────────────────────────────────
    with t2:
        c1, c2 = st.columns(2)

        with c1:
            acc_df = pd.DataFrame(
                [{"Model": k, "Accuracy": v["accuracy"]} for k, v in results.items()]
            )
            fig_acc = px.bar(
                acc_df, x="Model", y="Accuracy",
                text_auto=".3f",
                color="Accuracy",
                color_continuous_scale="RdYlGn",
                range_y=[0.75, 1.0],
                title=f"Accuracy — {prep} · {n_docs:,} docs",
            )
            fig_acc.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        with c2:
            cm = results[model_name]["cm"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual"},
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                title=f"Confusion Matrix — {model_name}",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Detailed metrics")
        df_metrics = metrics_table(results)
        st.dataframe(
            df_metrics.style
                .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                         "Recall": "{:.4f}", "F1": "{:.4f}"})
                .highlight_max(subset=["Accuracy", "F1"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 · Word Analysis
    # ─────────────────────────────────────────────────────────────────────────
    with t3:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Feature weights")
            feat_df = top_features(vec, model)
            if feat_df is not None:
                fig_fw = px.bar(
                    feat_df.sort_values("weight"),
                    x="weight", y="word",
                    color="type",
                    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
                    orientation="h",
                    title=f"Top tokens by TF-IDF weight ({model_name})",
                )
                fig_fw.update_layout(
                    height=620,
                    yaxis_title="",
                    xaxis_title="Coefficient",
                    legend_title="Sentiment",
                )
                st.plotly_chart(fig_fw, use_container_width=True)
            else:
                st.info("Feature weights not available for Naive Bayes in this view.")

        with c2:
            st.subheader("Word cloud")
            sent_choice = st.radio("Sentiment filter", ["Positive", "Negative"], horizontal=True)
            mask      = y == (1 if sent_choice == "Positive" else 0)
            text_blob = " ".join(processed[mask][:3_000])

            wc = WordCloud(
                width=700, height=420,
                background_color="white",
                colormap="Greens" if sent_choice == "Positive" else "Reds",
                max_words=120,
                collocations=False,
            ).generate(text_blob)

            fig_wc, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{sent_choice} reviews — {prep}", fontsize=13)
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        # Dataset summary
        st.divider()
        st.subheader("Dataset summary")
        df_info = load_df(n_docs)
        pos_n   = int((df_info["sentiment"] == "positive").sum())

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total reviews",  f"{len(df_info):,}")
        s2.metric("Positive",       f"{pos_n:,}")
        s3.metric("Negative",       f"{len(df_info) - pos_n:,}")
        s4.metric("Vocab size",     f"{len(vec.vocabulary_):,}")

        # N-gram frequency chart (top 20 unigrams from TF-IDF vocab)
        st.subheader("Top 30 vocabulary terms by document frequency")
        df_vocab = pd.DataFrame({
            "term": list(vec.vocabulary_.keys()),
            "index": list(vec.vocabulary_.values()),
        })
        # Use sum of TF-IDF column as a proxy for frequency
        col_sums = np.asarray(
            vec.transform(processed).sum(axis=0)
        ).ravel()
        df_vocab["score"] = df_vocab["index"].map(lambda i: float(col_sums[i]))
        top30 = df_vocab.nlargest(30, "score")

        fig_top = px.bar(
            top30.sort_values("score"), x="score", y="term",
            orientation="h",
            title="Top 30 terms (TF-IDF column sum across all docs)",
            labels={"score": "Aggregate TF-IDF score", "term": ""},
        )
        fig_top.update_layout(height=550)
        st.plotly_chart(fig_top, use_container_width=True)


if __name__ == "__main__":
    main()
