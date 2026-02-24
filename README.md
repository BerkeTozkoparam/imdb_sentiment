# 🎬 IMDB Sentiment Analyzer

An interactive NLP web app that classifies movie reviews as **positive** or **negative** using classical machine learning — no deep learning required.

---

## Features

- **Live Prediction** — paste any movie review, get instant sentiment with a confidence gauge
- **3 Classifiers** — Logistic Regression, Naive Bayes, Linear SVM compared side by side
- **2 Preprocessing Modes** — switch between Stemming and Lemmatization from the sidebar
- **Word Analysis** — TF-IDF feature weights, word clouds, top vocabulary terms
- **Confusion Matrix & Metrics** — accuracy, precision, recall, F1 for each model
- **Adjustable dataset size** — 1k to 10k reviews via sidebar slider

---

## Tech Stack

| Layer | Tools |
|---|---|
| Preprocessing | NLTK · PorterStemmer · WordNetLemmatizer + POS tagging |
| Vectorization | TF-IDF (unigram + bigram, 20k features) |
| Classifiers | Logistic Regression · Naive Bayes · Linear SVM |
| Visualization | Plotly · Matplotlib · WordCloud |
| UI | Streamlit |
| Dataset | IMDB 10K Movie Reviews (bundled) |

---

## Run Locally

```bash
git clone https://github.com/BerkeTozkoparan/imdb_sentiment.git
cd imdb_sentiment
pip install -r requirements.txt
streamlit run app.py
```

---

## How It Works

```
Raw Review
    │
    ▼
Lowercase + Remove punctuation/digits
    │
    ▼
Tokenize → Stem (Porter) or Lemmatize (WordNet + POS)
    │
    ▼
TF-IDF Matrix (1–2 gram, max 20k features)
    │
    ├── Logistic Regression ──┐
    ├── Naive Bayes           ├──▶ Sentiment: Positive / Negative
    └── Linear SVM ───────────┘
```

- Train/test split: 80/20, stratified
- Models cached with `@st.cache_resource` — train once per session
- Dataset balanced: 50% positive, 50% negative

---

## Project Structure

```
imdb_sentiment/
├── app.py           # Streamlit app
├── imdb_10k.csv     # Bundled dataset (10k balanced reviews)
└── requirements.txt
```
