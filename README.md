# 🎬 IMDB Sentiment Analyzer

An interactive NLP web app that analyzes movie review sentiment using classical machine learning. Built with Streamlit.

## Features

- **Live Prediction** — paste any movie review and get instant positive/negative sentiment with a confidence gauge
- **Model Comparison** — Logistic Regression, Naive Bayes, and Linear SVM accuracy side by side
- **Word Analysis** — TF-IDF feature weights, word clouds, and top vocabulary terms
- **Flexible pipeline** — switch between Stemming and Lemmatization, tune dataset size from the sidebar

## Tech Stack

| Layer | Tools |
|---|---|
| Preprocessing | NLTK (PorterStemmer, WordNetLemmatizer + POS tagging) |
| Vectorization | TF-IDF (unigram + bigram, 20k features) |
| Classifiers | Logistic Regression · Naive Bayes · Linear SVM |
| Visualization | Plotly · Matplotlib · WordCloud |
| UI | Streamlit |
| Dataset | [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |

## Demo

![App Screenshot](https://raw.githubusercontent.com/BerkeTozkoparam/imdb_sentiment/main/screenshot.png)

## Run Locally

```bash
git clone https://github.com/BerkeTozkoparam/imdb_sentiment.git
cd imdb_sentiment
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. Raw reviews are lowercased, punctuation/digits stripped
2. Tokens are stemmed (Porter) or lemmatized (WordNet + POS)
3. TF-IDF matrix built with bigram support
4. Three classifiers trained on an 80/20 train-test split
5. Results cached with `@st.cache_resource` — models train once per session
