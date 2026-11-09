# Smart Recommendation Agent Using Collaborative and Content-Based Filtering

A Streamlit movie discovery application combining TF-IDF content matching with item-based collaborative filtering. It uses the MovieLens ml-latest-small dataset from GroupLens Research and automatically downloads the dataset on first launch.

## Features

- Seven sections: dashboard, hybrid recommendation agent, content-based discovery, collaborative recommendations, analytics and catalog, offline evaluation, and project overview.
- Personalized recommendations for MovieLens profiles and new viewers, with adjustable hybrid weights and explanations.
- Genre and tag TF-IDF profiles, sparse item-based adjusted cosine similarity, and a normalized weighted hybrid score.
- Interactive Plotly charts, searchable and filterable catalog, and per-user 80/20 offline evaluation with Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR, RMSE, and MAE.
- Movie cards include cached Wikimedia thumbnail lookups, with a graceful fallback when artwork is unavailable.
- Automated unit tests for data preparation, recommendation models, hybrid scoring, and evaluation.

## Requirements

Python 3.10 or later and internet access on first launch. Dataset files are downloaded into `data/` and are not committed to version control.

## Installation and launch

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Run tests

```powershell
python -m pytest -q
```

## Project structure

```text
app.py
requirements.txt
README.md
.gitignore
data/                       # downloaded MovieLens files
src/
  __init__.py
  data_loader.py
  content_based.py
  collaborative.py
  hybrid.py
  evaluation.py
  utils.py
tests/
  test_data_loader.py
  test_content_based.py
  test_collaborative.py
  test_hybrid.py
  test_evaluation.py
```

## Dataset

MovieLens ml-latest-small contains 9,742 movies, 100,836 ratings, and 610 users. It is provided by GroupLens Research at [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/). The app downloads the archive from GroupLens when local `movies.csv` and `ratings.csv` are absent.

Poster thumbnails are looked up by title through the Wikipedia REST API and cached for seven days. Images may be absent for some films and remain subject to their source licensing.

## Evaluation notes

The interactive benchmark samples up to 100 eligible users reproducibly and independently holds out 20% of each user's ratings (users need at least two ratings). Ratings of 4.0 or above are treated as relevant for ranking metrics. RMSE and MAE are calculated on held-out ratings that have a prediction from the respective model. `evaluate_models(..., max_users=None)` can be used for a full 610-user run outside the interactive demo.