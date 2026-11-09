"""Smart Recommendation Agent Streamlit application."""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.collaborative import CollaborativeRecommender
from src.content_based import ContentRecommender
from src.data_loader import load_movielens, movie_catalog
from src.evaluation import evaluate_models
from src.hybrid import hybrid_recommendations, recommendation_explanation
from src.posters import get_movie_poster


st.set_page_config(
    page_title="Smart Recommendation Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#f3f5ff; --muted:#a4acc3; --panel:#151a2a; --line:#28304a; --blue:#6ca8ff; --pink:#ef78c7; --mint:#68dfc3; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(ellipse at 82% 4%,rgba(82,95,177,.18),transparent 36%),linear-gradient(145deg,#0b1020 0%,#101527 52%,#111322 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#131a2d,#101421); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1 { font-family:'Space Grotesk',sans-serif; font-size:1.13rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; }
    h1 { letter-spacing:0 !important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,36,59,.96),rgba(21,27,44,.96)); border:1px solid #29334e; padding:19px 20px; border-radius:8px; box-shadow:0 12px 30px rgba(0,0,0,.12); }
    [data-testid="stMetricLabel"] { color:#aab4cf !important; }
    [data-testid="stMetricValue"] { color:#f6f7ff !important; font-family:'Space Grotesk',sans-serif; }
    .eyebrow { color:#90a8d7; text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:1.5px; }
    .hero { border:1px solid #293451; border-radius:10px; padding:30px 34px; background:linear-gradient(112deg,rgba(40,64,111,.55),rgba(48,34,81,.32) 60%,rgba(24,33,54,.7)); margin-bottom:22px; }
    .hero h1 { font-size:2.1rem; margin:.35rem 0 .45rem; }
    .hero p, .subtle { color:var(--muted); }
    .movie-card { background:linear-gradient(135deg,#171e31,#141a29); border:1px solid #2a3550; border-radius:8px; padding:15px 16px; height:100%; min-height:128px; margin-bottom:12px; transition:transform .18s ease,border-color .18s ease; }
    .movie-card { display:flex; align-items:flex-start; gap:13px; }
    .movie-poster { width:72px; height:108px; object-fit:cover; border-radius:4px; flex:0 0 72px; background:#222a3c; }
    .poster-fallback { width:72px; height:108px; flex:0 0 72px; display:flex; align-items:center; justify-content:center; border-radius:4px; background:linear-gradient(145deg,#293b59,#342844); color:#c4cce0; font-size:.66rem; text-align:center; }
    .movie-card-copy { min-width:0; }
    .movie-card:hover { transform:translateY(-3px); border-color:#6c8ed1; }
    .movie-title { color:#f6f7ff; font-weight:700; font-size:.98rem; line-height:1.35; }
    .movie-meta { color:#a8b2ca; font-size:.8rem; margin-top:7px; }
    .score { color:#8ee4cf; font-size:.82rem; font-weight:700; margin-top:9px; }
    .stButton > button { border-radius:7px; border:1px solid #465b8a; background:linear-gradient(110deg,#397bd8,#7659bb); color:white; font-weight:700; }
    .stButton > button:hover { border-color:#a1bfff; color:white; }
    div[data-baseweb="tab-list"] { gap:8px; }
    div[data-baseweb="tab"] { background:#151b2b; border-radius:6px; }
    hr { border-color:#28304a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading MovieLens data...")
def get_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies, ratings, tags = load_movielens()
    return movies, ratings, tags, movie_catalog(movies, ratings, tags)


@st.cache_data(ttl=604800, show_spinner=False)
def get_cached_poster(title: str) -> str | None:
    return get_movie_poster(title)


@st.cache_resource(show_spinner="Building genre and tag profiles...")
def get_content_model(_movies: pd.DataFrame, _tags: pd.DataFrame, fingerprint: str) -> ContentRecommender:
    del fingerprint
    return ContentRecommender(_movies, _tags)


@st.cache_resource(show_spinner="Learning community rating patterns...")
def get_collaborative_model(_ratings: pd.DataFrame, fingerprint: str) -> CollaborativeRecommender:
    del fingerprint
    return CollaborativeRecommender(_ratings)


def title_for(movie_id: int, catalog: pd.DataFrame) -> str:
    match = catalog.loc[catalog["movieId"] == movie_id, "title"]
    return str(match.iloc[0]) if not match.empty else f"Movie {movie_id}"


def movie_picker_options(catalog: pd.DataFrame) -> dict[str, int]:
    return {
        f"{row.title}  ·  {row.genres.replace('|', ' / ')}": int(row.movieId)
        for row in catalog.itertuples()
    }


def render_movie_cards(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    score_column: str = "score",
    limit: int = 12,
) -> None:
    if recommendations.empty:
        st.info("No recommendations match those choices yet. Try another profile or genre.")
        return
    details = recommendations.drop(columns=["title", "genres"], errors="ignore").merge(
        catalog[["movieId", "title", "genres"]], on="movieId", how="left"
    )
    details = details.head(limit)
    columns = st.columns(3)
    for index, row in enumerate(details.itertuples()):
        raw_score = getattr(row, score_column, 0)
        score = float(raw_score) if pd.notna(raw_score) else 0.0
        reason = recommendation_explanation(pd.Series(row._asdict()))
        poster_url = get_cached_poster(str(row.title))
        poster = (
            f'<img class="movie-poster" src="{html.escape(poster_url, quote=True)}" '
            f'alt="Poster for {html.escape(str(row.title), quote=True)}" loading="lazy">'
            if poster_url else '<div class="poster-fallback">No poster</div>'
        )
        card = (
            '<div class="movie-card">'
            f'{poster}<div class="movie-card-copy">'
            f'<div class="movie-title">{html.escape(str(row.title))}</div>'
            f'<div class="movie-meta">{html.escape(str(row.genres).replace("|", " · "))}</div>'
            f'<div class="score">Score&nbsp; {score:.2f}</div>'
            f'<div class="movie-meta">{html.escape(reason)}</div></div>'
            '</div>'
        )
        with columns[index % 3]:
            st.markdown(card, unsafe_allow_html=True)


def weighted_popularity(catalog: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(catalog.loc[catalog["rating_count"] > 0, "avg_rating"].mean())
    threshold = float(catalog["rating_count"].quantile(0.75))
    rated = catalog[catalog["rating_count"] > 0].copy()
    denominator = rated["rating_count"] + threshold
    rated["popular_score"] = (
        rated["rating_count"] / denominator * rated["avg_rating"]
        + threshold / denominator * global_mean
    )
    return rated


def page_home(catalog: pd.DataFrame, ratings: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">MovieLens · intelligent discovery</div>'
        '<h1>Find your next favorite.</h1><p>Recommendations shaped by what you love and what the community loves.</p></div>',
        unsafe_allow_html=True,
    )
    average_rating = ratings["rating"].mean()
    genres = sorted({genre for value in catalog["genres"] for genre in value.split("|") if genre != "(no genres listed)"})
    metrics = st.columns(5)
    values = [
        ("Movies", f"{len(catalog):,}"),
        ("Active users", f"{ratings['userId'].nunique():,}"),
        ("Ratings", f"{len(ratings):,}"),
        ("Genres", f"{len(genres)}"),
        ("Average rating", f"{average_rating:.2f} / 5"),
    ]
    for column, (label, value) in zip(metrics, values):
        column.metric(label, value)

    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Popular with the community")
        popular = weighted_popularity(catalog).sort_values("rating_count", ascending=False).head(6)
        popular["source"] = "popularity"
        render_movie_cards(popular, catalog, score_column="popular_score", limit=6)
    with right:
        st.subheader("Highest rated")
        highly_rated = catalog[catalog["rating_count"] >= 20].nlargest(8, "avg_rating").copy()
        if highly_rated.empty:
            highly_rated = catalog[catalog["rating_count"] > 0].nlargest(8, "avg_rating")
        st.dataframe(
            highly_rated[["title", "genres", "avg_rating", "rating_count"]].rename(
                columns={"avg_rating": "Rating", "rating_count": "Ratings"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={"Rating": st.column_config.NumberColumn(format="%.2f")},
        )


def page_agent(
    catalog: pd.DataFrame,
    ratings: pd.DataFrame,
    content_model: ContentRecommender,
    collaborative_model: CollaborativeRecommender,
) -> None:
    st.title("Smart Recommendation Agent")
    st.caption("Blend your personal taste with signals from the MovieLens community.")
    existing, new = st.tabs(["MovieLens profile", "New viewer"])
    users = sorted(ratings["userId"].unique().tolist())
    options = movie_picker_options(catalog)
    genres = sorted({genre for value in catalog["genres"] for genre in value.split("|") if genre != "(no genres listed)"})

    with existing:
        user_id = st.selectbox("Choose a MovieLens user", users, format_func=lambda value: f"Viewer {value}")
        preferences = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4.0)]
        favorites = preferences.sort_values("rating", ascending=False)["movieId"].head(30).tolist()
        st.caption(f"Profile built from {len(preferences)} positively rated movies.")
        build_agent_results(user_id, favorites, [], catalog, ratings, content_model, collaborative_model)

    with new:
        favorite_labels = st.multiselect(
            "Choose a few movies you enjoy", list(options), max_selections=10,
            placeholder="Search titles, for example: Toy Story",
        )
        selected_genres = st.multiselect("Favorite genres", genres, placeholder="Select any genres you like")
        build_agent_results(
            None, [options[label] for label in favorite_labels], selected_genres,
            catalog, ratings, content_model, collaborative_model,
        )


def build_agent_results(
    user_id: int | None,
    favorites: list[int],
    preferred_genres: list[str],
    catalog: pd.DataFrame,
    ratings: pd.DataFrame,
    content_model: ContentRecommender,
    collaborative_model: CollaborativeRecommender,
) -> None:
    controls = st.columns([1, 1, 1.4])
    with controls[0]:
        content_weight = st.slider("Content weight", 0.0, 1.0, 0.5, 0.05, key=f"content_{user_id}")
    collaborative_weight = 1.0 - content_weight
    controls[1].metric("Collaborative weight", f"{collaborative_weight:.0%}")
    with controls[2]:
        top_k = st.select_slider("Recommendations", options=[5, 10, 15, 20], value=10, key=f"topk_{user_id}")
    st.caption(f"The weights always total 100% · content {content_weight:.0%} / community {collaborative_weight:.0%}.")

    content_scores = content_model.recommend_for_profile(
        favorites, preferred_genres, top_k=len(catalog), exclude_ids=set(favorites)
    )
    if user_id is not None:
        collaborative_scores = collaborative_model.predict_for_user(user_id)
        seen_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    else:
        popularity = weighted_popularity(catalog)
        collaborative_scores = popularity[["movieId", "popular_score"]].rename(columns={"popular_score": "score"})
        collaborative_scores["score"] = collaborative_scores["score"].rank(pct=True)
        collaborative_scores["source"] = "popularity"
        collaborative_scores = collaborative_scores.sort_values("score", ascending=False).reset_index(drop=True)
        seen_ids = set(favorites)
        if not favorites and not preferred_genres:
            content_scores = collaborative_scores[["movieId", "score"]].copy()
            content_scores["reason"] = "Popular with MovieLens viewers."
    collaborative_scores = collaborative_scores[~collaborative_scores["movieId"].isin(seen_ids)]
    content_scores = content_scores[~content_scores["movieId"].isin(seen_ids)]
    if user_id is not None:
        collaborative_scores = collaborative_scores.copy()
        collaborative_scores["source"] = "collaborative"
    hybrid = hybrid_recommendations(
        content_scores, collaborative_scores, content_weight, collaborative_weight,
        top_k=top_k, exclude_ids=seen_ids,
    )
    if user_id is None and not favorites and not preferred_genres:
        hybrid["reason"] = "Popular with MovieLens viewers."
    content_top = content_scores.head(top_k).copy()
    collaborative_top = collaborative_scores.head(top_k).copy()
    st.markdown("#### Your recommendations")
    hybrid_tab, content_tab, community_tab = st.tabs(["Hybrid", "Based on your taste", "Community picks"])
    with hybrid_tab:
        render_movie_cards(hybrid, catalog, limit=top_k)
    with content_tab:
        render_movie_cards(content_top, catalog, limit=top_k)
    with community_tab:
        render_movie_cards(collaborative_top, catalog, score_column="score", limit=top_k)

    st.markdown("#### Side-by-side comparison")
    comparison = st.columns(3)
    for column, heading, frame in zip(
        comparison,
        ["Hybrid agent", "Content-based", "Collaborative"],
        [hybrid, content_top, collaborative_top],
    ):
        with column:
            st.markdown(f"**{heading}**")
            if frame.empty:
                st.caption("No results")
            else:
                for row in frame.head(5).itertuples():
                    st.write(f"{title_for(int(row.movieId), catalog)} · {float(row.score):.2f}")


def page_content(catalog: pd.DataFrame, content_model: ContentRecommender) -> None:
    st.title("Content-Based Recommendations")
    st.caption("Find films with similar genres and community-assigned tags using TF-IDF and cosine similarity.")
    options = movie_picker_options(catalog)
    selected = st.selectbox("Choose a movie", list(options), key="similar_content_movie")
    top_k = st.select_slider("Similar movies", options=[5, 10, 15, 20], value=10, key="similar_content_k")
    results = content_model.similar_movies(options[selected], top_k)
    results = results.merge(catalog[["movieId", "title", "genres"]], on="movieId", how="left")
    results["source"] = "content_similarity"
    if not results.empty:
        fig = px.bar(results.sort_values("score"), x="score", y="title", orientation="h", color="score", color_continuous_scale=["#526b9d", "#e77bc1"])
        fig.update_layout(**dark_chart_layout(), height=430, coloraxis_showscale=False, xaxis_title="Cosine similarity", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        render_movie_cards(results, catalog, limit=top_k)


def page_collaborative(
    catalog: pd.DataFrame,
    ratings: pd.DataFrame,
    collaborative_model: CollaborativeRecommender,
) -> None:
    st.title("Collaborative Recommendations")
    st.caption("Adjusted cosine similarity spots movies that are rated similarly after accounting for each viewer's rating habits.")
    user_ids = sorted(ratings["userId"].unique().tolist())
    left, right = st.columns(2)
    with left:
        user_id = st.selectbox("Analyze viewer", user_ids, format_func=lambda value: f"Viewer {value}", key="cf_user")
    with right:
        selected_title = st.selectbox("Find community-similar movies", list(movie_picker_options(catalog)), key="similar_cf_movie")
    user_ratings = ratings[ratings["userId"] == user_id]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ratings made", f"{len(user_ratings):,}")
    c2.metric("Personal average", f"{user_ratings['rating'].mean():.2f} / 5")
    c3.metric("Community neighbors", f"{int((ratings.groupby('movieId')['userId'].nunique() > 1).sum()):,} movies")
    predicted = collaborative_model.recommend(user_id, 10)
    predicted["source"] = "collaborative"
    st.subheader("Predicted favorites you haven't rated")
    render_movie_cards(predicted, catalog, score_column="score", limit=10)

    similar = collaborative_model.similar_movies(movie_picker_options(catalog)[selected_title], 10)
    similar = similar.merge(catalog[["movieId", "title", "genres"]], on="movieId", how="left")
    similar["source"] = "collaborative"
    st.subheader(f"Viewers who rated {selected_title.split('  ·  ')[0]} also connect with")
    render_movie_cards(similar, catalog, limit=10)
    rating_counts = user_ratings["rating"].value_counts().reindex([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], fill_value=0)
    fig = px.bar(x=rating_counts.index.astype(str), y=rating_counts.values, labels={"x": "Rating", "y": "Number of movies"}, color=rating_counts.values, color_continuous_scale=["#6da4e6", "#ed7bc4"])
    fig.update_layout(**dark_chart_layout(), coloraxis_showscale=False, title="Viewer rating pattern")
    st.plotly_chart(fig, use_container_width=True)


def dark_chart_layout() -> dict:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#d9deef", "family": "DM Sans"},
        "margin": {"l": 16, "r": 16, "t": 35, "b": 20},
        "xaxis": {"gridcolor": "#28304a", "zerolinecolor": "#28304a"},
        "yaxis": {"gridcolor": "#28304a", "zerolinecolor": "#28304a"},
    }


def page_analytics(catalog: pd.DataFrame, ratings: pd.DataFrame) -> None:
    st.title("Analytics & Movie Catalog")
    st.caption("Explore rating behavior, genre coverage, and the complete searchable catalog.")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(ratings, x="rating", nbins=10, color_discrete_sequence=["#75a9f8"], title="Movie rating distribution")
        fig.update_layout(**dark_chart_layout(), bargap=0.12)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        genre_counts = catalog.assign(genre=catalog["genres"].str.split("|")).explode("genre")
        genre_counts = genre_counts[genre_counts["genre"] != "(no genres listed)"]["genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Movies"]
        fig = px.bar(genre_counts, x="Movies", y="Genre", orientation="h", color="Movies", color_continuous_scale=["#526b9d", "#e77bc1"], title="Movies by genre")
        layout = dark_chart_layout()
        layout["yaxis"]["categoryorder"] = "total ascending"
        fig.update_layout(**layout, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    top_rated = catalog.nlargest(15, "rating_count").sort_values("rating_count")
    fig = px.bar(top_rated, x="rating_count", y="title", orientation="h", color="avg_rating", color_continuous_scale=["#6da4e6", "#ed7bc4"], title="15 most-rated movies")
    fig.update_layout(**dark_chart_layout(), yaxis_title="", xaxis_title="Ratings", height=520)
    left.plotly_chart(fig, use_container_width=True)
    activity = ratings.groupby("userId").size().rename("ratings").reset_index()
    fig = px.histogram(activity, x="ratings", nbins=30, color_discrete_sequence=["#68dfc3"], title="User activity distribution")
    fig.update_layout(**dark_chart_layout(), xaxis_title="Ratings per user", yaxis_title="Users")
    right.plotly_chart(fig, use_container_width=True)

    st.subheader("Movie catalog")
    genres = sorted({genre for value in catalog["genres"] for genre in value.split("|") if genre != "(no genres listed)"})
    filter_cols = st.columns([1.5, 1, 1])
    search = filter_cols[0].text_input("Search titles", placeholder="Start typing a movie name")
    selected_genre = filter_cols[1].selectbox("Genre", ["All genres", *genres])
    years = catalog["year"].dropna()
    year_range = (int(years.min()), int(years.max())) if not years.empty else (1900, 2026)
    selected_years = filter_cols[2].slider("Release year", *year_range, value=year_range)
    filtered = catalog.copy()
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]
    if selected_genre != "All genres":
        filtered = filtered[filtered["genres"].str.contains(selected_genre, regex=False)]
    if not years.empty:
        filtered = filtered[filtered["year"].between(*selected_years) | filtered["year"].isna()]
    st.caption(f"Showing {len(filtered):,} of {len(catalog):,} movies")
    st.dataframe(
        filtered[["title", "genres", "year", "avg_rating", "rating_count"]].sort_values("rating_count", ascending=False),
        hide_index=True, use_container_width=True, height=440,
        column_config={"year": st.column_config.NumberColumn(format="%d"), "avg_rating": st.column_config.NumberColumn("Average rating", format="%.2f"), "rating_count": st.column_config.NumberColumn("Ratings", format="%d")},
    )


@st.cache_data(show_spinner=False)
def run_evaluation(_movies: pd.DataFrame, _ratings: pd.DataFrame, _tags: pd.DataFrame) -> pd.DataFrame:
    return evaluate_models(_movies, _ratings, _tags, k=10, max_users=100)


def page_evaluation(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> None:
    st.title("Offline Evaluation & Benchmarks")
    st.caption("An 80/20 per-user holdout compares ranking quality and rating prediction error.")
    st.info("The benchmark uses a reproducible sample of up to 100 users. Ratings of 4 stars or higher are relevant, with an 80/20 split per user. Results are cached for this session.")
    if st.button("Run benchmark", type="primary"):
        try:
            result = run_evaluation(movies, ratings, tags)
            st.session_state["evaluation_result"] = result
        except (ValueError, MemoryError) as error:
            st.error(f"Evaluation could not finish: {error}")
    result = st.session_state.get("evaluation_result")
    if result is None:
        st.caption("No benchmark has been run yet.")
        return
    st.dataframe(result, hide_index=True, use_container_width=True, column_config={metric: st.column_config.NumberColumn(format="%.4f") for metric in result.columns if metric != "Algorithm"})
    left, right = st.columns(2)
    ranking_metrics = ["Precision@K", "Recall@K", "NDCG@K", "Hit Rate@K", "MRR"]
    chart_data = result.melt(id_vars="Algorithm", value_vars=ranking_metrics, var_name="Metric", value_name="Score")
    fig = px.bar(chart_data, x="Metric", y="Score", color="Algorithm", barmode="group", color_discrete_sequence=["#71a8f4", "#e879bf", "#69ddc2"], title="Ranking metrics (higher is better)")
    fig.update_layout(**dark_chart_layout())
    left.plotly_chart(fig, use_container_width=True)
    error_data = result.melt(id_vars="Algorithm", value_vars=["RMSE", "MAE"], var_name="Metric", value_name="Error")
    fig = px.bar(error_data, x="Metric", y="Error", color="Algorithm", barmode="group", color_discrete_sequence=["#71a8f4", "#e879bf", "#69ddc2"], title="Rating prediction error (lower is better)")
    fig.update_layout(**dark_chart_layout())
    right.plotly_chart(fig, use_container_width=True)


def page_about() -> None:
    st.title("About the Project")
    st.markdown("""
### Problem statement
Movie catalogs are large, while each viewer has a distinct and evolving taste. This project combines movie attributes with community behavior to make recommendations that are personal, explainable, and useful to new viewers.

### Objectives
- Turn MovieLens ratings, genres, and tags into an interactive discovery experience.
- Compare content-based, item-based collaborative, and hybrid recommendations.
- Make recommendation logic inspectable through score explanations and offline evaluation.

### System architecture
`MovieLens CSVs → Pandas data preparation → TF-IDF content model + sparse adjusted-cosine model → normalized weighted hybrid → Streamlit pages and Plotly analytics`

### Algorithms
**Content-based filtering** builds a TF-IDF vector from each movie's genres and user tags. A profile is the mean vector of favorite movies, compared against the catalog with cosine similarity.

**Collaborative filtering** subtracts each viewer's mean rating from their ratings, then compares movie columns with cosine similarity. Predictions combine neighboring-item residuals with the viewer's mean rating.

**Hybrid recommendation** normalizes each model's candidate scores and blends them using the selected weights. Movies already rated by the selected profile are excluded.

### Mathematics
Cosine similarity: $\\displaystyle \\operatorname{sim}(i,j)=\\frac{\\mathbf{x}_i\\cdot\\mathbf{x}_j}{\\|\\mathbf{x}_i\\|\\|\\mathbf{x}_j\\|}$

Adjusted rating: $\\displaystyle r'_{u,i}=r_{u,i}-\\bar{r}_u$

Hybrid score: $\\displaystyle s(u,i)=w_c s_c(u,i)+w_f s_f(u,i),\\quad w_c+w_f=1$

Weighted prediction: $\\displaystyle \\hat r_{u,i}=\\bar r_u+\\frac{\\sum_{j\\in I_u}\\max(0,\\operatorname{sim}(i,j))r'_{u,j}}{\\sum_{j\\in I_u}\\max(0,\\operatorname{sim}(i,j))}$

### Technology stack
Python, Streamlit, Pandas, NumPy, scikit-learn, SciPy, Plotly, and pytest. MovieLens ml-latest-small is downloaded from GroupLens on first launch.

### Limitations
The dataset is historical and relatively small; tags are sparse; implicit feedback and contextual signals are not modeled. New users need favorite movies or genres to personalize content results, while collaborative cold-start falls back to community popularity.

### Future enhancements
Add richer movie metadata and poster art, implicit-feedback ranking, time-aware preferences, user accounts, and larger-scale approximate-neighbor retrieval.
""")


def main() -> None:
    st.sidebar.markdown("# 🎬 Smart Recs")
    st.sidebar.caption("Collaborative + content intelligence")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Home Dashboard", "Smart Recommendation Agent", "Content-Based Recommendations",
            "Collaborative Recommendations", "Analytics & Movie Catalog",
            "Offline Evaluation & Benchmarks", "About Project",
        ],
        label_visibility="collapsed",
    )
    try:
        movies, ratings, tags, catalog = get_dataset()
    except RuntimeError as error:
        st.error(str(error))
        st.info("The app downloads the MovieLens ml-latest-small archive automatically on first launch.")
        st.stop()

    fingerprint = f"{len(ratings)}-{len(movies)}"
    content_model = None
    collaborative_model = None
    if page in {"Smart Recommendation Agent", "Content-Based Recommendations"}:
        content_model = get_content_model(movies, tags, fingerprint)
    if page in {"Smart Recommendation Agent", "Collaborative Recommendations"}:
        collaborative_model = get_collaborative_model(ratings, fingerprint)

    if page == "Home Dashboard":
        page_home(catalog, ratings)
    elif page == "Smart Recommendation Agent":
        page_agent(catalog, ratings, content_model, collaborative_model)
    elif page == "Content-Based Recommendations":
        page_content(catalog, content_model)
    elif page == "Collaborative Recommendations":
        page_collaborative(catalog, ratings, collaborative_model)
    elif page == "Analytics & Movie Catalog":
        page_analytics(catalog, ratings)
    elif page == "Offline Evaluation & Benchmarks":
        page_evaluation(movies, ratings, tags)
    else:
        page_about()


if __name__ == "__main__":
    main()