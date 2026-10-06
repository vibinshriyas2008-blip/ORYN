"""
ORYN API — Adaptive Hybrid Movie Recommendation Engine
FastAPI backend serving real ML models trained on MovieLens dataset.
"""
import sys
import os
import json
import re
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────

class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float  # 0.5 – 5.0


class LikeRequest(BaseModel):
    user_id: int
    movie_id: int


class RecommendRequest(BaseModel):
    user_id: int
    n: int = 10
    include_explanation: bool = True
    diversity: bool = True


class OnboardingRequest(BaseModel):
    user_id: int
    genres: List[str]
    liked_movie_ids: List[int]
    ratings: dict = {}  # movie_id -> rating


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ORYN API",
    version="1.0.0",
    description="Adaptive Hybrid Movie Recommendation Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# In-memory state
# ─────────────────────────────────────────────────────────────

users_db: dict = {}
user_ratings: dict = {}
user_likes: dict = {}
next_user_id: int = 1

movies_dict: dict = {}
all_genres: set = set()
model_status: dict = {"status": "uninitialized"}

# ML models (loaded at startup)
pop_model = None
cf_model = None
cb_model = None
hybrid_model = None

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-latest-small")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def generate_poster_color(movie_id: int) -> str:
    """Deterministic HSL colour from movie_id for poster placeholder."""
    h = (movie_id * 137.5) % 360
    s = 50 + (movie_id % 30)
    l = 35 + (movie_id % 20)
    return f"hsl({h:.0f},{s:.0f}%,{l:.0f}%)"


GENRE_DESCRIPTIONS = {
    "Action": "A pulse-pounding action adventure full of thrills and excitement.",
    "Comedy": "A laugh-out-loud comedy that will leave you smiling.",
    "Drama": "A gripping drama exploring the human condition.",
    "Horror": "A terrifying horror film that will keep you on the edge of your seat.",
    "Sci-Fi": "A mind-bending science fiction journey into the unknown.",
    "Romance": "A heartfelt romance story about love and connection.",
    "Thriller": "A tense, suspenseful thriller that keeps you guessing.",
    "Animation": "A beautifully animated film for all ages.",
    "Documentary": "An eye-opening documentary exploring fascinating real-world topics.",
    "Fantasy": "An enchanting fantasy epic in a world of magic and wonder.",
    "Crime": "A gripping crime story full of twists and moral complexity.",
    "Mystery": "A captivating mystery that unfolds clue by clue.",
    "Adventure": "An epic adventure full of discovery and excitement.",
}


def make_description(genres: List[str], year: Optional[int]) -> str:
    """Generate a genre-based description for a movie."""
    for g in genres:
        if g in GENRE_DESCRIPTIONS:
            base = GENRE_DESCRIPTIONS[g]
            if year:
                return f"{base} ({year})"
            return base
    year_str = f" ({year})" if year else ""
    genre_str = ", ".join(genres[:2]) if genres else "General"
    return f"A {genre_str.lower()} film{year_str}."


def _explanation_for(movie_id: int, user_prefs: dict, interaction_count: int,
                     cf_score: float, cb_score: float) -> str:
    """Generate a human-readable explanation for a recommendation."""
    genres = movies_dict.get(movie_id, {}).get("genres", [])
    liked = user_prefs.get("liked_movie_ids", [])

    if interaction_count == 0:
        return "Popular with ORYN users"
    if interaction_count <= 5 and liked:
        liked_titles = [movies_dict[m]["title"] for m in liked[:1] if m in movies_dict]
        if liked_titles:
            return f"Because you liked {liked_titles[0]}"
        return "Matches your taste preferences"
    if cf_score > cb_score:
        return "Popular with users like you"
    if genres:
        pref_genres = user_prefs.get("genres", [])
        for g in genres:
            if g in pref_genres:
                return f"Matches your {g} taste"
        return f"Similar to your recent watches"
    return "Recommended by ORYN AI"


# ─────────────────────────────────────────────────────────────
# Startup — load data & models
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global movies_dict, all_genres, model_status
    global pop_model, cf_model, cb_model, hybrid_model

    try:
        # ── Load movie data ──────────────────────────────────
        movies_csv = os.path.join(DATA_DIR, "movies.csv")
        ratings_csv = os.path.join(DATA_DIR, "ratings.csv")
        tags_csv = os.path.join(DATA_DIR, "tags.csv")

        if not os.path.exists(movies_csv):
            model_status = {"status": "error", "message": "MovieLens data not found. Run scripts/download_data.py"}
            return

        movies_df = pd.read_csv(movies_csv)
        ratings_df = pd.read_csv(ratings_csv)

        # Rating aggregates
        rating_counts = ratings_df.groupby("movieId").size()
        rating_means = ratings_df.groupby("movieId")["rating"].mean()
        global_mean = rating_means.mean()
        global_count = rating_counts.mean()

        tags_df = None
        if os.path.exists(tags_csv):
            raw_tags = pd.read_csv(tags_csv)
            tags_df = raw_tags.groupby("movieId")["tag"].apply(lambda x: list(x.dropna().astype(str))).reset_index()
            tags_dict = dict(zip(tags_df["movieId"], tags_df["tag"]))
        else:
            tags_dict = {}

        for _, row in movies_df.iterrows():
            mid = int(row["movieId"])
            title = str(row["title"])
            genres_str = str(row.get("genres", ""))

            year_match = re.search(r"\((\d{4})\)", title)
            year = int(year_match.group(1)) if year_match else None
            clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

            genres = (
                [g for g in genres_str.split("|") if g and g != "(no genres listed)"]
                if genres_str else []
            )
            for g in genres:
                all_genres.add(g)

            rc = int(rating_counts.get(mid, 0))
            rm = float(rating_means.get(mid, global_mean))

            # Bayesian average rating
            v = rc
            R = rm
            C = global_count
            m = global_mean
            bayesian = (v / (v + C)) * R + (C / (v + C)) * m if (v + C) > 0 else m

            movies_dict[mid] = {
                "id": mid,
                "title": clean_title,
                "year": year,
                "genres": genres,
                "rating": round(rm, 2),
                "bayesian_rating": round(bayesian, 4),
                "rating_count": rc,
                "description": make_description(genres, year),
                "tags": tags_dict.get(mid, [])[:10],
                "poster_color": generate_poster_color(mid),
            }

        # ── Load ML models ───────────────────────────────────
        import joblib

        def try_load(path, name):
            if os.path.exists(path):
                try:
                    m = joblib.load(path)
                    print(f"[ORYN] Loaded {name} from {path}")
                    return m
                except Exception as e:
                    print(f"[ORYN] Failed to load {name}: {e}")
            return None

        pop_model = try_load(os.path.join(MODELS_DIR, "popularity.pkl"), "Popularity")
        cf_model  = try_load(os.path.join(MODELS_DIR, "cf.pkl"), "CF")
        cb_model  = try_load(os.path.join(MODELS_DIR, "cb.pkl"), "CB")
        hybrid_model = try_load(os.path.join(MODELS_DIR, "hybrid.pkl"), "Hybrid")

        # ── If models missing, train on the fly ─────────────
        if hybrid_model is None:
            print("[ORYN] Models not found — training on the fly (may take ~30s)...")
            from src.data_loader import load_ratings, load_movies, load_tags
            from src.preprocessing import get_interaction_matrix, split_train_test
            from src.collaborative import CollaborativeFilter
            from src.content_based import ContentBasedFilter
            from src.popularity import PopularityRecommender
            from src.hybrid import HybridRecommender

            ratings = load_ratings()
            movies = load_movies()
            tags = load_tags()
            train_df, _ = split_train_test(ratings)

            pop_model = PopularityRecommender()
            pop_model.fit(train_df)

            cf_model = CollaborativeFilter(n_components=50)
            matrix, uid_map, mid_map = get_interaction_matrix(train_df)
            cf_model.fit(matrix, uid_map, mid_map)

            cb_model = ContentBasedFilter()
            cb_model.fit(movies, tags)

            hybrid_model = HybridRecommender()
            hybrid_model.fit(train_df, movies, tags)

            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(pop_model, os.path.join(MODELS_DIR, "popularity.pkl"))
            joblib.dump(cf_model, os.path.join(MODELS_DIR, "cf.pkl"))
            joblib.dump(cb_model, os.path.join(MODELS_DIR, "cb.pkl"))
            joblib.dump(hybrid_model, os.path.join(MODELS_DIR, "hybrid.pkl"))
            print("[ORYN] Models trained and saved.")

        model_status = {
            "status": "ready",
            "movies_count": len(movies_dict),
            "genres_count": len(all_genres),
            "models": {
                "popularity": pop_model is not None,
                "collaborative": cf_model is not None,
                "content_based": cb_model is not None,
                "hybrid": hybrid_model is not None,
            },
        }
        print(f"[ORYN] Startup complete — {len(movies_dict)} movies loaded.")

    except Exception as e:
        import traceback
        model_status = {"status": "error", "message": str(e), "trace": traceback.format_exc()}
        print(f"[ORYN] Startup error: {e}")


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "ORYN API", "version": "1.0.0", "status": "operational",
            "tagline": "Discover What Feels Made for You."}


@app.get("/health")
def health():
    return model_status


# ── Auth ─────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(user: UserCreate):
    global next_user_id
    for u in users_db.values():
        if u["email"] == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    uid = next_user_id
    next_user_id += 1
    users_db[uid] = {
        "id": uid,
        "name": user.name,
        "email": user.email,
        "password_hash": hashlib.sha256(user.password.encode()).hexdigest(),
        "genres": [],
        "liked_movies": [],
        "ratings": {},
        "created_at": datetime.now().isoformat(),
        "interaction_count": 0,
    }
    user_ratings[uid] = {}
    user_likes[uid] = set()
    token = f"oryn_tok_{uid}_{uuid.uuid4().hex}"
    safe_user = {k: v for k, v in users_db[uid].items() if k != "password_hash"}
    return {"user_id": uid, "token": token, "user": safe_user}


@app.post("/auth/login")
def login(user: UserLogin):
    pwd_hash = hashlib.sha256(user.password.encode()).hexdigest()
    for uid, u in users_db.items():
        if u["email"] == user.email and u["password_hash"] == pwd_hash:
            token = f"oryn_tok_{uid}_{uuid.uuid4().hex}"
            safe_user = {k: v for k, v in u.items() if k != "password_hash"}
            return {"user": safe_user, "token": token}
    raise HTTPException(status_code=401, detail="Invalid email or password")


# ── Movies ───────────────────────────────────────────────────

@app.get("/movies")
def get_movies(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    genre: str = "",
    search: str = "",
    sort: str = "rating",
):
    movies_list = list(movies_dict.values())

    if genre:
        gl = genre.lower()
        movies_list = [m for m in movies_list if any(g.lower() == gl for g in m["genres"])]
    if search:
        sl = search.lower()
        movies_list = [m for m in movies_list if sl in m["title"].lower()]

    if sort == "rating":
        movies_list.sort(key=lambda x: x.get("bayesian_rating", 0), reverse=True)
    elif sort == "newest":
        movies_list.sort(key=lambda x: x.get("year") or 0, reverse=True)
    elif sort == "popular":
        movies_list.sort(key=lambda x: x.get("rating_count", 0), reverse=True)
    elif sort == "title":
        movies_list.sort(key=lambda x: x.get("title", ""))

    start = (page - 1) * limit
    return {
        "total": len(movies_list),
        "page": page,
        "limit": limit,
        "results": movies_list[start: start + limit],
    }


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    if movie_id not in movies_dict:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie = movies_dict[movie_id]
    similar = []
    if cb_model is not None:
        try:
            sim_raw = cb_model.recommend_from_movie(movie_id, n=8)
            similar = [
                {**movies_dict[mid], "similarity_score": round(float(sc), 4)}
                for mid, sc in sim_raw
                if mid in movies_dict
            ]
        except Exception:
            pass

    if not similar:
        # Fallback: genre-based similarity
        tg = set(movie["genres"])
        scored = [
            (len(tg & set(m["genres"])), m)
            for mid, m in movies_dict.items()
            if mid != movie_id
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        similar = [m for _, m in scored[:8] if _[0] > 0]

    return {"movie": movie, "similar": similar}


@app.get("/popular")
def get_popular(n: int = Query(20, ge=1, le=100), genre: str = ""):
    if pop_model is not None:
        recs = pop_model.recommend(n=n * 3)  # get extra to allow genre filter
        results = []
        for mid, score in recs:
            if mid not in movies_dict:
                continue
            m = movies_dict[mid]
            if genre and not any(g.lower() == genre.lower() for g in m["genres"]):
                continue
            results.append({**m, "pop_score": round(float(score), 4)})
            if len(results) >= n:
                break
        return results

    # Fallback
    movies_list = list(movies_dict.values())
    if genre:
        movies_list = [m for m in movies_list if any(g.lower() == genre.lower() for g in m["genres"])]
    movies_list.sort(key=lambda x: x.get("bayesian_rating", 0), reverse=True)
    return movies_list[:n]


@app.get("/similar/{movie_id}")
def get_similar(movie_id: int, n: int = Query(10, ge=1, le=50)):
    if movie_id not in movies_dict:
        raise HTTPException(status_code=404, detail="Movie not found")

    if cb_model is not None:
        try:
            sim_raw = cb_model.recommend_from_movie(movie_id, n=n)
            return [
                {**movies_dict[mid], "similarity_score": round(float(sc), 4)}
                for mid, sc in sim_raw
                if mid in movies_dict
            ]
        except Exception:
            pass

    # Fallback
    tg = set(movies_dict[movie_id]["genres"])
    scored = [(len(tg & set(m["genres"])), m) for mid, m in movies_dict.items() if mid != movie_id]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for cnt, m in scored[:n] if cnt > 0]


# ── Recommendations ──────────────────────────────────────────

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user = users_db[req.user_id]
    interaction_count = user.get("interaction_count", 0)
    liked_ids = list(user_likes.get(req.user_id, set()))
    rated_ids = list(user_ratings.get(req.user_id, {}).keys())
    all_seen = set(liked_ids + rated_ids)

    user_prefs = {
        "genres": user.get("genres", []),
        "liked_movie_ids": liked_ids + user.get("liked_movies", []),
    }

    recs_raw = []

    if hybrid_model is not None:
        try:
            recs_raw = hybrid_model.recommend(
                user_id=req.user_id,
                interaction_count=interaction_count,
                n=req.n * 2,
                user_prefs=user_prefs,
            )
        except Exception:
            pass

    # Fallback to popularity
    if not recs_raw and pop_model is not None:
        raw_pop = pop_model.recommend(n=req.n * 2, exclude_ids=list(all_seen))
        recs_raw = [
            {"movie_id": mid, "score": sc, "cf_score": 0.0, "cb_score": sc,
             "explanation": "Popular with ORYN users"}
            for mid, sc in raw_pop
        ]

    # Final fallback to static sort
    if not recs_raw:
        ml = sorted(movies_dict.values(), key=lambda x: x.get("bayesian_rating", 0), reverse=True)
        recs_raw = [
            {"movie_id": m["id"], "score": m["bayesian_rating"], "cf_score": 0.0,
             "cb_score": m["bayesian_rating"], "explanation": "Highly rated on ORYN"}
            for m in ml
            if m["id"] not in all_seen
        ]

    # Re-rank for diversity if requested
    if req.diversity and recs_raw and len(recs_raw) > req.n:
        try:
            from src.reranking import rerank
            movies_df_local = pd.DataFrame(list(movies_dict.values()))
            reranked = rerank(
                [(r["movie_id"], r["score"]) for r in recs_raw],
                movies_df_local,
                n=req.n,
            )
            reranked_ids = [mid for mid, _ in reranked]
            recs_raw = [r for r in recs_raw if r["movie_id"] in reranked_ids]
            id_order = {mid: i for i, mid in enumerate(reranked_ids)}
            recs_raw.sort(key=lambda r: id_order.get(r["movie_id"], 999))
        except Exception:
            recs_raw = recs_raw[:req.n]
    else:
        recs_raw = recs_raw[:req.n]

    results = []
    for r in recs_raw:
        mid = r["movie_id"]
        if mid not in movies_dict:
            continue
        m = movies_dict[mid]
        cf_s = r.get("cf_score", 0.0)
        cb_s = r.get("cb_score", 0.0)
        explanation = r.get("explanation") or _explanation_for(
            mid, user_prefs, interaction_count, cf_s, cb_s
        )
        results.append({
            **m,
            "score": round(float(r["score"]), 4),
            "cf_score": round(float(cf_s), 4),
            "cb_score": round(float(cb_s), 4),
            "explanation": explanation,
            "is_novel": m.get("rating_count", 0) < 50,
        })

    return results


# ── User Interactions ─────────────────────────────────────────

@app.post("/rate")
def rate_movie(req: RatingRequest):
    if req.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    if not (0.5 <= req.rating <= 5.0):
        raise HTTPException(status_code=422, detail="Rating must be between 0.5 and 5.0")
    user_ratings.setdefault(req.user_id, {})[req.movie_id] = req.rating
    users_db[req.user_id]["ratings"][req.movie_id] = req.rating
    users_db[req.user_id]["interaction_count"] += 1
    return {"status": "success", "rating": req.rating}


@app.post("/like")
def like_movie(req: LikeRequest):
    if req.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    user_likes.setdefault(req.user_id, set()).add(req.movie_id)
    if req.movie_id not in users_db[req.user_id]["liked_movies"]:
        users_db[req.user_id]["liked_movies"].append(req.movie_id)
    users_db[req.user_id]["interaction_count"] += 1
    return {"status": "success"}


@app.post("/onboarding")
def onboarding(req: OnboardingRequest):
    if req.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    users_db[req.user_id]["genres"] = req.genres
    users_db[req.user_id]["liked_movies"] = req.liked_movie_ids
    users_db[req.user_id]["ratings"] = req.ratings
    for mid in req.liked_movie_ids:
        user_likes.setdefault(req.user_id, set()).add(mid)
    for mid, r in req.ratings.items():
        user_ratings.setdefault(req.user_id, {})[mid] = float(r)
    users_db[req.user_id]["interaction_count"] = (
        len(req.liked_movie_ids) + len(req.ratings)
    )

    # Return initial recommendations
    from fastapi.testclient import TestClient  # lazy import to avoid circularity
    try:
        user_prefs = {"genres": req.genres, "liked_movie_ids": req.liked_movie_ids}
        interaction_count = users_db[req.user_id]["interaction_count"]
        recs_raw = []
        if hybrid_model is not None:
            recs_raw = hybrid_model.recommend(
                user_id=req.user_id,
                interaction_count=interaction_count,
                n=10,
                user_prefs=user_prefs,
            )
        if not recs_raw and pop_model is not None:
            pop_raw = pop_model.recommend(n=10)
            recs_raw = [{"movie_id": mid, "score": sc, "cf_score": 0.0,
                         "cb_score": sc, "explanation": "Popular on ORYN"}
                        for mid, sc in pop_raw]

        recs = [
            {**movies_dict[r["movie_id"]], "explanation": r.get("explanation", "Recommended for you")}
            for r in recs_raw[:10]
            if r["movie_id"] in movies_dict
        ]
    except Exception:
        recs = []

    return {"status": "success", "recommendations": recs}


# ── User Profile ─────────────────────────────────────────────

@app.get("/user/{user_id}")
def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    safe = {k: v for k, v in users_db[user_id].items() if k != "password_hash"}
    # Enrich liked_movies with full info
    safe["liked_movies_detail"] = [
        movies_dict[mid] for mid in safe.get("liked_movies", []) if mid in movies_dict
    ]
    return safe


@app.get("/user/{user_id}/ratings")
def get_user_ratings(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    rated = user_ratings.get(user_id, {})
    return [
        {**movies_dict[int(mid)], "user_rating": r}
        for mid, r in rated.items()
        if int(mid) in movies_dict
    ]


@app.get("/user/{user_id}/likes")
def get_user_likes(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return [movies_dict[mid] for mid in user_likes.get(user_id, set()) if mid in movies_dict]


# ── Analytics & Metadata ─────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    eval_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(eval_file):
        try:
            with open(eval_file) as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read metrics: {e}")
    raise HTTPException(status_code=404, detail="Metrics not yet computed. Run scripts/train_models.py first.")


@app.get("/genres")
def get_genres():
    return sorted(all_genres)
