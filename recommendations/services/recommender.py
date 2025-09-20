from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Movie, Rating, Tag


def _safe_to_df(queryset, columns: List[str]) -> pd.DataFrame:
    rows = list(queryset.values(*columns))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


class CollaborativeFiltering:
    def __init__(self):
        self._rating_matrix = None
        self._user_index = None
        self._item_index = None

    def _build_matrix(self) -> None:
        ratings_df = _safe_to_df(Rating.objects.all(), ["user_id", "movie_id", "rating"])
        if ratings_df.empty:
            self._rating_matrix = pd.DataFrame()
            return
        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating")
        self._rating_matrix = pivot
        self._user_index = pivot.index
        self._item_index = pivot.columns

    def user_based_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        if self._rating_matrix is None:
            self._build_matrix()
        if self._rating_matrix.empty or user_id not in self._rating_matrix.index:
            return []
        ratings = self._rating_matrix
        sims = cosine_similarity(ratings.fillna(0))
        user_pos = list(ratings.index).index(user_id)
        user_sims = sims[user_pos]
        user_ratings = ratings.iloc[user_pos]
        unrated_mask = user_ratings.isna()
        rated_matrix = ratings.fillna(0)
        scores = np.dot(user_sims, rated_matrix.values) / (np.abs(user_sims).sum() + 1e-8)
        scores_series = pd.Series(scores, index=ratings.columns)
        recs = scores_series[unrated_mask].sort_values(ascending=False).head(top_n)
        return list(recs.index)

    def item_based_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        if self._rating_matrix is None:
            self._build_matrix()
        if self._rating_matrix.empty or user_id not in self._rating_matrix.index:
            return []
        ratings = self._rating_matrix
        item_sims = cosine_similarity(ratings.fillna(0).T)
        item_sims_df = pd.DataFrame(item_sims, index=ratings.columns, columns=ratings.columns)
        user_ratings = ratings.loc[user_id]
        scores = item_sims_df.mul(user_ratings, axis=0).sum(axis=0)
        scores = scores[user_ratings.isna()].sort_values(ascending=False).head(top_n)
        return list(scores.index)


class ContentBasedFiltering:
    def __init__(self):
        self._tfidf = None
        self._matrix = None
        self._movie_index = None

    def _build_features(self) -> None:
        movies_df = _safe_to_df(Movie.objects.all(), ["movie_id", "genres"])
        tags_df = _safe_to_df(Tag.objects.all(), ["movie_id", "tag"])
        if movies_df.empty:
            self._matrix = None
            return
        if not tags_df.empty:
            tags_agg = tags_df.groupby("movie_id")["tag"].apply(lambda s: " ".join(map(str, s))).rename("tags")
            movies_df = movies_df.merge(tags_agg, on="movie_id", how="left")
        else:
            movies_df["tags"] = ""
        movies_df["text"] = movies_df["genres"].fillna("") + " " + movies_df["tags"].fillna("")
        vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_#]+")
        self._matrix = vectorizer.fit_transform(movies_df["text"].fillna(""))
        self._tfidf = vectorizer
        self._movie_index = movies_df["movie_id"].tolist()

    def similar_items_top_n(self, liked_movie_ids: List[int], top_n: int = 10) -> List[int]:
        if self._matrix is None:
            self._build_features()
        if self._matrix is None or not liked_movie_ids:
            return []
        id_to_pos = {mid: i for i, mid in enumerate(self._movie_index)}
        valid_positions = [id_to_pos[m] for m in liked_movie_ids if m in id_to_pos]
        if not valid_positions:
            return []
        sims = cosine_similarity(self._matrix[valid_positions], self._matrix).mean(axis=0)
        sims = np.asarray(sims).ravel()
        ranked_pos = np.argsort(-sims)
        ranked_ids = [self._movie_index[p] for p in ranked_pos if self._movie_index[p] not in liked_movie_ids]
        return ranked_ids[:top_n]


class HybridRecommender:
    def __init__(self, w_cf: float = 0.6, w_content: float = 0.4):
        self.cf = CollaborativeFiltering()
        self.cb = ContentBasedFiltering()
        self.w_cf = w_cf
        self.w_content = w_content

    def _user_liked_movies(self, user_id: int, min_rating: float = 4.0) -> List[int]:
        qs = Rating.objects.filter(user_id=user_id, rating__gte=min_rating).values_list("movie_id", flat=True)
        return list(qs)

    def get_hybrid_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict]:
        cf_user = self.cf.user_based_top_n(user_id, top_n=top_n * 3)
        cf_item = self.cf.item_based_top_n(user_id, top_n=top_n * 3)
        liked = self._user_liked_movies(user_id)
        cb = self.cb.similar_items_top_n(liked, top_n=top_n * 3)

        # score aggregation
        def to_scores(ids: List[int], base: float) -> Dict[int, float]:
            return {mid: base / (i + 1) for i, mid in enumerate(ids)}

        scores = {}
        for m, s in to_scores(cf_user, self.w_cf).items():
            scores[m] = scores.get(m, 0) + s
        for m, s in to_scores(cf_item, self.w_cf).items():
            scores[m] = scores.get(m, 0) + s
        for m, s in to_scores(cb, self.w_content).items():
            scores[m] = scores.get(m, 0) + s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Filter out movies that don't exist in database and user's liked movies
        existing_movies = set(Movie.objects.values_list('movie_id', flat=True))
        top_ids = [m for m, _ in ranked if m not in liked and m in existing_movies][:top_n]
        movies = {m.movie_id: m for m in Movie.objects.filter(movie_id__in=top_ids)}
        return [
            {
                "movie_id": mid,
                "title": movies.get(mid).title if mid in movies and movies.get(mid) else f"Movie {mid}",
            }
            for mid in top_ids
        ]


def train_test_split_evaluate(test_ratio: float = 0.2, top_n: int = 10, max_users: int = 100) -> Dict[str, float]:
    """
    Evaluate recommender with train/test split using advanced collaborative filtering.
    
    Args:
        test_ratio: Fraction of data to use for testing
        top_n: Number of recommendations to generate
        max_users: Maximum number of users to evaluate (for performance)
    """
    ratings_df = _safe_to_df(Rating.objects.all(), ["user_id", "movie_id", "rating"])
    if ratings_df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "users_evaluated": 0}
    
    # Use random split for better evaluation
    shuffled = ratings_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split = int(len(shuffled) * (1 - test_ratio))
    train_df = shuffled.iloc[:split]
    test_df = shuffled.iloc[split:]
    
    if test_df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "users_evaluated": 0}
    
    # Group test ratings by user
    user_to_true = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    
    # Limit number of users for performance
    user_ids = list(user_to_true.keys())
    if len(user_ids) > max_users:
        user_ids = user_ids[:max_users]
    
    # Build user-item matrix from training data
    train_matrix = train_df.pivot_table(index="user_id", columns="movie_id", values="rating")
    
    # Calculate user similarities using cosine similarity
    user_similarities = cosine_similarity(train_matrix.fillna(0))
    user_sim_df = pd.DataFrame(user_similarities, index=train_matrix.index, columns=train_matrix.index)
    
    # Calculate item similarities for item-based collaborative filtering
    item_similarities = cosine_similarity(train_matrix.fillna(0).T)
    item_sim_df = pd.DataFrame(item_similarities, index=train_matrix.columns, columns=train_matrix.columns)
    
    precisions = []
    recalls = []
    
    for user_id in user_ids:
        true_set = user_to_true[user_id]
        
        if user_id not in train_matrix.index:
            # User not in training data, use popularity
            movie_popularity = train_df.groupby("movie_id")["rating"].agg(['count', 'mean']).reset_index()
            movie_popularity = movie_popularity.sort_values(['count', 'mean'], ascending=[False, False])
            popular_movies = movie_popularity.head(top_n * 2)['movie_id'].tolist()
            pred_ids = popular_movies[:top_n]
        else:
            # Advanced collaborative filtering approach
            user_ratings = train_matrix.loc[user_id]
            user_sims = user_sim_df.loc[user_id]
            
            # Get movies not rated by user
            unrated_movies = user_ratings[user_ratings.isna()].index
            
            # Calculate predicted ratings using both user-based and item-based CF
            predictions = []
            
            for movie_id in unrated_movies:
                # User-based collaborative filtering
                user_pred = 0
                user_weight = 0
                
                # Find users who rated this movie
                movie_ratings = train_matrix[movie_id].dropna()
                if len(movie_ratings) > 0:
                    common_users = movie_ratings.index.intersection(user_sims.index)
                    if len(common_users) > 0:
                        similarities = user_sims[common_users]
                        ratings = movie_ratings[common_users]
                        
                        # Weighted average with similarity threshold
                        valid_sims = similarities[similarities > 0.1]  # Only consider similar users
                        if len(valid_sims) > 0:
                            valid_ratings = ratings[valid_sims.index]
                            user_pred = (valid_sims * valid_ratings).sum() / valid_sims.sum()
                            user_weight = len(valid_sims)
                
                # Item-based collaborative filtering
                item_pred = 0
                item_weight = 0
                
                # Find movies similar to this one that user has rated
                if movie_id in item_sim_df.index:
                    movie_sims = item_sim_df.loc[movie_id]
                    rated_movies = user_ratings[user_ratings.notna()].index
                    common_movies = movie_sims.index.intersection(rated_movies)
                    
                    if len(common_movies) > 0:
                        similarities = movie_sims[common_movies]
                        ratings = user_ratings[common_movies]
                        
                        # Weighted average with similarity threshold
                        valid_sims = similarities[similarities > 0.1]
                        if len(valid_sims) > 0:
                            valid_ratings = ratings[valid_sims.index]
                            item_pred = (valid_sims * valid_ratings).sum() / valid_sims.sum()
                            item_weight = len(valid_sims)
                
                # Combine user-based and item-based predictions
                if user_weight > 0 and item_weight > 0:
                    # Weighted combination
                    total_weight = user_weight + item_weight
                    combined_pred = (user_pred * user_weight + item_pred * item_weight) / total_weight
                elif user_weight > 0:
                    combined_pred = user_pred
                elif item_weight > 0:
                    combined_pred = item_pred
                else:
                    # Fallback to global average
                    movie_avg = train_df[train_df['movie_id'] == movie_id]['rating'].mean()
                    user_avg = train_df[train_df['user_id'] == user_id]['rating'].mean()
                    combined_pred = (movie_avg + user_avg) / 2 if not pd.isna(movie_avg) and not pd.isna(user_avg) else 3.0
                
                predictions.append((movie_id, combined_pred))
            
            # Sort by predicted rating and take top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            pred_ids = [movie_id for movie_id, _ in predictions[:top_n]]
        
        pred_set = set(pred_ids)
        tp = len(true_set & pred_set)
        precision = tp / max(len(pred_set), 1)
        recall = tp / max(len(true_set), 1)
        precisions.append(precision)
        recalls.append(recall)
    
    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "users_evaluated": len(user_ids)}


