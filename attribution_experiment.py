import argparse
import json
import math
import time
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
from math import comb

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from run_benchmarks import (
    ece_score,
    fit_gru_count,
    fit_set_transformer,
    fit_transformer_count,
    predict_gru_count_tail_probs,
    predict_set_transformer,
    predict_transformer_count_tail_probs,
    poisson_binomial_tail,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_10M_URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
MOVIELENS_20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
MODEL_ORDER = [
    "MeanIndep",
    "ARRollout",
    "SeqGRUCount",
    "SeqTXCount",
    "SeqTXCountPM",
    "ObjMLP",
    "ObjSetTF",
    "ObjSetTFPM",
    "SeqTXGlobal",
    "ObjSetGlobal",
    "TwoTowerIndep",
    "TwoTowerDropout",
    "TwoTowerEnsMean",
    "TwoTowerEnsMix",
    "ANP",
    "FlowLatent",
    "GlobalGrid",
]
DEFAULT_MODELS = [
    "MeanIndep",
    "ARRollout",
    "SeqGRUCount",
    "SeqTXCount",
    "SeqTXCountPM",
    "ObjMLP",
    "ObjSetTF",
    "ObjSetTFPM",
    "SeqTXGlobal",
    "ObjSetGlobal",
    "TwoTowerIndep",
    "TwoTowerDropout",
    "TwoTowerEnsMean",
    "TwoTowerEnsMix",
    "ANP",
    "FlowLatent",
    "GlobalGrid",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def binary_metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "acc": float(np.mean((prob >= 0.5) == y_true)),
        "nll": float(-np.mean(y_true * np.log(prob) + (1 - y_true) * np.log1p(-prob))),
        "brier": float(np.mean((prob - y_true) ** 2)),
        "ece": float(ece_score(y_true, prob)),
    }


def triage_metrics(y_true, prob, tau=0.72, reveal_cost=0.30):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    confident = np.maximum(prob, 1.0 - prob) >= tau
    pred = prob >= 0.5
    classify_utility = (pred == y_true).astype(np.float64)
    utility = np.where(confident, classify_utility, 1.0 - reveal_cost)
    out = binary_metrics(y_true, prob)
    out.update(
        {
            "utility": float(np.mean(utility)),
            "regret": float(1.0 - np.mean(utility)),
            "defer_rate": float(np.mean(~confident)),
        }
    )
    return out


def posterior_metrics(y_true, prob, tau=0.72, ref_prob=None, taxscore_bin=-1):
    out = triage_metrics(y_true, prob, tau=tau)
    out["num_examples"] = int(len(y_true))
    out["taxscore_bin"] = int(taxscore_bin)
    if ref_prob is None:
        out["tail_tv_proxy"] = 0.0
    else:
        out["tail_tv_proxy"] = float(
            np.mean(np.abs(np.asarray(prob, dtype=np.float64) - np.asarray(ref_prob, dtype=np.float64)))
        )
    return out


def choose_tau(y_val, prob, reveal_cost=0.30):
    best_tau, best = 0.72, -1e9
    for tau in np.arange(0.55, 0.96, 0.05):
        util = triage_metrics(y_val, prob, tau=float(tau), reveal_cost=reveal_cost)["utility"]
        if util > best:
            best_tau, best = float(tau), util
    return best_tau


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def aggregate_metric_dict(records):
    keys = records[0].keys()
    return {key: mean_std([record[key] for record in records]) for key in keys}


def aggregate_any(values):
    first = values[0]
    if isinstance(first, (int, float, np.integer, np.floating, bool)):
        return mean_std(values)
    if isinstance(first, dict):
        keys = first.keys()
        return {key: aggregate_any([value[key] for value in values]) for key in keys}
    if isinstance(first, list):
        if not first:
            return []
        return [aggregate_any([value[idx] for value in values]) for idx in range(len(first))]
    return first


def model_param_count(model):
    if model is None:
        return 0
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        return int(sum(param.numel() for param in model.parameters()))
    if hasattr(model, "coefs_"):
        total = 0
        for coef in model.coefs_:
            total += int(np.prod(coef.shape))
        for bias in getattr(model, "intercepts_", []):
            total += int(np.prod(bias.shape))
        return total
    return 0


def audit_record(model=None, seconds=0.0, best_epoch=None, mc_samples=0, rollouts=0, members=1):
    return {
        "params": int(model_param_count(model)),
        "training_seconds": float(seconds),
        "best_epoch": -1 if best_epoch is None else int(best_epoch),
        "mc_samples": int(mc_samples),
        "rollouts": int(rollouts),
        "ensemble_members": int(members),
    }


def timed_fit(fn, *args, **kwargs):
    start = time.perf_counter()
    model = fn(*args, **kwargs)
    seconds = time.perf_counter() - start
    return model, seconds


def parse_models(raw):
    if raw is None or raw == "" or raw == "default":
        return list(DEFAULT_MODELS)
    if isinstance(raw, (list, tuple)):
        tokens = []
        for item in raw:
            tokens.extend(str(item).split(","))
    else:
        tokens = str(raw).split(",")
    requested = []
    aliases = {
        "seqtx_count": "SeqTXCount",
        "seqtx_count_pm": "SeqTXCountPM",
        "seqtx_global": "SeqTXGlobal",
        "anp": "ANP",
        "flow": "FlowLatent",
        "flow_latent": "FlowLatent",
        "param_matched": "SeqTXCountPM",
        "obj_param_matched": "ObjSetTFPM",
        "global_grid": "GlobalGrid",
        "mean_indep": "MeanIndep",
        "two_tower_indep": "TwoTowerIndep",
        "twotower_indep": "TwoTowerIndep",
        "two_tower_dropout": "TwoTowerDropout",
        "twotower_dropout": "TwoTowerDropout",
        "two_tower_ens_mean": "TwoTowerEnsMean",
        "twotower_ens_mean": "TwoTowerEnsMean",
        "two_tower_ens_mix": "TwoTowerEnsMix",
        "twotower_ens_mix": "TwoTowerEnsMix",
        "all": "all",
    }
    for token in tokens:
        key = token.strip()
        if not key:
            continue
        mapped = aliases.get(key.lower(), key)
        if mapped == "all":
            return list(DEFAULT_MODELS)
        if mapped not in MODEL_ORDER:
            raise ValueError(f"Unknown model '{key}'. Known models: {', '.join(MODEL_ORDER)}")
        if mapped not in requested:
            requested.append(mapped)
    return requested or list(DEFAULT_MODELS)


def download_movielens_1m(cache_dir=".data_cache/movielens"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-1m.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_1M_URL, zip_path)
    extract_dir = cache_dir / "ml-1m"
    if not (extract_dir / "ratings.dat").exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(cache_dir)
    return extract_dir


def download_movielens_20m(cache_dir=".data_cache/movielens"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-20m.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_20M_URL, zip_path)
    extract_dir = cache_dir / "ml-20m"
    if not (extract_dir / "ratings.csv").exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(cache_dir)
    return extract_dir


def download_movielens_10m(cache_dir=".data_cache/movielens"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-10m.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_10M_URL, zip_path)
    extract_dir = cache_dir / "ml-10M100K"
    if not (extract_dir / "ratings.dat").exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(cache_dir)
    return extract_dir


def load_movielens_1m(cache_dir=".data_cache/movielens"):
    root = download_movielens_1m(cache_dir)
    genre_index = {genre: idx for idx, genre in enumerate(GENRES)}
    movie_features = {}
    with (root / "movies.dat").open("r", encoding="latin-1") as handle:
        for line in handle:
            movie_id, _, genres = line.rstrip("\n").split("::")
            feat = np.zeros(len(GENRES), dtype=np.float32)
            for genre in genres.split("|"):
                if genre in genre_index:
                    feat[genre_index[genre]] = 1.0
            movie_features[int(movie_id)] = feat

    ratings_by_user = {}
    all_ratings = []
    with (root / "ratings.dat").open("r", encoding="latin-1") as handle:
        for line in handle:
            user, movie, rating, _ = line.rstrip("\n").split("::")
            user_id = int(user)
            movie_id = int(movie)
            rating_value = float(rating)
            positive = int(rating_value >= 4.0)
            ratings_by_user.setdefault(user_id, []).append((movie_id, rating_value, positive))
            all_ratings.append((user_id, movie_id, rating_value, positive))
    return ratings_by_user, movie_features, all_ratings


def load_movielens_10m(cache_dir=".data_cache/movielens"):
    root = download_movielens_10m(cache_dir)
    genre_index = {genre: idx for idx, genre in enumerate(GENRES)}
    movie_features = {}
    with (root / "movies.dat").open("r", encoding="latin-1") as handle:
        for line in handle:
            movie_id, _, genres = line.rstrip("\n").split("::")
            feat = np.zeros(len(GENRES), dtype=np.float32)
            for genre in genres.split("|"):
                if genre in genre_index:
                    feat[genre_index[genre]] = 1.0
            movie_features[int(movie_id)] = feat

    ratings_by_user = {}
    all_ratings = []
    with (root / "ratings.dat").open("r", encoding="latin-1") as handle:
        for line in handle:
            user, movie, rating, _ = line.rstrip("\n").split("::")
            user_id = int(user)
            movie_id = int(movie)
            rating_value = float(rating)
            positive = int(rating_value >= 4.0)
            ratings_by_user.setdefault(user_id, []).append((movie_id, rating_value, positive))
            all_ratings.append((user_id, movie_id, rating_value, positive))
    return ratings_by_user, movie_features, all_ratings


def load_movielens_20m(cache_dir=".data_cache/movielens"):
    root = download_movielens_20m(cache_dir)
    genre_index = {genre: idx for idx, genre in enumerate(GENRES)}
    movie_features = {}
    with (root / "movies.csv").open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            movie_id = int(parts[0])
            genres = parts[-1]
            feat = np.zeros(len(GENRES), dtype=np.float32)
            for genre in genres.split("|"):
                if genre in genre_index:
                    feat[genre_index[genre]] = 1.0
            movie_features[movie_id] = feat

    ratings_by_user = {}
    all_ratings = []
    with (root / "ratings.csv").open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            user, movie, rating, _ = line.rstrip("\n").split(",")
            user_id = int(user)
            movie_id = int(movie)
            rating_value = float(rating)
            positive = int(rating_value >= 4.0)
            ratings_by_user.setdefault(user_id, []).append((movie_id, rating_value, positive))
            all_ratings.append((user_id, movie_id, rating_value, positive))
    return ratings_by_user, movie_features, all_ratings


def load_movielens_dataset(dataset="ml1m", cache_dir=".data_cache/movielens"):
    key = str(dataset).lower().replace("-", "").replace("_", "")
    if key in {"ml10m", "10m", "movielens10m"}:
        return "MovieLens10M", *load_movielens_10m(cache_dir)
    if key in {"ml20m", "20m", "movielens20m"}:
        return "MovieLens20M", *load_movielens_20m(cache_dir)
    if key in {"ml1m", "1m", "movielens1m"}:
        return "MovieLens1M", *load_movielens_1m(cache_dir)
    raise ValueError(f"Unknown MovieLens dataset '{dataset}'. Use ml1m or ml20m.")


def split_users(ratings_by_user, min_items, seed, max_users=None):
    users = np.asarray([u for u, rows in ratings_by_user.items() if len(rows) >= min_items])
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    if max_users is not None and int(max_users) > 0:
        users = users[: int(max_users)]
    n = len(users)
    train = users[: int(0.60 * n)]
    val = users[int(0.60 * n) : int(0.75 * n)]
    test = users[int(0.75 * n) :]
    return train.tolist(), val.tolist(), test.tolist()


def movie_popularity_logits(ratings_by_user, train_users):
    pos = {}
    total = {}
    for user in train_users:
        for movie_id, _, positive in ratings_by_user[user]:
            pos[movie_id] = pos.get(movie_id, 0) + int(positive)
            total[movie_id] = total.get(movie_id, 0) + 1
    global_pos = sum(pos.values())
    global_total = sum(total.values())
    global_rate = (global_pos + 2.0) / (global_total + 4.0)
    logits = {}
    for movie_id in total:
        rate = (pos[movie_id] + 2.0 * global_rate) / (total[movie_id] + 2.0)
        logits[movie_id] = float(logit(rate))
    return logits, float(logit(global_rate)), float(global_rate)


def rating_bin_index(rating_value):
    if rating_value <= 2.0:
        return 0
    if rating_value <= 3.0:
        return 1
    return 2


def movie_rating_bin_probs(ratings_by_user, train_users):
    counts = {}
    global_counts = np.ones(3, dtype=np.float64)
    for user in train_users:
        for movie_id, rating_value, _ in ratings_by_user[user]:
            if movie_id not in counts:
                counts[movie_id] = np.ones(3, dtype=np.float64)
            idx = rating_bin_index(rating_value)
            counts[movie_id][idx] += 1.0
            global_counts[idx] += 1.0
    global_probs = global_counts / np.sum(global_counts)
    probs = {}
    for movie_id, row in counts.items():
        # Mild shrinkage toward the global rating-bin distribution.
        probs[movie_id] = (row + 3.0 * global_probs) / (np.sum(row) + 3.0)
    return probs, global_probs


def item_feature(movie_id, rating_value, positive, observed, movie_features, movie_logits, default_logit):
    genre = movie_features.get(movie_id, np.zeros(len(GENRES), dtype=np.float32))
    movie_prior = sigmoid(movie_logits.get(movie_id, default_logit))
    return np.concatenate(
        [
            genre,
            np.asarray(
                [
                    movie_prior,
                    (rating_value - 1.0) / 4.0 if observed else 0.0,
                    float(positive) if observed else 0.0,
                    float(observed),
                    1.0 - float(observed),
                ],
                dtype=np.float32,
            ),
        ]
    )


def sample_views(
    ratings_by_user,
    movie_features,
    movie_logits,
    movie_bin_probs,
    default_logit,
    default_bin_probs,
    users,
    k,
    hidden_n,
    c,
    views_per_user,
    seed,
    policy="random",
    critical_margin=0.08,
    max_attempts=80,
    prefix_weight=0.20,
    selector_mode="default",
):
    if selector_mode not in {"default", "prior_only", "prefix_only"}:
        raise ValueError(f"Unknown selector_mode: {selector_mode}")
    rng = np.random.default_rng(seed)
    seqs, labels, counts, meta = [], [], [], []
    threshold = int(math.ceil(c * hidden_n))
    for user in users:
        rows = ratings_by_user[user]
        for _ in range(views_per_user):
            best_score, best_obs, best_hidden, best_hidden_prior, best_obs_rate = None, None, None, None, None
            for _attempt in range(max_attempts if policy == "critical" else 1):
                chosen = rng.choice(len(rows), size=k + hidden_n, replace=False)
                chosen_rows = [rows[int(i)] for i in chosen]
                rng.shuffle(chosen_rows)
                obs_candidate = chosen_rows[:k]
                hidden_candidate = chosen_rows[k:]
                if policy != "critical":
                    best_obs, best_hidden = obs_candidate, hidden_candidate
                    break
                hidden_prior = np.mean(
                    [sigmoid(movie_logits.get(movie_id, default_logit)) for movie_id, _, _ in hidden_candidate]
                )
                obs_rate = np.mean([positive for _, _, positive in obs_candidate])
                prior_term = abs(float(hidden_prior) - c)
                prefix_term = abs(float(obs_rate) - 0.5)
                if selector_mode == "prior_only":
                    score = prior_term
                elif selector_mode == "prefix_only":
                    score = prefix_term
                else:
                    score = prior_term + float(prefix_weight) * prefix_term
                if best_score is None or score < best_score:
                    best_score = score
                    best_obs, best_hidden = obs_candidate, hidden_candidate
                    best_hidden_prior, best_obs_rate = float(hidden_prior), float(obs_rate)
                if selector_mode == "prior_only":
                    accept = prior_term <= critical_margin
                elif selector_mode == "prefix_only":
                    accept = 0.0 < obs_rate < 1.0
                else:
                    accept = prior_term <= critical_margin and 0.0 < obs_rate < 1.0
                if accept:
                    break
            obs = best_obs
            hidden = best_hidden
            hidden_count = int(sum(row[2] for row in hidden))
            label = int(hidden_count >= threshold)
            items = []
            obs_pos = []
            obs_bins = []
            obs_bin_base = []
            hidden_movie_logits = []
            hidden_pos = []
            hidden_bins = []
            hidden_bin_base = []
            for movie_id, rating_value, positive in obs:
                items.append(
                    item_feature(
                        movie_id,
                        rating_value,
                        positive,
                        True,
                        movie_features,
                        movie_logits,
                        default_logit,
                    )
                )
                obs_pos.append(int(positive))
                obs_bins.append(rating_bin_index(rating_value))
                obs_bin_base.append(movie_bin_probs.get(movie_id, default_bin_probs))
            for movie_id, rating_value, positive in hidden:
                items.append(
                    item_feature(
                        movie_id,
                        rating_value,
                        positive,
                        False,
                        movie_features,
                        movie_logits,
                        default_logit,
                    )
                )
                hidden_movie_logits.append(movie_logits.get(movie_id, default_logit))
                hidden_pos.append(int(positive))
                hidden_bins.append(rating_bin_index(rating_value))
                hidden_bin_base.append(movie_bin_probs.get(movie_id, default_bin_probs))
            seqs.append(np.stack(items, axis=0))
            labels.append(label)
            counts.append(hidden_count)
            meta.append(
                {
                    "obs_pos": np.asarray(obs_pos, dtype=np.int8),
                    "obs_bins": np.asarray(obs_bins, dtype=np.int8),
                    "obs_bin_base": np.asarray(obs_bin_base, dtype=np.float64),
                    "hidden_logits": np.asarray(hidden_movie_logits, dtype=np.float64),
                    "hidden_pos": np.asarray(hidden_pos, dtype=np.int8),
                    "hidden_bins": np.asarray(hidden_bins, dtype=np.int8),
                    "hidden_bin_base": np.asarray(hidden_bin_base, dtype=np.float64),
                    "user_items": len(rows),
                    "selection_hidden_prior": float(best_hidden_prior if best_hidden_prior is not None else np.nan),
                    "selection_obs_rate": float(best_obs_rate if best_obs_rate is not None else np.nan),
                    "selection_score": float(best_score if best_score is not None else np.nan),
                    "selector_mode": selector_mode,
                    "selector_prefix_weight": float(prefix_weight),
                    "selector_critical_margin": float(critical_margin),
                    "selector_candidate_budget": int(max_attempts),
                }
            )
    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(counts, dtype=np.int64),
        meta,
    )


def standardize_sequences(train_seq, val_seq, test_seq):
    scaler = StandardScaler()
    obs = train_seq.reshape(-1, train_seq.shape[-1])
    scaler.fit(obs)
    def transform(x):
        y = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        # Preserve binary masks exactly after scaling by appending original masks.
        y[..., -2:] = x[..., -2:]
        return y.astype(np.float32)
    return transform(train_seq), transform(val_seq), transform(test_seq)


def aggregate_features(seqs, k):
    obs = seqs[:, :k, :]
    hidden = seqs[:, k:, :]
    parts = [
        obs.mean(axis=1),
        obs.std(axis=1),
        hidden.mean(axis=1),
        hidden.std(axis=1),
    ]
    return np.concatenate(parts, axis=1)


def posterior_weights(obs_pos, obs_logits, grid, prior_scale=1.6):
    log_prior = -0.5 * (grid / prior_scale) ** 2
    log_w = log_prior.copy()
    for y, base in zip(obs_pos, obs_logits):
        p = sigmoid(base + grid)
        log_w += y * np.log(np.clip(p, 1e-8, 1.0)) + (1 - y) * np.log(
            np.clip(1.0 - p, 1e-8, 1.0)
        )
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / np.sum(w)


def grid_tail_prob(meta, k, threshold, grid):
    # Observed logits are stored in the observed features through movie prior.
    # Reconstruct them from prior probabilities.
    # The first len(GENRES) entries are genres, then prior, rating, positive, masks.
    raise RuntimeError("grid_tail_prob should not be called directly.")


def grid_predictions(seqs, metas, k, threshold, grid):
    out_mixture = []
    out_indep = []
    tax_scores = []
    mean_qs = []
    cov_sums = []
    pos_cov_sums = []
    neg_cov_sums = []
    neg_pair_fracs = []
    for seq, meta in zip(seqs, metas):
        obs_prior = np.clip(seq[:k, len(GENRES)], 1e-5, 1.0 - 1e-5)
        obs_logits = logit(obs_prior)
        weights = posterior_weights(meta["obs_pos"], obs_logits, grid)
        hidden_logits = meta["hidden_logits"]
        component_tails = []
        component_means = []
        component_probs = []
        for theta in grid:
            probs = sigmoid(hidden_logits + theta)
            component_probs.append(probs)
            component_means.append(float(np.mean(probs)))
            component_tails.append(poisson_binomial_tail(probs, threshold))
        component_probs = np.asarray(component_probs)
        component_tails = np.asarray(component_tails)
        mixture_tail = float(np.dot(weights, component_tails))
        marginal_probs = np.dot(weights, component_probs)
        centered = component_probs - marginal_probs[None, :]
        cov = (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
        offdiag = cov[~np.eye(cov.shape[0], dtype=bool)]
        cov_sum = float(np.sum(offdiag))
        pos_cov = float(np.sum(np.clip(offdiag, 0.0, None)))
        neg_cov = float(np.sum(np.clip(offdiag, None, 0.0)))
        indep_tail = poisson_binomial_tail(marginal_probs, threshold)
        mean_q = float(np.mean(marginal_probs))
        latent_var = float(np.dot(weights, (np.asarray(component_means) - mean_q) ** 2))
        margin = abs(mean_q - threshold / len(hidden_logits)) + 0.03
        tax_score = len(hidden_logits) * latent_var / (margin * margin)
        out_mixture.append(mixture_tail)
        out_indep.append(indep_tail)
        tax_scores.append(tax_score)
        mean_qs.append(mean_q)
        cov_sums.append(cov_sum)
        pos_cov_sums.append(pos_cov)
        neg_cov_sums.append(neg_cov)
        neg_pair_fracs.append(float(np.mean(offdiag < -1e-10)) if len(offdiag) else 0.0)
    return (
        np.asarray(out_mixture, dtype=np.float64),
        np.asarray(out_indep, dtype=np.float64),
        np.asarray(tax_scores, dtype=np.float64),
        np.asarray(mean_qs, dtype=np.float64),
        {
            "cov_sum": np.asarray(cov_sums, dtype=np.float64),
            "positive_cov_sum": np.asarray(pos_cov_sums, dtype=np.float64),
            "negative_cov_sum": np.asarray(neg_cov_sums, dtype=np.float64),
            "negative_pair_fraction": np.asarray(neg_pair_fracs, dtype=np.float64),
        },
    )


def ar_rollout_predictions(metas, k, hidden_n, threshold, global_rate, n_rollouts=96, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for meta in metas:
        obs_pos = meta["obs_pos"]
        alpha0 = 2.0 * global_rate + float(np.sum(obs_pos))
        beta0 = 2.0 * (1.0 - global_rate) + float(len(obs_pos) - np.sum(obs_pos))
        wins = 0
        for _ in range(n_rollouts):
            alpha, beta = alpha0, beta0
            count = 0
            for _ in range(hidden_n):
                p = alpha / (alpha + beta)
                y = int(rng.random() < p)
                count += y
                alpha += y
                beta += 1 - y
            wins += int(count >= threshold)
        out.append(wins / n_rollouts)
    return np.asarray(out, dtype=np.float64)


if TORCH_AVAILABLE:

    class TransformerGlobalTail(nn.Module):
        def __init__(self, input_dim, max_len, hidden_n, threshold, grid, hidden_dim=96, num_heads=4):
            super().__init__()
            self.hidden_n = hidden_n
            self.threshold = threshold
            self.register_buffer("grid", torch.tensor(grid, dtype=torch.float32))
            self.embed = nn.Linear(input_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.weight_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(grid)))
            self.item_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        def forward(self, x):
            h = self.embed(x)
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = h + self.pos_emb[:, : h.shape[1], :]
            h = self.encoder(h)
            weights = torch.softmax(self.weight_head(h[:, 0, :]), dim=-1)
            hidden_h = h[:, -self.hidden_n :, :]
            base = self.item_head(hidden_h).squeeze(-1)
            probs = torch.sigmoid(base[:, None, :] + self.grid[None, :, None])
            tails = differentiable_tail(probs, self.threshold)
            return torch.sum(weights * tails, dim=-1)


    class SetGlobalTail(nn.Module):
        def __init__(self, input_dim, hidden_n, threshold, grid, hidden_dim=96):
            super().__init__()
            self.hidden_n = hidden_n
            self.threshold = threshold
            self.register_buffer("grid", torch.tensor(grid, dtype=torch.float32))
            self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.weight_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(grid)),
            )
            self.item_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        def forward(self, x):
            h = self.embed(x)
            pooled = h.mean(dim=1)
            weights = torch.softmax(self.weight_head(pooled), dim=-1)
            hidden_h = h[:, -self.hidden_n :, :]
            base = self.item_head(hidden_h).squeeze(-1)
            probs = torch.sigmoid(base[:, None, :] + self.grid[None, :, None])
            tails = differentiable_tail(probs, self.threshold)
            return torch.sum(weights * tails, dim=-1)


    class AttentiveNPTail(nn.Module):
        def __init__(self, input_dim, hidden_n, threshold, latent_dim=8, hidden_dim=96, num_heads=4):
            super().__init__()
            self.hidden_n = hidden_n
            self.threshold = threshold
            self.latent_dim = latent_dim
            self.context_embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.target_embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.posterior = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )
            self.item_head = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim + latent_dim),
                nn.Linear(2 * hidden_dim + latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def _features(self, x):
            context = x[:, :-self.hidden_n, :]
            target = x[:, -self.hidden_n :, :]
            ctx = self.context_embed(context)
            tgt = self.target_embed(target)
            attended, _ = self.attn(tgt, ctx, ctx, need_weights=False)
            pooled = ctx.mean(dim=1)
            stats = self.posterior(pooled)
            mu, logvar = stats.chunk(2, dim=-1)
            logvar = torch.clamp(logvar, -5.0, 2.0)
            return tgt, attended, mu, logvar

        def forward(self, x, mc_samples=16):
            tgt, attended, mu, logvar = self._features(x)
            batch = x.shape[0]
            eps = torch.randn(batch, mc_samples, self.latent_dim, device=x.device, dtype=x.dtype)
            z = mu[:, None, :] + torch.exp(0.5 * logvar)[:, None, :] * eps
            z = z[:, :, None, :].expand(-1, -1, self.hidden_n, -1)
            item = torch.cat(
                [
                    tgt[:, None, :, :].expand(-1, mc_samples, -1, -1),
                    attended[:, None, :, :].expand(-1, mc_samples, -1, -1),
                    z,
                ],
                dim=-1,
            )
            logits = self.item_head(item).squeeze(-1)
            probs = torch.sigmoid(logits)
            tails = differentiable_tail(probs, self.threshold)
            return tails.mean(dim=-1)


    class ConditionalCoupling(nn.Module):
        def __init__(self, latent_dim, cond_dim, mask):
            super().__init__()
            self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
            self.net = nn.Sequential(
                nn.Linear(latent_dim + cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 2 * latent_dim),
            )

        def forward(self, z, cond):
            masked = z * self.mask
            inp = torch.cat([masked, cond], dim=-1)
            shift, log_scale = self.net(inp).chunk(2, dim=-1)
            log_scale = 0.6 * torch.tanh(log_scale)
            return masked + (1.0 - self.mask) * (z * torch.exp(log_scale) + shift)


    class FlowLatentTail(nn.Module):
        def __init__(self, input_dim, hidden_n, threshold, latent_dim=8, hidden_dim=96, num_heads=4):
            super().__init__()
            self.hidden_n = hidden_n
            self.threshold = threshold
            self.latent_dim = latent_dim
            self.context_embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.target_embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.cond_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            masks = []
            for layer_idx in range(4):
                mask = np.asarray([(j + layer_idx) % 2 for j in range(latent_dim)], dtype=np.float32)
                masks.append(mask)
            self.couplings = nn.ModuleList([ConditionalCoupling(latent_dim, hidden_dim, mask) for mask in masks])
            self.item_head = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim + latent_dim),
                nn.Linear(2 * hidden_dim + latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, mc_samples=16):
            context = x[:, :-self.hidden_n, :]
            target = x[:, -self.hidden_n :, :]
            ctx = self.context_embed(context)
            tgt = self.target_embed(target)
            attended, _ = self.attn(tgt, ctx, ctx, need_weights=False)
            cond = self.cond_head(ctx.mean(dim=1))
            batch = x.shape[0]
            z = torch.randn(batch, mc_samples, self.latent_dim, device=x.device, dtype=x.dtype)
            cond_s = cond[:, None, :].expand(-1, mc_samples, -1).reshape(batch * mc_samples, -1)
            z_flat = z.reshape(batch * mc_samples, -1)
            for coupling in self.couplings:
                z_flat = coupling(z_flat, cond_s)
            z = z_flat.reshape(batch, mc_samples, self.latent_dim)
            z = z[:, :, None, :].expand(-1, -1, self.hidden_n, -1)
            item = torch.cat(
                [
                    tgt[:, None, :, :].expand(-1, mc_samples, -1, -1),
                    attended[:, None, :, :].expand(-1, mc_samples, -1, -1),
                    z,
                ],
                dim=-1,
            )
            logits = self.item_head(item).squeeze(-1)
            probs = torch.sigmoid(logits)
            tails = differentiable_tail(probs, self.threshold)
            return tails.mean(dim=-1)


    class TwoTowerItemTail(nn.Module):
        def __init__(self, input_dim, hidden_n, hidden_dim=64, dropout=0.1):
            super().__init__()
            self.hidden_n = hidden_n
            self.user_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.item_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.score = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            context = x[:, :-self.hidden_n, :]
            target = x[:, -self.hidden_n :, :]
            user = self.user_net(context.mean(dim=1))
            item = self.item_net(target)
            user_items = user[:, None, :].expand(-1, self.hidden_n, -1)
            pair = torch.cat([user_items, item, user_items * item], dim=-1)
            return self.score(pair).squeeze(-1)


def differentiable_tail(probs, threshold):
    # probs: [batch, K, n]
    batch, k_components, n_items = probs.shape
    dp = probs.new_zeros(batch, k_components, n_items + 1)
    dp[:, :, 0] = 1.0
    for idx in range(n_items):
        p = probs[:, :, idx]
        new = probs.new_zeros(batch, k_components, n_items + 1)
        new[:, :, 1:] += dp[:, :, :-1] * p[:, :, None]
        new[:, :, :-1] += dp[:, :, :-1] * (1.0 - p[:, :, None])
        dp = new
    return dp[:, :, threshold:].sum(dim=-1)


def fit_global_tail(model_cls, train_seq, train_labels, val_seq, val_labels, seed, hidden_n, threshold, grid):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for global latent heads.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_cls == "seq":
        model = TransformerGlobalTail(train_seq.shape[-1], train_seq.shape[1], hidden_n, threshold, grid).to(device)
    else:
        model = SetGlobalTail(train_seq.shape[-1], hidden_n, threshold, grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=192, shuffle=True)
    val_ds = TensorDataset(
        torch.from_numpy(val_seq.astype(np.float32)),
        torch.from_numpy(val_labels.astype(np.float32)),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    best_state, best_val, stale = None, None, 0
    for _ in range(10):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = torch.clamp(model(xb), 1e-5, 1.0 - 1e-5)
            loss = -(yb * torch.log(pred) + (1.0 - yb) * torch.log1p(-pred)).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_count = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = torch.clamp(model(xb), 1e-5, 1.0 - 1e-5)
                loss = -(yb * torch.log(pred) + (1.0 - yb) * torch.log1p(-pred))
                total_loss += float(loss.sum().item())
                total_count += int(yb.numel())
            val_loss = total_loss / max(total_count, 1)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def fit_stochastic_latent_tail(
    model_cls,
    train_seq,
    train_labels,
    val_seq,
    val_labels,
    seed,
    hidden_n,
    threshold,
    latent_dim=8,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for stochastic latent heads.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_cls == "anp":
        model = AttentiveNPTail(train_seq.shape[-1], hidden_n, threshold, latent_dim=latent_dim).to(device)
    elif model_cls == "flow":
        model = FlowLatentTail(train_seq.shape[-1], hidden_n, threshold, latent_dim=latent_dim).to(device)
    else:
        raise ValueError(f"Unknown stochastic latent model class: {model_cls}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=160, shuffle=True)
    val_ds = TensorDataset(
        torch.from_numpy(val_seq.astype(np.float32)),
        torch.from_numpy(val_labels.astype(np.float32)),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    best_state, best_val, stale = None, None, 0
    for _ in range(9):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = torch.clamp(model(xb, mc_samples=16), 1e-5, 1.0 - 1e-5)
            loss = -(yb * torch.log(pred) + (1.0 - yb) * torch.log1p(-pred)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_count = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = torch.clamp(model(xb, mc_samples=32), 1e-5, 1.0 - 1e-5)
                loss = -(yb * torch.log(pred) + (1.0 - yb) * torch.log1p(-pred))
                total_loss += float(loss.sum().item())
                total_count += int(yb.numel())
            val_loss = total_loss / max(total_count, 1)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_global_tail(model, seq, batch_size=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            xb = torch.from_numpy(seq[start : start + batch_size].astype(np.float32)).to(device)
            out.append(model(xb).cpu().numpy())
    model.cpu()
    return np.concatenate(out, axis=0)


def predict_stochastic_latent_tail(model, seq, seed=0, mc_samples=64, batch_size=192):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            xb = torch.from_numpy(seq[start : start + batch_size].astype(np.float32)).to(device)
            out.append(model(xb, mc_samples=mc_samples).cpu().numpy())
    model.cpu()
    return np.concatenate(out, axis=0)


def hidden_label_matrix(metas, key="hidden_pos"):
    return np.stack([np.asarray(meta[key], dtype=np.float32) for meta in metas], axis=0)


def tail_from_item_probs(item_probs, threshold):
    return np.asarray([poisson_binomial_tail(row, threshold) for row in np.asarray(item_probs)], dtype=np.float64)


def fit_two_tower_item(train_seq, train_item_y, val_seq, val_item_y, seed, hidden_n, hidden_dim=64, epochs=8):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for two-tower experiments.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerItemTail(train_seq.shape[-1], hidden_n, hidden_dim=hidden_dim, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_item_y.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=192, shuffle=True)
    val_x = torch.from_numpy(val_seq.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_item_y.astype(np.float32)).to(device)
    best_state, best_val, best_epoch, stale = None, None, -1, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_epoch = best_epoch
    return model.cpu()


def predict_two_tower_item_probs(model, seq, batch_size=256, dropout=False, seed=0):
    if dropout:
        model.train()
    else:
        model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            xb = torch.from_numpy(seq[start : start + batch_size].astype(np.float32)).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    model.cpu()
    return np.concatenate(out, axis=0)


def predict_two_tower_tail(model, seq, threshold, batch_size=256):
    return tail_from_item_probs(predict_two_tower_item_probs(model, seq, batch_size=batch_size), threshold)


def predict_two_tower_dropout_tail(model, seq, threshold, seed=0, mc_samples=24):
    tails = []
    for s in range(mc_samples):
        probs = predict_two_tower_item_probs(model, seq, dropout=True, seed=seed + 17 * s)
        tails.append(tail_from_item_probs(probs, threshold))
    return np.mean(np.stack(tails, axis=0), axis=0)


def fit_two_tower_ensemble(train_seq, train_item_y, val_seq, val_item_y, seed, hidden_n, members=3):
    models = []
    seconds = 0.0
    for idx in range(members):
        model, elapsed = timed_fit(
            fit_two_tower_item,
            train_seq,
            train_item_y,
            val_seq,
            val_item_y,
            seed + 101 * idx,
            hidden_n,
            hidden_dim=64,
            epochs=7,
        )
        models.append(model)
        seconds += elapsed
    return models, seconds


def predict_two_tower_ensemble(models, seq, threshold, mode="mixture"):
    item_probs = [predict_two_tower_item_probs(model, seq) for model in models]
    if mode == "mean":
        return tail_from_item_probs(np.mean(np.stack(item_probs, axis=0), axis=0), threshold)
    tails = [tail_from_item_probs(probs, threshold) for probs in item_probs]
    return np.mean(np.stack(tails, axis=0), axis=0)


def reliability_curve(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (prob >= edges[idx]) & (prob <= edges[idx + 1])
        else:
            mask = (prob >= edges[idx]) & (prob < edges[idx + 1])
        if not np.any(mask):
            rows.append({"bin": idx, "count": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        rows.append({"bin": idx, "count": int(np.sum(mask)), "confidence": conf, "accuracy": acc, "gap": abs(conf - acc)})
    return rows


def brier_decomposition(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    base = float(np.mean(y_true))
    reliability = 0.0
    resolution = 0.0
    for row in reliability_curve(y_true, prob, n_bins=n_bins):
        if row["count"] == 0:
            continue
        w = row["count"] / len(y_true)
        reliability += w * (row["confidence"] - row["accuracy"]) ** 2
        resolution += w * (row["accuracy"] - base) ** 2
    uncertainty = base * (1.0 - base)
    return {
        "brier": float(np.mean((prob - y_true) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def summarize_array(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def signed_covariance_audit(cov_diag, mask=None):
    mask_arr = np.ones_like(cov_diag["cov_sum"], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return {
        key: summarize_array(np.asarray(value)[mask_arr])
        for key, value in cov_diag.items()
    }


def gain_by_taxscore_bins(y_true, test_probs, test_taxscore, baseline="SeqTXCount", targets=("ANP", "FlowLatent")):
    if baseline not in test_probs:
        return {}
    q = np.quantile(test_taxscore, [0.0, 0.25, 0.50, 0.75, 1.0])
    rows = []
    for idx in range(4):
        if idx == 3:
            mask = test_taxscore >= q[idx]
        else:
            mask = (test_taxscore >= q[idx]) & (test_taxscore < q[idx + 1])
        if not np.any(mask):
            continue
        base_metrics = triage_metrics(y_true[mask], test_probs[baseline][mask])
        row = {
            "bin": idx,
            "count": int(np.sum(mask)),
            "mean_taxscore": float(np.mean(test_taxscore[mask])),
            "baseline": baseline,
        }
        for target in targets:
            if target not in test_probs:
                continue
            target_metrics = triage_metrics(y_true[mask], test_probs[target][mask])
            row[target] = {
                "nll_gain": float(base_metrics["nll"] - target_metrics["nll"]),
                "ece_gain": float(base_metrics["ece"] - target_metrics["ece"]),
                "utility_gain": float(target_metrics["utility"] - base_metrics["utility"]),
                "regret_reduction": float(base_metrics["regret"] - target_metrics["regret"]),
            }
        rows.append(row)
    x = np.asarray([row["mean_taxscore"] for row in rows], dtype=np.float64)
    out = {"rows": rows}
    for target in targets:
        vals = np.asarray([row[target]["utility_gain"] for row in rows if target in row], dtype=np.float64)
        xs = np.asarray([row["mean_taxscore"] for row in rows if target in row], dtype=np.float64)
        if len(vals) >= 2 and np.std(vals) > 0 and np.std(xs) > 0:
            out[f"{target}_taxscore_utility_corr"] = float(np.corrcoef(xs, vals)[0, 1])
        else:
            out[f"{target}_taxscore_utility_corr"] = 0.0
    return out


def sign_test_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = int(len(diffs))
    if n == 0:
        return 1.0
    wins = int(np.sum(diffs > 0))
    tail = min(wins, n - wins)
    prob = sum(comb(n, k) for k in range(tail + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * prob))


def paired_seed_summary(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "wins": 0,
            "n": 0,
            "sign_p": 1.0,
        }
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    # Two-sided 95% t critical values for n=1..10; normal fallback thereafter.
    tcrit = {
        1: 0.0,
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(n, 1.96)
    half = tcrit * std / np.sqrt(n) if n > 1 else 0.0
    mean = float(np.mean(vals))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
        "wins": int(np.sum(vals > 0)),
        "n": n,
        "sign_p": sign_test_pvalue(vals),
    }


def metric_gain(base, target, metric):
    if metric in {"nll", "ece", "brier", "regret"}:
        return float(base[metric] - target[metric])
    return float(target[metric] - base[metric])


def stat_mean(value):
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def statistical_audit_for_config(payloads, cfg, baseline="SeqTXCount"):
    targets = [
        "SeqTXCountPM",
        "SeqTXGlobal",
        "TwoTowerEnsMean",
        "TwoTowerEnsMix",
        "ANP",
        "FlowLatent",
        "GlobalGrid",
    ]
    metrics = ["nll", "ece", "utility", "regret"]
    out = {}
    scopes = ["overall", "high"]
    tail_slices = payloads[0]["configs"][cfg].get("tail_slices", {})
    scopes.extend([name for name in ["top20", "top10"] if name in tail_slices])
    for scope in scopes:
        out[scope] = {}
        for target in targets:
            if target not in payloads[0]["configs"][cfg]["metrics"]:
                continue
            out[scope][target] = {}
            for metric in metrics:
                diffs = []
                for payload in payloads:
                    if scope == "overall":
                        base = payload["configs"][cfg]["metrics"][baseline]
                        tgt = payload["configs"][cfg]["metrics"][target]
                    elif scope == "high":
                        base = payload["configs"][cfg]["bins"]["high"]["metrics"][baseline]
                        tgt = payload["configs"][cfg]["bins"]["high"]["metrics"][target]
                    else:
                        base = payload["configs"][cfg]["tail_slices"][scope]["metrics"][baseline]
                        tgt = payload["configs"][cfg]["tail_slices"][scope]["metrics"][target]
                    diffs.append(metric_gain(base, tgt, metric))
                out[scope][target][f"{metric}_gain"] = paired_seed_summary(diffs)
    return out


def proxy_gain_correlation_for_config(aggregate_cfg, target="ANP", metric="nll"):
    rows = []
    for bin_name in ["low", "mid", "high"]:
        bin_row = aggregate_cfg["bins"][bin_name]
        metrics = bin_row["metrics"]
        if "SeqTXCount" not in metrics or target not in metrics:
            continue
        base = metrics["SeqTXCount"]
        tgt = metrics[target]
        rows.append(
            {
                "bin": bin_name,
                "taxscore": stat_mean(bin_row["mean_taxscore"]),
                "finite_ratio": stat_mean(bin_row["mean_n2_over_user_items"]),
                "nll_gain": stat_mean(base["nll"]) - stat_mean(tgt["nll"]),
                "utility_gain": stat_mean(tgt["utility"]) - stat_mean(base["utility"]),
            }
        )
    out = {"rows": rows}
    for proxy in ["taxscore", "finite_ratio"]:
        for gain_name in ["nll_gain", "utility_gain"]:
            x = np.asarray([row[proxy] for row in rows], dtype=np.float64)
            y = np.asarray([row[gain_name] for row in rows], dtype=np.float64)
            key = f"{proxy}_{gain_name}_corr"
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                out[key] = float(np.corrcoef(x, y)[0, 1])
            else:
                out[key] = 0.0
    return out


def taxscore_tail_slices(y_true, test_probs, test_taxscore, grid_tail, tau, test_meta, hidden_n):
    """Pre-specified deployment triage slices selected only by TaxScore."""
    slices = {}
    for name, frac in [("top30", 0.30), ("top20", 0.20), ("top10", 0.10), ("top05", 0.05)]:
        cutoff = float(np.quantile(test_taxscore, 1.0 - frac))
        mask = test_taxscore >= cutoff
        if not np.any(mask):
            continue
        slices[name] = {
            "fraction": frac,
            "cutoff": cutoff,
            "count": int(np.sum(mask)),
            "mean_taxscore": float(np.mean(test_taxscore[mask])),
            "mean_user_items": float(np.mean([m["user_items"] for m, keep in zip(test_meta, mask) if keep])),
            "mean_n2_over_user_items": float(
                np.mean([(hidden_n * hidden_n) / m["user_items"] for m, keep in zip(test_meta, mask) if keep])
            ),
            "metrics": {
                model: posterior_metrics(
                    y_true[mask],
                    prob[mask],
                    tau=tau[model],
                    ref_prob=grid_tail[mask],
                    taxscore_bin=-1,
                )
                for model, prob in test_probs.items()
            },
        }
    return slices


def deployment_switch_audit(
    val_y,
    test_y,
    val_probs,
    test_probs,
    val_taxscore,
    test_taxscore,
    grid_tail_test,
    baseline="SeqTXCount",
    targets=("ANP", "FlowLatent"),
    fractions=(0.05, 0.10, 0.20, 0.30),
):
    """Audit a deployable policy: use baseline everywhere, switch flagged slates to a richer head."""
    if baseline not in val_probs or baseline not in test_probs:
        return {}
    base_tau = choose_tau(val_y, val_probs[baseline])
    base_all = posterior_metrics(test_y, test_probs[baseline], tau=base_tau, ref_prob=grid_tail_test)
    audit = {
        "baseline": baseline,
        "baseline_all": base_all,
        "selection": "TaxScore threshold chosen on validation quantile; no hidden test label is used.",
        "targets": {},
    }
    for target in targets:
        if target not in val_probs or target not in test_probs:
            continue
        rows = []
        for frac in fractions:
            cutoff = float(np.quantile(val_taxscore, 1.0 - frac))
            val_mask = val_taxscore >= cutoff
            test_mask = test_taxscore >= cutoff
            if not np.any(test_mask) or not np.any(val_mask):
                continue
            mixed_val = np.asarray(val_probs[baseline], dtype=np.float64).copy()
            mixed_test = np.asarray(test_probs[baseline], dtype=np.float64).copy()
            mixed_val[val_mask] = np.asarray(val_probs[target], dtype=np.float64)[val_mask]
            mixed_test[test_mask] = np.asarray(test_probs[target], dtype=np.float64)[test_mask]
            mixed_tau = choose_tau(val_y, mixed_val)
            switched = posterior_metrics(test_y, mixed_test, tau=mixed_tau, ref_prob=grid_tail_test)
            base_flagged = posterior_metrics(
                test_y[test_mask],
                np.asarray(test_probs[baseline], dtype=np.float64)[test_mask],
                tau=base_tau,
                ref_prob=grid_tail_test[test_mask],
            )
            target_tau = choose_tau(val_y, val_probs[target])
            target_flagged = posterior_metrics(
                test_y[test_mask],
                np.asarray(test_probs[target], dtype=np.float64)[test_mask],
                tau=target_tau,
                ref_prob=grid_tail_test[test_mask],
            )
            rows.append(
                {
                    "budget_fraction": float(frac),
                    "validation_cutoff": cutoff,
                    "test_fraction": float(np.mean(test_mask)),
                    "test_count": int(np.sum(test_mask)),
                    "tau": float(mixed_tau),
                    "switched": switched,
                    "full_nll_gain": float(base_all["nll"] - switched["nll"]),
                    "full_ece_gain": float(base_all["ece"] - switched["ece"]),
                    "full_utility_gain": float(switched["utility"] - base_all["utility"]),
                    "full_regret_reduction": float(base_all["regret"] - switched["regret"]),
                    "flagged_nll_gain": float(base_flagged["nll"] - target_flagged["nll"]),
                    "flagged_ece_gain": float(base_flagged["ece"] - target_flagged["ece"]),
                    "flagged_utility_gain": float(target_flagged["utility"] - base_flagged["utility"]),
                    "flagged_regret_reduction": float(base_flagged["regret"] - target_flagged["regret"]),
                    "baseline_flagged": base_flagged,
                    "target_flagged": target_flagged,
                }
            )
        audit["targets"][target] = rows
    return audit



def exact_top_fraction_mask(score, fraction):
    score = np.asarray(score, dtype=np.float64)
    count = max(1, min(len(score), int(round(float(fraction) * len(score)))))
    order = np.argsort(score, kind="mergesort")
    mask = np.zeros(len(score), dtype=bool)
    mask[order[-count:]] = True
    return mask


def random_fraction_mask(length, fraction, rng):
    count = max(1, min(int(length), int(round(float(fraction) * int(length)))))
    mask = np.zeros(int(length), dtype=bool)
    mask[rng.choice(int(length), size=count, replace=False)] = True
    return mask


def matched_routing_attribution(
    val_y,
    test_y,
    val_probs,
    test_probs,
    val_taxscore,
    test_taxscore,
    grid_tail_test,
    seed=0,
    baseline="SeqTXCount",
    targets=("ANP", "FlowLatent"),
    fractions=(0.20, 0.30),
    random_repeats=100,
):
    """Exact-budget attribution: TaxScore vs cheap uncertainty vs random routing."""
    if baseline not in val_probs or baseline not in test_probs:
        return {}
    base_val = np.asarray(val_probs[baseline], dtype=np.float64)
    base_test = np.asarray(test_probs[baseline], dtype=np.float64)
    base_tau = choose_tau(val_y, base_val)
    base_all = posterior_metrics(test_y, base_test, tau=base_tau, ref_prob=grid_tail_test)
    val_uncertainty = base_val * (1.0 - base_val)
    test_uncertainty = base_test * (1.0 - base_test)

    def evaluate_masks(target, val_mask, test_mask):
        target_val = np.asarray(val_probs[target], dtype=np.float64)
        target_test = np.asarray(test_probs[target], dtype=np.float64)
        mixed_val = base_val.copy()
        mixed_test = base_test.copy()
        mixed_val[val_mask] = target_val[val_mask]
        mixed_test[test_mask] = target_test[test_mask]
        mixed_tau = choose_tau(val_y, mixed_val)
        switched = posterior_metrics(test_y, mixed_test, tau=mixed_tau, ref_prob=grid_tail_test)
        target_tau = choose_tau(val_y, target_val)
        base_flagged = posterior_metrics(
            test_y[test_mask], base_test[test_mask], tau=base_tau, ref_prob=grid_tail_test[test_mask]
        )
        target_flagged = posterior_metrics(
            test_y[test_mask], target_test[test_mask], tau=target_tau, ref_prob=grid_tail_test[test_mask]
        )
        return {
            "test_count": int(np.sum(test_mask)),
            "test_fraction": float(np.mean(test_mask)),
            "mixed_tau": float(mixed_tau),
            "full_nll_gain": float(base_all["nll"] - switched["nll"]),
            "full_utility_gain": float(switched["utility"] - base_all["utility"]),
            "flagged_nll_gain": float(base_flagged["nll"] - target_flagged["nll"]),
            "flagged_utility_gain": float(target_flagged["utility"] - base_flagged["utility"]),
        }

    result = {
        "baseline": baseline,
        "selection_definitions": {
            "TaxScore": "exact top fraction by the pre-specified hidden-target-free TaxScore",
            "CheapUncertainty": "exact top fraction by SeqTXCount p(1-p)",
            "Random": f"mean over {int(random_repeats)} exact-budget random masks per seed",
        },
        "targets": {},
    }
    for target in targets:
        if target not in val_probs or target not in test_probs:
            continue
        target_rows = {}
        for fraction in fractions:
            tax_val = exact_top_fraction_mask(val_taxscore, fraction)
            tax_test = exact_top_fraction_mask(test_taxscore, fraction)
            unc_val = exact_top_fraction_mask(val_uncertainty, fraction)
            unc_test = exact_top_fraction_mask(test_uncertainty, fraction)
            selector_rows = {
                "TaxScore": evaluate_masks(target, tax_val, tax_test),
                "CheapUncertainty": evaluate_masks(target, unc_val, unc_test),
            }
            rng = np.random.default_rng(seed + int(10000 * fraction) + 1009 * len(target))
            random_rows = []
            for _ in range(int(random_repeats)):
                rv = random_fraction_mask(len(val_y), fraction, rng)
                rt = random_fraction_mask(len(test_y), fraction, rng)
                random_rows.append(evaluate_masks(target, rv, rt))
            keys = [
                "test_count",
                "test_fraction",
                "mixed_tau",
                "full_nll_gain",
                "full_utility_gain",
                "flagged_nll_gain",
                "flagged_utility_gain",
            ]
            selector_rows["Random"] = {
                key: {
                    "mean": float(np.mean([row[key] for row in random_rows])),
                    "std": float(np.std([row[key] for row in random_rows], ddof=0)),
                }
                for key in keys
            }
            target_rows[f"{int(round(100 * fraction))}%"] = selector_rows
        result["targets"][target] = target_rows
    return result


def tilted_bin_probs(base_probs, theta):
    scores = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    logits = np.log(np.clip(base_probs, 1e-8, 1.0)) + theta * scores
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def posterior_weights_bins(obs_bins, obs_base_probs, grid, prior_scale=1.6):
    log_prior = -0.5 * (grid / prior_scale) ** 2
    log_w = log_prior.copy()
    obs_bins = np.asarray(obs_bins, dtype=np.int64)
    scores = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    for idx, base in zip(obs_bins, obs_base_probs):
        logits = np.log(np.clip(np.asarray(base, dtype=np.float64), 1e-8, 1.0))[None, :] + grid[:, None] * scores[None, :]
        logits -= np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        log_w += np.log(np.clip(probs[:, idx], 1e-8, 1.0))
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / np.sum(w)


def partition_event_mc(weights, component_probs, rng, samples=192, high_frac=0.60, low_frac=0.20):
    # component_probs: [K, n, 3] for low/mid/high rating bins.
    k_components, n_items, _ = component_probs.shape
    high_threshold = int(math.ceil(high_frac * n_items))
    low_cap = int(math.floor(low_frac * n_items))
    component_idx = rng.choice(k_components, size=samples, p=weights)
    wins = 0
    for comp in component_idx:
        probs = component_probs[comp]
        uniforms = rng.random(n_items)
        cdf_low = probs[:, 0]
        cdf_mid = probs[:, 0] + probs[:, 1]
        low_count = int(np.sum(uniforms < cdf_low))
        high_count = int(np.sum(uniforms >= cdf_mid))
        wins += int(high_count >= high_threshold and low_count <= low_cap)
    mixture_prob = wins / samples
    marginal = np.tensordot(weights, component_probs, axes=(0, 0))
    wins = 0
    for _ in range(samples):
        uniforms = rng.random(n_items)
        cdf_low = marginal[:, 0]
        cdf_mid = marginal[:, 0] + marginal[:, 1]
        low_count = int(np.sum(uniforms < cdf_low))
        high_count = int(np.sum(uniforms >= cdf_mid))
        wins += int(high_count >= high_threshold and low_count <= low_cap)
    indep_prob = wins / samples
    return float(np.clip(mixture_prob, 1e-4, 1.0 - 1e-4)), float(np.clip(indep_prob, 1e-4, 1.0 - 1e-4))


def nonbinary_partition_demo(metas, grid, seed=0, max_examples=800):
    rng = np.random.default_rng(seed)
    if len(metas) > max_examples:
        indices = rng.choice(len(metas), size=max_examples, replace=False)
    else:
        indices = np.arange(len(metas))
    labels, mix_probs, indep_probs, laplace_gap, cov_gap = [], [], [], [], []
    h = np.asarray([-0.35, 0.0, 0.55], dtype=np.float64)
    for idx in indices:
        meta = metas[int(idx)]
        weights = posterior_weights_bins(meta["obs_bins"], meta["obs_bin_base"], grid)
        component_probs = np.stack(
            [tilted_bin_probs(meta["hidden_bin_base"], theta) for theta in grid], axis=0
        )
        mix_prob, indep_prob = partition_event_mc(weights, component_probs, rng)
        hidden_bins = np.asarray(meta["hidden_bins"], dtype=np.int64)
        high_count = int(np.sum(hidden_bins == 2))
        low_count = int(np.sum(hidden_bins == 0))
        label = int(high_count >= math.ceil(0.60 * len(hidden_bins)) and low_count <= math.floor(0.20 * len(hidden_bins)))
        marginal = np.tensordot(weights, component_probs, axes=(0, 0))
        comp_lap = np.prod(np.sum(component_probs * np.exp(h)[None, None, :], axis=-1), axis=-1)
        mix_lap = float(np.dot(weights, comp_lap))
        indep_lap = float(np.prod(np.sum(marginal * np.exp(h)[None, :], axis=-1)))
        comp_high = component_probs[:, :, 2]
        mean_high = np.tensordot(weights, comp_high, axes=(0, 0))
        centered = comp_high - mean_high[None, :]
        cov = (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
        offdiag = cov[~np.eye(cov.shape[0], dtype=bool)]
        labels.append(label)
        mix_probs.append(mix_prob)
        indep_probs.append(indep_prob)
        laplace_gap.append(mix_lap - indep_lap)
        cov_gap.append(float(np.sum(offdiag)))
    labels = np.asarray(labels, dtype=np.int64)
    mix_probs = np.asarray(mix_probs, dtype=np.float64)
    indep_probs = np.asarray(indep_probs, dtype=np.float64)
    return {
        "config": {
            "rating_bins": ["1-2", "3", "4-5"],
            "event": "high_count >= 0.60 n and low_count <= 0.20 n",
            "examples": int(len(labels)),
            "mc_samples": 192,
        },
        "GlobalBinMixture": binary_metrics(labels, mix_probs),
        "MeanIndependentBins": binary_metrics(labels, indep_probs),
        "mean_abs_event_gap": float(np.mean(np.abs(mix_probs - indep_probs))),
        "mean_laplace_gap": float(np.mean(laplace_gap)),
        "mean_high_bin_cov_gap": float(np.mean(cov_gap)),
    }


def hypergeom_pmf(N, K, n):
    lo = max(0, n - (N - K))
    hi = min(n, K)
    pmf = np.zeros(n + 1, dtype=np.float64)
    denom = comb(N, n)
    for r in range(lo, hi + 1):
        pmf[r] = comb(K, r) * comb(N - K, n - r) / denom
    return pmf


def binom_pmf(n, q):
    return np.asarray([comb(n, r) * (q ** r) * ((1.0 - q) ** (n - r)) for r in range(n + 1)], dtype=np.float64)


def finite_exchangeability_stress(seed=0):
    _ = np.random.default_rng(seed)
    q0, q1, alpha, n, c = 0.2, 0.8, 0.35, 24, 0.5
    threshold = int(math.ceil(c * n))
    rows = []
    for N in [32, 48, 64, 96, 128, 192, 384, 768, 1536]:
        k0 = int(round(q0 * N))
        k1 = int(round(q1 * N))
        finite = (1.0 - alpha) * hypergeom_pmf(N, k0, n) + alpha * hypergeom_pmf(N, k1, n)
        surrogate = (1.0 - alpha) * binom_pmf(n, k0 / N) + alpha * binom_pmf(n, k1 / N)
        tv = 0.5 * float(np.sum(np.abs(finite - surrogate)))
        threshold_gap = abs(float(np.sum(finite[threshold:]) - np.sum(surrogate[threshold:])))
        log_no_collision = sum(math.log1p(-j / N) for j in range(n))
        delta = 1.0 - math.exp(log_no_collision)
        rows.append(
            {
                "N": N,
                "n": n,
                "n2_over_N": float((n * n) / N),
                "tv_finite_vs_surrogate": tv,
                "threshold_gap": threshold_gap,
                "collision_bound": float(delta),
                "simple_bound": float(n * (n - 1) / (2 * N)),
            }
        )
    return {"config": {"q0": q0, "q1": q1, "alpha": alpha, "n": n, "c": c}, "rows": rows}


def run_config(seed, config, requested_models=None):
    requested_models = list(requested_models or DEFAULT_MODELS)
    dataset_name, ratings_by_user, movie_features, all_ratings = load_movielens_dataset(
        config.get("dataset", "ml1m")
    )
    k = int(config["k"])
    hidden_n = int(config["n"])
    c = float(config["c"])
    policy = str(config.get("policy", "random"))
    critical_margin = float(config.get("critical_margin", 0.08))
    critical_attempts = int(config.get("critical_attempts", 80))
    prefix_weight = float(config.get("prefix_weight", 0.20))
    selector_mode = str(config.get("selector_mode", "default"))
    threshold = int(math.ceil(c * hidden_n))
    train_users, val_users, test_users = split_users(
        ratings_by_user, k + hidden_n, seed, max_users=config.get("max_users")
    )
    movie_logits, default_logit, global_rate = movie_popularity_logits(ratings_by_user, train_users)
    movie_bin_probs, default_bin_probs = movie_rating_bin_probs(ratings_by_user, train_users)

    train_seq, train_y, train_count, train_meta = sample_views(
        ratings_by_user,
        movie_features,
        movie_logits,
        movie_bin_probs,
        default_logit,
        default_bin_probs,
        train_users,
        k,
        hidden_n,
        c,
        views_per_user=3,
        seed=seed + 11,
        policy=policy,
        critical_margin=critical_margin,
        max_attempts=critical_attempts,
        prefix_weight=prefix_weight,
        selector_mode=selector_mode,
    )
    val_seq, val_y, val_count, val_meta = sample_views(
        ratings_by_user,
        movie_features,
        movie_logits,
        movie_bin_probs,
        default_logit,
        default_bin_probs,
        val_users,
        k,
        hidden_n,
        c,
        views_per_user=2,
        seed=seed + 12,
        policy=policy,
        critical_margin=critical_margin,
        max_attempts=critical_attempts,
        prefix_weight=prefix_weight,
        selector_mode=selector_mode,
    )
    test_seq, test_y, test_count, test_meta = sample_views(
        ratings_by_user,
        movie_features,
        movie_logits,
        movie_bin_probs,
        default_logit,
        default_bin_probs,
        test_users,
        k,
        hidden_n,
        c,
        views_per_user=2,
        seed=seed + 13,
        policy=policy,
        critical_margin=critical_margin,
        max_attempts=critical_attempts,
        prefix_weight=prefix_weight,
        selector_mode=selector_mode,
    )
    val_seq_raw = val_seq.copy()
    test_seq_raw = test_seq.copy()
    train_seq, val_seq, test_seq = standardize_sequences(train_seq, val_seq, test_seq)

    grid_main = np.linspace(-3.0, 3.0, int(config.get("K", 16)))
    grid_global = np.linspace(-3.0, 3.0, int(config.get("K_global", config.get("K", 16))))
    grid_tail_val, mean_tail_val, val_taxscore, _, val_cov_diag = grid_predictions(
        val_seq_raw, val_meta, k, threshold, grid_main
    )
    grid_tail_test, mean_tail_test, test_taxscore, mean_q, test_cov_diag = grid_predictions(
        test_seq_raw, test_meta, k, threshold, grid_main
    )
    ar_val = ar_rollout_predictions(val_meta, k, hidden_n, threshold, global_rate, seed=seed + 100)
    ar_test = ar_rollout_predictions(test_meta, k, hidden_n, threshold, global_rate, seed=seed + 101)

    obj_train = aggregate_features(train_seq, k)
    obj_val = aggregate_features(val_seq, k)
    obj_test = aggregate_features(test_seq, k)
    val_probs = {
        "MeanIndep": mean_tail_val,
        "ARRollout": ar_val,
        "GlobalGrid": grid_tail_val,
    }
    test_probs = {
        "MeanIndep": mean_tail_test,
        "ARRollout": ar_test,
        "GlobalGrid": grid_tail_test,
    }
    capacity_audit = {
        "MeanIndep": audit_record(seconds=0.0),
        "ARRollout": audit_record(seconds=0.0, rollouts=96),
        "GlobalGrid": audit_record(seconds=0.0, mc_samples=len(grid_main)),
    }
    train_item_y = hidden_label_matrix(train_meta)
    val_item_y = hidden_label_matrix(val_meta)
    if "ObjMLP" in requested_models:
        start = time.perf_counter()
        obj_mlp = MLPClassifier(
            hidden_layer_sizes=(96, 48),
            random_state=seed + 30,
            max_iter=160,
            alpha=1e-4,
            early_stopping=True,
            n_iter_no_change=8,
        )
        obj_mlp.fit(obj_train, train_y)
        capacity_audit["ObjMLP"] = audit_record(obj_mlp, time.perf_counter() - start)
        val_probs["ObjMLP"] = obj_mlp.predict_proba(obj_val)[:, 1]
        test_probs["ObjMLP"] = obj_mlp.predict_proba(obj_test)[:, 1]
    if "SeqGRUCount" in requested_models:
        seq_gru, elapsed = timed_fit(
            fit_gru_count,
            train_seq,
            train_count,
            val_seq,
            val_count,
            seed + 40,
            n_counts=hidden_n + 1,
            hidden_dim=72,
            epochs=8,
            batch_size=256,
        )
        capacity_audit["SeqGRUCount"] = audit_record(seq_gru, elapsed)
        val_probs["SeqGRUCount"] = predict_gru_count_tail_probs(seq_gru, val_seq, threshold)
        test_probs["SeqGRUCount"] = predict_gru_count_tail_probs(seq_gru, test_seq, threshold)
    if "SeqTXCount" in requested_models:
        seq_tx, elapsed = timed_fit(
            fit_transformer_count,
            train_seq,
            train_count,
            val_seq,
            val_count,
            seed + 50,
            n_counts=hidden_n + 1,
            hidden_dim=72,
            epochs=8,
            batch_size=256,
        )
        capacity_audit["SeqTXCount"] = audit_record(seq_tx, elapsed)
        val_probs["SeqTXCount"] = predict_transformer_count_tail_probs(seq_tx, val_seq, threshold)
        test_probs["SeqTXCount"] = predict_transformer_count_tail_probs(seq_tx, test_seq, threshold)
    if "SeqTXCountPM" in requested_models:
        seq_tx_pm, elapsed = timed_fit(
            fit_transformer_count,
            train_seq,
            train_count,
            val_seq,
            val_count,
            seed + 55,
            n_counts=hidden_n + 1,
            hidden_dim=128,
            epochs=10,
            batch_size=192,
        )
        capacity_audit["SeqTXCountPM"] = audit_record(seq_tx_pm, elapsed)
        val_probs["SeqTXCountPM"] = predict_transformer_count_tail_probs(seq_tx_pm, val_seq, threshold)
        test_probs["SeqTXCountPM"] = predict_transformer_count_tail_probs(seq_tx_pm, test_seq, threshold)
    if "ObjSetTF" in requested_models:
        obj_set, elapsed = timed_fit(
            fit_set_transformer,
            train_seq,
            train_y,
            val_seq,
            val_y,
            seed + 60,
            hidden_dim=64,
            epochs=8,
            batch_size=256,
        )
        capacity_audit["ObjSetTF"] = audit_record(obj_set, elapsed)
        val_probs["ObjSetTF"] = predict_set_transformer(obj_set, val_seq)
        test_probs["ObjSetTF"] = predict_set_transformer(obj_set, test_seq)
    if "ObjSetTFPM" in requested_models:
        obj_set_pm, elapsed = timed_fit(
            fit_set_transformer,
            train_seq,
            train_y,
            val_seq,
            val_y,
            seed + 65,
            hidden_dim=96,
            epochs=10,
            batch_size=192,
        )
        capacity_audit["ObjSetTFPM"] = audit_record(obj_set_pm, elapsed)
        val_probs["ObjSetTFPM"] = predict_set_transformer(obj_set_pm, val_seq)
        test_probs["ObjSetTFPM"] = predict_set_transformer(obj_set_pm, test_seq)
    if "SeqTXGlobal" in requested_models:
        seq_global, elapsed = timed_fit(
            fit_global_tail,
            "seq", train_seq, train_y, val_seq, val_y, seed + 70, hidden_n, threshold, grid_global
        )
        capacity_audit["SeqTXGlobal"] = audit_record(seq_global, elapsed, mc_samples=len(grid_global))
        val_probs["SeqTXGlobal"] = predict_global_tail(seq_global, val_seq)
        test_probs["SeqTXGlobal"] = predict_global_tail(seq_global, test_seq)
    if "ObjSetGlobal" in requested_models:
        set_global, elapsed = timed_fit(
            fit_global_tail,
            "set", train_seq, train_y, val_seq, val_y, seed + 80, hidden_n, threshold, grid_global
        )
        capacity_audit["ObjSetGlobal"] = audit_record(set_global, elapsed, mc_samples=len(grid_global))
        val_probs["ObjSetGlobal"] = predict_global_tail(set_global, val_seq)
        test_probs["ObjSetGlobal"] = predict_global_tail(set_global, test_seq)
    if "TwoTowerIndep" in requested_models or "TwoTowerDropout" in requested_models:
        tower, elapsed = timed_fit(
            fit_two_tower_item,
            train_seq,
            train_item_y,
            val_seq,
            val_item_y,
            seed + 84,
            hidden_n,
            hidden_dim=64,
            epochs=8,
        )
        if "TwoTowerIndep" in requested_models:
            capacity_audit["TwoTowerIndep"] = audit_record(tower, elapsed, best_epoch=getattr(tower, "_best_epoch", None))
            val_probs["TwoTowerIndep"] = predict_two_tower_tail(tower, val_seq, threshold)
            test_probs["TwoTowerIndep"] = predict_two_tower_tail(tower, test_seq, threshold)
        if "TwoTowerDropout" in requested_models:
            capacity_audit["TwoTowerDropout"] = audit_record(
                tower, elapsed, best_epoch=getattr(tower, "_best_epoch", None), mc_samples=24
            )
            val_probs["TwoTowerDropout"] = predict_two_tower_dropout_tail(tower, val_seq, threshold, seed=seed + 85, mc_samples=24)
            test_probs["TwoTowerDropout"] = predict_two_tower_dropout_tail(tower, test_seq, threshold, seed=seed + 86, mc_samples=24)
    if "TwoTowerEnsMean" in requested_models or "TwoTowerEnsMix" in requested_models:
        ensemble, elapsed = fit_two_tower_ensemble(train_seq, train_item_y, val_seq, val_item_y, seed + 87, hidden_n)
        if "TwoTowerEnsMean" in requested_models:
            capacity_audit["TwoTowerEnsMean"] = audit_record(
                None, elapsed, members=len(ensemble), mc_samples=0
            )
            capacity_audit["TwoTowerEnsMean"]["params"] = int(sum(model_param_count(model) for model in ensemble))
            val_probs["TwoTowerEnsMean"] = predict_two_tower_ensemble(ensemble, val_seq, threshold, mode="mean")
            test_probs["TwoTowerEnsMean"] = predict_two_tower_ensemble(ensemble, test_seq, threshold, mode="mean")
        if "TwoTowerEnsMix" in requested_models:
            capacity_audit["TwoTowerEnsMix"] = audit_record(
                None, elapsed, members=len(ensemble), mc_samples=len(ensemble)
            )
            capacity_audit["TwoTowerEnsMix"]["params"] = int(sum(model_param_count(model) for model in ensemble))
            val_probs["TwoTowerEnsMix"] = predict_two_tower_ensemble(ensemble, val_seq, threshold, mode="mixture")
            test_probs["TwoTowerEnsMix"] = predict_two_tower_ensemble(ensemble, test_seq, threshold, mode="mixture")
    latent_dim = int(config.get("latent_dim", 8))
    if "ANP" in requested_models:
        anp, elapsed = timed_fit(
            fit_stochastic_latent_tail,
            "anp", train_seq, train_y, val_seq, val_y, seed + 90, hidden_n, threshold, latent_dim=latent_dim
        )
        capacity_audit["ANP"] = audit_record(anp, elapsed, mc_samples=64)
        val_probs["ANP"] = predict_stochastic_latent_tail(anp, val_seq, seed=seed + 91, mc_samples=64)
        test_probs["ANP"] = predict_stochastic_latent_tail(anp, test_seq, seed=seed + 92, mc_samples=64)
    if "FlowLatent" in requested_models:
        flow, elapsed = timed_fit(
            fit_stochastic_latent_tail,
            "flow", train_seq, train_y, val_seq, val_y, seed + 95, hidden_n, threshold, latent_dim=latent_dim
        )
        capacity_audit["FlowLatent"] = audit_record(flow, elapsed, mc_samples=64)
        val_probs["FlowLatent"] = predict_stochastic_latent_tail(flow, val_seq, seed=seed + 96, mc_samples=64)
        test_probs["FlowLatent"] = predict_stochastic_latent_tail(flow, test_seq, seed=seed + 97, mc_samples=64)

    val_probs = {model: prob for model, prob in val_probs.items() if model in requested_models}
    test_probs = {model: prob for model, prob in test_probs.items() if model in requested_models}

    tau = {model: choose_tau(val_y, prob) for model, prob in val_probs.items()}
    metrics = {
        model: posterior_metrics(test_y, prob, tau=tau[model], ref_prob=grid_tail_test, taxscore_bin=-1)
        for model, prob in test_probs.items()
    }

    q = np.quantile(test_taxscore, [0.0, 1 / 3, 2 / 3, 1.0])
    bins = {}
    for idx, name in enumerate(["low", "mid", "high"]):
        if idx == 2:
            mask = test_taxscore >= q[idx]
        else:
            mask = (test_taxscore >= q[idx]) & (test_taxscore < q[idx + 1])
        bins[name] = {
            "count": int(np.sum(mask)),
            "mean_taxscore": float(np.mean(test_taxscore[mask])),
            "mean_user_items": float(np.mean([m["user_items"] for m, keep in zip(test_meta, mask) if keep])),
            "mean_n2_over_user_items": float(
                np.mean([(hidden_n * hidden_n) / m["user_items"] for m, keep in zip(test_meta, mask) if keep])
            ),
            "metrics": {
                model: posterior_metrics(
                    test_y[mask],
                    prob[mask],
                    tau=tau[model],
                    ref_prob=grid_tail_test[mask],
                    taxscore_bin=idx,
                )
                for model, prob in test_probs.items()
            },
        }
    tail_slices = taxscore_tail_slices(test_y, test_probs, test_taxscore, grid_tail_test, tau, test_meta, hidden_n)

    resolution_rows = []
    for K in [2, 4, 8, 16, 32]:
        grid = np.linspace(-3.0, 3.0, K)
        grid_tail, mean_tail, _, _, _ = grid_predictions(test_seq_raw, test_meta, k, threshold, grid)
        resolution_rows.append(
            {
                "K": K,
                "GlobalGrid": binary_metrics(test_y, grid_tail)["nll"],
                "MeanIndep": binary_metrics(test_y, mean_tail)["nll"],
            }
        )

    return {
        "config": {
            "dataset": dataset_name,
            "max_users": int(config["max_users"]) if config.get("max_users") else -1,
            "k": k,
            "hidden_n": hidden_n,
            "c": c,
            "policy": policy,
            "selector_mode": selector_mode,
            "prefix_weight": prefix_weight,
            "critical_margin": critical_margin,
            "critical_attempts": critical_attempts,
            "threshold": threshold,
            "taxscore_grid_size": int(len(grid_main)),
            "seqtxglobal_components": int(len(grid_global)),
            "train_users": len(train_users),
            "val_users": len(val_users),
            "test_users": len(test_users),
            "train_samples": len(train_y),
            "val_samples": len(val_y),
            "test_samples": len(test_y),
        },
        "slice_protocol": {
            "policy": policy,
            "visible_features": [
                "hidden movie genres",
                "training-set movie popularity logits",
                "observed prefix ratings",
            ],
            "forbidden_features": [
                "hidden ratings",
                "hidden positives",
                "future user outcomes",
                "model predictions",
                "test labels",
            ],
            "selector_mode": selector_mode,
            "prefix_weight": prefix_weight,
            "critical_margin": critical_margin,
            "max_attempts": critical_attempts,
            "boundary_score": {
                "default": "|p_prior-c| + prefix_weight |y_obs-1/2|",
                "prior_only": "|p_prior-c|",
                "prefix_only": "|y_obs-1/2|",
            }[selector_mode],
            "proxy_direction": "smaller boundary score is more decision-critical; routing TaxScore bins use the separate validation-fixed posterior-grid proxy",
            "mean_selection_hidden_prior": float(np.mean([m["selection_hidden_prior"] for m in test_meta])),
            "mean_selection_obs_rate": float(np.mean([m["selection_obs_rate"] for m in test_meta])),
            "mean_selection_boundary_score": float(np.mean([m["selection_score"] for m in test_meta])),
        },
        "metrics": metrics,
        "bins": bins,
        "tail_slices": tail_slices,
        "resolution_rows": resolution_rows,
        "capacity_audit": {model: capacity_audit[model] for model in metrics if model in capacity_audit},
        "calibration_diagnostics": {
            model: {
                "reliability_curve": reliability_curve(test_y, prob),
                "brier_decomposition": brier_decomposition(test_y, prob),
            }
            for model, prob in test_probs.items()
        },
        "deployment_switch_audit": deployment_switch_audit(
            val_y,
            test_y,
            val_probs,
            test_probs,
            val_taxscore,
            test_taxscore,
            grid_tail_test,
        ),
        "matched_routing_attribution": matched_routing_attribution(
            val_y,
            test_y,
            val_probs,
            test_probs,
            val_taxscore,
            test_taxscore,
            grid_tail_test,
            seed=seed + 5000,
        ),
        "taxscore_gain_correlation": gain_by_taxscore_bins(test_y, test_probs, test_taxscore),
        "signed_covariance_audit": {
            "overall": signed_covariance_audit(test_cov_diag),
            "high": signed_covariance_audit(test_cov_diag, test_taxscore >= q[2]),
        },
        "nonbinary_partition_demo": nonbinary_partition_demo(test_meta, grid_main, seed=seed + 777),
        "posterior_interface_only": {
            "GlobalGrid": posterior_metrics(test_y, grid_tail_test, tau=choose_tau(val_y, grid_tail_val), ref_prob=grid_tail_test),
            "MeanIndep": posterior_metrics(test_y, mean_tail_test, tau=choose_tau(val_y, mean_tail_val), ref_prob=grid_tail_test),
        },
        "param_matched_ablation": {
            model: metrics[model]
            for model in [
                "SeqTXCount",
                "SeqTXCountPM",
                "ObjSetTF",
                "ObjSetTFPM",
                "TwoTowerIndep",
                "TwoTowerEnsMean",
                "TwoTowerEnsMix",
                "SeqTXGlobal",
                "ANP",
                "FlowLatent",
            ]
            if model in metrics
        },
        "high_dim_latent_ablation": {
            model: metrics[model]
            for model in [
                "SeqTXCount",
                "TwoTowerIndep",
                "TwoTowerEnsMean",
                "TwoTowerEnsMix",
                "GlobalGrid",
                "SeqTXGlobal",
                "ANP",
                "FlowLatent",
            ]
            if model in metrics
        },
    }


def aggregate_payloads(payloads):
    configs = payloads[0]["configs"].keys()
    result = {
        "configs": {},
        "tail_slices": {},
        "posterior_interface_benchmark": {},
        "high_dim_latent_ablation": {},
        "param_matched_ablation": {},
        "slice_protocol": {},
        "capacity_audit": {},
        "calibration_diagnostics": {},
        "deployment_switch_audit": {},
        "taxscore_gain_correlation": {},
        "diagnostic_proxy_audit": {},
        "movielens_proxy_robustness": {},
        "statistical_audit": {},
        "signed_covariance_audit": {},
        "nonbinary_partition_demo": {},
    }
    for cfg in configs:
        models = [
            model
            for model in MODEL_ORDER
            if model in payloads[0]["configs"][cfg]["metrics"]
        ]
        result["configs"][cfg] = {
            "config": payloads[0]["configs"][cfg]["config"],
            "metrics": {},
            "bins": {},
            "tail_slices": {},
            "resolution_rows": [],
        }
        for model in models:
            records = [p["configs"][cfg]["metrics"][model] for p in payloads]
            result["configs"][cfg]["metrics"][model] = aggregate_metric_dict(records)
        for bin_name in ["low", "mid", "high"]:
            bin_records = [p["configs"][cfg]["bins"][bin_name] for p in payloads]
            result["configs"][cfg]["bins"][bin_name] = {
                "count": mean_std([r["count"] for r in bin_records]),
                "mean_taxscore": mean_std([r["mean_taxscore"] for r in bin_records]),
                "mean_user_items": mean_std([r["mean_user_items"] for r in bin_records]),
                "mean_n2_over_user_items": mean_std([r["mean_n2_over_user_items"] for r in bin_records]),
                "metrics": {},
            }
            for model in models:
                records = [r["metrics"][model] for r in bin_records]
                result["configs"][cfg]["bins"][bin_name]["metrics"][model] = aggregate_metric_dict(records)
        for slice_name in ["top30", "top20", "top10", "top05"]:
            if slice_name not in payloads[0]["configs"][cfg].get("tail_slices", {}):
                continue
            slice_records = [p["configs"][cfg]["tail_slices"][slice_name] for p in payloads]
            result["configs"][cfg]["tail_slices"][slice_name] = {
                "fraction": mean_std([r["fraction"] for r in slice_records]),
                "cutoff": mean_std([r["cutoff"] for r in slice_records]),
                "count": mean_std([r["count"] for r in slice_records]),
                "mean_taxscore": mean_std([r["mean_taxscore"] for r in slice_records]),
                "mean_user_items": mean_std([r["mean_user_items"] for r in slice_records]),
                "mean_n2_over_user_items": mean_std([r["mean_n2_over_user_items"] for r in slice_records]),
                "metrics": {},
            }
            for model in models:
                records = [r["metrics"][model] for r in slice_records]
                result["configs"][cfg]["tail_slices"][slice_name]["metrics"][model] = aggregate_metric_dict(records)
            result["tail_slices"].setdefault(cfg, {})[slice_name] = result["configs"][cfg]["tail_slices"][slice_name]
        for idx in range(len(payloads[0]["configs"][cfg]["resolution_rows"])):
            rows = [p["configs"][cfg]["resolution_rows"][idx] for p in payloads]
            entry = aggregate_metric_dict(rows)
            entry["K"] = payloads[0]["configs"][cfg]["resolution_rows"][idx]["K"]
            result["configs"][cfg]["resolution_rows"].append(entry)
        for key in ["posterior_interface_only", "high_dim_latent_ablation", "param_matched_ablation"]:
            result[key.replace("_only", "_benchmark") if key == "posterior_interface_only" else key][cfg] = {}
            models_key = payloads[0]["configs"][cfg].get(key, {}).keys()
            for model in models_key:
                records = [p["configs"][cfg][key][model] for p in payloads]
                target_key = key.replace("_only", "_benchmark") if key == "posterior_interface_only" else key
                result[target_key][cfg][model] = aggregate_metric_dict(records)
        for key in [
            "slice_protocol",
            "capacity_audit",
            "calibration_diagnostics",
            "deployment_switch_audit",
            "taxscore_gain_correlation",
            "signed_covariance_audit",
            "nonbinary_partition_demo",
        ]:
            result[key][cfg] = aggregate_any([p["configs"][cfg][key] for p in payloads])
        result["statistical_audit"][cfg] = statistical_audit_for_config(payloads, cfg)
        result["diagnostic_proxy_audit"][cfg] = {
            "ANP": proxy_gain_correlation_for_config(result["configs"][cfg], target="ANP"),
            "FlowLatent": proxy_gain_correlation_for_config(result["configs"][cfg], target="FlowLatent"),
            "interpretation": (
                "TaxScore is compared with the crude finite-population ratio n^2/N_user. "
                "A weaker finite-ratio correlation is a negative control: slate size alone "
                "does not substitute for an ambiguity-and-margin diagnostic."
            ),
        }
        if "SeqTXCount" in result["configs"][cfg]["metrics"] and "FlowLatent" in result["configs"][cfg]["metrics"]:
            protocol = result["slice_protocol"][cfg]
            top20 = result["configs"][cfg].get("tail_slices", {}).get("top20", {})
            deploy_rows = (
                result["deployment_switch_audit"]
                .get(cfg, {})
                .get("targets", {})
                .get("FlowLatent", [])
            )
            switch20 = next(
                (row for row in deploy_rows if abs(stat_mean(row.get("budget_fraction", 0.0)) - 0.20) < 1e-6),
                {},
            )
            proxy_corr = proxy_gain_correlation_for_config(result["configs"][cfg], target="FlowLatent")
            if top20:
                base = top20["metrics"]["SeqTXCount"]
                target = top20["metrics"]["FlowLatent"]
                result["movielens_proxy_robustness"][cfg] = {
                    "dataset": result["configs"][cfg]["config"].get("dataset", ""),
                    "hidden_n": result["configs"][cfg]["config"].get("hidden_n", 0),
                    "max_users": result["configs"][cfg]["config"].get("max_users", -1),
                    "c": result["configs"][cfg]["config"].get("c", 0.0),
                    "selector_mode": protocol.get("selector_mode", "default"),
                    "prefix_weight": protocol.get("prefix_weight", 0.20),
                    "critical_margin": protocol.get("critical_margin", 0.08),
                    "candidate_budget": protocol.get("max_attempts", 80),
                    "corr_proxy_bin_nll_gain": proxy_corr.get("taxscore_nll_gain_corr", 0.0),
                    "top20_nll_gain": stat_mean(base["nll"]) - stat_mean(target["nll"]),
                    "top20_utility_gain": stat_mean(target["utility"]) - stat_mean(base["utility"]),
                    "full_switch_utility_gain": stat_mean(switch20.get("full_utility_gain", 0.0)),
                }
    if "finite_exchangeability_stress" in payloads[0]:
        result["finite_exchangeability_stress"] = {
            "config": payloads[0]["finite_exchangeability_stress"]["config"],
            "rows": [],
        }
        for idx in range(len(payloads[0]["finite_exchangeability_stress"]["rows"])):
            rows = [p["finite_exchangeability_stress"]["rows"][idx] for p in payloads]
            entry = aggregate_metric_dict(rows)
            entry["N"] = payloads[0]["finite_exchangeability_stress"]["rows"][idx]["N"]
            entry["n"] = payloads[0]["finite_exchangeability_stress"]["rows"][idx]["n"]
            result["finite_exchangeability_stress"]["rows"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=str, default="movielens_tail_results.json")
    parser.add_argument("--aggregate", nargs="*")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config in name:k:n:c format, e.g. hard:5:50:0.7",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="default",
        help="Comma-separated model list or 'default'. Aliases include seqtx_count, seqtx_global, anp, flow, two_tower_indep, two_tower_ens_mix, param_matched, global_grid, mean_indep.",
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--critical-margin", type=float, default=0.08)
    parser.add_argument("--critical-attempts", type=int, default=80)
    parser.add_argument("--prefix-weight", type=float, default=0.20)
    parser.add_argument(
        "--selector-mode",
        type=str,
        default="default",
        choices=["default", "prior_only", "prefix_only"],
    )
    parser.add_argument("--dataset", type=str, default="ml1m", choices=["ml1m", "ml10m", "ml20m"])
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Optional cap on eligible users after shuffling. Use 0 for all eligible users.",
    )
    parser.add_argument("--skip-finite-stress", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.aggregate]
        result = aggregate_payloads(payloads)
    else:
        if args.seed is None:
            raise ValueError("--seed is required unless --aggregate is provided.")
        config_specs = args.config or ["main:5:50:0.7", "cold:2:100:0.7"]
        requested_models = parse_models(args.models)
        result = {"seed": args.seed, "models": requested_models, "configs": {}}
        for spec in config_specs:
            parts = spec.split(":")
            if len(parts) == 4:
                name, k, n, c = parts
                policy = "random"
                critical_margin = args.critical_margin
                critical_attempts = args.critical_attempts
                prefix_weight = args.prefix_weight
                selector_mode = args.selector_mode
            elif len(parts) == 5:
                name, k, n, c, policy = parts
                critical_margin = args.critical_margin
                critical_attempts = args.critical_attempts
                prefix_weight = args.prefix_weight
                selector_mode = args.selector_mode
            elif len(parts) == 9:
                name, k, n, c, policy, critical_margin, critical_attempts, prefix_weight, selector_mode = parts
                critical_margin = float(critical_margin)
                critical_attempts = int(critical_attempts)
                prefix_weight = float(prefix_weight)
            else:
                raise ValueError(
                    "--config must be name:k:n:c, name:k:n:c:policy, "
                    "or name:k:n:c:policy:critical_margin:critical_attempts:prefix_weight:selector_mode"
                )
            result["configs"][name] = run_config(
                args.seed,
                {
                    "name": name,
                    "k": int(k),
                    "n": int(n),
                    "c": float(c),
                    "K": 16,
                    "K_global": 64,
                    "latent_dim": args.latent_dim,
                    "policy": policy,
                    "dataset": args.dataset,
                    "max_users": args.max_users,
                    "critical_margin": critical_margin,
                    "critical_attempts": critical_attempts,
                    "prefix_weight": prefix_weight,
                    "selector_mode": selector_mode,
                },
                requested_models=requested_models,
            )
        if not args.skip_finite_stress:
            result["finite_exchangeability_stress"] = finite_exchangeability_stress(args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
