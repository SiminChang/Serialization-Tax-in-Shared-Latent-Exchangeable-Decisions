import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def ece_score(y_true, prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins, right=False) - 1
    value = 0.0
    for b in range(n_bins):
        mask = idx == b
        if np.any(mask):
            conf = float(np.mean(prob[mask]))
            acc = float(np.mean(y_true[mask]))
            value += float(np.mean(mask)) * abs(conf - acc)
    return value


def poisson_binomial_tail(probabilities, threshold):
    dp = np.zeros(len(probabilities) + 1, dtype=np.float64)
    dp[0] = 1.0
    for p in probabilities:
        new = np.zeros_like(dp)
        new[1:] += dp[:-1] * p
        new[:-1] += dp[:-1] * (1.0 - p)
        dp = new
    return float(np.sum(dp[threshold:]))


@dataclass
class BagData:
    x_obs: np.ndarray
    z_obs: np.ndarray
    x_full: np.ndarray
    z_full: np.ndarray
    bag_label: int
    latent_h: int


MIL_CSV_URLS = {
    "birds_brown_creeper": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/birds_brown_creeper.csv",
    "corel_dogs": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/corel_dogs.csv",
    "elephant": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/elephant.csv",
    "musk1": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/musk1.csv",
    "musk2": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/musk2.csv",
    "protein": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/protein.csv",
    "ucsb_breast_cancer": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/ucsb_breast_cancer.csv",
    "web_recommendation_1": "https://raw.githubusercontent.com/rosasalberto/mil/master/mil/data/datasets/csv/web_recommendation_1.csv",
}


def split_binary_pools(X, y):
    neg = X[y == 0]
    pos = X[y == 1]
    return neg, pos


def cached_download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(url, destination)
    return destination


def load_mil_csv_groups(dataset_name, min_items=1, cache_dir=".data_cache/mil"):
    if dataset_name not in MIL_CSV_URLS:
        raise ValueError(f"Unknown MIL dataset: {dataset_name}")
    cache_path = cached_download(MIL_CSV_URLS[dataset_name], Path(cache_dir) / f"{dataset_name}.csv")
    bags = {}
    labels = {}
    with cache_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            label = int(float(row[0]))
            bag_id = row[1]
            feat = np.asarray(row[2:], dtype=np.float32)
            bags.setdefault(bag_id, []).append(feat)
            labels[bag_id] = label

    groups = {}
    for bag_id, feats in bags.items():
        x = np.asarray(feats, dtype=np.float32)
        if len(x) < min_items:
            continue
        groups[str(bag_id)] = {"x": x, "label": int(labels[bag_id])}
    feature_dim = next(iter(groups.values()))["x"].shape[1]
    return groups, feature_dim


def sample_hierarchical_bags(X, y, n_bags, m, k, q0, q1, seed):
    rng = np.random.default_rng(seed)
    neg_pool, pos_pool = split_binary_pools(X, y)
    bags = []
    for _ in range(n_bags):
        h = int(rng.integers(0, 2))
        q = q1 if h == 1 else q0
        x_items = []
        z_items = []
        for _ in range(m):
            z = int(rng.random() < q)
            pool = pos_pool if z == 1 else neg_pool
            x_items.append(pool[int(rng.integers(0, len(pool)))])
            z_items.append(z)
        x_items = np.asarray(x_items, dtype=np.float32)
        z_items = np.asarray(z_items, dtype=np.int8)
        perm = rng.permutation(m)
        x_items = x_items[perm]
        z_items = z_items[perm]
        bags.append(
            BagData(
                x_obs=x_items[:k],
                z_obs=z_items[:k],
                x_full=x_items,
                z_full=z_items,
                bag_label=int(np.sum(z_items) >= (m // 2 + 1)),
                latent_h=h,
            )
        )
    return bags


def seq_features(bags, t):
    return np.stack([bag.x_full[:t].reshape(-1) for bag in bags], axis=0)


def seq_features_permuted(bags, t, perm):
    return np.stack([bag.x_full[:t][perm].reshape(-1) for bag in bags], axis=0)


def obj_features(bags, t, item_model=None):
    feats = []
    for bag in bags:
        obs = bag.x_full[:t]
        parts = [obs.mean(axis=0), obs.std(axis=0)]
        if item_model is not None:
            prob = item_model.predict_proba(obs)[:, 1]
            score_summary = np.asarray(
                [
                    float(np.mean(prob)),
                    float(np.std(prob)),
                    float(np.min(prob)),
                    float(np.max(prob)),
                ],
                dtype=np.float64,
            )
            parts.append(score_summary)
        feats.append(np.concatenate(parts))
    return np.stack(feats, axis=0)


def bag_labels(bags):
    return np.asarray([bag.bag_label for bag in bags], dtype=np.int64)


def local_independence_probs(item_model, bags, m, t):
    threshold = m // 2 + 1
    out = []
    for bag in bags:
        obs = bag.x_full[:t]
        p_obs = item_model.predict_proba(obs)[:, 1]
        q_bar = float(np.mean(p_obs))
        probs = list(map(float, p_obs)) + [q_bar] * (m - t)
        out.append(poisson_binomial_tail(probs, threshold))
    return np.asarray(out, dtype=np.float64)


def remaining_threshold_labels(bags, t, threshold_count):
    return np.asarray(
        [int(np.sum(bag.z_full[t:]) >= threshold_count) for bag in bags], dtype=np.int64
    )


def remaining_positive_counts(bags, t):
    return np.asarray([int(np.sum(bag.z_full[t:])) for bag in bags], dtype=np.int64)


def local_completion_tail_probs(item_model, bags, t, hidden_count, threshold_count):
    out = []
    for bag in bags:
        obs = bag.x_full[:t]
        p_obs = item_model.predict_proba(obs)[:, 1]
        q_bar = float(np.mean(p_obs))
        out.append(poisson_binomial_tail([q_bar] * hidden_count, threshold_count))
    return np.asarray(out, dtype=np.float64)


def probability_metrics(y_true, prob):
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y_true = np.asarray(y_true, dtype=np.int64)
    return {
        "acc": float(np.mean((prob >= 0.5) == y_true)),
        "nll": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier": float(np.mean((prob - y_true) ** 2)),
        "ece": float(ece_score(y_true, prob)),
        "sampled_utility": float(np.mean(y_true * prob + (1 - y_true) * (1 - prob))),
    }


def subset_probability_metrics(y_true, prob, mask):
    mask = np.asarray(mask, dtype=bool)
    return {
        "count": int(np.sum(mask)),
        "prevalence": float(np.mean(np.asarray(y_true)[mask])),
        **probability_metrics(np.asarray(y_true)[mask], np.asarray(prob)[mask]),
    }


def posterior_event_metrics(exact_prob, pred_prob):
    exact_prob = np.clip(np.asarray(exact_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pred_prob = np.clip(np.asarray(pred_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    event_tv = np.abs(pred_prob - exact_prob)
    oracle_action = exact_prob >= 0.5
    pred_action = pred_prob >= 0.5
    disagreement = pred_action != oracle_action
    regret = np.abs(2.0 * exact_prob - 1.0) * disagreement.astype(np.float64)
    return {
        "event_tv": float(np.mean(event_tv)),
        "event_tv_p90": float(np.quantile(event_tv, 0.90)),
        "oracle_decision_disagreement": float(np.mean(disagreement)),
        "oracle_regret": float(np.mean(regret)),
    }


def subset_posterior_event_metrics(exact_prob, prob, mask):
    mask = np.asarray(mask, dtype=bool)
    return posterior_event_metrics(np.asarray(exact_prob)[mask], np.asarray(prob)[mask])


def posterior_alpha_from_prefix_count(s, k, q0, q1, prior=0.5):
    like0 = (1.0 - prior) * (q0 ** s) * ((1.0 - q0) ** (k - s))
    like1 = prior * (q1 ** s) * ((1.0 - q1) ** (k - s))
    return float(like1 / (like0 + like1))


def oracle_prefix_variance_from_bags(bags, k, q0, q1, prior=0.5):
    out = []
    for bag in bags:
        s = int(np.sum(bag.z_full[:k]))
        alpha = posterior_alpha_from_prefix_count(s, k, q0, q1, prior=prior)
        out.append(alpha * (1.0 - alpha) * ((q1 - q0) ** 2))
    return np.asarray(out, dtype=np.float64)


def binomial_tail(n, q, threshold):
    probs = [
        math.comb(n, r) * (q ** r) * ((1.0 - q) ** (n - r)) for r in range(n + 1)
    ]
    return float(np.sum(probs[int(threshold) :]))


def oracle_threshold_tail_probs_from_bags(
    bags, k, hidden_count, threshold_count, q0, q1, prior=0.5
):
    tail0 = binomial_tail(hidden_count, q0, threshold_count)
    tail1 = binomial_tail(hidden_count, q1, threshold_count)
    out = []
    for bag in bags:
        s = int(np.sum(bag.z_full[:k]))
        alpha = posterior_alpha_from_prefix_count(s, k, q0, q1, prior=prior)
        out.append((1.0 - alpha) * tail0 + alpha * tail1)
    return np.asarray(out, dtype=np.float64)


def oracle_independent_tail_probs_from_bags(
    bags, k, hidden_count, threshold_count, q0, q1, prior=0.5
):
    out = []
    for bag in bags:
        s = int(np.sum(bag.z_full[:k]))
        alpha = posterior_alpha_from_prefix_count(s, k, q0, q1, prior=prior)
        q_bar = (1.0 - alpha) * q0 + alpha * q1
        out.append(binomial_tail(hidden_count, q_bar, threshold_count))
    return np.asarray(out, dtype=np.float64)


def oracle_taxscore_from_bags(bags, k, hidden_count, threshold_count, q0, q1, prior=0.5):
    scores = []
    threshold_frac = threshold_count / hidden_count
    for bag in bags:
        s = int(np.sum(bag.z_full[:k]))
        alpha = posterior_alpha_from_prefix_count(s, k, q0, q1, prior=prior)
        q_bar = (1.0 - alpha) * q0 + alpha * q1
        v = alpha * (1.0 - alpha) * ((q1 - q0) ** 2)
        margin = abs(q_bar - threshold_frac) + 0.03
        scores.append(hidden_count * v / (margin * margin))
    return np.asarray(scores, dtype=np.float64)


def fit_item_model(X_train, y_train, seed=0, max_items=None):
    if max_items is not None and len(X_train) > max_items:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=max_items, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model


def fit_item_model_from_bags(bags, seed=0, max_items=None):
    X = np.concatenate([bag.x_full for bag in bags], axis=0)
    y = np.concatenate([bag.z_full for bag in bags], axis=0)
    return fit_item_model(X, y, seed=seed, max_items=max_items)


def fit_prefix_mlp(features, labels, seed, hidden_layer_sizes=(32, 16), max_iter=150):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            early_stopping=True,
            max_iter=max_iter,
            random_state=seed,
        ),
    )
    model.fit(features, labels)
    return model


def fit_raw_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def set_tensor_features(bags, t, raw_scaler):
    return np.stack(
        [raw_scaler.transform(bag.x_full[:t]).astype(np.float32) for bag in bags], axis=0
    )


def seq_tensor_features_permuted(bags, t, raw_scaler, perm):
    return np.stack(
        [raw_scaler.transform(bag.x_full[:t][perm]).astype(np.float32) for bag in bags], axis=0
    )


def seed_torch(seed):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for SetTransformer experiments.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if TORCH_AVAILABLE:
    class SetAttentionBlock(nn.Module):
        def __init__(self, hidden_dim, num_heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            return self.norm2(x + self.ff(x))


    class SetTransformerBinary(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_heads=4, depth=2, dropout=0.1):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.blocks = nn.ModuleList(
                [SetAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(depth)]
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pool = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            h = self.embed(x)
            for block in self.blocks:
                h = block(h)
            query = self.pool_query.expand(h.shape[0], -1, -1)
            pooled, _ = self.pool(query, h, h, need_weights=False)
            return self.head(pooled[:, 0, :]).squeeze(-1)


    class SequenceGRUBinary(nn.Module):
        def __init__(self, input_dim, hidden_dim=96, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)


    class SequenceTransformerBinary(nn.Module):
        def __init__(
            self,
            input_dim,
            max_len,
            hidden_dim=96,
            num_heads=4,
            depth=2,
            dropout=0.1,
        ):
            super().__init__()
            self.max_len = max_len
            self.embed = nn.Linear(input_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            h = self.embed(x)
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = h + self.pos_emb[:, : h.shape[1], :]
            h = self.encoder(h)
            return self.head(h[:, 0, :]).squeeze(-1)


    class SequenceGRUCount(nn.Module):
        def __init__(self, input_dim, n_counts, hidden_dim=96, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_counts),
            )

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])


    class SequenceTransformerCount(nn.Module):
        def __init__(
            self,
            input_dim,
            max_len,
            n_counts,
            hidden_dim=96,
            num_heads=4,
            depth=2,
            dropout=0.1,
        ):
            super().__init__()
            self.max_len = max_len
            self.embed = nn.Linear(input_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_counts),
            )

        def forward(self, x):
            h = self.embed(x)
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = h + self.pos_emb[:, : h.shape[1], :]
            h = self.encoder(h)
            return self.head(h[:, 0, :])


def fit_set_transformer(
    train_sets,
    train_labels,
    val_sets,
    val_labels,
    seed,
    hidden_dim=64,
    epochs=25,
    batch_size=256,
    learning_rate=1e-3,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for SetTransformer experiments.")
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetTransformerBinary(train_sets.shape[-1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(
        torch.from_numpy(train_sets.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val_sets.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_labels.astype(np.float32)).to(device)

    best_state = None
    best_val = None
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_set_transformer(model, sets, batch_size=512):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for SetTransformer experiments.")
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(sets), batch_size):
            xb = torch.from_numpy(sets[start : start + batch_size].astype(np.float32)).to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


def fit_gru_binary(
    train_seq,
    train_labels,
    val_seq,
    val_labels,
    seed,
    hidden_dim=96,
    epochs=25,
    batch_size=256,
    learning_rate=1e-3,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for GRU experiments.")
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceGRUBinary(train_seq.shape[-1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val_seq.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_labels.astype(np.float32)).to(device)

    best_state = None
    best_val = None
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_gru_binary(model, sequences, batch_size=512):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for GRU experiments.")
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.from_numpy(sequences[start : start + batch_size].astype(np.float32)).to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


def fit_transformer_binary(
    train_seq,
    train_labels,
    val_seq,
    val_labels,
    seed,
    hidden_dim=96,
    epochs=25,
    batch_size=256,
    learning_rate=1e-3,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for transformer experiments.")
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTransformerBinary(
        train_seq.shape[-1], max_len=train_seq.shape[1], hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val_seq.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_labels.astype(np.float32)).to(device)

    best_state = None
    best_val = None
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_transformer_binary(model, sequences, batch_size=512):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for transformer experiments.")
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.from_numpy(sequences[start : start + batch_size].astype(np.float32)).to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


def fit_gru_count(
    train_seq,
    train_counts,
    val_seq,
    val_counts,
    seed,
    n_counts,
    hidden_dim=96,
    epochs=25,
    batch_size=256,
    learning_rate=1e-3,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for GRU experiments.")
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceGRUCount(train_seq.shape[-1], n_counts=n_counts, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_counts.astype(np.int64)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val_seq.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_counts.astype(np.int64)).to(device)

    best_state = None
    best_val = None
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def fit_transformer_count(
    train_seq,
    train_counts,
    val_seq,
    val_counts,
    seed,
    n_counts,
    hidden_dim=96,
    epochs=25,
    batch_size=256,
    learning_rate=1e-3,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for transformer experiments.")
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTransformerCount(
        train_seq.shape[-1],
        max_len=train_seq.shape[1],
        n_counts=n_counts,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(
        torch.from_numpy(train_seq.astype(np.float32)),
        torch.from_numpy(train_counts.astype(np.int64)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val_seq.astype(np.float32)).to(device)
    val_y = torch.from_numpy(val_counts.astype(np.int64)).to(device)

    best_state = None
    best_val = None
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_gru_count_tail_probs(model, sequences, threshold_count, batch_size=512):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for GRU experiments.")
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.from_numpy(sequences[start : start + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)[:, threshold_count:].sum(dim=-1).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


def predict_transformer_count_tail_probs(model, sequences, threshold_count, batch_size=512):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for transformer experiments.")
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb = torch.from_numpy(sequences[start : start + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)[:, threshold_count:].sum(dim=-1).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


def rollout_training_data(bags, t, max_rows, seed):
    hidden_n = bags[0].x_full.shape[0] - t
    prefix = seq_features(bags, t)
    rows = []
    labels = []
    for idx, bag in enumerate(bags):
        pos = 0
        pref = prefix[idx]
        for step, z in enumerate(bag.z_full[t:]):
            rows.append(
                np.concatenate(
                    [
                        pref,
                        np.asarray(
                            [step / max(hidden_n, 1), pos / max(hidden_n, 1)],
                            dtype=np.float32,
                        ),
                    ]
                )
            )
            labels.append(int(z))
            pos += int(z)
    rows = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if max_rows is not None and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=max_rows, replace=False)
        rows = rows[idx]
        labels = labels[idx]
    return rows, labels


def fit_rollout_model(
    bags,
    t,
    seed,
    hidden_layer_sizes=(64, 32),
    max_iter=120,
    max_rows=200000,
):
    features, labels = rollout_training_data(bags, t, max_rows=max_rows, seed=seed)
    return fit_prefix_mlp(
        features,
        labels,
        seed=seed,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
    )


def rollout_tail_probs(rollout_model, bags, t, threshold_count, chunk_size=256):
    hidden_n = bags[0].x_full.shape[0] - t
    prefix = seq_features(bags, t).astype(np.float32)
    out = np.zeros(len(bags), dtype=np.float64)
    threshold_count = int(threshold_count)
    for start in range(0, len(bags), chunk_size):
        pref = prefix[start : start + chunk_size]
        dist = np.zeros((len(pref), hidden_n + 1), dtype=np.float64)
        dist[:, 0] = 1.0
        for step in range(hidden_n):
            counts = np.arange(step + 1, dtype=np.float32)
            repeated = np.repeat(pref, step + 1, axis=0)
            step_feat = np.full((len(pref) * (step + 1), 1), step / max(hidden_n, 1), dtype=np.float32)
            count_feat = (
                np.tile(counts[None, :], (len(pref), 1)).reshape(-1, 1) / max(hidden_n, 1)
            )
            feat = np.concatenate([repeated, step_feat, count_feat.astype(np.float32)], axis=1)
            p = rollout_model.predict_proba(feat)[:, 1].reshape(len(pref), step + 1)
            new_dist = np.zeros_like(dist)
            new_dist[:, : step + 1] += dist[:, : step + 1] * (1.0 - p)
            new_dist[:, 1 : step + 2] += dist[:, : step + 1] * p
            dist = new_dist
        out[start : start + len(pref)] = np.sum(dist[:, threshold_count:], axis=1)
    return out


def fit_threshold_predictors(
    train_bags,
    val_bags,
    k,
    threshold_count,
    seed,
    raw_scaler,
    item_model=None,
    item_model_max_items=100000,
):
    y_train_thr = remaining_threshold_labels(train_bags, k, threshold_count)
    y_val_thr = remaining_threshold_labels(val_bags, k, threshold_count)
    y_train_count = remaining_positive_counts(train_bags, k)
    y_val_count = remaining_positive_counts(val_bags, k)
    hidden_n = train_bags[0].x_full.shape[0] - k
    if item_model is None:
        item_model = fit_item_model_from_bags(
            train_bags, seed=seed + 2000, max_items=item_model_max_items
        )
    seq_model = fit_prefix_mlp(
        seq_features(train_bags, k),
        y_train_thr,
        seed + 1,
        hidden_layer_sizes=(64, 32),
        max_iter=120,
    )
    rollout_model = fit_rollout_model(
        train_bags,
        k,
        seed=seed + 2,
        hidden_layer_sizes=(64, 32),
        max_iter=120,
        max_rows=200000,
    )
    seq_count_model = fit_gru_count(
        set_tensor_features(train_bags, k, raw_scaler),
        y_train_count,
        set_tensor_features(val_bags, k, raw_scaler),
        y_val_count,
        seed=seed + 21,
        n_counts=hidden_n + 1,
        hidden_dim=96,
        epochs=20,
        batch_size=256,
    )
    seq_tx_count_model = fit_transformer_count(
        set_tensor_features(train_bags, k, raw_scaler),
        y_train_count,
        set_tensor_features(val_bags, k, raw_scaler),
        y_val_count,
        seed=seed + 22,
        n_counts=hidden_n + 1,
        hidden_dim=96,
        epochs=20,
        batch_size=256,
    )
    obj_model = fit_prefix_mlp(
        obj_features(train_bags, k, item_model),
        y_train_thr,
        seed + 3,
        hidden_layer_sizes=(64, 32),
        max_iter=120,
    )
    obj_set_model = fit_set_transformer(
        set_tensor_features(train_bags, k, raw_scaler),
        y_train_thr,
        set_tensor_features(val_bags, k, raw_scaler),
        y_val_thr,
        seed=seed + 4,
        hidden_dim=64,
        epochs=20,
        batch_size=256,
    )
    return {
        "item_model": item_model,
        "seq_model": seq_model,
        "rollout_model": rollout_model,
        "seq_count_model": seq_count_model,
        "seq_tx_count_model": seq_tx_count_model,
        "obj_model": obj_model,
        "obj_set_model": obj_set_model,
    }


def threshold_predictor_probs(models, bags, k, threshold_count, seed):
    hidden_n = bags[0].x_full.shape[0] - k
    raw_scaler = models["raw_scaler"]
    return {
        "IndepPB": local_completion_tail_probs(
            models["item_model"], bags, k, hidden_n, threshold_count
        ),
        "SeqMLP": models["seq_model"].predict_proba(seq_features(bags, k))[:, 1],
        "SeqPermAvg": permutation_ensemble_probs(models["seq_model"], bags, k, seed + 7000),
        "SeqGRUCount": permutation_gru_count_tail_probs(
            models["seq_count_model"],
            bags,
            k,
            raw_scaler,
            seed + 7100,
            threshold_count=threshold_count,
        ),
        "SeqTXCount": permutation_transformer_count_tail_probs(
            models["seq_tx_count_model"],
            bags,
            k,
            raw_scaler,
            seed + 7150,
            threshold_count=threshold_count,
        ),
        "ARRollout": rollout_tail_probs(models["rollout_model"], bags, k, threshold_count),
        "ObjMLP": models["obj_model"].predict_proba(obj_features(bags, k, models["item_model"]))[:, 1],
        "ObjSetTF": predict_set_transformer(
            models["obj_set_model"], set_tensor_features(bags, k, raw_scaler)
        ),
    }


def fit_active_threshold_model(model_name, train_bags, val_bags, k, threshold_count, seed, raw_scaler):
    y_train_thr = remaining_threshold_labels(train_bags, k, threshold_count)
    y_val_thr = remaining_threshold_labels(val_bags, k, threshold_count)
    y_train_count = remaining_positive_counts(train_bags, k)
    y_val_count = remaining_positive_counts(val_bags, k)
    hidden_n = train_bags[0].x_full.shape[0] - k
    if model_name == "IndepPB":
        return {
            "model_name": model_name,
            "item_model": fit_item_model_from_bags(
                train_bags, seed=seed + 10, max_items=100000
            ),
        }
    if model_name == "SeqPermAvg":
        return {
            "model_name": model_name,
            "seq_model": fit_prefix_mlp(
                seq_features(train_bags, k),
                y_train_thr,
                seed=seed + 11,
                hidden_layer_sizes=(64, 32),
                max_iter=120,
            ),
        }
    if model_name == "ARRollout":
        return {
            "model_name": model_name,
            "rollout_model": fit_rollout_model(
                train_bags,
                k,
                seed=seed + 12,
                hidden_layer_sizes=(64, 32),
                max_iter=120,
                max_rows=200000,
            ),
        }
    if model_name == "SeqGRUCount":
        return {
            "model_name": model_name,
            "raw_scaler": raw_scaler,
            "seq_count_model": fit_gru_count(
                set_tensor_features(train_bags, k, raw_scaler),
                y_train_count,
                set_tensor_features(val_bags, k, raw_scaler),
                y_val_count,
                seed=seed + 16,
                n_counts=hidden_n + 1,
                hidden_dim=96,
                epochs=20,
                batch_size=256,
            ),
        }
    if model_name == "SeqTXCount":
        return {
            "model_name": model_name,
            "raw_scaler": raw_scaler,
            "seq_tx_count_model": fit_transformer_count(
                set_tensor_features(train_bags, k, raw_scaler),
                y_train_count,
                set_tensor_features(val_bags, k, raw_scaler),
                y_val_count,
                seed=seed + 17,
                n_counts=hidden_n + 1,
                hidden_dim=96,
                epochs=20,
                batch_size=256,
            ),
        }
    if model_name == "ObjMLP":
        item_model = fit_item_model_from_bags(train_bags, seed=seed + 13, max_items=100000)
        return {
            "model_name": model_name,
            "item_model": item_model,
            "obj_model": fit_prefix_mlp(
                obj_features(train_bags, k, item_model),
                y_train_thr,
                seed=seed + 14,
                hidden_layer_sizes=(64, 32),
                max_iter=120,
            ),
        }
    if model_name == "ObjSetTF":
        return {
            "model_name": model_name,
            "obj_set_model": fit_set_transformer(
                set_tensor_features(train_bags, k, raw_scaler),
                y_train_thr,
                set_tensor_features(val_bags, k, raw_scaler),
                y_val_thr,
                seed=seed + 15,
                hidden_dim=64,
                epochs=20,
                batch_size=256,
            ),
            "raw_scaler": raw_scaler,
        }
    raise ValueError(model_name)


def active_threshold_probs(model_state, bags, k, threshold_count, seed):
    hidden_n = bags[0].x_full.shape[0] - k
    model_name = model_state["model_name"]
    if model_name == "IndepPB":
        return local_completion_tail_probs(
            model_state["item_model"], bags, k, hidden_n, threshold_count
        )
    if model_name == "SeqPermAvg":
        return permutation_ensemble_probs(model_state["seq_model"], bags, k, seed + 9000)
    if model_name == "ARRollout":
        return rollout_tail_probs(model_state["rollout_model"], bags, k, threshold_count)
    if model_name == "SeqGRUCount":
        return permutation_gru_count_tail_probs(
            model_state["seq_count_model"],
            bags,
            k,
            model_state["raw_scaler"],
            seed + 9100,
            threshold_count=threshold_count,
        )
    if model_name == "SeqTXCount":
        return permutation_transformer_count_tail_probs(
            model_state["seq_tx_count_model"],
            bags,
            k,
            model_state["raw_scaler"],
            seed + 9125,
            threshold_count=threshold_count,
        )
    if model_name == "ObjMLP":
        return model_state["obj_model"].predict_proba(
            obj_features(bags, k, model_state["item_model"])
        )[:, 1]
    if model_name == "ObjSetTF":
        return predict_set_transformer(
            model_state["obj_set_model"], set_tensor_features(bags, k, model_state["raw_scaler"])
        )
    raise ValueError(model_name)


def query_most_uncertain(prob, batch_size):
    order = np.argsort(np.abs(np.asarray(prob, dtype=np.float64) - 0.5))
    return order[:batch_size]


def permutation_list(t, seed, max_random=11):
    base = tuple(range(t))
    if t <= 1:
        return [np.asarray(base, dtype=np.int64)]
    if t <= 4:
        perms = [np.asarray(p, dtype=np.int64) for p in itertools.permutations(range(t))]
        return perms
    rng = np.random.default_rng(seed)
    perms = {base}
    while len(perms) < max_random + 1:
        perms.add(tuple(rng.permutation(t).tolist()))
    return [np.asarray(p, dtype=np.int64) for p in perms]


def permutation_ensemble_probs(seq_model, bags, t, seed):
    perms = permutation_list(t, seed)
    total = np.zeros(len(bags), dtype=np.float64)
    for perm in perms:
        total += seq_model.predict_proba(seq_features_permuted(bags, t, perm))[:, 1]
    return total / len(perms)


def permutation_gru_binary_probs(gru_model, bags, t, raw_scaler, seed):
    perms = permutation_list(t, seed)
    total = np.zeros(len(bags), dtype=np.float64)
    for perm in perms:
        total += predict_gru_binary(
            gru_model, seq_tensor_features_permuted(bags, t, raw_scaler, perm)
        )
    return total / len(perms)


def permutation_gru_count_tail_probs(gru_model, bags, t, raw_scaler, seed, threshold_count):
    perms = permutation_list(t, seed)
    total = np.zeros(len(bags), dtype=np.float64)
    for perm in perms:
        total += predict_gru_count_tail_probs(
            gru_model,
            seq_tensor_features_permuted(bags, t, raw_scaler, perm),
            threshold_count=threshold_count,
        )
    return total / len(perms)


def permutation_transformer_binary_probs(model, bags, t, raw_scaler, seed):
    perms = permutation_list(t, seed)
    total = np.zeros(len(bags), dtype=np.float64)
    for perm in perms:
        total += predict_transformer_binary(
            model, seq_tensor_features_permuted(bags, t, raw_scaler, perm)
        )
    return total / len(perms)


def permutation_transformer_count_tail_probs(model, bags, t, raw_scaler, seed, threshold_count):
    perms = permutation_list(t, seed)
    total = np.zeros(len(bags), dtype=np.float64)
    for perm in perms:
        total += predict_transformer_count_tail_probs(
            model,
            seq_tensor_features_permuted(bags, t, raw_scaler, perm),
            threshold_count=threshold_count,
        )
    return total / len(perms)


def load_musk_groups(min_items=8):
    return load_mil_csv_groups("musk2", min_items=min_items)


def split_group_names(groups, seed):
    names = sorted(groups.keys())
    labels = np.asarray([groups[name]["label"] for name in names], dtype=np.int64)
    train_names, tmp_names, train_labels, tmp_labels = train_test_split(
        names, labels, test_size=0.4, random_state=seed, stratify=labels
    )
    val_names, test_names = train_test_split(
        tmp_names, test_size=0.5, random_state=seed, stratify=tmp_labels
    )
    return list(train_names), list(val_names), list(test_names)


def sample_group_views(groups, group_names, m, views_per_group, seed):
    rng = np.random.default_rng(seed)
    bags = []
    for name in group_names:
        x_group = groups[name]["x"]
        label = groups[name]["label"]
        for _ in range(views_per_group):
            idx = rng.choice(len(x_group), size=m, replace=False)
            x_items = x_group[idx]
            perm = rng.permutation(m)
            x_items = x_items[perm]
            z_dummy = np.zeros(m, dtype=np.int8)
            bags.append(
                BagData(
                    x_obs=x_items,
                    z_obs=z_dummy,
                    x_full=x_items,
                    z_full=z_dummy,
                    bag_label=label,
                    latent_h=label,
                )
            )
    return bags


def run_grouped_stop_reveal(
    seed,
    dataset_name,
    min_items,
    m,
    train_views,
    val_views,
    test_views,
    cost,
    cost_grid,
    thresholds,
):
    groups, feature_dim = load_mil_csv_groups(dataset_name, min_items=min_items)
    train_names, val_names, test_names = split_group_names(groups, seed)

    train_bags = sample_group_views(groups, train_names, m, train_views, seed + 1)
    val_bags = sample_group_views(groups, val_names, m, val_views, seed + 2)
    test_bags = sample_group_views(groups, test_names, m, test_views, seed + 3)
    raw_scaler = fit_raw_scaler(np.concatenate([groups[name]["x"] for name in train_names], axis=0))

    y_train_bag = bag_labels(train_bags)
    y_val_bag = bag_labels(val_bags)
    y_test_bag = bag_labels(test_bags)

    seq_gru_models = {}
    seq_tx_models = {}
    obj_models = {}
    obj_set_models = {}
    for t in range(1, m + 1):
        train_seq = set_tensor_features(train_bags, t, raw_scaler)
        val_seq = set_tensor_features(val_bags, t, raw_scaler)
        seq_gru_models[t] = fit_gru_binary(
            train_seq,
            y_train_bag,
            val_seq,
            y_val_bag,
            seed=seed + 10 + t,
            hidden_dim=96,
            epochs=20,
            batch_size=128,
        )
        seq_tx_models[t] = fit_transformer_binary(
            train_seq,
            y_train_bag,
            val_seq,
            y_val_bag,
            seed=seed + 60 + t,
            hidden_dim=96,
            epochs=20,
            batch_size=128,
        )
        obj_models[t] = fit_prefix_mlp(
            obj_features(train_bags, t, None),
            y_train_bag,
            seed + 100 + t,
            hidden_layer_sizes=(64, 32),
            max_iter=160,
        )
        obj_set_models[t] = fit_set_transformer(
            train_seq,
            y_train_bag,
            val_seq,
            y_val_bag,
            seed=seed + 200 + t,
            hidden_dim=64,
            epochs=20,
            batch_size=128,
        )

    def model_prob(name, bags, t):
        if name == "SeqGRU":
            return permutation_gru_binary_probs(
                seq_gru_models[t], bags, t, raw_scaler, seed + 6000 + t
            )
        if name == "SeqTX":
            return permutation_transformer_binary_probs(
                seq_tx_models[t], bags, t, raw_scaler, seed + 6050 + t
            )
        if name == "ObjMLP":
            return obj_models[t].predict_proba(obj_features(bags, t, None))[:, 1]
        if name == "ObjSetTF":
            return predict_set_transformer(obj_set_models[t], set_tensor_features(bags, t, raw_scaler))
        raise ValueError(name)

    model_names = ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]
    prefix_metrics = {
        name: probability_metrics(y_test_bag, model_prob(name, test_bags, 1)) for name in model_names
    }

    val_prob_cache = {name: {} for name in model_names}
    test_prob_cache = {name: {} for name in model_names}
    for name in model_names:
        for t in range(1, m + 1):
            val_prob_cache[name][t] = model_prob(name, val_bags, t)
            test_prob_cache[name][t] = model_prob(name, test_bags, t)

    def evaluate_policy(prob_cache, labels, tau, eval_cost):
        utility = []
        reveals = []
        for i in range(len(labels)):
            for t in range(1, m + 1):
                prob = float(prob_cache[t][i])
                if max(prob, 1.0 - prob) >= tau or t == m:
                    pred = int(prob >= 0.5)
                    utility.append((1.0 if pred == labels[i] else 0.0) - eval_cost * (t - 1))
                    reveals.append(t)
                    break
        return float(np.mean(utility)), float(np.mean(reveals))

    policy_results = {}
    for name in model_names:
        best_tau = None
        best_utility = None
        for tau in thresholds:
            mean_utility, _ = evaluate_policy(val_prob_cache[name], y_val_bag, float(tau), cost)
            if best_utility is None or mean_utility > best_utility:
                best_utility = mean_utility
                best_tau = float(tau)
        mean_utility, mean_reveals = evaluate_policy(
            test_prob_cache[name], y_test_bag, best_tau, cost
        )
        policy_results[name] = {
            "utility": mean_utility,
            "avg_reveals": mean_reveals,
            "best_tau": best_tau,
        }

    policy_curve = {name: [] for name in model_names}
    for name in model_names:
        for eval_cost in cost_grid:
            best_tau = None
            best_utility = None
            for tau in thresholds:
                mean_utility, _ = evaluate_policy(
                    val_prob_cache[name], y_val_bag, float(tau), float(eval_cost)
                )
                if best_utility is None or mean_utility > best_utility:
                    best_utility = mean_utility
                    best_tau = float(tau)
            test_utility, test_reveals = evaluate_policy(
                test_prob_cache[name], y_test_bag, best_tau, float(eval_cost)
            )
            policy_curve[name].append(
                {
                    "cost": float(eval_cost),
                    "utility": test_utility,
                    "avg_reveals": test_reveals,
                    "best_tau": best_tau,
                }
            )

    return {
        "config": {
            "dataset": dataset_name,
            "m": m,
            "min_items": min_items,
            "feature_dim": feature_dim,
            "cost": cost,
            "cost_grid": cost_grid.tolist(),
            "threshold_grid": thresholds.tolist(),
            "n_groups_total": len(groups),
            "n_train_groups": len(train_names),
            "n_val_groups": len(val_names),
            "n_test_groups": len(test_names),
            "train_views_per_group": train_views,
            "val_views_per_group": val_views,
            "test_views_per_group": test_views,
        },
        "prefix_1_metrics": prefix_metrics,
        "policy_metrics": policy_results,
        "policy_curve": policy_curve,
    }


def run_musk_stop_reveal(seed):
    return run_grouped_stop_reveal(
        seed=seed,
        dataset_name="musk2",
        min_items=6,
        m=6,
        train_views=24,
        val_views=8,
        test_views=8,
        cost=0.04,
        cost_grid=np.arange(0.0, 0.101, 0.02),
        thresholds=np.arange(0.55, 0.91, 0.05),
    )


def run_protein_stop_reveal(seed):
    return run_grouped_stop_reveal(
        seed=seed,
        dataset_name="protein",
        min_items=16,
        m=16,
        train_views=16,
        val_views=8,
        test_views=8,
        cost=0.04,
        cost_grid=np.arange(0.0, 0.101, 0.02),
        thresholds=np.arange(0.55, 0.91, 0.05),
    )


def run_birds_stop_reveal(seed):
    return run_grouped_stop_reveal(
        seed=seed,
        dataset_name="birds_brown_creeper",
        min_items=16,
        m=16,
        train_views=4,
        val_views=2,
        test_views=2,
        cost=0.04,
        cost_grid=np.arange(0.0, 0.101, 0.02),
        thresholds=np.arange(0.55, 0.91, 0.05),
    )


def run_breast_benchmark(seed):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    m = 5
    k = 2
    q0 = 0.2
    q1 = 0.8

    train_bags = sample_hierarchical_bags(X_train, y_train, 5000, m, k, q0, q1, seed + 1)
    val_bags = sample_hierarchical_bags(X_val, y_val, 1200, m, k, q0, q1, seed + 2)
    test_bags = sample_hierarchical_bags(X_test, y_test, 1200, m, k, q0, q1, seed + 3)

    y_train_bag = bag_labels(train_bags)
    y_val_bag = bag_labels(val_bags)
    y_test_bag = bag_labels(test_bags)

    item_model = fit_item_model(X_train, y_train, seed=seed + 1000)
    seq_model = fit_prefix_mlp(seq_features(train_bags, k), y_train_bag, seed)
    obj_model = fit_prefix_mlp(
        obj_features(train_bags, k, item_model), y_train_bag, seed + 10
    )

    results = {}
    probs = {
        "IndepPB": local_independence_probs(item_model, test_bags, m, k),
        "SeqMLP": seq_model.predict_proba(seq_features(test_bags, k))[:, 1],
        "SeqPermAvg": permutation_ensemble_probs(seq_model, test_bags, k, seed + 5000),
        "ObjMLP": obj_model.predict_proba(obj_features(test_bags, k, item_model))[:, 1],
    }
    for name, prob in probs.items():
        results[name] = probability_metrics(y_test_bag, prob)

    return {
        "config": {
            "dataset": "breast_cancer",
            "m": m,
            "k": k,
            "q0": q0,
            "q1": q1,
            "n_train_bags": len(train_bags),
            "n_val_bags": len(val_bags),
            "n_test_bags": len(test_bags),
        },
        "validation_bag_prevalence": float(np.mean(y_val_bag)),
        "test_bag_prevalence": float(np.mean(y_test_bag)),
        "metrics": results,
    }


def run_digits_benchmark(seed):
    X, y = load_digits(return_X_y=True)
    y = (y >= 5).astype(np.int64)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    m = 7
    q0 = 0.2
    q1 = 0.8
    cost = 0.06
    cost_grid = np.arange(0.0, 0.121, 0.02)
    thresholds = np.arange(0.55, 0.91, 0.05)

    train_bags = sample_hierarchical_bags(X_train, y_train, 4000, m, m, q0, q1, seed + 1)
    val_bags = sample_hierarchical_bags(X_val, y_val, 1000, m, m, q0, q1, seed + 2)
    test_bags = sample_hierarchical_bags(X_test, y_test, 1000, m, m, q0, q1, seed + 3)

    y_train_bag = bag_labels(train_bags)
    y_val_bag = bag_labels(val_bags)
    y_test_bag = bag_labels(test_bags)

    item_model = fit_item_model(X_train, y_train, seed=seed + 1000)
    seq_models = {}
    obj_models = {}
    for t in range(1, m + 1):
        seq_models[t] = fit_prefix_mlp(seq_features(train_bags, t), y_train_bag, seed + t)
        obj_models[t] = fit_prefix_mlp(
            obj_features(train_bags, t, item_model), y_train_bag, seed + 100 + t
        )

    def model_prob(name, bags, t):
        if name == "IndepPB":
            return local_independence_probs(item_model, bags, m, t)
        if name == "SeqMLP":
            return seq_models[t].predict_proba(seq_features(bags, t))[:, 1]
        if name == "SeqPermAvg":
            return permutation_ensemble_probs(seq_models[t], bags, t, seed + 5000 + t)
        if name == "ObjMLP":
            return obj_models[t].predict_proba(obj_features(bags, t, item_model))[:, 1]
        raise ValueError(name)

    prefix_metrics = {}
    model_names = ["IndepPB", "SeqMLP", "SeqPermAvg", "ObjMLP"]
    for name in model_names:
        prefix_metrics[name] = probability_metrics(y_test_bag, model_prob(name, test_bags, 1))

    val_prob_cache = {name: {} for name in model_names}
    test_prob_cache = {name: {} for name in model_names}
    for name in model_names:
        for t in range(1, m + 1):
            val_prob_cache[name][t] = model_prob(name, val_bags, t)
            test_prob_cache[name][t] = model_prob(name, test_bags, t)

    def evaluate_policy(prob_cache, labels, tau, eval_cost):
        utility = []
        reveals = []
        for i in range(len(labels)):
            for t in range(1, m + 1):
                prob = float(prob_cache[t][i])
                if max(prob, 1.0 - prob) >= tau or t == m:
                    pred = int(prob >= 0.5)
                    utility.append((1.0 if pred == labels[i] else 0.0) - eval_cost * (t - 1))
                    reveals.append(t)
                    break
        return float(np.mean(utility)), float(np.mean(reveals))

    policy_results = {}
    for name in model_names:
        best_tau = None
        best_utility = None
        for tau in thresholds:
            mean_utility, _ = evaluate_policy(val_prob_cache[name], y_val_bag, float(tau), cost)
            if best_utility is None or mean_utility > best_utility:
                best_utility = mean_utility
                best_tau = float(tau)

        mean_utility, mean_reveals = evaluate_policy(
            test_prob_cache[name], y_test_bag, best_tau, cost
        )
        policy_results[name] = {
            "utility": mean_utility,
            "avg_reveals": mean_reveals,
            "best_tau": best_tau,
        }

    policy_curve = {name: [] for name in model_names}
    for name in model_names:
        for eval_cost in cost_grid:
            best_tau = None
            best_utility = None
            for tau in thresholds:
                mean_utility, _ = evaluate_policy(
                    val_prob_cache[name], y_val_bag, float(tau), float(eval_cost)
                )
                if best_utility is None or mean_utility > best_utility:
                    best_utility = mean_utility
                    best_tau = float(tau)
            test_utility, test_reveals = evaluate_policy(
                test_prob_cache[name], y_test_bag, best_tau, float(eval_cost)
            )
            policy_curve[name].append(
                {
                    "cost": float(eval_cost),
                    "utility": test_utility,
                    "avg_reveals": test_reveals,
                    "best_tau": best_tau,
                }
            )

    return {
        "config": {
            "dataset": "digits",
            "m": m,
            "q0": q0,
            "q1": q1,
            "cost": cost,
            "cost_grid": cost_grid.tolist(),
            "threshold_grid": thresholds.tolist(),
            "n_train_bags": len(train_bags),
            "n_val_bags": len(val_bags),
            "n_test_bags": len(test_bags),
        },
        "prefix_1_metrics": prefix_metrics,
        "policy_metrics": policy_results,
        "policy_curve": policy_curve,
    }


def run_covtype_benchmark(seed):
    X, y = fetch_covtype(return_X_y=True)
    y = (y == 2).astype(np.int64)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    m = 21
    k = 5
    q0 = 0.2
    q1 = 0.8

    train_bags = sample_hierarchical_bags(X_train, y_train, 12000, m, k, q0, q1, seed + 1)
    val_bags = sample_hierarchical_bags(X_val, y_val, 3000, m, k, q0, q1, seed + 2)
    test_bags = sample_hierarchical_bags(X_test, y_test, 3000, m, k, q0, q1, seed + 3)

    y_train_bag = bag_labels(train_bags)
    y_val_bag = bag_labels(val_bags)
    y_test_bag = bag_labels(test_bags)

    item_model = fit_item_model(X_train, y_train, seed=seed + 2000, max_items=100000)
    seq_model = fit_prefix_mlp(
        seq_features(train_bags, k),
        y_train_bag,
        seed,
        hidden_layer_sizes=(64, 32),
        max_iter=120,
    )
    obj_model = fit_prefix_mlp(
        obj_features(train_bags, k, item_model),
        y_train_bag,
        seed + 10,
        hidden_layer_sizes=(64, 32),
        max_iter=120,
    )

    probs = {
        "IndepPB": local_independence_probs(item_model, test_bags, m, k),
        "SeqMLP": seq_model.predict_proba(seq_features(test_bags, k))[:, 1],
        "SeqPermAvg": permutation_ensemble_probs(seq_model, test_bags, k, seed + 5000),
        "ObjMLP": obj_model.predict_proba(obj_features(test_bags, k, item_model))[:, 1],
    }
    results = {name: probability_metrics(y_test_bag, prob) for name, prob in probs.items()}

    return {
        "config": {
            "dataset": "covtype",
            "positive_class": "cover_type==2",
            "m": m,
            "k": k,
            "q0": q0,
            "q1": q1,
            "n_raw_train": int(len(X_train)),
            "n_raw_val": int(len(X_val)),
            "n_raw_test": int(len(X_test)),
            "n_train_bags": len(train_bags),
            "n_val_bags": len(val_bags),
            "n_test_bags": len(test_bags),
        },
        "validation_bag_prevalence": float(np.mean(y_val_bag)),
        "test_bag_prevalence": float(np.mean(y_test_bag)),
        "metrics": results,
    }


def run_covtype_threshold_benchmark(seed):
    X, y = fetch_covtype(return_X_y=True)
    y = (y == 2).astype(np.int64)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    k = 3
    q0 = 0.2
    q1 = 0.8
    threshold_fraction = 0.5
    hidden_sizes = [16, 48, 96]
    train_bags_n = 8000
    val_bags_n = 2000
    test_bags_n = 2000
    item_model = fit_item_model(X_train, y_train, seed=seed + 2000, max_items=100000)
    raw_scaler = fit_raw_scaler(X_train)
    rows = []
    ambiguity_rows = []

    for idx, hidden_n in enumerate(hidden_sizes):
        m = k + hidden_n
        threshold_count = int(np.ceil(threshold_fraction * hidden_n))
        train_bags = sample_hierarchical_bags(
            X_train, y_train, train_bags_n, m, k, q0, q1, seed + 10 * idx + 1
        )
        val_bags = sample_hierarchical_bags(
            X_val, y_val, val_bags_n, m, k, q0, q1, seed + 10 * idx + 2
        )
        test_bags = sample_hierarchical_bags(
            X_test, y_test, test_bags_n, m, k, q0, q1, seed + 10 * idx + 3
        )

        y_train_thr = remaining_threshold_labels(train_bags, k, threshold_count)
        y_test_thr = remaining_threshold_labels(test_bags, k, threshold_count)
        models = fit_threshold_predictors(
            train_bags,
            val_bags,
            k=k,
            threshold_count=threshold_count,
            seed=seed + 100 * idx,
            raw_scaler=raw_scaler,
            item_model=item_model,
            item_model_max_items=100000,
        )
        models["raw_scaler"] = raw_scaler
        probs = threshold_predictor_probs(
            models, test_bags, k=k, threshold_count=threshold_count, seed=seed + idx
        )
        oracle_v = oracle_prefix_variance_from_bags(test_bags, k=k, q0=q0, q1=q1)
        oracle_tail = oracle_threshold_tail_probs_from_bags(
            test_bags,
            k=k,
            hidden_count=hidden_n,
            threshold_count=threshold_count,
            q0=q0,
            q1=q1,
        )
        median_v = float(np.median(oracle_v))
        low_mask = oracle_v <= median_v
        high_mask = oracle_v > median_v
        if not np.any(high_mask):
            unique_v = np.unique(oracle_v)
            split_v = unique_v[-1]
            low_mask = oracle_v < split_v
            high_mask = oracle_v >= split_v
        row = {
            "hidden_n": hidden_n,
            "m": m,
            "threshold_fraction": threshold_fraction,
            "threshold_count": threshold_count,
            "label_prevalence": float(np.mean(y_test_thr)),
            "oracle_tail_mean": float(np.mean(oracle_tail)),
            "metrics": {
                name: {
                    **probability_metrics(y_test_thr, prob),
                    **posterior_event_metrics(oracle_tail, prob),
                }
                for name, prob in probs.items()
            },
        }
        rows.append(row)
        ambiguity_rows.append(
            {
                "hidden_n": hidden_n,
                "oracle_v_median": median_v,
                "low_v_mean": float(np.mean(oracle_v[low_mask])),
                "high_v_mean": float(np.mean(oracle_v[high_mask])),
                "low_nv_mean": float(hidden_n * np.mean(oracle_v[low_mask])),
                "high_nv_mean": float(hidden_n * np.mean(oracle_v[high_mask])),
                "low_n2v_mean": float(hidden_n * (hidden_n - 1) * np.mean(oracle_v[low_mask])),
                "high_n2v_mean": float(hidden_n * (hidden_n - 1) * np.mean(oracle_v[high_mask])),
                "slice_metrics": {
                    "low_v": {
                        name: {
                            **subset_probability_metrics(y_test_thr, prob, low_mask),
                            **subset_posterior_event_metrics(oracle_tail, prob, low_mask),
                        }
                        for name, prob in probs.items()
                    },
                    "high_v": {
                        name: {
                            **subset_probability_metrics(y_test_thr, prob, high_mask),
                            **subset_posterior_event_metrics(oracle_tail, prob, high_mask),
                        }
                        for name, prob in probs.items()
                    },
                },
            }
        )

    return {
        "config": {
            "dataset": "covtype_threshold",
            "positive_class": "cover_type==2",
            "k": k,
            "hidden_sizes": hidden_sizes,
            "threshold_fraction": threshold_fraction,
            "q0": q0,
            "q1": q1,
            "n_raw_train": int(len(X_train)),
            "n_raw_val": int(len(X_val)),
            "n_raw_test": int(len(X_test)),
            "n_train_bags": train_bags_n,
            "n_val_bags": val_bags_n,
            "n_test_bags": test_bags_n,
        },
        "rows": rows,
        "ambiguity_rows": ambiguity_rows,
    }


def run_batch_inspection_benchmark(seed):
    """Semi-synthetic quality-control batches built from real UCI item features."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    k = 4
    hidden_n = 60
    m = k + hidden_n
    q0 = 0.10
    q1 = 0.55
    threshold_fraction = 0.35
    threshold_count = int(np.ceil(threshold_fraction * hidden_n))
    train_bags = sample_hierarchical_bags(X_train, y_train, 4500, m, k, q0, q1, seed + 1)
    val_bags = sample_hierarchical_bags(X_val, y_val, 1200, m, k, q0, q1, seed + 2)
    test_bags = sample_hierarchical_bags(X_test, y_test, 1200, m, k, q0, q1, seed + 3)

    y_train_thr = remaining_threshold_labels(train_bags, k, threshold_count)
    y_val_thr = remaining_threshold_labels(val_bags, k, threshold_count)
    y_test_thr = remaining_threshold_labels(test_bags, k, threshold_count)
    item_model = fit_item_model(X_train, y_train, seed=seed + 700, max_items=None)
    obj_model = fit_prefix_mlp(
        obj_features(train_bags, k, item_model),
        y_train_thr,
        seed + 710,
        hidden_layer_sizes=(48, 24),
        max_iter=120,
    )
    seq_model = fit_prefix_mlp(
        seq_features(train_bags, k),
        y_train_thr,
        seed + 720,
        hidden_layer_sizes=(48, 24),
        max_iter=120,
    )

    oracle_shared = oracle_threshold_tail_probs_from_bags(
        test_bags, k, hidden_n, threshold_count, q0, q1
    )
    oracle_indep = oracle_independent_tail_probs_from_bags(
        test_bags, k, hidden_n, threshold_count, q0, q1
    )
    taxscore = oracle_taxscore_from_bags(test_bags, k, hidden_n, threshold_count, q0, q1)
    probs = {
        "OracleIndep": oracle_indep,
        "OracleShared": oracle_shared,
        "SeqMLP": seq_model.predict_proba(seq_features(test_bags, k))[:, 1],
        "ObjMLP": obj_model.predict_proba(obj_features(test_bags, k, item_model))[:, 1],
    }

    q = np.quantile(taxscore, [0.0, 1 / 3, 2 / 3, 1.0])
    bins = {}
    for idx, name in enumerate(["low", "mid", "high"]):
        if idx == 2:
            mask = taxscore >= q[idx]
        else:
            mask = (taxscore >= q[idx]) & (taxscore < q[idx + 1])
        bins[name] = {
            "count": int(np.sum(mask)),
            "mean_taxscore": float(np.mean(taxscore[mask])),
            "metrics": {
                model: {
                    **subset_probability_metrics(y_test_thr, prob, mask),
                    **subset_posterior_event_metrics(oracle_shared, prob, mask),
                }
                for model, prob in probs.items()
            },
        }

    return {
        "config": {
            "dataset": "breast_cancer_batch_inspection",
            "role": "semi-synthetic quality-control mechanism check",
            "k": k,
            "hidden_n": hidden_n,
            "m": m,
            "q0": q0,
            "q1": q1,
            "threshold_fraction": threshold_fraction,
            "threshold_count": threshold_count,
            "n_train_batches": len(train_bags),
            "n_val_batches": len(val_bags),
            "n_test_batches": len(test_bags),
        },
        "test_prevalence": float(np.mean(y_test_thr)),
        "overall_metrics": {
            model: {
                **probability_metrics(y_test_thr, prob),
                **posterior_event_metrics(oracle_shared, prob),
            }
            for model, prob in probs.items()
        },
        "taxscore_bins": bins,
    }


def run_covtype_active_learning(seed):
    X, y = fetch_covtype(return_X_y=True)
    y = (y == 2).astype(np.int64)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    k = 3
    hidden_n = 96
    m = k + hidden_n
    q0 = 0.2
    q1 = 0.8
    threshold_fraction = 0.5
    threshold_count = int(np.ceil(threshold_fraction * hidden_n))
    budget_schedule = [256, 512, 1024, 2048]
    pool_bags_n = 6144
    val_bags_n = 1000
    test_bags_n = 2000
    raw_scaler = fit_raw_scaler(X_train)

    pool_bags = sample_hierarchical_bags(X_train, y_train, pool_bags_n, m, k, q0, q1, seed + 1)
    val_bags = sample_hierarchical_bags(X_val, y_val, val_bags_n, m, k, q0, q1, seed + 2)
    test_bags = sample_hierarchical_bags(X_test, y_test, test_bags_n, m, k, q0, q1, seed + 3)
    y_test_thr = remaining_threshold_labels(test_bags, k, threshold_count)

    model_names = ["IndepPB", "SeqGRUCount", "SeqTXCount", "ARRollout", "ObjMLP", "ObjSetTF"]
    rng = np.random.default_rng(seed + 1000)
    base_perm = rng.permutation(pool_bags_n)
    curves = {name: [] for name in model_names}

    for model_idx, model_name in enumerate(model_names):
        selected = list(base_perm[: budget_schedule[0]])
        remaining = list(base_perm[budget_schedule[0] :])
        for round_idx, budget in enumerate(budget_schedule):
            labeled_bags = [pool_bags[i] for i in selected]
            model_state = fit_active_threshold_model(
                model_name,
                labeled_bags,
                val_bags,
                k=k,
                threshold_count=threshold_count,
                seed=seed + 200 * model_idx + 20 * round_idx,
                raw_scaler=raw_scaler,
            )
            prob_test = active_threshold_probs(
                model_state, test_bags, k=k, threshold_count=threshold_count, seed=seed + round_idx
            )
            entry = {"budget_bags": budget}
            entry.update(probability_metrics(y_test_thr, prob_test))
            curves[model_name].append(entry)

            if round_idx == len(budget_schedule) - 1:
                continue
            query_batch = budget_schedule[round_idx + 1] - budget_schedule[round_idx]
            remaining_bags = [pool_bags[i] for i in remaining]
            prob_pool = active_threshold_probs(
                model_state,
                remaining_bags,
                k=k,
                threshold_count=threshold_count,
                seed=seed + 500 + round_idx,
            )
            query_local = query_most_uncertain(prob_pool, query_batch)
            query_global = [remaining[i] for i in query_local]
            query_set = set(query_global)
            selected.extend(query_global)
            remaining = [idx for idx in remaining if idx not in query_set]

    return {
        "config": {
            "dataset": "covtype_active_learning",
            "positive_class": "cover_type==2",
            "k": k,
            "hidden_n": hidden_n,
            "m": m,
            "threshold_fraction": threshold_fraction,
            "threshold_count": threshold_count,
            "q0": q0,
            "q1": q1,
            "budget_schedule": budget_schedule,
            "pool_bags": pool_bags_n,
            "val_bags": val_bags_n,
            "test_bags": test_bags_n,
        },
        "curves": curves,
    }


def run_synthetic_scaling():
    from math import comb

    q0 = 0.2
    q1 = 0.8
    alpha = 0.2
    q_bar = alpha * q1 + (1.0 - alpha) * q0
    v = alpha * (1.0 - alpha) * (q1 - q0) ** 2

    def dist_from_alpha(n, alpha_val):
        exact_dist = []
        local_dist = []
        q_bar_val = alpha_val * q1 + (1.0 - alpha_val) * q0
        for r in range(n + 1):
            exact_dist.append(
                comb(n, r)
                * (
                    alpha_val * (q1**r) * ((1.0 - q1) ** (n - r))
                    + (1.0 - alpha_val) * (q0**r) * ((1.0 - q0) ** (n - r))
                )
            )
            local_dist.append(comb(n, r) * (q_bar_val**r) * ((1.0 - q_bar_val) ** (n - r)))
        return np.asarray(exact_dist, dtype=np.float64), np.asarray(local_dist, dtype=np.float64), q_bar_val

    n_rows = []
    for n, c in [(6, 4 / 6), (10, 7 / 10), (20, 13 / 20), (40, 25 / 40)]:
        threshold = int(np.ceil(c * n))
        exact_dist, local_dist, _ = dist_from_alpha(n, alpha)
        exact = float(np.sum(exact_dist[threshold:]))
        local = float(np.sum(local_dist[threshold:]))
        var_gap = float(n * (n - 1) * v)
        tv_exact = float(0.5 * np.sum(np.abs(exact_dist - local_dist)))
        n_rows.append(
            {
                "n": n,
                "threshold_fraction": c,
                "exact": exact,
                "local": local,
                "gap": exact - local,
                "tv_exact": tv_exact,
                "posterior_variance": v,
                "tv_lower_bound": v,
                "variance_gap": var_gap,
                "variance_gap_lower_bound": var_gap,
            }
        )

    alpha_rows = []
    n_alpha = 40
    c_alpha = 0.5
    threshold_alpha = int(np.ceil(c_alpha * n_alpha))
    for alpha_val in np.linspace(0.05, 0.45, 9):
        v_val = alpha_val * (1.0 - alpha_val) * (q1 - q0) ** 2
        exact_dist, local_dist, q_bar_val = dist_from_alpha(n_alpha, alpha_val)
        exact = float(np.sum(exact_dist[threshold_alpha:]))
        local = float(np.sum(local_dist[threshold_alpha:]))
        var_gap = float(n_alpha * (n_alpha - 1) * v_val)
        tv_exact = float(0.5 * np.sum(np.abs(exact_dist - local_dist)))
        alpha_rows.append(
            {
                "alpha": float(alpha_val),
                "q_bar": float(q_bar_val),
                "posterior_variance": float(v_val),
                "exact": float(exact),
                "local": float(local),
                "gap": float(exact - local),
                "tv_exact": tv_exact,
                "tv_lower_bound": float(v_val),
                "variance_gap": var_gap,
                "variance_gap_lower_bound": var_gap,
            }
        )

    return {
        "config": {"alpha": alpha, "q0": q0, "q1": q1, "q_bar": q_bar, "v": v},
        "n_sweep": {
            "config": {"alpha": alpha, "q0": q0, "q1": q1, "q_bar": q_bar, "v": v},
            "rows": n_rows,
        },
        "alpha_sweep": {
            "config": {"n": n_alpha, "c": c_alpha, "q0": q0, "q1": q1},
            "rows": alpha_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=[
            "synthetic",
            "breast",
            "digits",
            "covtype",
            "covtype_threshold",
            "covtype_active_learning",
            "batch_inspection",
            "musk_stop_reveal",
            "birds_stop_reveal",
            "protein_stop_reveal",
        ],
        choices=[
            "synthetic",
            "breast",
            "digits",
            "covtype",
            "covtype_threshold",
            "covtype_active_learning",
            "batch_inspection",
            "musk_stop_reveal",
            "birds_stop_reveal",
            "protein_stop_reveal",
        ],
        help="Subset of benchmarks to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where JSON outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if "synthetic" in args.benchmarks:
        results["synthetic_scaling"] = run_synthetic_scaling()
    if "breast" in args.benchmarks:
        results["breast_benchmark"] = run_breast_benchmark(args.seed)
    if "digits" in args.benchmarks:
        results["digits_benchmark"] = run_digits_benchmark(args.seed)
    if "covtype" in args.benchmarks:
        results["covtype_benchmark"] = run_covtype_benchmark(args.seed)
    if "covtype_threshold" in args.benchmarks:
        results["covtype_threshold_benchmark"] = run_covtype_threshold_benchmark(args.seed)
    if "covtype_active_learning" in args.benchmarks:
        results["covtype_active_learning"] = run_covtype_active_learning(args.seed)
    if "batch_inspection" in args.benchmarks:
        results["batch_inspection_benchmark"] = run_batch_inspection_benchmark(args.seed)
    if "musk_stop_reveal" in args.benchmarks:
        results["musk_stop_reveal"] = run_musk_stop_reveal(args.seed)
    if "birds_stop_reveal" in args.benchmarks:
        results["birds_stop_reveal"] = run_birds_stop_reveal(args.seed)
    if "protein_stop_reveal" in args.benchmarks:
        results["protein_stop_reveal"] = run_protein_stop_reveal(args.seed)

    output_path = output_dir / "benchmark_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
