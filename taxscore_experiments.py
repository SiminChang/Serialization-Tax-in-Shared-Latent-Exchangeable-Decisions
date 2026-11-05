import argparse
import json
import math
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
from sklearn.datasets import fetch_covtype

from run_benchmarks import (
    BagData,
    binomial_tail,
    ece_score,
    fit_gru_binary,
    fit_prefix_mlp,
    fit_raw_scaler,
    fit_set_transformer,
    fit_transformer_binary,
    load_mil_csv_groups,
    obj_features,
    oracle_prefix_variance_from_bags,
    oracle_threshold_tail_probs_from_bags,
    permutation_gru_binary_probs,
    permutation_transformer_binary_probs,
    posterior_alpha_from_prefix_count,
    predict_set_transformer,
    sample_group_views,
    sample_hierarchical_bags,
    set_tensor_features,
    split_group_names,
)


MODEL_ORDER = ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def aggregate_metric_dict(records):
    keys = records[0].keys()
    return {key: mean_std([record[key] for record in records]) for key in keys}


def beta_binomial_tail_grid(n, threshold, mean, var, grid_size=4000):
    max_var = mean * (1.0 - mean)
    if var <= 1e-10 or var >= max_var - 1e-10:
        return binomial_tail(n, mean, threshold)
    common = max_var / var - 1.0
    a = mean * common
    b = (1.0 - mean) * common
    q = np.linspace(1e-5, 1.0 - 1e-5, grid_size)
    log_pdf = (
        (a - 1.0) * np.log(q)
        + (b - 1.0) * np.log1p(-q)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    pdf = np.exp(log_pdf - np.max(log_pdf))
    tails = np.asarray([binomial_tail(n, float(x), threshold) for x in q], dtype=np.float64)
    return float(np.trapz(tails * pdf, q) / np.trapz(pdf, q))


def run_head_resolution():
    q0, q1, alpha, c, n = 0.43, 0.61, 0.35, 0.5, 50
    threshold = int(np.ceil(c * n))
    exact = (1.0 - alpha) * binomial_tail(n, q0, threshold) + alpha * binomial_tail(
        n, q1, threshold
    )
    qbar = (1.0 - alpha) * q0 + alpha * q1
    var = alpha * (1.0 - alpha) * (q1 - q0) ** 2
    mean_completion = binomial_tail(n, qbar, threshold)
    beta_head = beta_binomial_tail_grid(n, threshold, qbar, var)

    rows = []
    for k in [2, 4, 8, 16, 32, 64]:
        grid = np.linspace(0.0, 1.0, k)
        p0 = float(grid[np.argmin(np.abs(grid - q0))])
        p1 = float(grid[np.argmin(np.abs(grid - q1))])
        grid_tail = (1.0 - alpha) * binomial_tail(n, p0, threshold) + alpha * binomial_tail(
            n, p1, threshold
        )
        rows.append(
            {
                "K": k,
                "mesh": float(1.0 / (k - 1)),
                "grid_error": abs(grid_tail - exact),
                "mean_completion_error": abs(mean_completion - exact),
                "beta_moment_error": abs(beta_head - exact),
            }
        )
    return {
        "config": {"q0": q0, "q1": q1, "alpha": alpha, "c": c, "n": n},
        "exact_tail": exact,
        "rows": rows,
    }


def run_controlled_taxscore(seed):
    x, y = fetch_covtype(return_X_y=True)
    y = (y == 2).astype(np.int64)
    k, q0, q1, c = 3, 0.2, 0.8, 0.5
    rows = []
    all_entries = []
    for idx, hidden_n in enumerate([16, 48, 96]):
        m = k + hidden_n
        threshold_count = int(np.ceil(c * hidden_n))
        bags = sample_hierarchical_bags(
            x, y, n_bags=5000, m=m, k=k, q0=q0, q1=q1, seed=seed + 1000 * idx + 11
        )
        exact_tail = oracle_threshold_tail_probs_from_bags(
            bags, k, hidden_n, threshold_count, q0, q1
        )
        oracle_v = oracle_prefix_variance_from_bags(bags, k, q0, q1)
        local_tail = []
        tax_scores = []
        for bag, v in zip(bags, oracle_v):
            s = int(np.sum(bag.z_full[:k]))
            alpha = posterior_alpha_from_prefix_count(s, k, q0, q1)
            qbar = (1.0 - alpha) * q0 + alpha * q1
            local_tail.append(binomial_tail(hidden_n, qbar, threshold_count))
            margin = abs(qbar - c) + 0.03
            tax_scores.append(hidden_n * float(v) / (margin * margin))
        local_tail = np.asarray(local_tail, dtype=np.float64)
        tax_scores = np.asarray(tax_scores, dtype=np.float64)
        event_tv = np.abs(local_tail - exact_tail)
        rows.append(
            {
                "hidden_n": hidden_n,
                "mean_taxscore": float(np.mean(tax_scores)),
                "mean_event_tv": float(np.mean(event_tv)),
                "p90_event_tv": float(np.quantile(event_tv, 0.90)),
                "mean_nv": float(hidden_n * np.mean(oracle_v)),
            }
        )
        for score, tv, v in zip(tax_scores, event_tv, oracle_v):
            all_entries.append((float(score), float(tv), float(hidden_n * v)))

    entries = np.asarray(all_entries, dtype=np.float64)
    quantiles = np.quantile(entries[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    bins = []
    for b in range(3):
        if b == 2:
            mask = entries[:, 0] >= quantiles[b]
        else:
            mask = (entries[:, 0] >= quantiles[b]) & (entries[:, 0] < quantiles[b + 1])
        bins.append(
            {
                "bin": b,
                "count": int(np.sum(mask)),
                "mean_taxscore": float(np.mean(entries[mask, 0])),
                "mean_event_tv": float(np.mean(entries[mask, 1])),
                "p90_event_tv": float(np.quantile(entries[mask, 1], 0.90)),
                "mean_nv": float(np.mean(entries[mask, 2])),
            }
        )
    return {"config": {"k": k, "q0": q0, "q1": q1, "c": c}, "rows": rows, "bins": bins}


def triage_utility(labels, prob, tau, reveal_cost):
    labels = np.asarray(labels, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    confident = np.maximum(prob, 1.0 - prob) >= tau
    pred = prob >= 0.5
    classify_utility = (pred == labels).astype(np.float64)
    utility = np.where(confident, classify_utility, 1.0 - reveal_cost)
    return float(np.mean(utility)), float(np.mean(~confident))


def binary_probability_metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    nll = -np.mean(y_true * np.log(prob) + (1 - y_true) * np.log1p(-prob))
    return {
        "acc": float(np.mean((prob >= 0.5) == y_true)),
        "nll": float(nll),
        "brier": float(np.mean((prob - y_true) ** 2)),
        "ece": float(ece_score(y_true, prob)),
        "sampled_utility": float(np.mean(y_true * prob + (1 - y_true) * (1 - prob))),
    }


def choose_tau(labels, prob, reveal_cost, grid):
    best_tau, best_utility = None, -np.inf
    for tau in grid:
        utility, _ = triage_utility(labels, prob, float(tau), reveal_cost)
        if utility > best_utility:
            best_tau, best_utility = float(tau), utility
    return best_tau


def load_movielens_tail_groups(min_items=30, cache_dir=".data_cache/movielens"):
    """Natural user-level slate task from MovieLens 100K."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-100k.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_URL, zip_path)
    extract_dir = cache_dir / "ml-100k"
    if not (extract_dir / "u.data").exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(cache_dir)

    genres = {}
    with (extract_dir / "u.item").open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            movie_id = int(parts[0])
            genres[movie_id] = np.asarray([float(x) for x in parts[-19:]], dtype=np.float32)

    by_user = {}
    with (extract_dir / "u.data").open("r", encoding="latin-1") as handle:
        for line in handle:
            user_id, movie_id, rating, _ = line.strip().split("\t")
            movie_id = int(movie_id)
            rating = float(rating)
            positive = float(rating >= 4.0)
            rating_scaled = (rating - 1.0) / 4.0
            feat = np.concatenate(
                [genres[movie_id], np.asarray([rating_scaled, positive], dtype=np.float32)]
            )
            by_user.setdefault(user_id, {"x": [], "z": []})
            by_user[user_id]["x"].append(feat.astype(np.float32))
            by_user[user_id]["z"].append(int(positive))

    groups = {}
    for user_id, payload in by_user.items():
        if len(payload["x"]) < min_items:
            continue
        x = np.asarray(payload["x"], dtype=np.float32)
        z = np.asarray(payload["z"], dtype=np.int8)
        groups[f"ml_user_{user_id}"] = {
            "x": x,
            "z": z,
            "label": int(np.mean(z) >= 0.55),
        }
    feature_dim = next(iter(groups.values()))["x"].shape[1]
    return groups, feature_dim


def sample_tail_views(groups, group_names, m, t_obs, views_per_group, seed):
    rng = np.random.default_rng(seed)
    bags = []
    hidden_n = m - t_obs
    threshold = int(np.ceil(0.5 * hidden_n))
    for name in group_names:
        x_group = groups[name]["x"]
        z_group = groups[name]["z"]
        for _ in range(views_per_group):
            idx = rng.choice(len(x_group), size=m, replace=False)
            perm = rng.permutation(m)
            x_items = x_group[idx][perm]
            z_items = z_group[idx][perm]
            label = int(np.sum(z_items[t_obs:]) >= threshold)
            bags.append(
                BagData(
                    x_obs=x_items[:t_obs],
                    z_obs=z_items[:t_obs],
                    x_full=x_items,
                    z_full=z_items,
                    bag_label=label,
                    latent_h=label,
                )
            )
    return bags


def model_bin_metrics(labels, prob, tau, reveal_cost, mask):
    labels = np.asarray(labels, dtype=np.int64)[mask]
    prob = np.asarray(prob, dtype=np.float64)[mask]
    utility, defer_rate = triage_utility(labels, prob, tau, reveal_cost)
    out = binary_probability_metrics(labels, prob)
    out.update({"utility": utility, "defer_rate": defer_rate})
    return out


def run_natural_taxscore(seed, datasets):
    out = {}
    tau_grid = np.arange(0.55, 0.96, 0.05)
    reveal_cost = 0.30
    for dataset in datasets:
        if dataset == "movielens_tail":
            groups, feature_dim = load_movielens_tail_groups(min_items=30)
        else:
            groups, feature_dim = load_mil_csv_groups(dataset, min_items=8)
        train_names, val_names, test_names = split_group_names(groups, seed)
        if dataset == "movielens_tail":
            t_obs = 3
            m = min(18, min(len(groups[name]["x"]) for name in train_names + val_names + test_names))
            train_bags = sample_tail_views(groups, train_names, m, t_obs, 4, seed + 1)
            val_bags = sample_tail_views(groups, val_names, m, t_obs, 2, seed + 2)
            test_bags = sample_tail_views(groups, test_names, m, t_obs, 2, seed + 3)
        else:
            t_obs = 1
            m = min(12, min(len(groups[name]["x"]) for name in train_names + val_names + test_names))
            train_bags = sample_group_views(groups, train_names, m, 8, seed + 1)
            val_bags = sample_group_views(groups, val_names, m, 4, seed + 2)
            test_bags = sample_group_views(groups, test_names, m, 4, seed + 3)
        raw_scaler = fit_raw_scaler(np.concatenate([groups[name]["x"] for name in train_names], axis=0))
        y_train = np.asarray([bag.bag_label for bag in train_bags], dtype=np.int64)
        y_val = np.asarray([bag.bag_label for bag in val_bags], dtype=np.int64)
        y_test = np.asarray([bag.bag_label for bag in test_bags], dtype=np.int64)
        train_seq = set_tensor_features(train_bags, t_obs, raw_scaler)
        val_seq = set_tensor_features(val_bags, t_obs, raw_scaler)
        test_seq = set_tensor_features(test_bags, t_obs, raw_scaler)

        seq_gru = fit_gru_binary(train_seq, y_train, val_seq, y_val, seed + 10, hidden_dim=64, epochs=8, batch_size=128)
        seq_tx = fit_transformer_binary(train_seq, y_train, val_seq, y_val, seed + 20, hidden_dim=64, epochs=8, batch_size=128)
        obj_mlp = fit_prefix_mlp(obj_features(train_bags, t_obs, None), y_train, seed + 30, hidden_layer_sizes=(64, 32), max_iter=120)
        obj_set = fit_set_transformer(train_seq, y_train, val_seq, y_val, seed + 40, hidden_dim=48, epochs=8, batch_size=128)

        val_probs = {
            "SeqGRU": permutation_gru_binary_probs(seq_gru, val_bags, t_obs, raw_scaler, seed + 100),
            "SeqTX": permutation_transformer_binary_probs(seq_tx, val_bags, t_obs, raw_scaler, seed + 200),
            "ObjMLP": obj_mlp.predict_proba(obj_features(val_bags, t_obs, None))[:, 1],
            "ObjSetTF": predict_set_transformer(obj_set, val_seq),
        }
        test_probs = {
            "SeqGRU": permutation_gru_binary_probs(seq_gru, test_bags, t_obs, raw_scaler, seed + 300),
            "SeqTX": permutation_transformer_binary_probs(seq_tx, test_bags, t_obs, raw_scaler, seed + 400),
            "ObjMLP": obj_mlp.predict_proba(obj_features(test_bags, t_obs, None))[:, 1],
            "ObjSetTF": predict_set_transformer(obj_set, test_seq),
        }
        best_tau = {
            name: choose_tau(y_val, prob, reveal_cost, tau_grid) for name, prob in val_probs.items()
        }
        ref = np.clip(test_probs["ObjSetTF"], 1e-4, 1.0 - 1e-4)
        margin = np.abs(ref - 0.5) + 0.05
        tax_proxy = (m - t_obs) * ref * (1.0 - ref) / (margin * margin)
        q = np.quantile(tax_proxy, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        bins = {}
        for b, name_bin in enumerate(["low", "mid", "high"]):
            if b == 2:
                mask = tax_proxy >= q[b]
            else:
                mask = (tax_proxy >= q[b]) & (tax_proxy < q[b + 1])
            bins[name_bin] = {
                "count": int(np.sum(mask)),
                "mean_taxscore": float(np.mean(tax_proxy[mask])),
                "metrics": {
                    model: model_bin_metrics(y_test, prob, best_tau[model], reveal_cost, mask)
                    for model, prob in test_probs.items()
                },
            }
        overall = {
            model: model_bin_metrics(
                y_test, prob, best_tau[model], reveal_cost, np.ones(len(y_test), dtype=bool)
            )
            for model, prob in test_probs.items()
        }
        out[dataset] = {
            "config": {
                "m": m,
                "t_obs": t_obs,
                "feature_dim": feature_dim,
                "n_train": len(train_bags),
                "n_val": len(val_bags),
                "n_test": len(test_bags),
                "reveal_cost": reveal_cost,
            },
            "overall": overall,
            "bins": bins,
        }
    return out


def aggregate_seed_payloads(payloads):
    result = {
        "posterior_head_resolution": payloads[0]["posterior_head_resolution"],
        "controlled_taxscore": {"rows": [], "bins": []},
        "natural_taxscore": {},
    }
    for idx in range(len(payloads[0]["controlled_taxscore"]["rows"])):
        rows = [p["controlled_taxscore"]["rows"][idx] for p in payloads]
        result["controlled_taxscore"]["rows"].append(aggregate_metric_dict(rows))
    for idx in range(len(payloads[0]["controlled_taxscore"]["bins"])):
        bins = [p["controlled_taxscore"]["bins"][idx] for p in payloads]
        entry = aggregate_metric_dict(bins)
        entry["bin"] = payloads[0]["controlled_taxscore"]["bins"][idx]["bin"]
        result["controlled_taxscore"]["bins"].append(entry)

    datasets = payloads[0]["natural_taxscore"].keys()
    for dataset in datasets:
        result["natural_taxscore"][dataset] = {
            "config": payloads[0]["natural_taxscore"][dataset]["config"],
            "overall": {},
            "bins": {},
        }
        for model in MODEL_ORDER:
            records = [p["natural_taxscore"][dataset]["overall"][model] for p in payloads]
            result["natural_taxscore"][dataset]["overall"][model] = aggregate_metric_dict(records)
        for bin_name in ["low", "mid", "high"]:
            bin_records = [p["natural_taxscore"][dataset]["bins"][bin_name] for p in payloads]
            result["natural_taxscore"][dataset]["bins"][bin_name] = {
                "count": mean_std([r["count"] for r in bin_records]),
                "mean_taxscore": mean_std([r["mean_taxscore"] for r in bin_records]),
                "metrics": {},
            }
            for model in MODEL_ORDER:
                records = [r["metrics"][model] for r in bin_records]
                result["natural_taxscore"][dataset]["bins"][bin_name]["metrics"][model] = (
                    aggregate_metric_dict(records)
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=str, default="taxscore_results.json")
    parser.add_argument("--aggregate", type=str, nargs="*")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "musk2",
            "birds_brown_creeper",
            "protein",
            "web_recommendation_1",
            "ucsb_breast_cancer",
            "movielens_tail",
        ],
    )
    args = parser.parse_args()

    if args.aggregate:
        payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.aggregate]
        result = aggregate_seed_payloads(payloads)
    else:
        if args.seed is None:
            raise ValueError("--seed is required unless --aggregate is provided.")
        result = {
            "seed": args.seed,
            "posterior_head_resolution": run_head_resolution(),
            "controlled_taxscore": run_controlled_taxscore(args.seed),
            "natural_taxscore": run_natural_taxscore(args.seed, args.datasets),
        }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
