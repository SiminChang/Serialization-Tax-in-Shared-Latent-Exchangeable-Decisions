import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype

from run_benchmarks import (
    binomial_tail,
    oracle_prefix_variance_from_bags,
    oracle_threshold_tail_probs_from_bags,
    posterior_alpha_from_prefix_count,
    posterior_event_metrics,
    sample_hierarchical_bags,
)


def mean_std(records, key):
    vals = np.asarray([record[key] for record in records], dtype=np.float64)
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}


def run_seed(seed):
    X, y = fetch_covtype(return_X_y=True)
    y = (y == 2).astype(np.int64)
    k = 3
    q0 = 0.2
    q1 = 0.8
    hidden_sizes = [16, 48, 96]
    n_bags = 5000
    rows = []
    for idx, hidden_n in enumerate(hidden_sizes):
        m = k + hidden_n
        threshold_count = int(np.ceil(0.5 * hidden_n))
        bags = sample_hierarchical_bags(
            X, y, n_bags, m, k, q0, q1, seed + 100 * idx + 17
        )
        exact_tail = oracle_threshold_tail_probs_from_bags(
            bags, k, hidden_n, threshold_count, q0, q1
        )
        local_tail = []
        for bag in bags:
            s = int(np.sum(bag.z_full[:k]))
            alpha = posterior_alpha_from_prefix_count(s, k, q0, q1)
            q_bar = (1.0 - alpha) * q0 + alpha * q1
            local_tail.append(binomial_tail(hidden_n, q_bar, threshold_count))
        local_tail = np.asarray(local_tail, dtype=np.float64)
        oracle_v = oracle_prefix_variance_from_bags(bags, k, q0, q1)
        metrics = posterior_event_metrics(exact_tail, local_tail)
        metrics.update(
            {
                "hidden_n": hidden_n,
                "threshold_count": threshold_count,
                "mean_v": float(np.mean(oracle_v)),
                "mean_nv": float(hidden_n * np.mean(oracle_v)),
                "mean_n2v": float(hidden_n * (hidden_n - 1) * np.mean(oracle_v)),
                "exact_tail_mean": float(np.mean(exact_tail)),
                "local_tail_mean": float(np.mean(local_tail)),
            }
        )
        rows.append(metrics)
    return rows


def aggregate(seed_rows):
    rows = []
    for idx in range(len(seed_rows[0])):
        exemplar = seed_rows[0][idx]
        entry = {
            "hidden_n": exemplar["hidden_n"],
            "threshold_count": exemplar["threshold_count"],
        }
        for key in exemplar:
            if key in {"hidden_n", "threshold_count"}:
                continue
            entry[key] = mean_std([rows_i[idx] for rows_i in seed_rows], key)
        rows.append(entry)
    return {
        "config": {
            "dataset": "covtype",
            "diagnostic": "exact posterior versus oracle mean-preserving independent completion",
            "k": 3,
            "q0": 0.2,
            "q1": 0.8,
            "n_bags_per_seed": 5000,
        },
        "aggregate_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="oracle_diagnostic_results.json")
    args = parser.parse_args()
    seed_rows = [run_seed(seed) for seed in args.seeds]
    result = {"seeds": args.seeds, "covtype_oracle_diagnostic": aggregate(seed_rows)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
