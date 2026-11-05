import argparse
import json
from pathlib import Path

import numpy as np


def mean_std_dict(records):
    out = {}
    keys = records[0].keys()
    for key in keys:
        vals = np.asarray([record[key] for record in records], dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
        }
    return out


def aggregate_curve_dict(records):
    out = []
    for idx in range(len(records[0])):
        keys = records[0][idx].keys()
        entry = {}
        for key in keys:
            vals = np.asarray([record[idx][key] for record in records], dtype=np.float64)
            entry[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }
        out.append(entry)
    return out


def aggregate_threshold_rows(records):
    out = []
    for idx in range(len(records[0])):
        exemplar = records[0][idx]
        entry = {
            "hidden_n": exemplar["hidden_n"],
            "m": exemplar["m"],
            "threshold_fraction": exemplar["threshold_fraction"],
            "threshold_count": exemplar["threshold_count"],
            "label_prevalence": {
                "mean": float(np.mean([record[idx]["label_prevalence"] for record in records])),
                "std": float(
                    np.std([record[idx]["label_prevalence"] for record in records], ddof=0)
                ),
            },
            "oracle_tail_mean": {
                "mean": float(np.mean([record[idx].get("oracle_tail_mean", np.nan) for record in records])),
                "std": float(
                    np.std([record[idx].get("oracle_tail_mean", np.nan) for record in records], ddof=0)
                ),
            },
            "metrics": {},
        }
        for model in exemplar["metrics"].keys():
            entry["metrics"][model] = mean_std_dict(
                [record[idx]["metrics"][model] for record in records]
            )
        out.append(entry)
    return out


def aggregate_ambiguity_rows(records):
    out = []
    for idx in range(len(records[0])):
        exemplar = records[0][idx]
        entry = {
            "hidden_n": exemplar["hidden_n"],
            "oracle_v_median": {
                "mean": float(np.mean([record[idx]["oracle_v_median"] for record in records])),
                "std": float(np.std([record[idx]["oracle_v_median"] for record in records], ddof=0)),
            },
            "low_v_mean": {
                "mean": float(np.mean([record[idx]["low_v_mean"] for record in records])),
                "std": float(np.std([record[idx]["low_v_mean"] for record in records], ddof=0)),
            },
            "high_v_mean": {
                "mean": float(np.mean([record[idx]["high_v_mean"] for record in records])),
                "std": float(np.std([record[idx]["high_v_mean"] for record in records], ddof=0)),
            },
            "low_nv_mean": {
                "mean": float(np.mean([record[idx]["low_nv_mean"] for record in records])),
                "std": float(np.std([record[idx]["low_nv_mean"] for record in records], ddof=0)),
            },
            "high_nv_mean": {
                "mean": float(np.mean([record[idx]["high_nv_mean"] for record in records])),
                "std": float(np.std([record[idx]["high_nv_mean"] for record in records], ddof=0)),
            },
            "low_n2v_mean": {
                "mean": float(np.mean([record[idx]["low_n2v_mean"] for record in records])),
                "std": float(np.std([record[idx]["low_n2v_mean"] for record in records], ddof=0)),
            },
            "high_n2v_mean": {
                "mean": float(np.mean([record[idx]["high_n2v_mean"] for record in records])),
                "std": float(np.std([record[idx]["high_n2v_mean"] for record in records], ddof=0)),
            },
            "slice_metrics": {},
        }
        for slice_name in exemplar["slice_metrics"].keys():
            entry["slice_metrics"][slice_name] = {
                model: mean_std_dict([record[idx]["slice_metrics"][slice_name][model] for record in records])
                for model in exemplar["slice_metrics"][slice_name].keys()
            }
        out.append(entry)
    return out


def load_seed_result(path):
    path = Path(path)
    if path.is_dir():
        path = path / "benchmark_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def default_input_paths(output_dir, seeds):
    return [Path(output_dir) / f"seed_{seed}" / "benchmark_results.json" for seed in seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Directories or JSON files produced by remote benchmark runs. "
        "Each directory must contain benchmark_results.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for the aggregate JSON.",
    )
    args = parser.parse_args()

    input_paths = (
        [Path(p) for p in args.input_dirs]
        if args.input_dirs
        else default_input_paths(args.output_dir, args.seeds)
    )
    seed_payloads = [load_seed_result(path) for path in input_paths]
    if len(seed_payloads) != len(args.seeds):
        raise ValueError("Number of input payloads must match the number of seeds.")

    per_seed = {str(seed): payload for seed, payload in zip(args.seeds, seed_payloads)}
    aggregate = {"seeds": args.seeds, "per_seed": per_seed}

    benchmark_names = set(seed_payloads[0].keys())
    for payload in seed_payloads[1:]:
        benchmark_names.intersection_update(payload.keys())

    if "synthetic_scaling" in benchmark_names:
        aggregate["synthetic_scaling"] = seed_payloads[0]["synthetic_scaling"]

    if "breast_benchmark" in benchmark_names:
        records = [payload["breast_benchmark"]["metrics"] for payload in seed_payloads]
        model_names = records[0].keys()
        aggregate["breast_benchmark"] = {
            "aggregate_metrics": {
                model: mean_std_dict([record[model] for record in records]) for model in model_names
            }
        }

    if "digits_benchmark" in benchmark_names:
        prefix_records = [payload["digits_benchmark"]["prefix_1_metrics"] for payload in seed_payloads]
        policy_records = [payload["digits_benchmark"]["policy_metrics"] for payload in seed_payloads]
        curve_records = [payload["digits_benchmark"]["policy_curve"] for payload in seed_payloads]
        model_names = prefix_records[0].keys()
        aggregate["digits_benchmark"] = {
            "aggregate_prefix_1_metrics": {
                model: mean_std_dict([record[model] for record in prefix_records])
                for model in model_names
            },
            "aggregate_policy_metrics": {
                model: mean_std_dict([record[model] for record in policy_records])
                for model in policy_records[0].keys()
            },
            "aggregate_policy_curve": {
                model: aggregate_curve_dict([record[model] for record in curve_records])
                for model in curve_records[0].keys()
            },
        }

    if "covtype_benchmark" in benchmark_names:
        records = [payload["covtype_benchmark"]["metrics"] for payload in seed_payloads]
        model_names = records[0].keys()
        aggregate["covtype_benchmark"] = {
            "aggregate_metrics": {
                model: mean_std_dict([record[model] for record in records]) for model in model_names
            }
        }

    if "covtype_threshold_benchmark" in benchmark_names:
        rows = [payload["covtype_threshold_benchmark"]["rows"] for payload in seed_payloads]
        aggregate["covtype_threshold_benchmark"] = {
            "config": seed_payloads[0]["covtype_threshold_benchmark"]["config"],
            "aggregate_rows": aggregate_threshold_rows(rows),
        }
        if "ambiguity_rows" in seed_payloads[0]["covtype_threshold_benchmark"]:
            ambiguity_rows = [
                payload["covtype_threshold_benchmark"]["ambiguity_rows"] for payload in seed_payloads
            ]
            aggregate["covtype_threshold_benchmark"]["aggregate_ambiguity_rows"] = (
                aggregate_ambiguity_rows(ambiguity_rows)
            )

    if "covtype_active_learning" in benchmark_names:
        curves = [payload["covtype_active_learning"]["curves"] for payload in seed_payloads]
        model_names = curves[0].keys()
        aggregate["covtype_active_learning"] = {
            "config": seed_payloads[0]["covtype_active_learning"]["config"],
            "aggregate_curves": {
                model: aggregate_curve_dict([curve[model] for curve in curves])
                for model in model_names
            },
        }

    if "batch_inspection_benchmark" in benchmark_names:
        records = [payload["batch_inspection_benchmark"] for payload in seed_payloads]
        model_names = records[0]["overall_metrics"].keys()
        aggregate["batch_inspection_benchmark"] = {
            "config": records[0]["config"],
            "test_prevalence": mean_std_dict(
                [{"test_prevalence": record["test_prevalence"]} for record in records]
            )["test_prevalence"],
            "aggregate_overall_metrics": {
                model: mean_std_dict([record["overall_metrics"][model] for record in records])
                for model in model_names
            },
            "aggregate_taxscore_bins": {},
        }
        for bin_name in ["low", "mid", "high"]:
            bin_records = [record["taxscore_bins"][bin_name] for record in records]
            aggregate["batch_inspection_benchmark"]["aggregate_taxscore_bins"][bin_name] = {
                "count": mean_std_dict([{"count": row["count"]} for row in bin_records])["count"],
                "mean_taxscore": mean_std_dict(
                    [{"mean_taxscore": row["mean_taxscore"]} for row in bin_records]
                )["mean_taxscore"],
                "metrics": {
                    model: mean_std_dict([row["metrics"][model] for row in bin_records])
                    for model in model_names
                },
            }

    if "musk_stop_reveal" in benchmark_names:
        prefix_records = [payload["musk_stop_reveal"]["prefix_1_metrics"] for payload in seed_payloads]
        policy_records = [payload["musk_stop_reveal"]["policy_metrics"] for payload in seed_payloads]
        curve_records = [payload["musk_stop_reveal"]["policy_curve"] for payload in seed_payloads]
        model_names = prefix_records[0].keys()
        aggregate["musk_stop_reveal"] = {
            "config": seed_payloads[0]["musk_stop_reveal"]["config"],
            "aggregate_prefix_1_metrics": {
                model: mean_std_dict([record[model] for record in prefix_records])
                for model in model_names
            },
            "aggregate_policy_metrics": {
                model: mean_std_dict([record[model] for record in policy_records])
                for model in policy_records[0].keys()
            },
            "aggregate_policy_curve": {
                model: aggregate_curve_dict([record[model] for record in curve_records])
                for model in curve_records[0].keys()
            },
        }

    if "birds_stop_reveal" in benchmark_names:
        prefix_records = [payload["birds_stop_reveal"]["prefix_1_metrics"] for payload in seed_payloads]
        policy_records = [payload["birds_stop_reveal"]["policy_metrics"] for payload in seed_payloads]
        curve_records = [payload["birds_stop_reveal"]["policy_curve"] for payload in seed_payloads]
        model_names = prefix_records[0].keys()
        aggregate["birds_stop_reveal"] = {
            "config": seed_payloads[0]["birds_stop_reveal"]["config"],
            "aggregate_prefix_1_metrics": {
                model: mean_std_dict([record[model] for record in prefix_records])
                for model in model_names
            },
            "aggregate_policy_metrics": {
                model: mean_std_dict([record[model] for record in policy_records])
                for model in policy_records[0].keys()
            },
            "aggregate_policy_curve": {
                model: aggregate_curve_dict([record[model] for record in curve_records])
                for model in curve_records[0].keys()
            },
        }

    if "protein_stop_reveal" in benchmark_names:
        prefix_records = [payload["protein_stop_reveal"]["prefix_1_metrics"] for payload in seed_payloads]
        policy_records = [payload["protein_stop_reveal"]["policy_metrics"] for payload in seed_payloads]
        curve_records = [payload["protein_stop_reveal"]["policy_curve"] for payload in seed_payloads]
        model_names = prefix_records[0].keys()
        aggregate["protein_stop_reveal"] = {
            "config": seed_payloads[0]["protein_stop_reveal"]["config"],
            "aggregate_prefix_1_metrics": {
                model: mean_std_dict([record[model] for record in prefix_records])
                for model in model_names
            },
            "aggregate_policy_metrics": {
                model: mean_std_dict([record[model] for record in policy_records])
                for model in policy_records[0].keys()
            },
            "aggregate_policy_curve": {
                model: aggregate_curve_dict([record[model] for record in curve_records])
                for model in curve_records[0].keys()
            },
        }

    output_path = Path(args.output) if args.output else Path(args.output_dir) / "aggregate_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved aggregated results to {output_path}")


if __name__ == "__main__":
    main()
