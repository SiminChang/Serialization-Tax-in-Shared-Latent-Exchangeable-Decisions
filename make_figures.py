import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "IndepPB": "#D55E00",
    "MeanIndep": "#D55E00",
    "SeqMLP": "#0072B2",
    "SeqPermAvg": "#CC79A7",
    "SeqGRU": "#56B4E9",
    "SeqGRUCount": "#1F4E79",
    "SeqTX": "#7E57C2",
    "SeqTXCount": "#6A3D9A",
    "SeqTXCountPM": "#4A148C",
    "ARRollout": "#E69F00",
    "ObjMLP": "#009E73",
    "ObjSetTF": "#111111",
    "ObjSetTFPM": "#555555",
    "SeqTXGlobal": "#009E73",
    "ObjSetGlobal": "#004D40",
    "TwoTowerIndep": "#8C564B",
    "TwoTowerDropout": "#A6761D",
    "TwoTowerEnsMean": "#B15928",
    "TwoTowerEnsMix": "#33A02C",
    "ANP": "#005AB5",
    "FlowLatent": "#DC3220",
    "GlobalGrid": "#000000",
}

MARKERS = {
    "IndepPB": "o",
    "MeanIndep": "o",
    "SeqMLP": "s",
    "SeqPermAvg": "^",
    "SeqGRU": "D",
    "SeqGRUCount": "P",
    "SeqTX": "v",
    "SeqTXCount": "X",
    "SeqTXCountPM": "x",
    "ARRollout": "*",
    "ObjMLP": "<",
    "ObjSetTF": ">",
    "ObjSetTFPM": "8",
    "SeqTXGlobal": "d",
    "ObjSetGlobal": "h",
    "TwoTowerIndep": "1",
    "TwoTowerDropout": "2",
    "TwoTowerEnsMean": "3",
    "TwoTowerEnsMix": "4",
    "ANP": "p",
    "FlowLatent": "H",
    "GlobalGrid": "P",
}

LINESTYLES = {
    "IndepPB": "-",
    "MeanIndep": "-",
    "SeqMLP": "--",
    "SeqPermAvg": "-.",
    "SeqGRU": ":",
    "SeqGRUCount": "-",
    "SeqTX": "--",
    "SeqTXCount": "-.",
    "SeqTXCountPM": "--",
    "ARRollout": ":",
    "ObjMLP": "-",
    "ObjSetTF": "--",
    "ObjSetTFPM": ":",
    "SeqTXGlobal": "-",
    "ObjSetGlobal": "--",
    "TwoTowerIndep": ":",
    "TwoTowerDropout": "-.",
    "TwoTowerEnsMean": "--",
    "TwoTowerEnsMix": "-",
    "ANP": "-",
    "FlowLatent": "-.",
    "GlobalGrid": "-.",
}

MOVIE_MAIN_CONFIGS = ["crit40", "crit80"]
MOVIE_SCALE_CONFIGS = ["ml10n120c60"]
MOVIE_AUDIT_CONFIGS = ["crit40", "crit80", "ml10n120c60"]
CONFIG_LABELS = {
    "crit40": "ML1M-40",
    "crit80": "ML1M-80",
    "crit80c75": "ML1M-80/.75",
    "ml10n120c60": "ML10M-120",
    "ml10k1n120c60": "ML10M-k1",
}


def load_results():
    candidates = [
        Path("aggregate_results.json"),
        Path("results/aggregate_results.json"),
        Path("benchmark_results.json"),
        Path("results/benchmark_results.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("No benchmark JSON found in working directory or results/.")


def add_band(ax, x, stats, model, key, ylabel, title):
    mean = [row["metrics"][model][key]["mean"] for row in stats]
    std = [row["metrics"][model][key]["std"] for row in stats]
    plot_method(ax, x, mean, model)
    ax.fill_between(
        x,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        color=COLORS[model],
        alpha=0.12,
    )
    ax.set_xlabel("Unrevealed size $n$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_method(ax, x, y, model, label=None):
    ax.plot(
        x,
        y,
        marker=MARKERS.get(model, "o"),
        linestyle=LINESTYLES.get(model, "-"),
        linewidth=2.2,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        color=COLORS[model],
        label=label or model,
    )


def plot_scaling_law(results):
    synthetic = results.get("synthetic_scaling")
    if synthetic is None:
        return

    n_rows = synthetic["n_sweep"]["rows"]
    alpha_rows = synthetic["alpha_sweep"]["rows"]

    n = [row["n"] for row in n_rows]
    exact = [row["exact"] for row in n_rows]
    local = [row["local"] for row in n_rows]
    gap = [abs(row["gap"]) for row in n_rows]
    tv_exact = [row["tv_exact"] for row in n_rows]
    tv_lb = [row["tv_lower_bound"] for row in n_rows]
    variance_gap = [row["variance_gap"] for row in n_rows]
    variance_gap_lb = [row["variance_gap_lower_bound"] for row in n_rows]

    variance = [row["posterior_variance"] for row in alpha_rows]
    alpha_exact = [row["exact"] for row in alpha_rows]
    alpha_local = [row["local"] for row in alpha_rows]
    alpha_gap = [abs(row["gap"]) for row in alpha_rows]
    alpha_tv_exact = [row["tv_exact"] for row in alpha_rows]
    alpha_tv_lb = [row["tv_lower_bound"] for row in alpha_rows]
    alpha_var_gap = [row["variance_gap"] for row in alpha_rows]
    alpha_var_gap_lb = [row["variance_gap_lower_bound"] for row in alpha_rows]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6))

    ax = axes[0, 0]
    ax.plot(n, exact, marker="o", linewidth=2, color="#0b6e4f", label="Exact posterior")
    ax.plot(n, local, marker="s", linewidth=2, color="#c84c09", label="Local completion")
    ax.set_xlabel("Unrevealed size $n$")
    ax.set_ylabel(r"$\Pr(Y_n=1 \mid O)$")
    ax.set_title("Threshold posterior")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[0, 1]
    ax.plot(n, gap, marker="o", linewidth=2, color="#7a1fa2", label="Posterior gap")
    ax.plot(n, tv_exact, marker="^", linewidth=2, color="#1f77b4", label="Exact TV")
    ax.plot(n, tv_lb, marker="d", linewidth=2, color="#444444", label="TV lower bound")
    ax.set_xlabel("Unrevealed size $n$")
    ax.set_ylabel("Gap")
    ax.set_title("Threshold and TV tax")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[0, 2]
    ax.plot(n, variance_gap, marker="o", linewidth=2, color="#c84c09", label="Variance gap")
    ax.plot(
        n,
        variance_gap_lb,
        marker="d",
        linewidth=2,
        color="#444444",
        label=r"$n(n-1)\mathrm{Var}(Q\mid O)$",
    )
    ax.set_xlabel("Unrevealed size $n$")
    ax.set_ylabel("Variance gap")
    ax.set_title("Quadratic variance law")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    ax.plot(
        variance,
        alpha_exact,
        marker="o",
        linewidth=2,
        color="#0b6e4f",
        label="Exact posterior",
    )
    ax.plot(
        variance,
        alpha_local,
        marker="s",
        linewidth=2,
        color="#c84c09",
        label="Local completion",
    )
    ax.set_xlabel(r"Latent variance $\mathrm{Var}(Q\mid O)$")
    ax.set_ylabel(r"$\Pr(Y_n=1 \mid O)$")
    ax.set_title("Threshold posterior")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="center left")

    ax = axes[1, 1]
    ax.plot(
        variance,
        alpha_gap,
        marker="o",
        linewidth=2,
        color="#7a1fa2",
        label="Posterior gap",
    )
    ax.plot(
        variance,
        alpha_tv_exact,
        marker="^",
        linewidth=2,
        color="#1f77b4",
        label="Exact TV",
    )
    ax.plot(
        variance,
        alpha_tv_lb,
        marker="d",
        linewidth=2,
        color="#444444",
        label="TV lower bound",
    )
    ax.set_xlabel(r"Latent variance $\mathrm{Var}(Q\mid O)$")
    ax.set_ylabel("Gap")
    ax.set_title("Uncertainty-driven tax")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 2]
    ax.plot(
        variance,
        alpha_var_gap,
        marker="o",
        linewidth=2,
        color="#c84c09",
        label="Variance gap",
    )
    ax.plot(
        variance,
        alpha_var_gap_lb,
        marker="d",
        linewidth=2,
        color="#444444",
        label=r"$n(n-1)\mathrm{Var}(Q\mid O)$",
    )
    ax.set_xlabel(r"Latent variance $\mathrm{Var}(Q\mid O)$")
    ax.set_ylabel("Variance gap")
    ax.set_title("Variance law")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig("figures/scaling_law.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_policy_tradeoff(results):
    digits = results.get("digits_benchmark")
    if digits is None or "aggregate_policy_curve" not in digits:
        return

    policy_curve = digits["aggregate_policy_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    for model in ["IndepPB", "SeqMLP", "SeqPermAvg", "ObjMLP"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        utility = [row["utility"]["mean"] for row in curve]
        plot_method(ax, costs, utility, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Utility")
    ax.set_title("Decision frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for model in ["IndepPB", "SeqMLP", "SeqPermAvg", "ObjMLP"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        reveals = [row["avg_reveals"]["mean"] for row in curve]
        plot_method(ax, costs, reveals, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Average reveals")
    ax.set_title("Information usage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("figures/policy_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_musk_policy_tradeoff(results):
    musk = results.get("musk_stop_reveal")
    if musk is None or "aggregate_policy_curve" not in musk:
        return

    policy_curve = musk["aggregate_policy_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        utility = [row["utility"]["mean"] for row in curve]
        plot_method(ax, costs, utility, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Utility")
    ax.set_title("Musk stop/reveal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        reveals = [row["avg_reveals"]["mean"] for row in curve]
        plot_method(ax, costs, reveals, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Average reveals")
    ax.set_title("Musk information usage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("figures/musk_policy_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_protein_policy_tradeoff(results):
    protein = results.get("protein_stop_reveal")
    if protein is None or "aggregate_policy_curve" not in protein:
        return

    policy_curve = protein["aggregate_policy_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        utility = [row["utility"]["mean"] for row in curve]
        plot_method(ax, costs, utility, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Utility")
    ax.set_title("Protein stop/reveal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        reveals = [row["avg_reveals"]["mean"] for row in curve]
        plot_method(ax, costs, reveals, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Average reveals")
    ax.set_title("Protein information usage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("figures/protein_policy_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_birds_policy_tradeoff(results):
    birds = results.get("birds_stop_reveal")
    if birds is None or "aggregate_policy_curve" not in birds:
        return

    policy_curve = birds["aggregate_policy_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        utility = [row["utility"]["mean"] for row in curve]
        plot_method(ax, costs, utility, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Utility")
    ax.set_title("Birds stop/reveal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for model in ["SeqGRU", "SeqTX", "ObjMLP", "ObjSetTF"]:
        if model not in policy_curve:
            continue
        curve = policy_curve[model]
        costs = [row["cost"]["mean"] for row in curve]
        reveals = [row["avg_reveals"]["mean"] for row in curve]
        plot_method(ax, costs, reveals, model)
    ax.set_xlabel("Reveal cost")
    ax.set_ylabel("Average reveals")
    ax.set_title("Birds information usage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("figures/birds_policy_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_risk(results):
    threshold = results.get("covtype_threshold_benchmark")
    if threshold is None or "aggregate_rows" not in threshold:
        return

    rows = threshold["aggregate_rows"]
    hidden_n = [row["hidden_n"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    model_order = [
        model
        for model in [
            "IndepPB",
            "SeqMLP",
            "SeqPermAvg",
            "SeqGRUCount",
            "SeqTXCount",
            "ARRollout",
            "ObjMLP",
            "ObjSetTF",
        ]
        if model in rows[0]["metrics"]
    ]
    for model in model_order:
        add_band(
            axes[0],
            hidden_n,
            rows,
            model,
            "acc",
            "Accuracy",
            "Threshold accuracy",
        )
        add_band(
            axes[1],
            hidden_n,
            rows,
            model,
            "nll",
            "NLL",
            "Threshold NLL",
        )
        add_band(
            axes[2],
            hidden_n,
            rows,
            model,
            "ece",
            "ECE",
            "Threshold calibration",
        )

    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig("figures/threshold_risk_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_active_learning(results):
    active = results.get("covtype_active_learning")
    if active is None or "aggregate_curves" not in active:
        return

    curves = active["aggregate_curves"]
    model_order = [
        model
        for model in ["IndepPB", "SeqGRUCount", "SeqTXCount", "ARRollout", "ObjMLP", "ObjSetTF"]
        if model in curves
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for model in model_order:
        curve = curves[model]
        x = [row["budget_bags"]["mean"] for row in curve]
        y = [row["acc"]["mean"] for row in curve]
        s = [row["acc"]["std"] for row in curve]
        plot_method(ax, x, y, model)
        ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color=COLORS[model], alpha=0.12)
    ax.set_xlabel("Labeled bags")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Active learning accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    for model in model_order:
        curve = curves[model]
        x = [row["budget_bags"]["mean"] for row in curve]
        y = [row["nll"]["mean"] for row in curve]
        s = [row["nll"]["std"] for row in curve]
        plot_method(ax, x, y, model)
        ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color=COLORS[model], alpha=0.12)
    ax.set_xlabel("Labeled bags")
    ax.set_ylabel("Test NLL")
    ax.set_title("Active learning calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("figures/active_learning_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_oracle_decision_tax(results):
    diagnostic = results.get("covtype_oracle_diagnostic")
    if diagnostic is None or "aggregate_rows" not in diagnostic:
        return

    rows = diagnostic["aggregate_rows"]
    n = [row["hidden_n"] for row in rows]
    mean_tv = [row["event_tv"]["mean"] for row in rows]
    mean_tv_std = [row["event_tv"]["std"] for row in rows]
    p90_tv = [row["event_tv_p90"]["mean"] for row in rows]
    nv = [row["mean_nv"]["mean"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    ax.plot(
        n,
        mean_tv,
        marker="o",
        linestyle="-",
        linewidth=2.2,
        markersize=5.5,
        color="#D55E00",
        label="Mean event TV",
    )
    ax.fill_between(
        n,
        [a - b for a, b in zip(mean_tv, mean_tv_std)],
        [a + b for a, b in zip(mean_tv, mean_tv_std)],
        color="#D55E00",
        alpha=0.15,
    )
    ax.plot(
        n,
        p90_tv,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=5.5,
        color="#111111",
        label="90th percentile",
    )
    ax.set_xlabel("Unrevealed size $n$")
    ax.set_ylabel(r"$|\widehat p(O)-p^\star(O)|$")
    ax.set_title("Threshold-event TV")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(
        nv,
        mean_tv,
        marker="o",
        linestyle="-",
        linewidth=2.2,
        markersize=5.5,
        color="#0072B2",
    )
    ax.set_xlabel(r"Mean ambiguity scale $n v(O)$")
    ax.set_ylabel("Mean event TV")
    ax.set_title("Decision scale")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("figures/oracle_decision_tax.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_taxscore_gap_curve(results):
    controlled = results.get("controlled_taxscore")
    natural = results.get("natural_taxscore")
    if controlled is None and natural is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    if controlled is not None and "bins" in controlled:
        bins = controlled["bins"]
        x = [row["mean_taxscore"]["mean"] for row in bins]
        y = [row["mean_event_tv"]["mean"] for row in bins]
        s = [row["mean_event_tv"]["std"] for row in bins]
        ax.plot(x, y, marker="o", linewidth=2.2, color="#D55E00", label="Controlled")
        ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color="#D55E00", alpha=0.15)
    ax.set_xlabel("Mean TaxScore")
    ax.set_ylabel("Mean event TV")
    ax.set_title("TaxScore predicts event error")
    ax.grid(alpha=0.25)

    ax = axes[1]
    if natural is not None:
        bin_names = ["low", "mid", "high"]
        gaps = []
        gap_err = []
        for bin_name in bin_names:
            vals = []
            for dataset in natural.values():
                metrics = dataset["bins"][bin_name]["metrics"]
                obj = max(metrics["ObjMLP"]["utility"]["mean"], metrics["ObjSetTF"]["utility"]["mean"])
                seq = max(metrics["SeqGRU"]["utility"]["mean"], metrics["SeqTX"]["utility"]["mean"])
                vals.append(obj - seq)
            gaps.append(float(np.mean(vals)))
            gap_err.append(float(np.std(vals, ddof=0)))
        x = np.arange(len(bin_names))
        ax.bar(x, gaps, yerr=gap_err, color=["#9bb7d4", "#e6b35c", "#d46a5f"])
        ax.axhline(0.0, color="#444444", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel("Natural-task TaxScore bin")
    ax.set_ylabel("Object minus sequence utility")
    ax.set_title("Natural decision diagnostic")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig("figures/taxscore_gap_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_head_resolution(results):
    head = results.get("posterior_head_resolution")
    if head is None or "rows" not in head:
        return

    rows = head["rows"]
    k = [row["K"] for row in rows]
    grid = [row["grid_error"] for row in rows]
    mean_completion = [row["mean_completion_error"] for row in rows]
    beta = [row["beta_moment_error"] for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.7))
    ax.plot(k, grid, marker="o", linewidth=2.2, color="#009E73", label="Grid mixture")
    ax.plot(k, mean_completion, marker="s", linewidth=2.2, linestyle="--", color="#D55E00", label="Mean completion")
    ax.plot(k, beta, marker="^", linewidth=2.2, linestyle="-.", color="#0072B2", label="Beta moment")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Posterior head grid size $K$")
    ax.set_ylabel("Threshold posterior error")
    ax.set_title("Resolution escapes the tax")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/posterior_head_resolution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_movielens_tail(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None or "configs" not in benchmark:
        return

    model_order = [
        "MeanIndep",
        "SeqTXCount",
        "SeqTXCountPM",
        "ObjSetTF",
        "ObjSetTFPM",
        "TwoTowerIndep",
        "TwoTowerDropout",
        "TwoTowerEnsMean",
        "TwoTowerEnsMix",
        "SeqTXGlobal",
        "ANP",
        "FlowLatent",
        "ObjSetGlobal",
        "GlobalGrid",
    ]
    labels = {
        "MeanIndep": "Indep",
        "SeqTXCount": "Seq\nTX",
        "SeqTXCountPM": "TX\nPM",
        "ObjSetTF": "Obj\nSet",
        "ObjSetTFPM": "Set\nPM",
        "TwoTowerIndep": "2T\nInd",
        "TwoTowerDropout": "2T\nDrop",
        "TwoTowerEnsMean": "2T\nMean",
        "TwoTowerEnsMix": "2T\nMix",
        "SeqTXGlobal": "TX\nLat",
        "ANP": "ANP",
        "FlowLatent": "Flow",
        "ObjSetGlobal": "Set\nLat",
        "GlobalGrid": "Grid",
    }
    configs = [name for name in MOVIE_AUDIT_CONFIGS if name in benchmark["configs"]]
    if not configs:
        configs = [name for name in ["main", "cold"] if name in benchmark["configs"]]
    if not configs:
        configs = list(benchmark["configs"].keys())[:2]

    fig, axes = plt.subplots(len(configs), 2, figsize=(8.2, 2.75 * len(configs)))
    if len(configs) == 1:
        axes = np.asarray([axes])

    for row_idx, cfg_name in enumerate(configs):
        cfg = benchmark["configs"][cfg_name]
        high = cfg["bins"]["high"]["metrics"]
        shown_models = [model for model in model_order if model in high]
        x = np.arange(len(shown_models))
        for col_idx, (metric, ylabel) in enumerate([("nll", "Tail NLL"), ("utility", "Decision utility")]):
            ax = axes[row_idx, col_idx]
            y = [high[model][metric]["mean"] for model in shown_models]
            err = [high[model][metric]["std"] for model in shown_models]
            colors = [COLORS[model] for model in shown_models]
            ax.bar(x, y, yerr=err, color=colors, alpha=0.88, capsize=2)
            ax.set_xticks(x)
            if row_idx == len(configs) - 1:
                ax.set_xticklabels([labels[m] for m in shown_models], rotation=0, ha="center", fontsize=7)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            ax.set_ylabel(ylabel)
            title = f"{CONFIG_LABELS.get(cfg_name, cfg_name)}: k={cfg['config']['k']}, n={cfg['config']['hidden_n']}, c={cfg['config']['c']}"
            ax.set_title(title if col_idx == 0 else "high TaxScore slice")
            ax.grid(axis="y", alpha=0.25)
        axes[row_idx, 0].text(
            0.02,
            0.95,
            f"mean TaxScore={cfg['bins']['high']['mean_taxscore']['mean']:.1f}",
            transform=axes[row_idx, 0].transAxes,
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig("figures/movielens_tail_risk.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_movielens_tail_violin(results, configs)

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.9))
    for cfg_name in configs:
        cfg = benchmark["configs"][cfg_name]
        rows = cfg["resolution_rows"]
        k = [row["K"] for row in rows]
        grid = [row["GlobalGrid"]["mean"] for row in rows]
        grid_err = [row["GlobalGrid"]["std"] for row in rows]
        indep = [row["MeanIndep"]["mean"] for row in rows]
        indep_err = [row["MeanIndep"]["std"] for row in rows]
        cfg_label = CONFIG_LABELS.get(cfg_name, cfg_name)
        ax.plot(k, grid, marker="o", linewidth=2.1, label=f"{cfg_label}: grid latent")
        ax.fill_between(k, np.asarray(grid) - np.asarray(grid_err), np.asarray(grid) + np.asarray(grid_err), alpha=0.12)
        ax.plot(k, indep, marker="s", linewidth=1.8, linestyle="--", label=f"{cfg_label}: independent")
        ax.fill_between(k, np.asarray(indep) - np.asarray(indep_err), np.asarray(indep) + np.asarray(indep_err), alpha=0.08)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent grid size $K$")
    ax.set_ylabel("Tail NLL")
    ax.set_title("Posterior resolution on MovieLens")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/movielens_resolution.pdf", bbox_inches="tight")
    plt.close(fig)

    finite = benchmark.get("finite_exchangeability_stress") or results.get("finite_exchangeability_stress")
    if finite and "rows" in finite:
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.8))
        rows = finite["rows"]
        x = [row["n2_over_N"]["mean"] if isinstance(row["n2_over_N"], dict) else row["n2_over_N"] for row in rows]
        tv = [
            row["tv_finite_vs_surrogate"]["mean"]
            if isinstance(row["tv_finite_vs_surrogate"], dict)
            else row["tv_finite_vs_surrogate"]
            for row in rows
        ]
        bound = [
            row["collision_bound"]["mean"] if isinstance(row["collision_bound"], dict) else row["collision_bound"]
            for row in rows
        ]
        ax.plot(x, tv, marker="o", linewidth=2.1, label="finite vs surrogate TV")
        ax.plot(x, bound, marker="s", linewidth=1.8, linestyle="--", label="collision bound")
        ax.set_xlabel(r"$n^2/N$")
        ax.set_ylabel("TV")
        ax.set_title("Finite-exchangeability correction")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig("figures/finite_exchangeability_stress.pdf", bbox_inches="tight")
        plt.close(fig)


def _movie_seed_metric_values(cfg_name, model, metric, bin_name="high"):
    compact = Path("results/plot_seed_metrics.json")
    if compact.exists():
        records = json.loads(compact.read_text(encoding="utf-8"))
        return records.get(cfg_name, {}).get(bin_name, {}).get(model, {}).get(metric, [])
    seed_specs = {
        "crit80": ("ml1_seed*.json", "crit80"),
        "ml10n120c60": ("ml10_seed*.json", "ml10n120c60"),
        "crit40": ("movielens_oral_aggregate.json", "crit40"),
    }
    if cfg_name not in seed_specs:
        return []
    pattern, key = seed_specs[cfg_name]
    values = []
    for path in sorted(Path(".").glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stat = payload["configs"][key]["bins"][bin_name]["metrics"][model][metric]
        except Exception:
            continue
        values.append(_mean_value(stat))
    return values


def plot_movielens_tail_violin(results, configs):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None:
        return
    configs = [cfg for cfg in ["crit80", "ml10n120c60"] if cfg in benchmark.get("configs", {})]
    if not configs:
        return
    models = ["SeqTXCount", "ANP", "FlowLatent"]
    labels = {"SeqTXCount": "SeqTX", "ANP": "ANP", "FlowLatent": "Flow"}
    metrics = [("nll", "Tail NLL", "lower is better"), ("utility", "Decision utility", "higher is better")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.35), sharex=False)
    rng_offsets = np.linspace(-0.075, 0.075, 5)
    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        positions, values, colors, xticklabels = [], [], [], []
        pos = 0.0
        for cfg_idx, cfg_name in enumerate(configs):
            cfg = benchmark["configs"][cfg_name]
            high = cfg["bins"]["high"]["metrics"]
            for model in models:
                if model not in high:
                    continue
                vals = _movie_seed_metric_values(cfg_name, model, metric)
                if not vals:
                    mean = high[model][metric]["mean"]
                    std = high[model][metric].get("std", 0.0)
                    vals = [mean - std, mean, mean + std]
                positions.append(pos)
                values.append(vals)
                colors.append(COLORS.get(model, "#333333"))
                xticklabels.append(labels[model])
                pos += 0.72
            if cfg_idx != len(configs) - 1:
                pos += 0.45
        viol = ax.violinplot(values, positions=positions, widths=0.50, showmeans=False, showextrema=False)
        for body, color in zip(viol["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
            body.set_linewidth(0.8)
        for p, vals, color in zip(positions, values, colors):
            offsets = rng_offsets[: len(vals)] if len(vals) <= len(rng_offsets) else np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(np.asarray(p) + offsets, vals, s=16, color=color, edgecolor="white", linewidth=0.35, zorder=3)
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.errorbar(p, mean, yerr=1.96 * sem, fmt="_", color="#111111", elinewidth=1.0, capsize=2, markersize=9, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        # group labels
        start = 0
        for cfg_name in configs:
            present = [m for m in models if m in benchmark["configs"][cfg_name]["bins"]["high"]["metrics"]]
            if not present:
                continue
            group_pos = positions[start : start + len(present)]
            ax.text(np.mean(group_pos), -0.22, CONFIG_LABELS.get(cfg_name, cfg_name), ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.5)
            start += len(present)
    axes[0].set_title("Tail calibration on high-TaxScore slates", fontsize=9)
    axes[1].set_title("Decision value on the same slates", fontsize=9)
    fig.tight_layout(w_pad=1.0)
    fig.savefig("figures/movielens_tail_risk_main_violin.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_value(x, default=0.0):
    if isinstance(x, dict) and "mean" in x:
        return x["mean"]
    if isinstance(x, (int, float)):
        return x
    return default


def _std_value(x, default=0.0):
    if isinstance(x, dict) and "std" in x:
        return x["std"]
    return default


def _fmt(x, digits=3):
    return f"{x:.{digits}f}"


def _fmt_pm(stat, digits=3):
    return f"{_fmt(_mean_value(stat), digits)} $\\pm$ {_fmt(_std_value(stat), digits)}"


def _variant_label(row):
    mode = row.get("selector_mode", "default")
    margin = _mean_value(row.get("critical_margin", 0.08))
    budget = _mean_value(row.get("candidate_budget", 80))
    weight = _mean_value(row.get("prefix_weight", 0.20))
    if mode == "prior_only":
        return "prior only"
    if mode == "prefix_only":
        return "prefix only"
    if abs(weight - 0.20) > 1e-9:
        return f"$w={_fmt(weight, 2)}$"
    if abs(margin - 0.08) > 1e-9:
        return f"$m={_fmt(margin, 2)}$"
    if abs(budget - 80) > 1e-9:
        return f"$B={_fmt(budget, 0)}$"
    return "default"


def write_movielens_audit_tables(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None or "configs" not in benchmark:
        return
    Path("figures").mkdir(exist_ok=True)
    lines = []

    configs = [cfg for cfg in MOVIE_AUDIT_CONFIGS if cfg in benchmark["configs"]]
    models = ["SeqTXCount", "SeqTXCountPM", "TwoTowerEnsMean", "TwoTowerEnsMix", "ANP", "FlowLatent"]
    label = {
        "SeqTXCount": "SeqTX",
        "SeqTXCountPM": "SeqTX-PM",
        "TwoTowerEnsMean": "2T-Mean",
        "TwoTowerEnsMix": "2T-Mix",
        "ANP": "ANP",
        "FlowLatent": "Flow",
    }

    lines.append(r"\begin{paperblock}")
    lines.append(r"\centering")
    lines.append(r"\captionof{table}{MovieLens tail-event NLL on full-distribution and high-TaxScore slices. Means and standard deviations are over five seeds. The high slice is the pre-specified decision-critical TaxScore bin. Dashes indicate baselines absent from the MovieLens 10M scale-up experiments.}")
    lines.append(r"\label{tab:movie-full-high}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Config & Slice & SeqTX & SeqTX-PM & 2T-Mean & 2T-Mix & ANP & Flow \\")
    lines.append(r"\midrule")
    for cfg in configs:
        for scope, scope_label in [("metrics", "All"), ("high", "High")]:
            row = [CONFIG_LABELS.get(cfg, cfg), scope_label]
            for model in models:
                metrics = (
                    benchmark["configs"][cfg]["metrics"]
                    if scope == "metrics"
                    else benchmark["configs"][cfg]["bins"]["high"]["metrics"]
                )
                if model not in metrics:
                    row.append("--")
                else:
                    row.append(_fmt_pm(metrics[model]["nll"]))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{paperblock}")
    lines.append("")

    tail_cfgs = [cfg for cfg in ["crit80", "ml10n120c60"] if benchmark["configs"].get(cfg, {}).get("tail_slices")]
    if tail_cfgs:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{Pre-specified high-TaxScore triage slices on MovieLens. The top slices are selected by the TaxScore diagnostic computed from visible metadata and observed prefix ratings.}")
        lines.append(r"\label{tab:movie-tail-triage}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{llrrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Config & Slice & Count & SeqTX NLL & SeqTX util. & ANP NLL & ANP util. & Flow NLL & Flow util. \\")
        lines.append(r"\midrule")
        for cfg in tail_cfgs:
            slices = benchmark["configs"][cfg]["tail_slices"]
            for slice_name, slice_label in [("top20", "Top 20\\%"), ("top10", "Top 10\\%")]:
                if slice_name not in slices:
                    continue
                row = slices[slice_name]
                metrics = row["metrics"]
                count = _mean_value(row["count"])
                def cell(model, metric):
                    return _fmt_pm(metrics[model][metric])
                lines.append(
                    " & ".join(
                        [
                            CONFIG_LABELS.get(cfg, cfg),
                            slice_label,
                            _fmt(count, 0),
                            cell("SeqTXCount", "nll"),
                            cell("SeqTXCount", "utility"),
                            cell("ANP", "nll"),
                            cell("ANP", "utility"),
                            cell("FlowLatent", "nll"),
                            cell("FlowLatent", "utility"),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"\addlinespace")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    deploy = benchmark.get("deployment_switch_audit")
    if deploy:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{Original budgeted switch gains over SeqTXCount. Mixed predictions use the rich head on slates exceeding a validation-score cutoff and the cheap head elsewhere. Full gains average all test slates. Flagged gains average the routed subset. The offline experiment evaluates both heads to form these comparisons. Latency is assessed separately from this payoff calculation. See Table~\ref{tab:matched-routing} for the separate exact-budget comparison and Table~\ref{tab:latency-audit} for head-only timings.}")
        lines.append(r"\label{tab:movie-deployment-switch}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{lllrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Config & Budget & Target & Test frac. & Full NLL gain & Full util. gain & Per-flagged util. gain \\")
        lines.append(r"\midrule")
        for cfg in [c for c in ["crit80", "ml10n120c60"] if c in deploy]:
            targets = deploy[cfg].get("targets", {})
            for target in ["ANP", "FlowLatent"]:
                rows = targets.get(target, [])
                for row in rows:
                    budget = _mean_value(row.get("budget_fraction"))
                    if min(abs(budget - keep) for keep in (0.10, 0.20, 0.30)) > 1e-6:
                        continue
                    lines.append(
                        " & ".join(
                            [
                                CONFIG_LABELS.get(cfg, cfg),
                                f"{_fmt(100 * budget, 0)}\\%",
                                label.get(target, target),
                                f"{_fmt(100 * _mean_value(row.get('test_fraction')), 1)}\\%",
                                _fmt_pm(row.get("full_nll_gain", {})),
                                _fmt_pm(row.get("full_utility_gain", {})),
                                _fmt_pm(row.get("flagged_utility_gain", {})),
                            ]
                        )
                        + r" \\"
                    )
                lines.append(r"\addlinespace")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    audit = benchmark.get("statistical_audit") or results.get("statistical_audit")
    if audit:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{Mean paired-seed gains over SeqTXCount on MovieLens. Positive NLL gain means lower NLL for the target model. Positive utility gain means higher utility. NLL wins count seeds with positive NLL gains out of five. These headline-sweep top-slice gains use the full five-seed audit aggregation. Robustness Tables~\ref{tab:movie-proxy-robustness} and~\ref{tab:movie-threshold-robustness} use separate Flow-only selector-sweep aggregations.}")
        lines.append(r"\label{tab:movie-paired-ci}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{lllrrr}")
        lines.append(r"\toprule")
        lines.append(r"Config & Slice & Target & NLL gain & Utility gain & NLL wins/5 \\")
        lines.append(r"\midrule")
        for cfg in configs:
            for scope in ["overall", "high", "top20", "top10"]:
                if scope not in audit[cfg]:
                    continue
                for target in ["SeqTXCountPM", "SeqTXGlobal", "ANP", "FlowLatent"]:
                    if target not in audit[cfg][scope]:
                        continue
                    nll = audit[cfg][scope][target]["nll_gain"]
                    util = audit[cfg][scope][target]["utility_gain"]
                    wins = f"{nll['wins']}/{nll['n']}"
                    lines.append(
                        " & ".join(
                            [
                                CONFIG_LABELS.get(cfg, cfg),
                                {"overall": "All", "high": "High", "top20": "Top20", "top10": "Top10"}.get(scope, scope),
                                label.get(target, target),
                                _fmt(_mean_value(nll)),
                                _fmt(_mean_value(util)),
                                wins,
                            ]
                        )
                        + r" \\"
                    )
            lines.append(r"\addlinespace")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    signed = benchmark.get("signed_covariance_audit") or results.get("signed_covariance_audit")
    proxies = benchmark.get("diagnostic_proxy_audit") or results.get("diagnostic_proxy_audit")
    if signed or proxies:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{MovieLens diagnostic checks. Signed covariances are reported on the high-TaxScore slice. Correlations compare TaxScore and the finite-ratio proxy $n^2/N_{\rm user}$ against the ANP NLL gains measured separately in the low-, medium-, and high-score diagnostic bins.}")
        lines.append(r"\label{tab:movie-diagnostic-audit}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{lrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Config & $S_+$ & $S_-$ & $S_+/n(n-1)$ & neg. pair frac. & corr(TaxScore, NLL gain) & corr($n^2/N$, NLL gain) \\")
        lines.append(r"\midrule")
        for cfg in configs:
            sp = sm = neg = 0.0
            if signed and cfg in signed:
                high = signed[cfg].get("high", signed[cfg].get("overall", {}))
                sp = _mean_value(high.get("positive_cov_sum", {}).get("mean", 0.0))
                sm = _mean_value(high.get("negative_cov_sum", {}).get("mean", 0.0))
                neg = _mean_value(high.get("negative_pair_fraction", {}).get("mean", 0.0))
            tcorr = fcorr = 0.0
            if proxies and cfg in proxies:
                tcorr = proxies[cfg].get("ANP", {}).get("taxscore_nll_gain_corr", 0.0)
                fcorr = proxies[cfg].get("ANP", {}).get("finite_ratio_nll_gain_corr", 0.0)
            n_map = {"crit40": 40, "crit80": 80, "ml10n120c60": 120}
            n_val = n_map.get(cfg, 0)
            norm_sp = sp / (n_val * (n_val - 1)) if n_val > 1 else 0.0
            lines.append(
                f"{CONFIG_LABELS.get(cfg, cfg)} & {_fmt(sp, 1)} & {_fmt(sm, 1)} & {_fmt(norm_sp, 3)} & {_fmt(neg, 3)} & {_fmt(tcorr, 2)} & {_fmt(fcorr, 2)} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    robust = results.get("movielens_proxy_robustness") or benchmark.get("movielens_proxy_robustness")
    if robust:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{MovieLens candidate-selection robustness. Variants perturb candidate budget $B$, prior-margin cutoff $m$, and prefix-ambiguity weight $w$. The subsequent grid-based TaxScore is unchanged. Hidden ratings, hidden positives, compared-head predictions, and test labels never enter candidate selection or scoring. Gains compare FlowLatent with SeqTXCount. Top-slice gains assess enrichment. Bin correlation is supplementary and can be negative, especially on the 1,200-user ML10M subset. The sign of switch utility additionally depends on action changes.}")
        lines.append(r"\label{tab:movie-proxy-robustness}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{llrrr}")
        lines.append(r"\toprule")
        lines.append(r"Setting & Variant & corr(bin, NLL gain) & Top-20 NLL/util. gain & Switch util. gain \\")
        lines.append(r"\midrule")
        def setting_key(item):
            _, row = item
            ds = str(row.get("dataset", ""))
            n = int(_mean_value(row.get("hidden_n", 0)))
            return (ds, n, _variant_label(row))
        for cfg, row in sorted(robust.items(), key=setting_key):
            ds = str(row.get("dataset", ""))
            n = int(_mean_value(row.get("hidden_n", 0)))
            setting = "ML10M-120" if "10" in ds or n >= 100 else "ML1M-80"
            max_users = int(_mean_value(row.get("max_users", -1)))
            if max_users > 0:
                setting = f"{setting} ({max_users} users)"
            lines.append(
                " & ".join(
                    [
                        setting,
                        _variant_label(row),
                        _fmt(_mean_value(row.get("corr_proxy_bin_nll_gain")), 2),
                        f"{_fmt(_mean_value(row.get('top20_nll_gain')), 3)}/{_fmt(_mean_value(row.get('top20_utility_gain')), 3)}",
                        _fmt(_mean_value(row.get("full_switch_utility_gain")), 3),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    threshold_robust = results.get("movielens_threshold_robustness") or benchmark.get("movielens_threshold_robustness")
    if threshold_robust:
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{ML1M-80 threshold robustness. The selector and model pair are fixed while the tail threshold $c$ changes. Gains compare FlowLatent with SeqTXCount. Positive values favor FlowLatent.}")
        lines.append(r"\label{tab:movie-threshold-robustness}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{4.0pt}")
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Threshold $c$ & corr(bin, NLL gain) & Top-20 NLL/util. gain & Switch util. gain \\")
        lines.append(r"\midrule")

        def threshold_key(item):
            cfg, row = item
            return (_mean_value(row.get("c", 0.0)), cfg)

        for cfg, row in sorted(threshold_robust.items(), key=threshold_key):
            c_val = _mean_value(row.get("c", 0.0))
            lines.append(
                " & ".join(
                    [
                        _fmt(c_val, 2),
                        _fmt(_mean_value(row.get("corr_proxy_bin_nll_gain")), 2),
                        f"{_fmt(_mean_value(row.get('top20_nll_gain')), 3)}/{_fmt(_mean_value(row.get('top20_utility_gain')), 3)}",
                        _fmt(_mean_value(row.get("full_switch_utility_gain")), 3),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    capacity = benchmark.get("capacity_audit") or results.get("capacity_audit")
    if capacity and "crit80" in capacity:
        cfg = "crit80"
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{Capacity and compute audit for the MovieLens $(k,n,c)=(2,80,0.7)$ setting. Parameter-matched direct count heads yield smaller high-TaxScore tail-calibration gains than shared-latent posterior heads. Train sec/run is the rounded per-seed training wall-clock segment recorded by the experiment script, so small equalities should be read at that reporting granularity.}")
        lines.append(r"\label{tab:movie-capacity-audit}")
        lines.append(r"\scriptsize")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Model & Params & Train sec/run & Samples / ensemble members & High NLL \\")
        lines.append(r"\midrule")
        high = benchmark["configs"][cfg]["bins"]["high"]["metrics"]
        for model in ["SeqTXCount", "SeqTXCountPM", "SeqTXGlobal", "TwoTowerEnsMean", "TwoTowerEnsMix", "ANP", "FlowLatent"]:
            if model not in capacity[cfg] or model not in high:
                continue
            rec = capacity[cfg][model]
            params = int(_mean_value(rec["params"]))
            seconds = _mean_value(rec["training_seconds"])
            mc = max(_mean_value(rec.get("mc_samples", 0.0)), _mean_value(rec.get("rollouts", 0.0)), _mean_value(rec.get("ensemble_members", 0.0)))
            lines.append(
                f"{label.get(model, model)} & {params:,} & {_fmt(seconds, 1)} & {_fmt(mc, 0)} & {_fmt_pm(high[model]['nll'])} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    batch = results.get("batch_inspection_benchmark")
    if batch:
        bins = batch.get("aggregate_taxscore_bins", {})
        lines.append(r"\begin{paperblock}")
        lines.append(r"\centering")
        lines.append(r"\captionof{table}{Semi-synthetic quality-control batch inspection. Real UCI Breast Cancer features form item pools, while a shared latent risk controls batch defect prevalence. This mechanism check reports NLL and event-TV, with smaller values indicating more accurate aggregate posteriors.}")
        lines.append(r"\label{tab:batch-inspection}")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\begin{tabular}{llrr}")
        lines.append(r"\toprule")
        lines.append(r"TaxScore bin & Interface & NLL & Event-TV \\")
        lines.append(r"\midrule")
        for bin_name, label_name in [("low", "Low"), ("mid", "Mid"), ("high", "High")]:
            if bin_name not in bins:
                continue
            metrics = bins[bin_name]["metrics"]
            for model, iface in [("OracleIndep", "indep. oracle"), ("OracleShared", "shared oracle"), ("ObjMLP", "object head")]:
                if model not in metrics:
                    continue
                rec = metrics[model]
                lines.append(f"{label_name} & {iface} & {_fmt_pm(rec['nll'])} & {_fmt_pm(rec['event_tv'])} \\\\")
            lines.append(r"\addlinespace")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{paperblock}")
        lines.append("")

    # Short tables stay intact. The long paired audit repeats its header across pages.
    content = "\n".join(lines)
    blocks = content.split(r"\begin{paperblock}")
    rebuilt = [blocks[0]]
    for block in blocks[1:]:
        if r"\label{tab:movie-paired-ci}" in block:
            caption = block.split(r"\captionof{table}{", 1)[1].split("\n", 1)[0][:-1]
            body = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
            header = r"Config & Slice & Target & NLL gain & Utility gain & NLL wins/5 \\"
            rebuilt.append("\n".join([
                r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
                r"\setlength{\LTcapwidth}{\linewidth}",
                r"\begin{longtable}{lllrrr}",
                r"\caption{\normalsize " + caption + r"}\label{tab:movie-paired-ci}\\",
                r"\toprule", header, r"\midrule\endfirsthead",
                r"\multicolumn{6}{l}{Table~\thetable\ (continued)}\\",
                r"\toprule", header, r"\midrule\endhead",
                r"\midrule\multicolumn{6}{r}{Continued on the next page}\\\endfoot",
                r"\bottomrule\endlastfoot", body,
                r"\end{longtable}\endgroup", "",
            ]))
        else:
            rebuilt.append(r"\begin{paperblock}" + "\n" + r"\begin{minipage}{0.985\linewidth}"
                           + block.replace(r"\end{paperblock}",
                                           r"\end{minipage}" + "\n" + r"\end{paperblock}"))
    content = "".join(rebuilt)
    Path("figures/movielens_audit_tables.tex").write_text(content, encoding="utf-8")


def plot_taxscore_gain_correlation(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None:
        return
    corr = benchmark.get("taxscore_gain_correlation")
    if not corr:
        return
    configs = [name for name in MOVIE_AUDIT_CONFIGS if name in corr]
    if not configs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.75))
    for target, ax, metric, ylabel in [
        ("ANP", axes[0], "nll_gain", "NLL gain over SeqTX"),
        ("ANP", axes[1], "utility_gain", "Utility gain over SeqTX"),
    ]:
        for cfg_name in configs:
            rows = corr[cfg_name].get("rows", [])
            x, y, yerr, sizes = [], [], [], []
            for row in rows:
                if target not in row:
                    continue
                x.append(_mean_value(row.get("mean_taxscore")))
                y.append(_mean_value(row[target].get(metric)))
                yerr.append(_std_value(row[target].get(metric)))
                sizes.append(26 + 0.10 * _mean_value(row.get("count")))
            if x:
                color = COLORS.get(target, "#333333") if cfg_name == configs[0] else "#444444"
                ax.errorbar(x, y, yerr=yerr, linewidth=1.7, capsize=2, color=color, alpha=0.72)
                ax.scatter(x, y, s=sizes, marker="o", color=color, edgecolor="white", linewidth=0.5, label=CONFIG_LABELS.get(cfg_name, cfg_name), zorder=3)
        ax.axhline(0.0, color="#444444", linewidth=1)
        ax.set_xlabel("Mean TaxScore bin")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title("TaxScore predicts calibration gains")
    axes[1].set_title("Decision gains concentrate in high bins")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/taxscore_gain_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_movielens_reliability(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None:
        return
    diagnostics = benchmark.get("calibration_diagnostics")
    if not diagnostics:
        return
    cfg_name = "crit80" if "crit80" in diagnostics else next(iter(diagnostics))
    models = [m for m in ["SeqTXCount", "SeqTXCountPM", "TwoTowerEnsMean", "TwoTowerEnsMix", "ANP", "FlowLatent"] if m in diagnostics[cfg_name]]
    if not models:
        return
    fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.0))
    for model in models:
        rows = diagnostics[cfg_name][model]["reliability_curve"]
        conf = [_mean_value(row["confidence"]) for row in rows if _mean_value(row["count"]) > 0]
        acc = [_mean_value(row["accuracy"]) for row in rows if _mean_value(row["count"]) > 0]
        if conf:
            ax.plot(conf, acc, marker=MARKERS.get(model, "o"), linewidth=2.0, label=model, color=COLORS.get(model))
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted threshold probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"Reliability on {cfg_name}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/movielens_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    x = np.arange(len(models))
    rel = [_mean_value(diagnostics[cfg_name][m]["brier_decomposition"]["reliability"]) for m in models]
    res = [_mean_value(diagnostics[cfg_name][m]["brier_decomposition"]["resolution"]) for m in models]
    ax.bar(x - 0.18, rel, width=0.36, color="#D55E00", label="Reliability error")
    ax.bar(x + 0.18, res, width=0.36, color="#0072B2", label="Resolution")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Brier decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/movielens_brier_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_movielens_diagnostics_main(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None:
        return
    corr = benchmark.get("taxscore_gain_correlation")
    diagnostics = benchmark.get("calibration_diagnostics")
    if not corr or not diagnostics:
        return

    configs = [name for name in MOVIE_AUDIT_CONFIGS if name in corr]
    cfg_name = "crit80" if "crit80" in diagnostics else next(iter(diagnostics))
    models = [
        m
        for m in ["SeqTXCount", "SeqTXCountPM", "TwoTowerEnsMean", "TwoTowerEnsMix", "ANP", "FlowLatent"]
        if m in diagnostics[cfg_name]
    ]
    if not configs or not models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.45), gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]})
    for ax, metric, ylabel, title in [
        (axes[0], "nll_gain", "NLL gain over SeqTX", "TaxScore vs. NLL gain"),
        (axes[1], "utility_gain", "Utility gain over SeqTX", "TaxScore vs. utility gain"),
    ]:
        for idx, cfg in enumerate(configs):
            rows = corr[cfg].get("rows", [])
            x, y, yerr, sizes = [], [], [], []
            for row in rows:
                if "ANP" not in row:
                    continue
                x.append(_mean_value(row.get("mean_taxscore")))
                y.append(_mean_value(row["ANP"].get(metric)))
                yerr.append(_std_value(row["ANP"].get(metric)))
                sizes.append(18 + 0.08 * _mean_value(row.get("count")))
            if x:
                color = "#005AB5" if idx == 0 else ("#666666" if idx == 1 else "#222222")
                ax.errorbar(x, y, yerr=yerr, linewidth=1.35, capsize=2, color=color, alpha=0.78)
                ax.scatter(
                    x,
                    y,
                    s=sizes,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    label=CONFIG_LABELS.get(cfg, cfg),
                    zorder=3,
                )
        ax.axhline(0.0, color="#444444", linewidth=0.9)
        ax.set_xlabel("Mean TaxScore bin", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7.5)

    ax = axes[2]
    for model in models:
        rows = diagnostics[cfg_name][model]["reliability_curve"]
        conf = [_mean_value(row["confidence"]) for row in rows if _mean_value(row["count"]) > 0]
        acc = [_mean_value(row["accuracy"]) for row in rows if _mean_value(row["count"]) > 0]
        if conf:
            ax.plot(
                conf,
                acc,
                marker=MARKERS.get(model, "o"),
                linewidth=1.65,
                markersize=4,
                label=model,
                color=COLORS.get(model),
            )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Predicted threshold probability", fontsize=8)
    ax.set_ylabel("Empirical frequency", fontsize=8)
    ax.set_title(f"Reliability on {CONFIG_LABELS.get(cfg_name, cfg_name)}", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")

    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout(w_pad=0.85)
    fig.savefig("figures/movielens_diagnostics_main.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_covtype_ambiguity_main(results):
    threshold = results.get("covtype_threshold_benchmark")
    if threshold is None or "aggregate_ambiguity_rows" not in threshold:
        return
    cov_rows = threshold["aggregate_ambiguity_rows"]
    if not cov_rows:
        return
    cov_row = max(cov_rows, key=lambda item: item["hidden_n"])
    fig, ax = plt.subplots(1, 1, figsize=(3.55, 2.25))
    xs = np.asarray([0, 1], dtype=np.float64)
    labels = [r"low $v(O)$", r"high $v(O)$"]
    cov_models = [model for model in ["IndepPB", "ObjSetTF"] if model in cov_row["slice_metrics"]["high_v"]]
    cov_colors = {"IndepPB": "#777777", "ObjSetTF": "#D55E00"}
    cov_markers = {"IndepPB": "o", "ObjSetTF": "s"}
    for model in cov_models:
        vals = [
            cov_row["slice_metrics"]["low_v"][model]["nll"]["mean"],
            cov_row["slice_metrics"]["high_v"][model]["nll"]["mean"],
        ]
        errs = [
            cov_row["slice_metrics"]["low_v"][model]["nll"]["std"],
            cov_row["slice_metrics"]["high_v"][model]["nll"]["std"],
        ]
        ax.errorbar(
            xs,
            vals,
            yerr=errs,
            marker=cov_markers.get(model, "o"),
            linewidth=1.75,
            capsize=2,
            markersize=4.8,
            color=cov_colors.get(model),
            label=model,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NLL", fontsize=8.5)
    ax.set_title("Covtype ambiguity split", fontsize=9.5)
    ax.text(
        0.05,
        0.94,
        f"$n={int(cov_row['hidden_n'])}$",
        transform=ax.transAxes,
        va="top",
        fontsize=7.8,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7.0, loc="upper center", bbox_to_anchor=(0.56, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/covtype_ambiguity_main.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_capacity_and_nonbinary(results):
    benchmark = results.get("movielens_tail_benchmark")
    if benchmark is None:
        return
    audit = benchmark.get("capacity_audit")
    configs = benchmark.get("configs", {})
    if audit:
        cfg_name = "crit80" if "crit80" in audit and "crit80" in configs else next(iter(audit))
        high = configs.get(cfg_name, {}).get("bins", {}).get("high", {}).get("metrics", {})
        models = [m for m in ["SeqTXCount", "SeqTXCountPM", "TwoTowerEnsMean", "TwoTowerEnsMix", "SeqTXGlobal", "ANP", "FlowLatent"] if m in audit[cfg_name] and m in high]
        if models:
            fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.0))
            params = [_mean_value(audit[cfg_name][m]["params"]) for m in models]
            nll = [high[m]["nll"]["mean"] for m in models]
            train_seconds = [_mean_value(audit[cfg_name][m].get("train_seconds"), 1.0) for m in models]
            max_seconds = max(train_seconds) if train_seconds else 1.0
            for model, px, py in zip(models, params, nll):
                size = 45 + 95 * (_mean_value(audit[cfg_name][model].get("train_seconds"), 1.0) / max_seconds)
                ax.scatter(px, py, s=size, color=COLORS.get(model, "#333333"), marker=MARKERS.get(model, "o"), alpha=0.86, edgecolor="white", linewidth=0.55, label=model)
            ax.set_xscale("log")
            ax.set_xlabel("Trainable parameters")
            ax.set_ylabel("High-TaxScore tail NLL")
            ax.set_title("Pareto audit: capacity, compute, and tail NLL")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig("figures/capacity_audit.pdf", bbox_inches="tight")
            plt.close(fig)

    demo = benchmark.get("nonbinary_partition_demo")
    if demo:
        configs = [name for name in ["crit40", "crit80", "ml10n120c60", "crit80c75", "main"] if name in demo]
        if configs:
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
            width = 0.35
            x = np.arange(len(configs))
            mix = [_mean_value(demo[c]["GlobalBinMixture"]["nll"]) for c in configs]
            indep = [_mean_value(demo[c]["MeanIndependentBins"]["nll"]) for c in configs]
            axes[0].bar(x - width / 2, mix, width, color="#009E73", label="Shared bin mixture")
            axes[0].bar(x + width / 2, indep, width, color="#D55E00", label="Independent bins")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
            axes[0].set_ylabel("Partition-event NLL")
            axes[0].set_title("Non-binary rating-bin demo")
            axes[0].legend(frameon=False, fontsize=8)
            gap = [_mean_value(demo[c]["mean_laplace_gap"]) for c in configs]
            cov = [_mean_value(demo[c]["mean_high_bin_cov_gap"]) for c in configs]
            axes[1].plot(x, gap, marker="o", linewidth=2.0, label="Laplace proxy gap")
            axes[1].plot(x, cov, marker="s", linewidth=2.0, label="High-bin covariance gap")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
            axes[1].set_title("Empirical-measure consequences")
            axes[1].grid(alpha=0.25)
            axes[1].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig("figures/nonbinary_partition_demo.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_ambiguity_slices(results):
    threshold = results.get("covtype_threshold_benchmark")
    if threshold is None or "aggregate_ambiguity_rows" not in threshold:
        return

    rows = threshold["aggregate_ambiguity_rows"]
    row = max(rows, key=lambda item: item["hidden_n"])
    model_order = [
        model
        for model in ["IndepPB", "SeqGRUCount", "SeqTXCount", "ARRollout", "ObjMLP", "ObjSetTF"]
        if model in row["slice_metrics"]["high_v"]
    ]
    x = np.arange(len(model_order))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, metric, ylabel, title in [
        (axes[0], "nll", "NLL", "High-v versus low-v NLL"),
        (axes[1], "ece", "ECE", "High-v versus low-v calibration"),
    ]:
        low = [row["slice_metrics"]["low_v"][model][metric]["mean"] for model in model_order]
        high = [row["slice_metrics"]["high_v"][model][metric]["mean"] for model in model_order]
        low_err = [row["slice_metrics"]["low_v"][model][metric]["std"] for model in model_order]
        high_err = [row["slice_metrics"]["high_v"][model][metric]["std"] for model in model_order]
        ax.bar(x - width / 2, low, width, yerr=low_err, color="#9bb7d4", label="low $v(O)$")
        ax.bar(x + width / 2, high, width, yerr=high_err, color="#d46a5f", label="high $v(O)$")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].text(
        0.02,
        0.96,
        f"$n={int(row['hidden_n'])}$, high $nv\\approx {row['high_nv_mean']['mean']:.1f}$",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/ambiguity_slices.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    Path("figures").mkdir(exist_ok=True)
    plt.style.use("default")
    results = load_results()
    plot_scaling_law(results)
    plot_policy_tradeoff(results)
    plot_musk_policy_tradeoff(results)
    plot_birds_policy_tradeoff(results)
    plot_protein_policy_tradeoff(results)
    plot_threshold_risk(results)
    plot_active_learning(results)
    plot_oracle_decision_tax(results)
    plot_taxscore_gap_curve(results)
    plot_head_resolution(results)
    plot_movielens_tail(results)
    plot_taxscore_gain_correlation(results)
    plot_movielens_reliability(results)
    plot_movielens_diagnostics_main(results)
    plot_covtype_ambiguity_main(results)
    plot_capacity_and_nonbinary(results)
    write_movielens_audit_tables(results)
    plot_ambiguity_slices(results)


if __name__ == "__main__":
    main()
