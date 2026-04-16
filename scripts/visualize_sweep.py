#!/usr/bin/env python3
"""Visualize results from a vlm_only sweep directory.

Reads results.json / oracle.json / summary.json from a sweep and produces
a set of figures saved under ``<sweep_dir>/figures/``.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASKET_RADIUS = 0.10  # from env.py (used for success threshold reference)


def load_sweep(sweep_dir: Path):
    results = json.loads((sweep_dir / "results.json").read_text())
    summary = json.loads((sweep_dir / "summary.json").read_text())
    oracle_path = sweep_dir / "oracle.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else []
    config = json.loads((sweep_dir / "config.json").read_text())
    return results, summary, oracle, config


def fig_success_and_distance(results, summary, oracle, out_path):
    budgets = sorted(int(b) for b in summary.keys())
    success_rates = [summary[str(b)]["success_rate"] for b in budgets]
    mean_dists = [summary[str(b)]["mean_distance_to_basket"] for b in budgets]

    oracle_sr = (sum(e["success"] for e in oracle) / len(oracle)) if oracle else None
    oracle_md = (sum(e["distance_to_basket"] for e in oracle) / len(oracle)) if oracle else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    bars = ax.bar([str(b) for b in budgets], success_rates,
                  color=["#95a5a6", "#3498db", "#2ecc71"][: len(budgets)],
                  alpha=0.85, edgecolor="black")
    for bar, sr in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, sr + 0.02, f"{sr:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    if oracle_sr is not None:
        ax.axhline(oracle_sr, linestyle="--", color="#e67e22", linewidth=2,
                   label=f"oracle = {oracle_sr:.0%}")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Probe budget")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs probe budget")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    # Box plot of distances, per budget
    data_per_budget = [
        [e["distance_to_basket"] for e in results if e["budget"] == b]
        for b in budgets
    ]
    bp = ax.boxplot(data_per_budget, tick_labels=[str(b) for b in budgets],
                    patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    colors = ["#95a5a6", "#3498db", "#2ecc71"][: len(budgets)]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # Scatter individual points
    for i, vals in enumerate(data_per_budget):
        xs = np.random.normal(i + 1, 0.05, size=len(vals))
        succ = [e["success"] for e in results if e["budget"] == budgets[i]]
        for x, y, s in zip(xs, vals, succ):
            ax.scatter(x, y, color="#27ae60" if s else "#c0392b",
                       s=28, alpha=0.85, edgecolor="black", linewidth=0.4,
                       zorder=5)
    if oracle_md is not None:
        ax.axhline(oracle_md, linestyle="--", color="#e67e22", linewidth=2,
                   label=f"oracle mean = {oracle_md:.3f}m")
    ax.axhline(BASKET_RADIUS, linestyle=":", color="black", linewidth=1,
               label=f"basket radius ({BASKET_RADIUS}m)")
    ax.set_xlabel("Probe budget")
    ax.set_ylabel("Distance to basket (m)")
    ax.set_title("Distance distribution (green=hit, red=miss)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_per_object_heatmap(results, oracle, out_path):
    budgets = sorted({e["budget"] for e in results})
    objects = sorted({e["object"] for e in results})

    mat = np.full((len(objects), len(budgets)), np.nan)
    succ = np.zeros_like(mat, dtype=bool)
    for e in results:
        i = objects.index(e["object"])
        j = budgets.index(e["budget"])
        mat[i, j] = e["distance_to_basket"]
        succ[i, j] = e["success"]

    oracle_dist = {e["object"]: e["distance_to_basket"] for e in oracle}

    fig_w = 1.2 * len(budgets) + (2.5 if oracle_dist else 0) + 2.5
    fig_h = 0.35 * len(objects) + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cols = list(budgets)
    data = mat.copy()
    extra_col = oracle_dist and len(oracle_dist) > 0
    if extra_col:
        oracle_col = np.array([oracle_dist.get(o, np.nan) for o in objects]).reshape(-1, 1)
        data = np.hstack([data, oracle_col])
        cols = cols + ["oracle"]

    vmax = np.nanmax(data)
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=vmax, aspect="auto")

    for i in range(len(objects)):
        for j in range(len(cols)):
            v = data[i, j]
            if np.isnan(v):
                continue
            marker = ""
            if j < len(budgets) and succ[i, j]:
                marker = " \u2713"  # check
            ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center",
                    color="black", fontsize=8)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"b={c}" if isinstance(c, int) else c for c in cols])
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects, fontsize=8)
    ax.set_title("Distance to basket per (object, budget)  — \u2713 = hit")
    fig.colorbar(im, ax=ax, label="distance (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_throw_params(results, oracle, out_path):
    budgets = sorted({e["budget"] for e in results})
    colors = {0: "#95a5a6", 3: "#3498db", 5: "#2ecc71"}
    default_palette = ["#95a5a6", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, b in enumerate(budgets):
        xs, ys, succ = [], [], []
        for e in results:
            if e["budget"] != b or e["aborted"]:
                continue
            xs.append(e["final_throw"]["theta"])
            ys.append(e["final_throw"]["v"])
            succ.append(e["success"])
        c = colors.get(b, default_palette[k % len(default_palette)])
        for idx, (x, y, s) in enumerate(zip(xs, ys, succ)):
            if s:
                ax.scatter(x, y, facecolor=c, edgecolor=c,
                           s=120, linewidth=1.5, marker="o", alpha=0.9,
                           label=f"budget={b}" if idx == 0 else None)
            else:
                ax.scatter(x, y, color=c, s=80, linewidth=1.8, marker="x",
                           alpha=0.9,
                           label=f"budget={b}" if idx == 0 else None)

    # Oracle marker
    if oracle:
        th = [e["throw"]["theta"] for e in oracle]
        v = [e["throw"]["v"] for e in oracle]
        ax.scatter(th, v, marker="*", color="#e67e22", s=220, edgecolor="black",
                   linewidth=0.7, label="oracle", zorder=10)

    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9, loc="best")
    ax.set_xlabel(r"launch angle $\theta$ (deg)")
    ax.set_ylabel("launch speed v (m/s)")
    ax.set_title("VLM final throws (filled=hit, x=miss)  vs  oracle (orange star)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_probe_usage(results, out_path):
    # Budgets with probes
    probed = [e for e in results if e["budget"] > 0 and e["probe_sequence"]]
    if not probed:
        print("  (no probes used, skipping probe figure)")
        return

    # Overall usage
    total_counter = Counter()
    for e in probed:
        total_counter.update(e["probe_sequence"])

    # Usage per budget
    per_budget = defaultdict(Counter)
    for e in probed:
        per_budget[e["budget"]].update(e["probe_sequence"])

    # Position-based usage: which probe at step 1, 2, 3, ...
    max_seq_len = max(len(e["probe_sequence"]) for e in probed)
    pos_counter = [Counter() for _ in range(max_seq_len)]
    for e in probed:
        for i, p in enumerate(e["probe_sequence"]):
            pos_counter[i][p] += 1

    probes = sorted(total_counter.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(probes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    counts = [total_counter[p] for p in probes]
    bars = ax.bar(probes, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.2, str(c),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Times selected")
    ax.set_title("Total probe selections")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    positions = np.arange(max_seq_len)
    bottom = np.zeros(max_seq_len)
    for p, col in zip(probes, colors):
        counts_at_pos = np.array([pos_counter[i].get(p, 0) for i in range(max_seq_len)])
        ax.bar(positions, counts_at_pos, bottom=bottom, label=p, color=col,
               edgecolor="black", linewidth=0.3, alpha=0.85)
        bottom += counts_at_pos
    ax.set_xticks(positions)
    ax.set_xticklabels([f"step {i+1}" for i in range(max_seq_len)])
    ax.set_ylabel("Times selected")
    ax.set_title("Probe choice by decision step")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_budget_deltas(results, out_path):
    """Per-object distance curve over budgets: does probing help this object?"""
    budgets = sorted({e["budget"] for e in results})
    per_obj = defaultdict(dict)
    for e in results:
        per_obj[e["object"]][e["budget"]] = e["distance_to_basket"]

    objects = sorted(per_obj.keys())
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(objects)))
    for obj, c in zip(objects, cmap):
        xs = budgets
        ys = [per_obj[obj].get(b, np.nan) for b in budgets]
        ax.plot(xs, ys, marker="o", color=c, alpha=0.85, label=obj, linewidth=1.5)

    ax.axhline(BASKET_RADIUS, linestyle=":", color="black", linewidth=1,
               label=f"basket radius ({BASKET_RADIUS}m)")
    ax.set_xticks(budgets)
    ax.set_xlabel("Probe budget")
    ax.set_ylabel("Distance to basket (m)")
    ax.set_title("Per-object distance vs budget")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=str, help="Path to sweep output directory")
    parser.add_argument("--out-subdir", type=str, default="figures",
                        help="Sub-directory of sweep_dir to write figures into")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    assert sweep_dir.is_dir(), f"Not a directory: {sweep_dir}"
    out_dir = sweep_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    results, summary, oracle, config = load_sweep(sweep_dir)
    print(f"Loaded {len(results)} episodes from {sweep_dir}")
    print(f"Writing figures to {out_dir}")

    fig_success_and_distance(results, summary, oracle,
                             out_dir / "01_success_and_distance.png")
    fig_per_object_heatmap(results, oracle,
                           out_dir / "02_per_object_heatmap.png")
    fig_budget_deltas(results, out_dir / "03_per_object_curves.png")
    fig_throw_params(results, oracle, out_dir / "04_throw_params.png")
    fig_probe_usage(results, out_dir / "05_probe_usage.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
