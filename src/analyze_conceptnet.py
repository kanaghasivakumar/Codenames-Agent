import json
from pathlib import Path
from collections import defaultdict
import argparse
import math
import statistics
import random

import matplotlib.pyplot as plt


def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_graph(graph):
    # graph: mapping from start -> list of edges with keys 'end','relation','weight'
    rel_weights = defaultdict(list)
    rel_entries = defaultdict(list)

    for start, edges in graph.items():
        for e in edges:
            rel = e.get("relation")
            w = e.get("normalized_weight")
            end = e.get("end")
            if rel is None or w is None:
                continue
            try:
                w = float(w)
            except Exception:
                continue
            rel_weights[rel].append(w)
            rel_entries[rel].append((start, end, w))

    stats = {}
    for rel, weights in rel_weights.items():
        mn = min(weights)
        mx = max(weights)
        mean = statistics.mean(weights) if weights else float("nan")
        stats[rel] = {"min": mn, "max": mx, "mean": mean, "count": len(weights)}

    return stats, rel_weights, rel_entries


def print_stats(stats):
    print("Relation statistics (min / mean / max) and counts:\n")
    for rel, s in sorted(stats.items(), key=lambda x: (-x[1]["count"], x[0])):
        print(f"{rel}: min={s['min']:.4g}, mean={s['mean']:.4g}, max={s['max']:.4g}  (n={s['count']})")


def print_examples(rel_entries, low_n=5, high_n=5):
    print("\nExamples of low- and high-weight edges per relation (random from bottom/top 5%):\n")
    for rel, entries in sorted(rel_entries.items(), key=lambda x: -len(x[1])):
        print(f"--- {rel} (n={len(entries)}) ---")
        sorted_by_w = sorted(entries, key=lambda t: t[2])
        # filter out self-edges where start == end (case-insensitive)
        filtered = [t for t in sorted_by_w if t[0].strip().lower() != t[1].strip().lower()]
        n = len(filtered)
        if n == 0:
            print("  (no non-self entries)\n")
            continue

        pool_k = max(1, int(math.ceil(0.05 * n)))
        low_pool = filtered[:pool_k]
        high_pool = filtered[-pool_k:]

        low_sample = low_pool if len(low_pool) <= low_n else random.sample(low_pool, low_n)
        high_sample = high_pool if len(high_pool) <= high_n else random.sample(high_pool, high_n)

        # sort samples for nicer display: low ascending, high descending
        low_sample = sorted(low_sample, key=lambda t: t[2])
        high_sample = sorted(high_sample, key=lambda t: t[2], reverse=True)

        print("  Low:")
        for a, b, w in low_sample:
            print(f"    {a} -> {b} : {w:.6g}")
        print("  High:")
        for a, b, w in high_sample:
            print(f"    {a} -> {b} : {w:.6g}")
        print()


def plot_distributions(rel_weights, out_path: Path, max_relations=None):
    rel_items = sorted(rel_weights.items(), key=lambda x: -len(x[1]))
    if max_relations:
        rel_items = rel_items[:max_relations]

    n = len(rel_items)
    if n == 0:
        print("No relations to plot.")
        return

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (rel, weights) in enumerate(rel_items):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        if len(weights) > 1:
            ax.hist(weights, bins=30, color="#4C72B0", edgecolor="black")
        else:
            ax.hist(weights, bins=1, color="#4C72B0", edgecolor="black")
        ax.set_title(f"{rel} (n={len(weights)})")
        ax.set_xlabel("weight")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)

    # turn off unused axes
    total_axes = nrows * ncols
    for empty_idx in range(n, total_axes):
        r = empty_idx // ncols
        c = empty_idx % ncols
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ConceptNet graph weights by relation")
    parser.add_argument("--path", default="data/conceptnet_graph.json", help="Path to conceptnet graph JSON")
    parser.add_argument("--out", default="plots/relation_weights.png", help="Output plot path")
    parser.add_argument("--max-relations", type=int, default=0, help="Limit number of relations to plot (0 = all)")
    parser.add_argument("--low-n", type=int, default=5, help="Number of low-weight examples to show per relation")
    parser.add_argument("--high-n", type=int, default=5, help="Number of high-weight examples to show per relation")
    args = parser.parse_args()

    graph_path = Path(args.path)
    out_path = Path(args.out)

    graph = load_graph(graph_path)
    stats, rel_weights, rel_entries = analyze_graph(graph)

    print_stats(stats)
    print_examples(rel_entries, low_n=args.low_n, high_n=args.high_n)

    max_rel = args.max_relations if args.max_relations > 0 else None
    plot_distributions(rel_weights, out_path, max_relations=max_rel)


if __name__ == "__main__":
    main()
