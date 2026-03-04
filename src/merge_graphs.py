"""
Merge ConceptNet and Wikidata graphs into a unified knowledge graph.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONCEPTNET_GRAPH = PROJECT_ROOT / "data" / "conceptnet_graph.json"
WIKIDATA_GRAPH = PROJECT_ROOT / "data" / "wikidata_graph.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "unified_graph.json"


def load_graph(path):
    """Load a graph from JSON file."""
    if not path.exists():
        print(f"Warning: {path} not found")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def merge_graphs():
    """Merge ConceptNet and Wikidata graphs."""
    print("=== Merging Knowledge Graphs ===\n")

    # Load graphs
    conceptnet = load_graph(CONCEPTNET_GRAPH)
    wikidata = load_graph(WIKIDATA_GRAPH)

    print(f"ConceptNet: {len(conceptnet)} words")
    print(f"Wikidata: {len(wikidata)} words")

    # Merge into unified graph
    unified = {}

    # Add ConceptNet edges (mark source if not present)
    cn_edges = 0
    for word, edges in conceptnet.items():
        unified[word] = []
        for edge in edges:
            if "source" not in edge:
                edge["source"] = "conceptnet"
            unified[word].append(edge)
            cn_edges += 1

    # Add Wikidata edges
    wd_edges = 0
    for word, edges in wikidata.items():
        if word not in unified:
            unified[word] = []
        for edge in edges:
            # Avoid duplicate edges
            existing = {(e["relation"], e["end"]) for e in unified[word]}
            if (edge["relation"], edge["end"]) not in existing:
                unified[word].append(edge)
                wd_edges += 1

    # Save unified graph
    with open(OUTPUT_FILE, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Unified graph: {len(unified)} words")
    print(f"ConceptNet edges: {cn_edges}")
    print(f"Wikidata edges added: {wd_edges}")
    print(f"Total edges: {cn_edges + wd_edges}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_graphs()
