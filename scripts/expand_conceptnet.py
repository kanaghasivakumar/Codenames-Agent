#!/usr/bin/env python3
"""
Expand ConceptNet Graph with Better Clue Words
===============================================
Queries ConceptNet API for intuitive clue words and adds
connections to board words.
"""

import json
import urllib.request
import urllib.error
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_DIR / "data" / "conceptnet_graph.json"
COMMON_WORDS_PATH = PROJECT_DIR / "data" / "common_words.txt"
CODENAMES_WORDS_PATH = PROJECT_DIR / "data" / "codenames_words.txt"

# ConceptNet API
CONCEPTNET_API = "https://api.conceptnet.io"

# Intuitive clue words that humans commonly use in Codenames
INTUITIVE_CLUES = [
    # Categories
    "mammal", "reptile", "insect", "predator", "prey", "pet", "wild",
    "continent", "nation", "capital", "european", "asian", "african",
    "celestial", "lunar", "solar", "cosmic", "orbital",
    "edible", "sweet", "sour", "bitter", "tasty", "delicious",
    "natural", "outdoor", "indoor", "underground", "underwater",

    # Actions
    "flying", "swimming", "running", "jumping", "climbing", "crawling",
    "hunting", "eating", "sleeping", "fighting", "playing", "working",
    "building", "breaking", "cutting", "burning", "freezing", "melting",

    # Properties
    "dangerous", "safe", "fast", "slow", "heavy", "light", "sharp", "dull",
    "loud", "quiet", "bright", "dark", "wet", "dry", "hard", "soft",
    "ancient", "modern", "old", "young", "alive", "dead",
    "round", "flat", "tall", "short", "thick", "thin",

    # Materials
    "wooden", "metallic", "plastic", "leather", "fabric", "ceramic",
    "golden", "silver", "bronze", "steel", "copper", "iron",

    # Colors (as clues)
    "crimson", "scarlet", "azure", "navy", "emerald", "golden",
    "white", "black", "grey", "brown", "purple", "pink",

    # Common associations
    "royal", "military", "medical", "musical", "magical", "mythical",
    "tropical", "arctic", "desert", "jungle", "forest", "marine",
    "domestic", "foreign", "eastern", "western", "northern", "southern",

    # Body parts (good for connecting)
    "wing", "tail", "horn", "claw", "teeth", "fur", "feather", "scale",

    # Time/Seasons
    "winter", "summer", "spring", "autumn", "seasonal", "annual", "daily",
    "morning", "evening", "midnight", "dawn", "dusk",

    # Size/Scale
    "tiny", "huge", "giant", "massive", "miniature", "enormous",

    # More intuitive connections
    "legendary", "famous", "secret", "hidden", "visible", "invisible",
    "natural", "artificial", "organic", "synthetic",
    "liquid", "solid", "gaseous", "frozen", "molten",
]


def load_data():
    """Load existing data files."""
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)

    with open(COMMON_WORDS_PATH, "r") as f:
        common_words = set(w.strip().lower() for w in f if w.strip())

    with open(CODENAMES_WORDS_PATH, "r") as f:
        board_words = [w.strip().lower() for w in f if w.strip()]

    return graph, common_words, board_words


def query_conceptnet(word, limit=100):
    """Query ConceptNet API for a word's relationships."""
    url = f"{CONCEPTNET_API}/c/en/{word}?limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CodenamesAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"  Error querying {word}: {e}")
    return None


def query_conceptnet_related(word, limit=50):
    """Query ConceptNet for words related to a concept."""
    url = f"{CONCEPTNET_API}/related/c/en/{word}?limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CodenamesAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"  Error querying related {word}: {e}")
    return None


def extract_connections(data, board_words_set):
    """Extract connections to board words from ConceptNet response."""
    connections = []

    if not data or "edges" not in data:
        return connections

    for edge in data["edges"]:
        start = edge.get("start", {})
        end = edge.get("end", {})
        rel = edge.get("rel", {})
        weight = edge.get("weight", 1.0)

        start_word = start.get("label", "").lower() if isinstance(start, dict) else ""
        end_word = end.get("label", "").lower() if isinstance(end, dict) else ""
        relation = rel.get("label", "") if isinstance(rel, dict) else ""

        # Check if either end connects to a board word
        if start_word in board_words_set:
            connections.append({
                "board_word": start_word,
                "clue": end_word,
                "relation": relation,
                "weight": weight
            })
        elif end_word in board_words_set:
            connections.append({
                "board_word": end_word,
                "clue": start_word,
                "relation": relation,
                "weight": weight
            })

    return connections


def main():
    print("=" * 60)
    print("ConceptNet Graph Expander for Codenames")
    print("=" * 60)
    print()

    # Load data
    print("Loading existing data...")
    graph, common_words, board_words = load_data()
    board_words_set = set(board_words)

    print(f"  Graph entries: {len(graph)}")
    print(f"  Common words: {len(common_words)}")
    print(f"  Board words: {len(board_words)}")
    print()

    # Track new additions
    new_edges_added = 0
    new_common_words = set()

    # Query ConceptNet for each intuitive clue
    print(f"Querying ConceptNet for {len(INTUITIVE_CLUES)} intuitive clue words...")
    print()

    for i, clue in enumerate(INTUITIVE_CLUES):
        print(f"[{i+1}/{len(INTUITIVE_CLUES)}] Checking '{clue}'...", end=" ")

        # Query ConceptNet
        data = query_conceptnet(clue)
        time.sleep(0.3)  # Rate limiting

        if not data:
            print("no data")
            continue

        # Find connections to board words
        connections = extract_connections(data, board_words_set)

        if connections:
            print(f"found {len(connections)} connections")

            for conn in connections:
                board_word = conn["board_word"]
                relation = conn["relation"]
                weight = conn["weight"]

                # Add edge to graph
                if board_word in graph:
                    # Check if this edge already exists
                    existing = [e for e in graph[board_word] if e.get("end", "").lower() == clue]
                    if not existing:
                        graph[board_word].append({
                            "end": clue,
                            "relation": relation,
                            "normalized_weight": min(weight / 5.0, 1.0)  # Normalize
                        })
                        new_edges_added += 1
                        print(f"    + {board_word.upper()} → {clue} ({relation})")

                # Track if clue needs to be added to common words
                if clue not in common_words:
                    new_common_words.add(clue)
        else:
            print("no board connections")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"New edges added to graph: {new_edges_added}")
    print(f"New common words to add: {len(new_common_words)}")

    if new_common_words:
        print(f"  Words: {sorted(new_common_words)[:20]}...")

    # Save updated graph
    if new_edges_added > 0:
        print()
        print("Saving updated graph...")
        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  Saved to {GRAPH_PATH}")

    # Save new common words
    if new_common_words:
        print("Adding new common words...")
        with open(COMMON_WORDS_PATH, "a") as f:
            for word in sorted(new_common_words):
                f.write(f"\n{word}")
        print(f"  Added {len(new_common_words)} words to {COMMON_WORDS_PATH}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
