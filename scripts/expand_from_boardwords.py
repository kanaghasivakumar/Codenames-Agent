#!/usr/bin/env python3
"""
Expand Graph by Querying Board Words on ConceptNet
===================================================
Queries ConceptNet API for each board word to find additional clue connections.
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

CONCEPTNET_API = "https://api.conceptnet.io"

# Relations we care about for Codenames clues
GOOD_RELATIONS = {
    "IsA", "HasA", "PartOf", "HasProperty", "RelatedTo",
    "AtLocation", "UsedFor", "CapableOf", "Causes", "SymbolOf",
    "SimilarTo", "MadeOf", "DefinedAs", "Synonym"
}


def load_data():
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)
    with open(COMMON_WORDS_PATH, "r") as f:
        common_words = set(w.strip().lower() for w in f if w.strip())
    with open(CODENAMES_WORDS_PATH, "r") as f:
        board_words = [w.strip().lower() for w in f if w.strip()]
    return graph, common_words, board_words


def query_conceptnet(word, limit=50, retries=3):
    """Query ConceptNet API with retry logic."""
    url = f"{CONCEPTNET_API}/c/en/{word}?limit={limit}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CodenamesAI/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 502 and attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def extract_clues(data, board_word, existing_clues):
    """Extract potential clue words from ConceptNet response."""
    new_clues = []

    if not data or "edges" not in data:
        return new_clues

    board_lower = board_word.lower()

    for edge in data["edges"]:
        rel = edge.get("rel", {})
        relation = rel.get("label", "") if isinstance(rel, dict) else ""

        if relation not in GOOD_RELATIONS:
            continue

        start = edge.get("start", {})
        end = edge.get("end", {})
        weight = edge.get("weight", 1.0)

        start_word = start.get("label", "").lower() if isinstance(start, dict) else ""
        end_word = end.get("label", "").lower() if isinstance(end, dict) else ""

        # Find the clue word (the one that's not the board word)
        if start_word == board_lower:
            clue = end_word
        elif end_word == board_lower:
            clue = start_word
        else:
            continue

        # Validate clue
        if not clue or not clue.isalpha() or len(clue) < 3:
            continue
        if clue == board_lower:
            continue
        if clue in existing_clues:
            continue

        new_clues.append({
            "clue": clue,
            "relation": relation,
            "weight": weight
        })

    return new_clues


def main():
    print("=" * 60)
    print("Expanding Graph from Board Words via ConceptNet")
    print("=" * 60)
    print()

    graph, common_words, board_words = load_data()

    print(f"Graph entries: {len(graph)}")
    print(f"Common words: {len(common_words)}")
    print(f"Board words: {len(board_words)}")
    print()

    # Test API first
    print("Testing ConceptNet API...")
    test = query_conceptnet("apple")
    if not test:
        print("  ConceptNet API is unavailable. Exiting.")
        return
    print("  API is working!")
    print()

    new_edges = 0
    new_common = set()

    # Sample 50 board words to expand
    import random
    sample = random.sample(board_words, min(50, len(board_words)))

    print(f"Querying ConceptNet for {len(sample)} board words...")
    print()

    for i, word in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] {word.upper()}...", end=" ", flush=True)

        data = query_conceptnet(word)
        time.sleep(0.5)  # Rate limiting

        if not data:
            print("no data")
            continue

        # Get existing clues for this word
        existing = set(e.get("end", "").lower() for e in graph.get(word, []))

        # Extract new clues
        clues = extract_clues(data, word, existing)

        if clues:
            print(f"found {len(clues)} new clues")
            for c in clues[:3]:  # Add top 3
                if word in graph:
                    graph[word].append({
                        "end": c["clue"],
                        "relation": c["relation"],
                        "normalized_weight": min(c["weight"] / 5.0, 1.0)
                    })
                    new_edges += 1

                    if c["clue"] not in common_words:
                        new_common.add(c["clue"])
        else:
            print("no new clues")

    print()
    print("=" * 60)
    print(f"New edges added: {new_edges}")
    print(f"New common words: {len(new_common)}")

    if new_edges > 0:
        print("\nSaving graph...")
        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)

    if new_common:
        print(f"Adding {len(new_common)} new common words...")
        with open(COMMON_WORDS_PATH, "a") as f:
            for w in sorted(new_common):
                f.write(f"\n{w}")

    print("\nDone!")


if __name__ == "__main__":
    main()
