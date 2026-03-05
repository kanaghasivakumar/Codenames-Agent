#!/usr/bin/env python3
"""
Parse ConceptNet Data Dump to Expand Graph
===========================================
Reads conceptnet-assertions-5.7.0.csv.gz to find more clue words
for Codenames board words.
"""

import gzip
import json
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DUMP_PATH = DATA_DIR / "conceptnet-assertions-5.7.0.csv.gz"
GRAPH_PATH = DATA_DIR / "conceptnet_graph.json"
COMMON_WORDS_PATH = DATA_DIR / "common_words.txt"
CODENAMES_WORDS_PATH = DATA_DIR / "codenames_words.txt"

# Relations we want for Codenames clues
GOOD_RELATIONS = {
    "IsA", "HasA", "PartOf", "HasProperty", "RelatedTo",
    "AtLocation", "UsedFor", "CapableOf", "Causes", "SymbolOf",
    "SimilarTo", "MadeOf", "DefinedAs", "Synonym", "DerivedFrom",
    "HasContext", "MannerOf", "InstanceOf", "Antonym"
}


def load_board_words():
    with open(CODENAMES_WORDS_PATH) as f:
        return set(w.strip().lower() for w in f if w.strip())


def load_common_words():
    with open(COMMON_WORDS_PATH) as f:
        return set(w.strip().lower() for w in f if w.strip())


def load_graph():
    with open(GRAPH_PATH) as f:
        return json.load(f)


def parse_conceptnet_dump(board_words, limit_rows=None):
    """
    Parse the ConceptNet CSV dump.

    Format: URI, relation, start, end, metadata_json
    Example: /a/[/r/IsA/,/c/en/shark/,/c/en/fish/]  /r/IsA  /c/en/shark  /c/en/fish  {"weight": 4.0, ...}
    """
    print(f"Parsing {DUMP_PATH}...")
    print(f"Looking for edges involving {len(board_words)} board words")
    print()

    new_edges = defaultdict(list)
    new_clue_words = set()
    rows_processed = 0
    edges_found = 0

    with gzip.open(DUMP_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            rows_processed += 1

            if limit_rows and rows_processed > limit_rows:
                break

            if rows_processed % 1000000 == 0:
                print(f"  Processed {rows_processed:,} rows, found {edges_found} relevant edges...")

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            uri, relation, start, end, metadata = parts[0], parts[1], parts[2], parts[3], parts[4]

            # Extract relation name
            rel_name = relation.split('/')[-1] if '/' in relation else relation
            if rel_name not in GOOD_RELATIONS:
                continue

            # Extract words (format: /c/en/word or /c/en/word/n)
            def extract_word(uri):
                if not uri.startswith('/c/en/'):
                    return None
                parts = uri.split('/')
                if len(parts) >= 4:
                    word = parts[3].lower().replace('_', ' ')
                    # Only single words, alphabetic
                    if ' ' not in word and word.isalpha() and len(word) >= 3:
                        return word
                return None

            start_word = extract_word(start)
            end_word = extract_word(end)

            if not start_word or not end_word:
                continue

            # Check if either word is a board word
            if start_word in board_words:
                board_word = start_word
                clue_word = end_word
            elif end_word in board_words:
                board_word = end_word
                clue_word = start_word
            else:
                continue

            # Skip self-references
            if board_word == clue_word:
                continue

            # Parse weight from metadata
            try:
                meta = json.loads(metadata)
                weight = meta.get('weight', 1.0)
            except:
                weight = 1.0

            # Add edge
            new_edges[board_word].append({
                "end": clue_word,
                "relation": rel_name,
                "weight": weight
            })
            new_clue_words.add(clue_word)
            edges_found += 1

    print(f"\nDone! Processed {rows_processed:,} rows")
    print(f"Found {edges_found} relevant edges")
    print(f"Found {len(new_clue_words)} unique clue words")

    return new_edges, new_clue_words


def main():
    print("=" * 60)
    print("ConceptNet Dump Parser for Codenames")
    print("=" * 60)
    print()

    if not DUMP_PATH.exists():
        print(f"ERROR: ConceptNet dump not found at {DUMP_PATH}")
        return

    # Load existing data
    board_words = load_board_words()
    common_words = load_common_words()
    graph = load_graph()

    print(f"Board words: {len(board_words)}")
    print(f"Common words: {len(common_words)}")
    print(f"Current graph edges: {sum(len(e) for e in graph.values())}")
    print()

    # Parse dump
    new_edges, new_clue_words = parse_conceptnet_dump(board_words)

    # Merge with existing graph
    print("\nMerging with existing graph...")
    edges_added = 0

    for board_word, edges in new_edges.items():
        if board_word not in graph:
            graph[board_word] = []

        # Get existing clues for this word
        existing_clues = set(e.get('end', '').lower() for e in graph[board_word])

        for edge in edges:
            clue = edge['end'].lower()
            if clue not in existing_clues:
                # Normalize weight
                norm_weight = min(edge['weight'] / 10.0, 1.0)
                graph[board_word].append({
                    "end": clue,
                    "relation": edge['relation'],
                    "normalized_weight": norm_weight
                })
                existing_clues.add(clue)
                edges_added += 1

    print(f"Added {edges_added} new edges to graph")

    # Find new common words to add
    new_common = new_clue_words - common_words - board_words
    print(f"New clue words not in common_words.txt: {len(new_common)}")

    # Save updated graph
    print("\nSaving updated graph...")
    with open(GRAPH_PATH, 'w') as f:
        json.dump(graph, f, indent=2)

    # Add new common words
    if new_common:
        print(f"Adding {len(new_common)} words to common_words.txt...")
        with open(COMMON_WORDS_PATH, 'a') as f:
            for word in sorted(new_common):
                f.write(f"\n{word}")

    # Final stats
    print()
    print("=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"Graph edges: {sum(len(e) for e in graph.values())}")
    print(f"Common words: {len(common_words) + len(new_common)}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
