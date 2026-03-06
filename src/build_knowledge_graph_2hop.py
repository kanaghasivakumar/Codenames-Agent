import gzip
import csv
import json
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
CONCEPTNET_CSV = Path("/Users/aasrithagopalam/Downloads/Codenames-Agent/data/conceptnet-assertions-5.7.0.csv.gz")
CODENAMES_WORDS_FILE = Path("/Users/aasrithagopalam/Downloads/Codenames-Agent/data/codenames_words.txt")
COMMON_WORDS_FILE = Path("/Users/aasrithagopalam/Downloads/Codenames-Agent/data/common_words.txt")

# Output
OUTPUT_FILE = Path("/Users/aasrithagopalam/Downloads/Codenames-Agent/data/conceptnet_graph.json")

RELEVANT_RELATIONS = {
    "/r/IsA",
    "/r/AtLocation",
    "/r/PartOf",
    "/r/Antonym",
    "/r/UsedFor",
    "/r/DistinctFrom",
    "/r/HasProperty",
    "/r/SimilarTo",
    "/r/CapableOf",
    "/r/Causes",
    "/r/MadeOf",
    "/r/ReceivesAction",
    "/r/HasPrerequisite",
    "/r/HasSubevent",
    "/r/CreatedBy",
    "/r/LocatedNear",
}

def load_word_set(filepath):
    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def is_single_word(word):
    """Check if a word is a single word (no spaces)."""
    return " " not in word

def get_weight_alpha(rel):
    """Get weight normalization alpha for a relation."""
    if rel in ('IsA', 'AtLocation'):
        return 0.0
    elif rel in ('PartOf', 'UsedFor', 'HasProperty', 'SimilarTo', 'CapableOf',
                 'Causes', 'MadeOf', 'HasSubevent', 'CreatedBy', 'LocatedNear'):
        return 0.8
    elif rel in ('Antonym', 'DistinctFrom', 'ReceivesAction', 'HasPrerequisite'):
        return 0.5
    return 0.5

def build_graph():
    if not CONCEPTNET_CSV.exists():
        print(f"Error: {CONCEPTNET_CSV} not found.")
        return

    print("--- Building 2-Hop Knowledge Graph ---")
    print("Rules:")
    print("  - Single-word intermediates: stored as endpoints + used for 2-hop")
    print("  - Multi-word intermediates: only stored in 2-hop paths (as 'intermediate' field)")
    print("  - All single-word targets get connected via any path")

    # Load vocabulary
    game_words = load_word_set(CODENAMES_WORDS_FILE)
    common_words = load_word_set(COMMON_WORDS_FILE)
    allowed_vocab = game_words.union(common_words)

    # Target common words (the 160+ words we specifically added)
    target_words = set([
        "android", "animator", "avengers", "baguette", "barista", "beatbox",
        "beyonce", "bibimbap", "blogger", "bratwurst", "burrito", "cappuccino", "cardio",
        "cheetah", "churro", "croissant", "cupcake", "dimsum", "donut", "dragonfly",
        "dumbledore", "dumpling", "emoji", "enchilada", "facebook", "falafel", "firefighter",
        "firefly", "firetruck", "fondue", "foodie", "forklift", "fortnite", "frappe",
        "freya", "frodo", "gamer", "gandalf", "geek", "gelato", "gigabyte", "goalkeeper",
        "godzilla", "guacamole", "hairstyle", "harry potter", "hashtag", "hiphop", "hipster",
        "hobbit", "hogwarts", "hoodie", "hotdog", "hummus", "influencer", "instagram",
        "iphone", "ironman", "jedi", "jetski", "karaoke", "katana", "kebab", "kimchi",
        "koala", "kungfu", "ladybug", "lamborghini", "lasagna", "latte", "linkedin",
        "logout", "loki", "lotr", "manicure", "masala", "megabyte", "midfielder",
        "milkshake", "minecraft", "minivan", "mocha", "naan", "nacho", "nerd", "netflix",
        "ninja", "nintendo", "origami", "pac man", "paella", "paypal", "pedicure",
        "photobomb", "pilates", "pita", "pixar", "pixie", "playstation", "podcast",
        "podcaster", "pokemon", "pretzel", "rambo", "ramen", "rapper", "reddit", "remix",
        "risotto", "rollerblades", "samosa", "sangria", "schnitzel", "seahorse", "selfie",
        "shawarma", "shrek", "simp", "simpsons", "sith", "skateboard", "skateboarding",
        "skype", "smartphone", "smoothie", "snapchat", "snowboarding", "sommelier",
        "spiderman", "spoiler", "star wars", "streamer", "strudel", "sumo",
        "surfboard", "swimsuit", "taekwondo", "tandoori", "tapas", "tempura", "teriyaki",
        "tetris", "triathlon", "twitter", "darth vader", "valkyrie", "vibe", "vlog",
        "waffle", "wasabi", "werewolf", "whatsapp", "wifi", "wikipedia", "wolverine",
        "wonderwoman", "wonton", "xbox", "yoda", "youtube",
        "arachnid", "carnivore", "condiment"
    ])

    print(f"\nLoaded: {len(game_words)} game words, {len(common_words)} common words")
    print(f"Target words to connect: {len(target_words)}")

    # Data structures
    # word -> set of (connected_word, relation, weight)
    all_connections = defaultdict(set)

    print("\nPass 1: Loading all ConceptNet connections...")

    with gzip.open(CONCEPTNET_CSV, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = 0

        for row in reader:
            rows += 1
            if rows % 5000000 == 0:
                print(f"   ...scanned {rows // 1000000}M rows")

            if len(row) < 5:
                continue

            rel = row[1]
            start_uri = row[2]
            end_uri = row[3]

            if not (start_uri.startswith("/c/en/") and end_uri.startswith("/c/en/")):
                continue

            if rel not in RELEVANT_RELATIONS:
                continue

            start_word = start_uri[6:].split("/")[0].replace("_", " ")
            end_word = end_uri[6:].split("/")[0].replace("_", " ")

            try:
                meta = json.loads(row[4])
                weight = meta.get('weight', 1.0)
            except:
                weight = 1.0

            clean_rel = rel.replace("/r/", "")

            # Store bidirectional connections
            all_connections[start_word].add((end_word, clean_rel, weight))
            all_connections[end_word].add((start_word, clean_rel, weight))

    print(f"\nPass 1 complete. Total words with connections: {len(all_connections)}")

    # Build the final graph
    print("\nBuilding graph with 2-hop paths...")

    knowledge_graph = {}
    for w in game_words:
        knowledge_graph[w] = []

    direct_edges = 0
    single_word_edges = 0
    two_hop_multiword = 0
    two_hop_singleword = 0
    targets_connected = set()

    # Track added edges to avoid duplicates
    seen_edges = defaultdict(set)  # game_word -> set of (end, intermediate)

    # For each game word
    for game_word in game_words:
        game_conns = all_connections.get(game_word, set())

        for connected, rel, weight in game_conns:
            alpha = get_weight_alpha(rel)
            normalized_weight = weight + alpha * (1 - weight)

            # Case 1: Direct connection to a single-word target
            if connected in target_words and is_single_word(connected):
                edge_key = (connected, None)
                if edge_key not in seen_edges[game_word]:
                    knowledge_graph[game_word].append({
                        "start": game_word,
                        "end": connected,
                        "relation": rel,
                        "weight": weight,
                        "normalized_weight": normalized_weight,
                        "path_type": "direct"
                    })
                    seen_edges[game_word].add(edge_key)
                    direct_edges += 1
                    targets_connected.add(connected)

            # Case 2: Single-word connection in allowed vocab -> endpoint
            elif is_single_word(connected) and connected in allowed_vocab:
                edge_key = (connected, None)
                if edge_key not in seen_edges[game_word]:
                    knowledge_graph[game_word].append({
                        "start": game_word,
                        "end": connected,
                        "relation": rel,
                        "weight": weight,
                        "normalized_weight": normalized_weight,
                        "path_type": "direct"
                    })
                    seen_edges[game_word].add(edge_key)
                    single_word_edges += 1

            # Case 3: Look for 2-hop paths through this intermediate to targets
            # This applies to ALL intermediates (single or multi-word)
            intermediate_conns = all_connections.get(connected, set())
            for target, rel2, weight2 in intermediate_conns:
                # Target must be a single-word target
                if target not in target_words or not is_single_word(target):
                    continue
                if target == game_word:
                    continue

                combined_weight = weight * weight2
                if combined_weight < 0.3:
                    continue

                # Determine if we store the intermediate
                if is_single_word(connected):
                    # Single-word intermediate: don't store it (it's already an endpoint above)
                    edge_key = (target, None)
                    if edge_key in seen_edges[game_word]:
                        continue

                    alpha2 = get_weight_alpha(rel2)
                    avg_alpha = (alpha + alpha2) / 2
                    norm_weight = combined_weight + avg_alpha * (1 - combined_weight)

                    knowledge_graph[game_word].append({
                        "start": game_word,
                        "end": target,
                        "relation": f"{rel}->{rel2}",
                        "via": connected,  # Note: this is a single word, shown for context
                        "weight": combined_weight,
                        "normalized_weight": norm_weight,
                        "path_type": "2-hop"
                    })
                    seen_edges[game_word].add(edge_key)
                    two_hop_singleword += 1
                    targets_connected.add(target)
                else:
                    # Multi-word intermediate: store it in the edge
                    edge_key = (target, connected)
                    if edge_key in seen_edges[game_word]:
                        continue

                    alpha2 = get_weight_alpha(rel2)
                    avg_alpha = (alpha + alpha2) / 2
                    norm_weight = combined_weight + avg_alpha * (1 - combined_weight)

                    knowledge_graph[game_word].append({
                        "start": game_word,
                        "end": target,
                        "relation": f"{rel}->{rel2}",
                        "intermediate": connected,  # Multi-word stored here
                        "weight": combined_weight,
                        "normalized_weight": norm_weight,
                        "path_type": "2-hop"
                    })
                    seen_edges[game_word].add(edge_key)
                    two_hop_multiword += 1
                    targets_connected.add(target)

    # Summary
    print(f"\n=== Build Complete ===")
    print(f"Direct edges to targets: {direct_edges}")
    print(f"Single-word endpoint edges: {single_word_edges}")
    print(f"2-hop via single-word intermediate: {two_hop_singleword}")
    print(f"2-hop via multi-word intermediate: {two_hop_multiword}")
    print(f"Target words connected: {len(targets_connected)}/{len(target_words)}")

    not_connected = target_words - targets_connected
    if not_connected:
        print(f"\nTargets NOT connected ({len(not_connected)}):")
        for w in sorted(not_connected):
            print(f"  {w}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_graph, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    build_graph()
