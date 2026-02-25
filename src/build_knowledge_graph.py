import gzip
import csv
import json
import os
from pathlib import Path

# --- Configuration ---
# Inputs
CONCEPTNET_CSV = Path("data/conceptnet-assertions-5.7.0.csv.gz")
CODENAMES_WORDS_FILE = Path("data/codenames_words.txt")
COMMON_WORDS_FILE = Path("data/common_words.txt")

# Output
OUTPUT_FILE = Path("data/conceptnet_graph.json")

# The Logical Relations we care about (ConceptNet uses specific IDs)
RELEVANT_RELATIONS = {
    "/r/IsA",
    "/r/UsedFor",
    "/r/AtLocation",
    "/r/HasProperty",
    "/r/PartOf",
    "/r/RelatedTo",
    "/r/Causes",
    "/r/CapableOf",
    "/r/Antonym",
    "/r/DistinctFrom"
}

def load_word_set(filepath):
    """
    Loads a set of words from a text file for fast O(1) lookups.
    Normalizes to lowercase.
    """
    if not filepath.exists():
        print(f"❌ Error: {filepath} not found.")
        return set()
    
    with open(filepath, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace, convert to lower
        return set(line.strip().lower() for line in f if line.strip())

def build_graph():
    """
    Streams the compressed ConceptNet CSV and filters for relevant edges.
    """
    # 1. Validation
    if not CONCEPTNET_CSV.exists():
        print(f"❌ Error: {CONCEPTNET_CSV} not found.")
        print("   Waiting for download to finish...")
        return

    print("--- Starting Knowledge Graph Build ---")

    # 2. Load Vocabulary
    game_words = load_word_set(CODENAMES_WORDS_FILE)
    common_words = load_word_set(COMMON_WORDS_FILE)
    
    # We allow an edge if it connects a Game Word to ANY allowed word (Game or Common)
    allowed_vocab = game_words.union(common_words)

    print(f"✅ Loaded Vocabulary: {len(game_words)} game words, {len(common_words)} common words.")
    print("⏳ Parsing ConceptNet CSV (this will take 2-5 minutes)...")

    knowledge_graph = {}
    
    # Initialize empty entries for all game words (ensures no KeyErrors later)
    for w in game_words:
        knowledge_graph[w] = []

    # 3. Stream the CSV (Efficient Memory Usage)
    # The file is Tab-Separated. Structure: URI, Relation, Start, End, JSON_Metadata
    try:
        with gzip.open(CONCEPTNET_CSV, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            rows_processed = 0
            edges_kept = 0
            
            for row in reader:
                rows_processed += 1
                if rows_processed % 1000000 == 0:
                    print(f"   ...scanned {rows_processed // 1000000}M rows (Kept {edges_kept} edges)")

                # Basic validation: Must have at least 5 columns
                if len(row) < 5: continue
                
                rel = row[1]
                start_uri = row[2]
                end_uri = row[3]
                
                # --- FILTER 1: English Only ---
                if not (start_uri.startswith("/c/en/") and end_uri.startswith("/c/en/")):
                    continue
                
                # --- FILTER 2: Relevant Relation ---
                if rel not in RELEVANT_RELATIONS:
                    continue

                # Extract raw words (remove "/c/en/" prefix)
                # ConceptNet uses underscores for spaces (e.g., "new_york")
                start_word = start_uri[6:].split("/")[0].replace("_", " ")
                end_word = end_uri[6:].split("/")[0].replace("_", " ")
                
                # --- FILTER 3: Relevance to Game ---
                # Connection logic:
                # One side MUST be a 'Game Word'.
                # The other side MUST be in our 'Allowed Vocabulary'.
                
                start_is_game = start_word in game_words
                end_is_game = end_word in game_words
                
                is_relevant = False
                
                if start_is_game and (end_word in allowed_vocab):
                    is_relevant = True
                elif end_is_game and (start_word in allowed_vocab):
                    is_relevant = True
                
                if not is_relevant:
                    continue

                # Extract Weight
                try:
                    meta = json.loads(row[4])
                    weight = meta.get('weight', 1.0)
                except:
                    weight = 1.0

                # --- STORE DATA ---
                # We normalize the relation string (remove "/r/")
                clean_rel = rel.replace("/r/", "")

                # If Start is the game word, store outgoing edge
                if start_is_game:
                    knowledge_graph[start_word].append({
                        "start": start_word,
                        "relation": clean_rel,
                        "end": end_word,
                        "weight": weight
                    })

                # If End is the game word, store incoming edge (reversed perspective)
                if end_is_game:
                    knowledge_graph[end_word].append({
                        "start": start_word,
                        "relation": clean_rel,
                        "end": end_word,
                        "weight": weight
                    })
                
                edges_kept += 1

    except EOFError:
        print("❌ Error: The .gz file seems corrupted or incomplete.")
        return

    # 4. Save to JSON
    print(f"✅ Parsing Complete. Saving {edges_kept} edges to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_graph, f, indent=2)
    
    print(f"🎉 Success! Knowledge Graph is ready.")

if __name__ == "__main__":
    build_graph()