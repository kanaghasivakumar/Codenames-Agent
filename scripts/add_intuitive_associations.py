#!/usr/bin/env python3
"""
Add Human-Intuitive Associations to ConceptNet Graph
=====================================================
Adds well-known associations that humans naturally make in Codenames.
These are verified common associations, not API-dependent.
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_DIR / "data" / "conceptnet_graph.json"
COMMON_WORDS_PATH = PROJECT_DIR / "data" / "common_words.txt"
CODENAMES_WORDS_PATH = PROJECT_DIR / "data" / "codenames_words.txt"

# Human-intuitive associations: clue → [board words it connects to]
# These are obvious associations that humans naturally make
INTUITIVE_ASSOCIATIONS = {
    # Animals - category clues
    "mammal": ["whale", "bear", "lion", "horse", "dog", "cat", "bat", "mouse", "elephant"],
    "reptile": ["dinosaur", "dragon", "lizard"],
    "bird": ["eagle", "hawk", "robin", "crane", "penguin", "turkey", "kiwi", "phoenix"],
    "fish": ["shark", "fish", "whale", "dolphin"],
    "insect": ["bug", "fly", "spider", "cricket", "ant", "bee"],
    "predator": ["shark", "lion", "eagle", "hawk", "wolf", "bear", "dragon"],
    "prey": ["mouse", "rabbit", "fish", "deer"],
    "pet": ["dog", "cat", "fish", "rabbit", "hamster"],
    "wild": ["lion", "bear", "tiger", "elephant", "jungle"],

    # Geography
    "continent": ["africa", "antarctica", "australia", "america", "europe", "asia"],
    "country": ["china", "india", "egypt", "mexico", "canada", "greece", "scotland"],
    "capital": ["beijing", "berlin", "london", "paris", "rome", "tokyo", "washington"],
    "island": ["australia", "japan", "england", "cuba", "hawaii", "iceland"],
    "mountain": ["alps", "everest", "olympus", "rock", "cliff"],
    "river": ["amazon", "nile", "stream", "bridge", "bank"],
    "ocean": ["pacific", "atlantic", "wave", "water", "beach", "shark", "whale"],
    "desert": ["egypt", "sand", "camel", "cactus", "pyramid"],
    "jungle": ["amazon", "tiger", "snake", "monkey", "vine", "tarzan"],
    "arctic": ["antarctica", "penguin", "polar", "ice", "snow", "cold"],
    "tropical": ["amazon", "jungle", "palm", "coconut", "island", "beach"],

    # Space
    "planet": ["saturn", "mercury", "jupiter", "mars", "earth", "moon"],
    "celestial": ["star", "moon", "sun", "comet", "meteor"],
    "lunar": ["moon", "night", "eclipse", "astronaut"],
    "solar": ["sun", "light", "energy", "eclipse", "system"],
    "cosmic": ["star", "space", "galaxy", "universe"],
    "orbit": ["satellite", "moon", "planet", "space"],
    "astronaut": ["space", "moon", "rocket", "shuttle"],

    # Food & Drink
    "fruit": ["apple", "orange", "lemon", "kiwi", "berry", "grape", "pumpkin"],
    "vegetable": ["carrot", "potato", "corn", "pumpkin", "olive"],
    "meat": ["ham", "beef", "chicken", "turkey", "steak"],
    "drink": ["water", "wine", "coffee", "tea", "juice"],
    "sweet": ["chocolate", "candy", "sugar", "honey", "cake"],
    "sour": ["lemon", "lime", "vinegar"],
    "spicy": ["pepper", "chili", "hot"],
    "citrus": ["orange", "lemon", "lime", "grapefruit"],

    # Nature
    "forest": ["tree", "bear", "deer", "wood", "cabin", "log"],
    "weather": ["snow", "rain", "wind", "storm", "cloud", "sun"],
    "season": ["spring", "summer", "fall", "winter"],
    "winter": ["snow", "ice", "cold", "christmas", "ski"],
    "summer": ["sun", "beach", "hot", "vacation", "pool"],
    "spring": ["flower", "rain", "green", "garden"],

    # Colors as clues
    "crimson": ["blood", "red", "rose", "apple"],
    "scarlet": ["red", "rose", "blood"],
    "azure": ["sky", "blue", "ocean"],
    "emerald": ["green", "gem", "ireland"],
    "golden": ["gold", "sun", "ring", "crown"],
    "silver": ["moon", "metal", "mirror", "second"],
    "bronze": ["medal", "statue", "third"],
    "ivory": ["elephant", "piano", "white", "tower"],

    # Materials
    "metal": ["iron", "gold", "silver", "steel", "copper", "coin", "robot"],
    "wooden": ["table", "chair", "stick", "log", "cabin", "board"],
    "glass": ["window", "mirror", "bottle", "screen"],
    "leather": ["belt", "boot", "saddle", "jacket"],
    "fabric": ["dress", "suit", "cotton", "silk"],

    # Military & Combat
    "military": ["soldier", "tank", "bomb", "war", "army", "general"],
    "weapon": ["gun", "knife", "sword", "bomb", "missile", "bow"],
    "war": ["soldier", "tank", "bomb", "battle", "army"],
    "armor": ["knight", "shield", "tank", "helmet"],
    "navy": ["ship", "submarine", "sailor", "ocean", "anchor"],
    "army": ["soldier", "tank", "general", "war", "march"],

    # Royalty & Medieval
    "royal": ["king", "queen", "crown", "palace", "throne"],
    "medieval": ["knight", "castle", "dragon", "sword", "king"],
    "crown": ["king", "queen", "royal", "throne"],
    "castle": ["knight", "king", "queen", "tower", "dragon"],

    # Music & Entertainment
    "musical": ["concert", "band", "opera", "piano", "guitar", "flute"],
    "instrument": ["piano", "guitar", "drum", "flute", "violin"],
    "orchestra": ["concert", "conductor", "violin", "symphony"],

    # Sports
    "sport": ["ball", "game", "team", "score", "win", "cricket"],
    "ball": ["football", "basketball", "tennis", "golf", "soccer"],
    "team": ["player", "coach", "game", "score"],
    "athlete": ["runner", "swimmer", "player"],
    "olympic": ["gold", "medal", "greece", "torch", "ring"],

    # Transportation
    "vehicle": ["car", "truck", "van", "bus", "motorcycle"],
    "aircraft": ["plane", "jet", "helicopter", "pilot"],
    "ship": ["boat", "anchor", "sail", "captain", "deck"],
    "flying": ["plane", "bird", "helicopter", "jet", "wing"],
    "sailing": ["boat", "ship", "wind", "anchor", "deck"],

    # Body parts (good connectors)
    "wing": ["bird", "plane", "angel", "chicken", "bat"],
    "tail": ["dog", "cat", "fish", "monkey", "comet"],
    "horn": ["unicorn", "bull", "ram", "rhino"],
    "claw": ["cat", "bear", "crab", "eagle"],
    "teeth": ["shark", "vampire", "smile", "dentist"],

    # Mythical & Fantasy
    "mythical": ["dragon", "unicorn", "phoenix", "mermaid", "centaur"],
    "magical": ["witch", "wizard", "wand", "spell"],
    "legendary": ["dragon", "phoenix", "giant", "hero"],
    "monster": ["dragon", "giant", "frankenstein", "vampire"],
    "fairy": ["tale", "godmother", "magic", "princess"],

    # Time
    "ancient": ["egypt", "rome", "greece", "pyramid", "dinosaur"],
    "modern": ["computer", "phone", "car", "city"],
    "night": ["moon", "star", "dark", "sleep", "owl"],
    "morning": ["sun", "breakfast", "coffee", "dawn"],
    "midnight": ["moon", "dark", "clock", "sleep"],

    # Actions as clues
    "flying": ["bird", "plane", "bat", "helicopter", "angel"],
    "swimming": ["fish", "shark", "whale", "pool", "ocean"],
    "running": ["race", "horse", "dog", "marathon"],
    "hunting": ["lion", "eagle", "gun", "deer", "bear"],
    "climbing": ["mountain", "tree", "rope", "rock"],
    "burning": ["fire", "sun", "flame", "torch"],
    "freezing": ["ice", "snow", "cold", "winter", "antarctica"],

    # Properties
    "dangerous": ["shark", "bomb", "poison", "gun", "fire"],
    "poisonous": ["snake", "spider", "poison", "venom"],
    "sharp": ["knife", "needle", "teeth", "blade"],
    "heavy": ["elephant", "whale", "tank", "rock", "iron"],
    "bright": ["sun", "star", "light", "diamond"],
    "dark": ["night", "shadow", "black", "cave"],
    "fast": ["jet", "car", "race", "cheetah", "rocket"],
    "slow": ["turtle", "snail"],
    "giant": ["elephant", "whale", "giant", "tower"],
    "tiny": ["ant", "mouse", "needle", "atom"],

    # Science
    "chemical": ["lab", "poison", "acid", "experiment"],
    "nuclear": ["bomb", "power", "radiation", "atom"],
    "electric": ["light", "power", "shock", "battery"],
    "magnetic": ["iron", "compass", "north", "pole"],

    # Medical
    "medical": ["doctor", "hospital", "nurse", "medicine"],
    "hospital": ["doctor", "nurse", "ambulance", "patient"],

    # Common associations
    "pirate": ["ship", "treasure", "skull", "hook", "captain"],
    "cowboy": ["horse", "west", "hat", "boot", "ranch"],
    "spy": ["secret", "agent", "code", "mission"],
    "detective": ["clue", "mystery", "magnifying", "case"],
    "scientist": ["lab", "experiment", "atom", "research"],
}


def load_data():
    """Load existing data files."""
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)

    with open(COMMON_WORDS_PATH, "r") as f:
        common_words = set(w.strip().lower() for w in f if w.strip())

    with open(CODENAMES_WORDS_PATH, "r") as f:
        board_words = set(w.strip().lower() for w in f if w.strip())

    return graph, common_words, board_words


def main():
    print("=" * 60)
    print("Adding Human-Intuitive Associations")
    print("=" * 60)
    print()

    # Load data
    print("Loading existing data...")
    graph, common_words, board_words = load_data()

    print(f"  Graph entries: {len(graph)}")
    print(f"  Common words: {len(common_words)}")
    print(f"  Board words: {len(board_words)}")
    print()

    # Track additions
    new_edges_added = 0
    new_common_words = set()
    connections_by_clue = {}

    # Process each intuitive association
    print("Processing intuitive associations...")
    print()

    for clue, targets in INTUITIVE_ASSOCIATIONS.items():
        clue_lower = clue.lower()
        connected_boards = []

        for target in targets:
            target_lower = target.lower()

            # Check if target is a board word
            if target_lower in board_words and target_lower in graph:
                # Check if this edge already exists
                existing = [e for e in graph[target_lower]
                           if e.get("end", "").lower() == clue_lower]

                if not existing:
                    # Add the edge: board_word → clue
                    graph[target_lower].append({
                        "end": clue_lower,
                        "relation": "RelatedTo",
                        "normalized_weight": 0.8  # High weight for intuitive associations
                    })
                    new_edges_added += 1
                    connected_boards.append(target_lower)

        if connected_boards:
            connections_by_clue[clue] = connected_boards
            print(f"  {clue:15} → {len(connected_boards)} boards: {connected_boards[:5]}")

        # Track if clue needs to be added to common words
        if clue_lower not in common_words:
            new_common_words.add(clue_lower)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"New edges added to graph: {new_edges_added}")
    print(f"Clues with connections: {len(connections_by_clue)}")
    print(f"New common words needed: {len(new_common_words)}")

    if new_common_words:
        print(f"  New words: {sorted(new_common_words)[:15]}...")

    # Save updated graph
    if new_edges_added > 0:
        print()
        print("Saving updated graph...")
        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  Saved to {GRAPH_PATH}")

    # Add new common words
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
