#!/usr/bin/env python3
"""
Add More Comprehensive Intuitive Associations
==============================================
Extended list covering more board words.
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_DIR / "data" / "conceptnet_graph.json"
COMMON_WORDS_PATH = PROJECT_DIR / "data" / "common_words.txt"
CODENAMES_WORDS_PATH = PROJECT_DIR / "data" / "codenames_words.txt"

# More comprehensive associations
MORE_ASSOCIATIONS = {
    # Places & Landmarks
    "asian": ["china", "india", "japan", "tokyo", "beijing", "panda", "rice"],
    "european": ["berlin", "london", "paris", "rome", "france", "germany"],
    "american": ["washington", "hollywood", "america", "buffalo", "texas"],
    "african": ["africa", "egypt", "lion", "elephant", "safari"],
    "landmark": ["pyramid", "tower", "statue", "bridge", "wall"],
    "monument": ["statue", "pyramid", "tower", "washington"],
    "city": ["berlin", "beijing", "london", "paris", "rome", "tokyo", "hollywood"],
    "temple": ["egypt", "greece", "india", "olympus"],
    "palace": ["king", "queen", "castle", "crown", "throne"],
    "ruins": ["rome", "greece", "egypt", "pyramid", "temple"],

    # Food & Cooking
    "kitchen": ["cook", "pan", "plate", "fork", "spoon", "knife", "oven"],
    "cooking": ["pan", "plate", "oven", "chef", "stove"],
    "baking": ["oven", "bread", "cake", "flour"],
    "restaurant": ["chef", "plate", "cook", "waiter"],
    "breakfast": ["egg", "toast", "coffee", "cereal"],
    "dinner": ["plate", "table", "fork", "knife"],
    "dessert": ["chocolate", "cake", "ice", "sweet"],
    "beverage": ["water", "coffee", "tea", "juice", "wine"],
    "alcohol": ["wine", "bar", "beer", "glass"],
    "snack": ["chip", "nut", "candy", "chocolate"],

    # Games & Entertainment
    "casino": ["dice", "card", "bet", "slot"],
    "gambling": ["dice", "card", "casino", "bet"],
    "chess": ["king", "queen", "knight", "board", "pawn"],
    "cards": ["deck", "diamond", "heart", "club", "spade"],
    "poker": ["card", "bet", "ace", "king", "bluff"],
    "movie": ["hollywood", "film", "star", "screen", "actor"],
    "theater": ["stage", "actor", "play", "curtain", "opera"],
    "cinema": ["hollywood", "film", "screen", "movie"],
    "magic": ["witch", "wand", "spell", "rabbit", "trick"],
    "circus": ["clown", "elephant", "lion", "ring", "tent"],

    # Clothing & Fashion
    "clothing": ["dress", "suit", "shirt", "pants", "coat"],
    "formal": ["suit", "tie", "dress", "ball", "gown"],
    "casual": ["shirt", "jeans", "shorts"],
    "footwear": ["boot", "shoe", "heel", "sock"],
    "accessory": ["belt", "hat", "ring", "watch", "tie"],
    "jewelry": ["ring", "diamond", "gold", "silver", "pearl"],
    "hat": ["cap", "crown", "top", "head"],

    # Nature & Weather
    "storm": ["rain", "wind", "thunder", "lightning", "cloud"],
    "sunny": ["sun", "beach", "summer", "bright"],
    "rainy": ["rain", "cloud", "umbrella", "wet"],
    "snowy": ["snow", "winter", "cold", "ice", "white"],
    "windy": ["wind", "storm", "sail", "kite"],
    "cloudy": ["cloud", "sky", "rain", "gray"],
    "earthquake": ["shake", "ground", "disaster"],
    "volcano": ["lava", "mountain", "fire", "ash"],
    "tsunami": ["wave", "ocean", "disaster", "water"],
    "flood": ["water", "rain", "river", "disaster"],

    # Plants & Gardening
    "garden": ["flower", "plant", "rose", "grass", "tree"],
    "farming": ["farm", "crop", "field", "harvest", "tractor"],
    "harvest": ["farm", "crop", "fall", "wheat", "corn"],
    "bloom": ["flower", "rose", "spring", "garden"],
    "seed": ["plant", "tree", "flower", "grow"],
    "leaf": ["tree", "plant", "fall", "green", "maple"],
    "branch": ["tree", "stick", "wood", "arm"],
    "root": ["tree", "plant", "ground", "tooth"],
    "vine": ["grape", "wine", "jungle", "plant"],
    "cactus": ["desert", "spike", "plant", "green"],

    # Buildings & Architecture
    "building": ["tower", "skyscraper", "office", "hotel", "house"],
    "skyscraper": ["tower", "city", "building", "tall"],
    "tower": ["castle", "church", "tall", "ivory"],
    "church": ["bell", "cross", "angel", "priest"],
    "hotel": ["room", "bed", "lobby", "vacation"],
    "office": ["work", "desk", "computer", "boss"],
    "factory": ["machine", "worker", "smoke", "robot"],
    "warehouse": ["box", "storage", "forklift"],
    "barn": ["farm", "horse", "hay", "animal"],
    "cabin": ["wood", "log", "forest", "mountain"],

    # Tools & Equipment
    "tool": ["hammer", "drill", "saw", "wrench"],
    "hammer": ["nail", "tool", "build", "hit"],
    "drill": ["hole", "tool", "dentist"],
    "saw": ["cut", "wood", "tool", "blade"],
    "wrench": ["bolt", "nut", "tool", "fix"],
    "scissors": ["cut", "paper", "hair"],
    "needle": ["thread", "sew", "sharp", "injection"],
    "rope": ["tie", "climb", "knot", "hang"],
    "chain": ["link", "lock", "metal", "bike"],
    "wheel": ["car", "bike", "roll", "tire", "spin"],

    # Technology
    "computer": ["screen", "keyboard", "mouse", "code", "software"],
    "phone": ["call", "ring", "cell", "mobile"],
    "internet": ["web", "computer", "online", "network"],
    "robot": ["machine", "metal", "android", "future"],
    "software": ["computer", "code", "program", "app"],
    "hardware": ["computer", "chip", "circuit"],
    "screen": ["computer", "phone", "tv", "movie"],
    "keyboard": ["computer", "type", "piano"],
    "camera": ["photo", "film", "lens", "picture"],
    "battery": ["power", "charge", "electric"],

    # Music
    "instrument": ["piano", "guitar", "drum", "flute", "violin"],
    "piano": ["key", "music", "concert", "play"],
    "guitar": ["string", "music", "rock", "play"],
    "drum": ["beat", "music", "stick", "band"],
    "violin": ["string", "music", "orchestra", "bow"],
    "flute": ["wind", "music", "blow"],
    "trumpet": ["brass", "music", "blow", "jazz"],
    "singer": ["voice", "song", "music", "concert"],
    "band": ["music", "rock", "concert", "drum"],
    "concert": ["music", "band", "stage", "audience"],

    # Sports (expanded)
    "football": ["ball", "goal", "kick", "field", "team"],
    "basketball": ["ball", "hoop", "court", "dunk"],
    "baseball": ["bat", "ball", "pitch", "diamond"],
    "tennis": ["ball", "racket", "court", "net"],
    "golf": ["ball", "club", "hole", "green"],
    "hockey": ["ice", "puck", "stick", "goal"],
    "soccer": ["ball", "goal", "kick", "field"],
    "boxing": ["ring", "punch", "glove", "fight"],
    "wrestling": ["fight", "ring", "pin"],
    "swimming": ["pool", "water", "dive", "race"],
    "skiing": ["snow", "mountain", "slope", "winter"],
    "surfing": ["wave", "ocean", "beach", "board"],
    "fishing": ["fish", "rod", "hook", "boat", "lake"],
    "hunting": ["gun", "deer", "forest", "shoot"],
    "racing": ["car", "fast", "speed", "track"],

    # Jobs & Professions
    "job": ["work", "office", "boss", "employee"],
    "doctor": ["hospital", "medicine", "nurse", "patient"],
    "nurse": ["hospital", "doctor", "patient", "care"],
    "teacher": ["school", "student", "class", "learn"],
    "lawyer": ["court", "judge", "case", "law"],
    "judge": ["court", "law", "gavel", "robe"],
    "police": ["cop", "badge", "arrest", "crime"],
    "firefighter": ["fire", "truck", "hose", "rescue"],
    "chef": ["cook", "kitchen", "food", "restaurant"],
    "pilot": ["plane", "fly", "sky", "captain"],
    "captain": ["ship", "boat", "plane", "leader"],
    "engineer": ["build", "machine", "train", "design"],
    "artist": ["paint", "brush", "canvas", "draw"],
    "writer": ["book", "pen", "story", "author"],
    "scientist": ["lab", "experiment", "research", "discover"],
    "farmer": ["farm", "crop", "field", "tractor"],
    "builder": ["build", "hammer", "construction", "house"],

    # Crime & Law
    "crime": ["police", "prison", "thief", "murder"],
    "prison": ["cell", "guard", "crime", "lock"],
    "thief": ["steal", "crime", "mask", "rob"],
    "murder": ["death", "crime", "kill", "weapon"],
    "detective": ["clue", "mystery", "solve", "case"],
    "evidence": ["clue", "proof", "crime", "case"],
    "trial": ["court", "judge", "jury", "lawyer"],

    # Health & Medicine
    "medicine": ["doctor", "pill", "hospital", "cure"],
    "surgery": ["doctor", "hospital", "knife", "cut"],
    "injection": ["needle", "medicine", "shot"],
    "pill": ["medicine", "drug", "swallow"],
    "bandage": ["wound", "wrap", "heal"],
    "fever": ["sick", "hot", "temperature"],
    "disease": ["sick", "doctor", "medicine", "virus"],
    "virus": ["sick", "disease", "infection", "computer"],

    # War & Military (expanded)
    "battle": ["war", "soldier", "fight", "army"],
    "combat": ["war", "fight", "soldier", "weapon"],
    "defense": ["shield", "guard", "protect", "wall"],
    "attack": ["fight", "war", "charge", "strike"],
    "victory": ["win", "war", "champion", "trophy"],
    "defeat": ["lose", "war", "battle"],
    "general": ["army", "war", "leader", "soldier"],
    "troops": ["army", "soldier", "march", "war"],
    "ammunition": ["gun", "bullet", "bomb", "war"],
    "explosive": ["bomb", "blast", "fire", "dynamite"],

    # Travel
    "vacation": ["beach", "hotel", "travel", "relax"],
    "travel": ["plane", "car", "trip", "map"],
    "journey": ["travel", "trip", "road", "adventure"],
    "adventure": ["travel", "explore", "jungle", "danger"],
    "explore": ["discover", "map", "jungle", "adventure"],
    "tourist": ["travel", "camera", "map", "vacation"],
    "passport": ["travel", "country", "stamp"],
    "luggage": ["travel", "bag", "suitcase"],
    "ticket": ["plane", "train", "movie", "travel"],
    "map": ["travel", "direction", "treasure", "country"],

    # Ocean & Sea
    "marine": ["ocean", "fish", "sea", "boat"],
    "underwater": ["ocean", "fish", "dive", "submarine"],
    "diving": ["ocean", "water", "scuba", "pool"],
    "beach": ["sand", "ocean", "sun", "wave"],
    "shore": ["beach", "ocean", "water", "sand"],
    "coral": ["reef", "ocean", "fish", "color"],
    "submarine": ["underwater", "navy", "ocean", "boat"],
    "anchor": ["ship", "boat", "ocean", "heavy"],
    "lighthouse": ["light", "ocean", "ship", "coast"],
    "sailor": ["ship", "navy", "ocean", "captain"],

    # Emotions & States
    "happy": ["smile", "joy", "laugh", "celebration"],
    "sad": ["cry", "tear", "blue", "down"],
    "angry": ["mad", "rage", "fire", "red"],
    "scared": ["fear", "ghost", "dark", "scream"],
    "love": ["heart", "romance", "kiss", "wedding"],
    "hate": ["anger", "enemy", "war"],
    "peace": ["calm", "dove", "quiet", "war"],
    "war": ["battle", "soldier", "fight", "bomb"],

    # Abstract concepts
    "power": ["energy", "electric", "strong", "king"],
    "energy": ["power", "battery", "sun", "electric"],
    "speed": ["fast", "car", "race", "quick"],
    "strength": ["strong", "muscle", "power", "giant"],
    "intelligence": ["smart", "brain", "genius", "mind"],
    "beauty": ["pretty", "queen", "rose", "model"],
    "wealth": ["rich", "gold", "money", "diamond"],
    "luck": ["fortune", "clover", "dice", "charm"],
    "fate": ["destiny", "fortune", "death", "life"],
    "time": ["clock", "hour", "minute", "watch"],
    "space": ["star", "planet", "rocket", "moon"],
    "death": ["dead", "skull", "grave", "ghost"],
    "life": ["alive", "birth", "heart", "live"],
}


def load_data():
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)
    with open(COMMON_WORDS_PATH, "r") as f:
        common_words = set(w.strip().lower() for w in f if w.strip())
    with open(CODENAMES_WORDS_PATH, "r") as f:
        board_words = set(w.strip().lower() for w in f if w.strip())
    return graph, common_words, board_words


def main():
    print("=" * 60)
    print("Adding More Intuitive Associations")
    print("=" * 60)
    print()

    graph, common_words, board_words = load_data()

    print(f"Graph entries: {len(graph)}")
    print(f"Board words: {len(board_words)}")
    print()

    new_edges = 0
    new_common = set()
    connections = {}

    for clue, targets in MORE_ASSOCIATIONS.items():
        clue_lower = clue.lower()
        connected = []

        for target in targets:
            target_lower = target.lower()

            if target_lower in board_words and target_lower in graph:
                existing = [e for e in graph[target_lower]
                           if e.get("end", "").lower() == clue_lower]

                if not existing:
                    graph[target_lower].append({
                        "end": clue_lower,
                        "relation": "RelatedTo",
                        "normalized_weight": 0.75
                    })
                    new_edges += 1
                    connected.append(target_lower)

        if connected:
            connections[clue] = connected
            print(f"  {clue:15} → {len(connected)} boards")

        if clue_lower not in common_words:
            new_common.add(clue_lower)

    print()
    print("=" * 60)
    print(f"New edges added: {new_edges}")
    print(f"Clues with new connections: {len(connections)}")
    print(f"New common words: {len(new_common)}")

    if new_edges > 0:
        print("\nSaving graph...")
        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)

    if new_common:
        with open(COMMON_WORDS_PATH, "a") as f:
            for w in sorted(new_common):
                f.write(f"\n{w}")
        print(f"Added {len(new_common)} words to common_words.txt")

    print("\nDone!")


if __name__ == "__main__":
    main()
