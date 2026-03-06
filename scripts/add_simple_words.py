#!/usr/bin/env python3
"""
Add Simple Common Words for Codenames
======================================
Only adds everyday words that real people use - no scientific jargon.
These are words you'd actually use as clues in Codenames.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
COMMON_WORDS_PATH = PROJECT_DIR / "data" / "common_words.txt"

# Simple words people actually use - good for Codenames clues
SIMPLE_WORDS = {
    # Pop Culture - Movies/TV
    "batman", "superman", "spiderman", "ironman", "hulk", "thor", "wolverine",
    "joker", "vader", "yoda", "gandalf", "frodo", "hobbit", "hogwarts", "dumbledore",
    "disney", "pixar", "marvel", "avengers", "jedi", "sith", "terminator",
    "jurassic", "godzilla", "kong", "tarzan", "shrek", "frozen", "simpsons",
    "starwars", "harrypotter", "lotr", "matrix", "avatar", "titanic", "rocky",
    "rambo", "bond", "batman", "superman", "wonderwoman", "aquaman", "flash",

    # Pop Culture - Games/Internet
    "pokemon", "mario", "zelda", "minecraft", "fortnite", "tetris", "pacman",
    "nintendo", "playstation", "xbox", "emoji", "meme", "viral", "hashtag",
    "selfie", "google", "facebook", "instagram", "twitter", "youtube", "netflix",
    "spotify", "amazon", "uber", "tiktok", "snapchat", "whatsapp", "reddit",
    "wikipedia", "ebay", "paypal", "zoom", "skype", "linkedin",

    # Food & Drinks
    "sushi", "ramen", "tofu", "wasabi", "teriyaki", "tempura",
    "pizza", "pasta", "lasagna", "risotto", "gelato", "espresso", "cappuccino",
    "taco", "burrito", "nacho", "salsa", "guacamole", "tortilla", "enchilada",
    "croissant", "baguette", "crepe", "souffle", "fondue",
    "hummus", "falafel", "kebab", "shawarma", "pita",
    "curry", "naan", "tandoori", "samosa", "chai", "masala",
    "kimchi", "bibimbap", "dumpling", "wonton", "dimsum",
    "pretzel", "bratwurst", "schnitzel", "strudel",
    "paella", "churro", "sangria", "tapas",
    "waffle", "pancake", "donut", "cupcake", "brownie", "cookie",
    "smoothie", "latte", "mocha", "frappe", "milkshake",
    "bacon", "steak", "burger", "hotdog", "fries", "ketchup", "mustard",
    "chocolate", "vanilla", "strawberry", "caramel", "cinnamon",
    "avocado", "mango", "pineapple", "coconut", "banana", "orange",

    # Countries & Nationalities
    "american", "british", "french", "german", "italian", "spanish",
    "japanese", "chinese", "korean", "indian", "russian", "brazilian",
    "mexican", "canadian", "australian", "african", "european", "asian",
    "swedish", "norwegian", "finnish", "danish", "dutch", "swiss",
    "polish", "greek", "turkish", "egyptian", "arabian", "persian",
    "thai", "vietnamese", "filipino", "indonesian", "malaysian",
    "argentinian", "chilean", "colombian", "peruvian", "cuban",
    "irish", "scottish", "welsh", "english", "portuguese", "belgian",

    # Famous Cities
    "vegas", "hollywood", "broadway", "manhattan", "brooklyn",
    "paris", "london", "tokyo", "berlin", "rome", "madrid", "barcelona",
    "moscow", "dubai", "singapore", "hongkong", "bangkok", "mumbai",
    "sydney", "melbourne", "toronto", "vancouver", "montreal",
    "chicago", "boston", "seattle", "miami", "denver", "austin",
    "amsterdam", "prague", "vienna", "budapest", "athens", "cairo",

    # Technology
    "smartphone", "laptop", "tablet", "wifi", "bluetooth", "usb",
    "iphone", "android", "app", "software", "hardware", "website",
    "download", "upload", "streaming", "podcast", "blog", "vlog",
    "robot", "drone", "laser", "radar", "sonar", "gps",
    "cyber", "virtual", "digital", "pixel", "megabyte", "gigabyte",
    "internet", "email", "password", "username", "login", "logout",
    "computer", "keyboard", "mouse", "monitor", "printer", "scanner",

    # Modern Slang/Common Words
    "awesome", "cool", "epic", "lame", "chill", "vibe", "mood",
    "photobomb", "binge", "spoiler", "trending", "viral",
    "hipster", "nerd", "geek", "gamer", "foodie", "influencer",
    "startup", "freelance", "remote", "hustle", "grind", "flex",
    "ghost", "catfish", "troll", "stan", "simp", "slay",

    # Sports & Fitness
    "yoga", "pilates", "cardio", "workout", "fitness", "gym",
    "marathon", "triathlon", "surfing", "skateboarding", "snowboarding",
    "karate", "judo", "taekwondo", "kungfu", "ninja", "samurai",
    "goalkeeper", "striker", "midfielder", "quarterback", "touchdown",
    "soccer", "football", "basketball", "baseball", "tennis", "golf",
    "hockey", "volleyball", "cricket", "rugby", "boxing", "wrestling",
    "olympics", "championship", "trophy", "medal", "champion",

    # Animals (common names people know)
    "dolphin", "penguin", "koala", "kangaroo", "panda", "gorilla",
    "cheetah", "leopard", "jaguar", "panther", "rhino", "hippo",
    "giraffe", "zebra", "elephant", "buffalo", "moose", "elk",
    "octopus", "jellyfish", "starfish", "seahorse", "lobster", "shrimp",
    "parrot", "flamingo", "peacock", "owl", "hummingbird", "woodpecker",
    "butterfly", "dragonfly", "firefly", "ladybug", "grasshopper",
    "puppy", "kitten", "bunny", "hamster", "goldfish", "turtle",
    "shark", "whale", "crocodile", "alligator", "python", "cobra",

    # Nature
    "rainforest", "savanna", "tundra", "glacier", "volcano", "canyon",
    "beach", "ocean", "mountain", "desert", "jungle", "forest",
    "waterfall", "island", "reef", "lagoon", "cave", "cliff",
    "sunrise", "sunset", "rainbow", "thunder", "lightning", "tornado",

    # Mythology & Fantasy
    "wizard", "witch", "vampire", "werewolf", "zombie", "ghost",
    "dragon", "unicorn", "phoenix", "griffin", "mermaid", "centaur",
    "goblin", "troll", "ogre", "elf", "dwarf", "fairy", "pixie",
    "zeus", "poseidon", "hades", "athena", "apollo", "hercules",
    "odin", "loki", "freya", "valkyrie", "viking",
    "pharaoh", "mummy", "sphinx", "pyramid", "cleopatra",
    "shogun", "geisha", "sumo", "origami", "katana",

    # Music
    "guitar", "piano", "violin", "drums", "saxophone", "trumpet",
    "rock", "jazz", "blues", "hiphop", "reggae", "techno", "disco",
    "karaoke", "concert", "festival", "orchestra", "symphony", "choir",
    "rapper", "beatbox", "remix", "acoustic", "electric",
    "beyonce", "madonna", "elvis", "beatles", "queen",

    # Fashion & Style
    "jeans", "hoodie", "sneakers", "sandals", "boots", "heels",
    "tuxedo", "gown", "bikini", "swimsuit", "lingerie", "pajamas",
    "sunglasses", "earrings", "necklace", "bracelet", "tattoo", "piercing",
    "hairstyle", "makeup", "manicure", "pedicure", "spa", "salon",
    "dress", "shirt", "pants", "jacket", "coat", "scarf", "hat",

    # Household
    "headphones", "charger", "remote", "battery",
    "microwave", "blender", "toaster", "dishwasher", "vacuum",
    "sofa", "couch", "mattress", "pillow", "blanket", "curtain",
    "bathtub", "shower", "toilet", "sink", "faucet", "towel",
    "garage", "basement", "attic", "balcony", "patio", "backyard",
    "doorbell", "mailbox", "driveway", "sidewalk", "fence", "gate",
    "kitchen", "bedroom", "bathroom", "living", "dining",

    # Vehicles
    "motorcycle", "scooter", "bicycle", "skateboard", "rollerblades",
    "helicopter", "airplane", "jet", "rocket", "spaceship", "submarine",
    "yacht", "sailboat", "kayak", "canoe", "surfboard", "jetski",
    "ambulance", "firetruck", "bulldozer", "crane", "forklift",
    "limousine", "convertible", "pickup", "minivan", "suv", "sedan",
    "tesla", "ferrari", "porsche", "lamborghini", "mustang",

    # Professions
    "astronaut", "scientist", "professor", "engineer", "architect",
    "chef", "waiter", "bartender", "barista", "sommelier",
    "surgeon", "nurse", "dentist", "therapist", "pharmacist",
    "lawyer", "judge", "detective", "sheriff", "firefighter",
    "pilot", "captain", "sailor", "mechanic", "electrician", "plumber",
    "photographer", "filmmaker", "animator", "designer", "illustrator",
    "journalist", "blogger", "podcaster", "streamer",
    "actor", "actress", "singer", "dancer", "comedian", "magician",

    # Emotions & States
    "happy", "sad", "angry", "scared", "excited", "nervous",
    "love", "hate", "fear", "joy", "peace", "chaos",
    "crazy", "lazy", "busy", "hungry", "tired", "sleepy",

    # Actions (common verbs as nouns)
    "dance", "sing", "jump", "run", "swim", "fly",
    "fight", "kiss", "hug", "laugh", "cry", "scream",
    "sleep", "dream", "think", "believe", "imagine",

    # Time & Events
    "birthday", "wedding", "funeral", "graduation", "anniversary",
    "christmas", "halloween", "thanksgiving", "easter", "valentine",
    "monday", "friday", "weekend", "holiday", "vacation",
    "morning", "evening", "midnight", "noon",

    # Colors (as nouns)
    "crimson", "scarlet", "maroon", "navy", "turquoise", "magenta",
    "ivory", "ebony", "amber", "jade", "ruby", "sapphire", "emerald",

    # Materials
    "leather", "silk", "cotton", "wool", "velvet", "denim",
    "marble", "granite", "bronze", "copper", "titanium", "chrome",

    # Body (common references)
    "brain", "heart", "muscle", "bone", "skin", "blood",
    "thumb", "fist", "elbow", "knee", "ankle", "spine",
}


def main():
    print("=" * 60)
    print("ADDING SIMPLE COMMON WORDS")
    print("=" * 60)
    print()

    # Load current words
    with open(COMMON_WORDS_PATH) as f:
        current_words = set(w.strip().lower() for w in f if w.strip())

    print(f"Current common_words.txt: {len(current_words)} words")

    # Find words to add
    words_to_add = set()
    for word in SIMPLE_WORDS:
        word = word.lower().strip()
        if word not in current_words and word.isalpha() and len(word) >= 3:
            words_to_add.add(word)

    print(f"Simple words to add: {len(words_to_add)}")
    print()

    # Show what we're adding
    print("Words being added:")
    for i, word in enumerate(sorted(words_to_add)):
        print(f"  {word}", end="")
        if (i + 1) % 8 == 0:
            print()
    print()
    print()

    # Add words
    if words_to_add:
        with open(COMMON_WORDS_PATH, 'a') as f:
            for word in sorted(words_to_add):
                f.write(f"\n{word}")
        print(f"Added {len(words_to_add)} words")

    # Final count
    with open(COMMON_WORDS_PATH) as f:
        final_count = len([w for w in f if w.strip()])

    print(f"Final common_words.txt: {final_count} words")
    print()
    print("Done! Only simple, everyday words added.")


if __name__ == "__main__":
    main()
