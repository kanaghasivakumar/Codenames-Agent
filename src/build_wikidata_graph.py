"""
Wikidata Knowledge Graph Builder for Codenames (v5 - Curated QIDs)

Uses MANUAL QID mapping for known polysemous words.
Falls back to popularity-based selection for others.

This is the correct approach for production-quality graphs.
"""

import json
import time
import requests
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
CODENAMES_WORDS_FILE = PROJECT_ROOT / "data" / "codenames_words.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "wikidata_graph.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# ============================================================
# CURATED QID MAPPING - Primary sense for Codenames
# These are the CORRECT Wikidata entities for each word
# ============================================================
CURATED_QIDS = {
    # Animals
    "bat": "Q28425",        # bat (mammal)
    "bear": "Q11090",       # bear (animal)
    "bug": "Q1390",         # insect
    "cat": "Q146",          # domestic cat
    "dog": "Q144",          # domestic dog
    "dragon": "Q7559",      # dragon (mythical)
    "eagle": "Q2092297",    # eagle (bird)
    "fish": "Q152",         # fish
    "hawk": "Q10885",       # hawk
    "horse": "Q726",        # horse
    "kangaroo": "Q39",      # kangaroo
    "mouse": "Q7380",       # mouse
    "octopus": "Q131250",   # octopus
    "penguin": "Q9103",     # penguin
    "shark": "Q7363",       # shark
    "snake": "Q2102",       # snake
    "spider": "Q1357",      # spider
    "turkey": "Q10870",     # turkey (bird)
    "whale": "Q42196",      # whale
    "wolf": "Q18498",       # wolf

    # Food
    "apple": "Q89",         # apple (fruit)
    "banana": "Q503",       # banana
    "berry": "Q13184",      # berry
    "bread": "Q7802",       # bread
    "carrot": "Q81",        # carrot
    "chocolate": "Q195",    # chocolate
    "lemon": "Q1093742",    # lemon
    "nut": "Q11009",        # nut (food)
    "olive": "Q3006889",    # olive
    "orange": "Q13191",     # orange (fruit)
    "pepper": "Q23425",     # pepper
    "potato": "Q10998",     # potato

    # Places
    "africa": "Q15",        # Africa
    "amazon": "Q3783",      # Amazon River
    "america": "Q30",       # United States
    "antarctica": "Q51",    # Antarctica
    "australia": "Q408",    # Australia (country)
    "berlin": "Q64",        # Berlin
    "brazil": "Q155",       # Brazil
    "canada": "Q16",        # Canada
    "china": "Q148",        # China
    "egypt": "Q79",         # Egypt
    "england": "Q21",       # England
    "europe": "Q46",        # Europe
    "france": "Q142",       # France
    "germany": "Q183",      # Germany
    "greece": "Q41",        # Greece
    "india": "Q668",        # India
    "japan": "Q17",         # Japan
    "london": "Q84",        # London
    "mexico": "Q96",        # Mexico
    "moscow": "Q649",       # Moscow
    "paris": "Q90",         # Paris
    "rome": "Q220",         # Rome
    "russia": "Q159",       # Russia
    "tokyo": "Q1490",       # Tokyo
    "washington": "Q61",    # Washington D.C.

    # Objects
    "ball": "Q18545",       # ball (object)
    "bank": "Q22687",       # bank (financial)
    "bar": "Q187456",       # bar (establishment)
    "bed": "Q42177",        # bed
    "bell": "Q101401",      # bell
    "belt": "Q614304",      # belt (clothing)
    "block": "Q11469",      # block
    "board": "Q815741",     # board
    "bolt": "Q189958",      # bolt (fastener)
    "boot": "Q190868",      # boot (footwear)
    "bottle": "Q80228",     # bottle
    "bow": "Q46311",        # bow (weapon)
    "box": "Q188075",       # box
    "bridge": "Q12280",     # bridge
    "brush": "Q14890",      # brush
    "button": "Q1346434",   # button
    "car": "Q1420",         # car
    "card": "Q47883",       # playing card
    "chair": "Q15026",      # chair
    "clock": "Q376",        # clock
    "diamond": "Q5283",     # diamond (gem)
    "dress": "Q200539",     # dress
    "fan": "Q127956",       # fan (device)
    "glass": "Q15006",      # glass
    "glove": "Q169031",     # glove
    "gold": "Q897",         # gold
    "guitar": "Q6607",      # guitar
    "gun": "Q12796",        # gun
    "hammer": "Q134787",    # hammer
    "iron": "Q677",         # iron (metal)
    "key": "Q175389",       # key
    "knife": "Q32489",      # knife
    "laser": "Q38867",      # laser
    "lemon": "Q500",        # lemon
    "light": "Q9128",       # light
    "lock": "Q37221",       # lock
    "match": "Q179415",     # match (fire)
    "microscope": "Q25253", # microscope
    "mirror": "Q35197",     # mirror
    "nail": "Q193572",      # nail (fastener)
    "needle": "Q80378",     # needle
    "net": "Q178802",       # net
    "paper": "Q11472",      # paper
    "pen": "Q862335",       # pen
    "piano": "Q5994",       # piano
    "pipe": "Q133343",      # pipe
    "pistol": "Q46311",     # pistol
    "plate": "Q57216",      # plate
    "ring": "Q46847",       # ring
    "rock": "Q8063",        # rock
    "screen": "Q2136937",   # screen
    "ship": "Q11446",       # ship
    "silver": "Q1090",      # silver
    "spring": "Q7942",      # spring (season)
    "star": "Q523",         # star (astronomy)
    "stick": "Q107293",     # stick
    "string": "Q131514",    # string
    "sword": "Q12791",      # sword
    "table": "Q14748",      # table
    "telephone": "Q11035",  # telephone
    "tower": "Q12518",      # tower
    "train": "Q870",        # train
    "tree": "Q10884",       # tree
    "trunk": "Q47461",      # trunk (luggage)
    "tube": "Q9690",        # tube
    "van": "Q193468",       # van
    "wall": "Q42948",       # wall
    "watch": "Q178794",     # watch
    "water": "Q283",        # water
    "wheel": "Q44679",      # wheel
    "wire": "Q161439",      # wire

    # Professions/People
    "agent": "Q189290",     # agent
    "angel": "Q235113",     # angel
    "doctor": "Q39631",     # physician
    "giant": "Q3696533",    # giant
    "guard": "Q15869269",   # guard
    "knight": "Q102083",    # knight
    "lawyer": "Q40348",     # lawyer
    "nurse": "Q186360",     # nurse
    "pilot": "Q158648",     # pilot
    "pirate": "Q7493",      # pirate
    "queen": "Q116",        # queen
    "king": "Q12097",       # king
    "soldier": "Q4991371",  # soldier
    "spy": "Q9352089",      # spy
    "teacher": "Q37226",    # teacher

    # Nature
    "beach": "Q40080",      # beach
    "cave": "Q35509",       # cave
    "cliff": "Q107679",     # cliff
    "cloud": "Q8074",       # cloud
    "coast": "Q93352",      # coast
    "desert": "Q8514",      # desert
    "forest": "Q4421",      # forest
    "ice": "Q23392",        # ice
    "island": "Q23442",     # island
    "jungle": "Q164546",    # jungle
    "lake": "Q23397",       # lake
    "moon": "Q405",         # Moon
    "mountain": "Q8502",    # mountain
    "ocean": "Q9430",       # ocean
    "river": "Q4022",       # river
    "snow": "Q7561",        # snow
    "sun": "Q525",          # Sun
    "volcano": "Q8072",     # volcano

    # Abstract/Other
    "air": "Q1931",         # air
    "bomb": "Q127197",      # bomb
    "center": "Q130879",    # center
    "circle": "Q17278",     # circle
    "contract": "Q17633",   # contract
    "court": "Q27686",      # court (law)
    "crash": "Q1756813",    # crash
    "cross": "Q20252",      # cross
    "death": "Q4",          # death
    "disease": "Q12136",    # disease
    "fire": "Q3196",        # fire
    "gas": "Q11432",        # gas
    "genius": "Q181741",    # genius
    "ghost": "Q21698",      # ghost
    "grace": "Q137073",     # grace
    "heart": "Q1072",       # heart
    "hole": "Q1637362",     # hole
    "hospital": "Q16917",   # hospital
    "hotel": "Q27686",      # hotel
    "love": "Q316",         # love
    "magic": "Q131539",     # magic
    "mind": "Q450",         # mind
    "moon": "Q405",         # Moon
    "music": "Q638",        # music
    "opera": "Q1344",       # opera
    "part": "Q15989253",    # part
    "party": "Q931191",     # party (event)
    "peace": "Q454",        # peace
    "pit": "Q3589",         # pit
    "poison": "Q110153",    # poison
    "pool": "Q192828",      # swimming pool
    "power": "Q25107",      # power
    "revolution": "Q10931",  # revolution
    "robot": "Q11012",      # robot
    "root": "Q111",         # root
    "satellite": "Q13396",  # satellite
    "school": "Q3914",      # school
    "shadow": "Q1419423",   # shadow
    "sleep": "Q7163",       # sleep
    "soul": "Q468",         # soul
    "space": "Q107",        # outer space
    "spirit": "Q7891",      # spirit
    "storm": "Q81054",      # storm
    "strike": "Q15711",     # strike
    "time": "Q11471",       # time
    "trip": "Q61509",       # trip
    "war": "Q198",          # war
    "wind": "Q8094",        # wind
    "witch": "Q328477",     # witch
}

# Properties to extract (removed HasPart/PartOf - too much anatomy noise)
PROPERTY_MAP = {
    "P31": "InstanceOf",
    "P279": "SubclassOf",
    "P366": "UsedFor",
    "P186": "MadeOf",
    "P106": "Occupation",
    "P136": "Genre",
    "P17": "LocatedIn",
    "P30": "Continent",
}

RELATION_WEIGHTS = {
    "InstanceOf": 4.0,
    "SubclassOf": 3.5,
    "UsedFor": 3.0,
    "MadeOf": 2.5,
    "Occupation": 3.5,
    "Genre": 3.0,
    "LocatedIn": 2.0,
    "Continent": 2.0,
}

BLACKLIST = {
    # Meta/generic
    "human", "entity", "object", "thing", "concept", "type", "class",
    "term", "specialized term", "glossary", "feature type", "theme",
    "wikimedia", "disambiguation", "redirect", "taxon",
    # Anatomy (too specific)
    "tooth", "ear", "eye", "nose", "bone", "intestine", "digestive system",
    "duodenum", "thorax", "nasal bone", "ethmoid bone", "human body",
    "breastfeeding", "breathing", "windshield", "furniture leg",
    # Linguistic
    "swadesh list", "common name", "word",
    # Roman/historical noise
    "roman governor", "ancient rome", "legatus pro praetore",
}


def load_words():
    with open(CODENAMES_WORDS_FILE, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


def query_wikidata(sparql):
    headers = {"User-Agent": "CodenamesBot/5.0", "Accept": "application/json"}
    try:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": sparql, "format": "json"},
            headers=headers,
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("results", {}).get("bindings", [])
        elif resp.status_code == 429:
            time.sleep(10)
    except Exception as e:
        print(f"  Query error: {e}")
    return []


def get_qid_for_word(word):
    """Get QID - curated first, then fallback to popularity."""
    if word in CURATED_QIDS:
        return CURATED_QIDS[word]

    # Fallback: get most popular entity
    sparql = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{word}"@en .
      ?item wikibase:sitelinks ?sitelinks .
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT 1
    """
    results = query_wikidata(sparql)
    if results:
        uri = results[0].get("item", {}).get("value", "")
        if "/entity/" in uri:
            return uri.split("/")[-1]
    return None


def get_relations_for_qid(qid, word):
    """Extract relations from specific QID."""
    props = " ".join([f"wdt:{p}" for p in PROPERTY_MAP.keys()])

    sparql = f"""
    SELECT DISTINCT ?prop ?targetLabel WHERE {{
      wd:{qid} ?prop ?target .
      VALUES ?prop {{ {props} }}
      ?target rdfs:label ?targetLabel .
      FILTER(LANG(?targetLabel) = "en")
    }}
    LIMIT 20
    """

    results = query_wikidata(sparql)
    edges = []

    for r in results:
        prop_uri = r.get("prop", {}).get("value", "")
        prop_id = prop_uri.split("/")[-1] if "/prop/direct/" in prop_uri else None
        target = r.get("targetLabel", {}).get("value", "").lower()

        if not prop_id or prop_id not in PROPERTY_MAP:
            continue
        if target in BLACKLIST:
            continue
        if len(target) < 2 or len(target) > 25:
            continue
        if not target.replace(" ", "").replace("-", "").isalpha():
            continue
        if target in word or word in target:
            continue

        relation = PROPERTY_MAP[prop_id]
        edges.append({
            "start": word,
            "relation": relation,
            "end": target,
            "weight": RELATION_WEIGHTS.get(relation, 2.0),
            "source": "wikidata",
            "qid": qid
        })

    return edges


def build_graph():
    print("=== Building Wikidata Graph (v5 - Curated QIDs) ===")
    print(f"Curated mappings: {len(CURATED_QIDS)} words\n")

    words = load_words()
    graph = defaultdict(list)
    curated_used = 0
    fallback_used = 0

    for i, word in enumerate(words):
        if (i + 1) % 20 == 0:
            edges = sum(len(e) for e in graph.values())
            print(f"Processing {i + 1}/{len(words)}... ({edges} edges)")

        is_curated = word in CURATED_QIDS
        qid = get_qid_for_word(word)
        time.sleep(0.2)

        if not qid:
            continue

        if is_curated:
            curated_used += 1
        else:
            fallback_used += 1

        edges = get_relations_for_qid(qid, word)
        time.sleep(0.2)

        # Dedupe
        seen = set()
        for e in edges:
            key = (e["relation"], e["end"])
            if key not in seen:
                seen.add(key)
                graph[word].append(e)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dict(graph), f, indent=2)

    words_with_edges = sum(1 for w in words if graph.get(w))
    total_edges = sum(len(e) for e in graph.values())

    print(f"\n=== Done ===")
    print(f"Words with edges: {words_with_edges}/{len(words)}")
    print(f"Total edges: {total_edges}")
    print(f"Curated QIDs used: {curated_used}")
    print(f"Fallback QIDs used: {fallback_used}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n=== Sample ===")
    for word in ["bat", "bank", "bar", "apple", "australia", "shark", "diamond", "doctor"]:
        if graph.get(word):
            qid = CURATED_QIDS.get(word, "fallback")
            print(f"\n{word} ({qid}):")
            for e in graph[word][:3]:
                print(f"  -> {e['relation']} -> {e['end']}")


if __name__ == "__main__":
    build_graph()
