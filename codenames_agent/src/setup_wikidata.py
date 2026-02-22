"""
Wikidata Entity Extraction for Codenames Agent
================================================

This script extracts entities and their relationships from Wikidata
to enable clue generation using pop culture, celebrities, places, etc.

Entity Types Extracted:
- People (musicians, actors, athletes, politicians, etc.)
- Places (cities, countries, landmarks)
- Organizations (companies, brands)
- Creative Works (movies, TV shows, books, songs)
- Concepts (genres, occupations, etc.)

Output: wikidata_entities.pkl containing:
- entities: dict mapping entity name -> properties
- relations: dict mapping entity -> [(relation, other_entity, weight), ...]
- categories: dict mapping category -> list of entities
"""

import os
import sys
import pickle
import time
import requests
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WIKIDATA_PKL = os.path.join(DATA_DIR, "wikidata_entities.pkl")
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests

# Categories to extract with their Wikidata property queries
ENTITY_CATEGORIES = {
    "musicians": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?genreLabel ?occupationLabel WHERE {
                ?item wdt:P106 wd:Q177220 .  # occupation: singer
                ?item wdt:P136 ?genre .       # has genre
                OPTIONAL { ?item wdt:P106 ?occupation . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 5000
        """,
        "relations": ["genre", "occupation"]
    },
    "actors": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?genreLabel WHERE {
                ?item wdt:P106 wd:Q33999 .   # occupation: actor
                OPTIONAL { ?item wdt:P136 ?genre . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 5000
        """,
        "relations": ["genre"]
    },
    "athletes": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?sportLabel ?teamLabel WHERE {
                ?item wdt:P106 wd:Q2066131 .  # occupation: athlete
                OPTIONAL { ?item wdt:P641 ?sport . }
                OPTIONAL { ?item wdt:P54 ?team . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 5000
        """,
        "relations": ["sport", "team"]
    },
    "movies": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?genreLabel ?directorLabel WHERE {
                ?item wdt:P31 wd:Q11424 .     # instance of: film
                ?item wdt:P577 ?date .         # has release date
                FILTER(YEAR(?date) >= 1990)
                OPTIONAL { ?item wdt:P136 ?genre . }
                OPTIONAL { ?item wdt:P57 ?director . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 5000
        """,
        "relations": ["genre", "director"]
    },
    "tv_shows": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?genreLabel WHERE {
                ?item wdt:P31 wd:Q5398426 .   # instance of: TV series
                OPTIONAL { ?item wdt:P136 ?genre . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 3000
        """,
        "relations": ["genre"]
    },
    "companies": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?industryLabel ?countryLabel WHERE {
                ?item wdt:P31 wd:Q4830453 .   # instance of: business
                OPTIONAL { ?item wdt:P452 ?industry . }
                OPTIONAL { ?item wdt:P17 ?country . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 3000
        """,
        "relations": ["industry", "country"]
    },
    "cities": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?countryLabel WHERE {
                ?item wdt:P31 wd:Q515 .       # instance of: city
                ?item wdt:P17 ?country .
                ?item wdt:P1082 ?pop .         # has population
                FILTER(?pop > 500000)          # major cities only
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 2000
        """,
        "relations": ["country"]
    },
    "countries": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?continentLabel WHERE {
                ?item wdt:P31 wd:Q6256 .      # instance of: country
                OPTIONAL { ?item wdt:P30 ?continent . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 500
        """,
        "relations": ["continent"]
    },
    "sports": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel WHERE {
                ?item wdt:P31 wd:Q31629 .     # instance of: sport
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 500
        """,
        "relations": []
    },
    "music_genres": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel WHERE {
                ?item wdt:P31 wd:Q188451 .    # instance of: music genre
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 500
        """,
        "relations": []
    },
    "foods": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?countryLabel WHERE {
                ?item wdt:P31 wd:Q2095 .      # instance of: food
                OPTIONAL { ?item wdt:P495 ?country . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 2000
        """,
        "relations": ["country"]
    },
    "video_games": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?genreLabel ?platformLabel WHERE {
                ?item wdt:P31 wd:Q7889 .      # instance of: video game
                OPTIONAL { ?item wdt:P136 ?genre . }
                OPTIONAL { ?item wdt:P400 ?platform . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 3000
        """,
        "relations": ["genre", "platform"]
    },
    "books": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?authorLabel ?genreLabel WHERE {
                ?item wdt:P31 wd:Q7725634 .   # instance of: literary work
                OPTIONAL { ?item wdt:P50 ?author . }
                OPTIONAL { ?item wdt:P136 ?genre . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 3000
        """,
        "relations": ["author", "genre"]
    },
    "fictional_characters": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?universeLabel WHERE {
                ?item wdt:P31 wd:Q95074 .     # instance of: fictional character
                OPTIONAL { ?item wdt:P1080 ?universe . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 3000
        """,
        "relations": ["universe"]
    },
    "landmarks": {
        "query": """
            SELECT DISTINCT ?item ?itemLabel ?countryLabel WHERE {
                ?item wdt:P31 wd:Q570116 .    # instance of: tourist attraction
                OPTIONAL { ?item wdt:P17 ?country . }
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 2000
        """,
        "relations": ["country"]
    }
}


def query_wikidata(sparql_query: str) -> List[Dict]:
    """Execute a SPARQL query against Wikidata endpoint."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "CodenamesAgent/1.0 (Educational Project)"
    }

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": sparql_query, "format": "json"},
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", {}).get("bindings", [])
    except requests.exceptions.RequestException as e:
        print(f"  Error querying Wikidata: {e}")
        return []


def clean_label(label: str) -> str:
    """Clean entity label for use as clue."""
    if not label:
        return ""
    # Remove parenthetical disambiguators
    if "(" in label:
        label = label.split("(")[0].strip()
    # Convert to lowercase, replace spaces with underscores for multi-word
    label = label.strip().lower()
    return label


def is_valid_clue_candidate(label: str) -> bool:
    """Check if label could be a valid Codenames clue."""
    if not label or len(label) < 2:
        return False
    # Skip labels that are just numbers or IDs
    if label.startswith("q") and label[1:].isdigit():
        return False
    # Skip very long labels
    if len(label) > 30:
        return False
    return True


def extract_entities() -> Tuple[Dict, Dict, Dict]:
    """Extract entities from Wikidata for all categories."""

    entities = {}           # name -> {category, properties}
    relations = defaultdict(list)  # entity -> [(relation, other, weight)]
    categories = defaultdict(list)  # category -> [entities]

    print("\n" + "=" * 60)
    print("EXTRACTING ENTITIES FROM WIKIDATA")
    print("=" * 60)

    for category, config in ENTITY_CATEGORIES.items():
        print(f"\n[{category.upper()}]")
        print(f"  Querying Wikidata...")

        results = query_wikidata(config["query"])

        if not results:
            print(f"  No results or error. Skipping.")
            time.sleep(REQUEST_DELAY)
            continue

        count = 0
        for item in results:
            # Get main entity label
            label = item.get("itemLabel", {}).get("value", "")
            label = clean_label(label)

            if not is_valid_clue_candidate(label):
                continue

            # Store entity
            if label not in entities:
                entities[label] = {
                    "category": category,
                    "properties": {}
                }
                categories[category].append(label)
                count += 1

            # Extract relations
            for rel in config["relations"]:
                rel_key = f"{rel}Label"
                if rel_key in item and item[rel_key].get("value"):
                    rel_value = clean_label(item[rel_key]["value"])
                    if is_valid_clue_candidate(rel_value):
                        # Add bidirectional relations
                        relations[label].append((rel, rel_value, 1.0))
                        relations[rel_value].append((f"has_{rel}", label, 1.0))

                        # Also add the related entity
                        if rel_value not in entities:
                            entities[rel_value] = {
                                "category": f"{category}_{rel}",
                                "properties": {}
                            }

        print(f"  Extracted {count} entities")
        time.sleep(REQUEST_DELAY)  # Rate limiting

    return dict(entities), dict(relations), dict(categories)


def add_category_relations(entities: Dict, relations: Dict, categories: Dict):
    """Add IsA relations based on categories."""
    print("\n[ADDING CATEGORY RELATIONS]")

    category_mapping = {
        "musicians": ["person", "artist", "celebrity", "music"],
        "actors": ["person", "artist", "celebrity", "entertainment"],
        "athletes": ["person", "sports", "celebrity"],
        "movies": ["film", "entertainment", "media"],
        "tv_shows": ["television", "entertainment", "media"],
        "companies": ["business", "organization", "brand"],
        "cities": ["place", "location", "urban"],
        "countries": ["place", "location", "nation"],
        "sports": ["activity", "game", "competition"],
        "music_genres": ["music", "genre", "style"],
        "foods": ["food", "cuisine", "eating"],
        "video_games": ["game", "entertainment", "gaming"],
        "books": ["literature", "reading", "writing"],
        "fictional_characters": ["character", "fiction", "story"],
        "landmarks": ["place", "location", "tourism"]
    }

    count = 0
    # Convert to defaultdict if needed
    from collections import defaultdict
    if not isinstance(relations, defaultdict):
        relations = defaultdict(list, relations)

    for category, entity_list in categories.items():
        if category in category_mapping:
            for entity in entity_list:
                for parent_category in category_mapping[category]:
                    relations[entity].append(("IsA", parent_category, 0.8))
                    relations[parent_category].append(("HasInstance", entity, 0.5))
                    count += 1

    print(f"  Added {count} category relations")
    return dict(relations)  # Return the updated relations


def save_data(entities: Dict, relations: Dict, categories: Dict):
    """Save extracted data to pickle file."""
    print("\n[SAVING DATA]")

    os.makedirs(DATA_DIR, exist_ok=True)

    data = {
        "entities": entities,
        "relations": relations,
        "categories": categories,
        "metadata": {
            "source": "Wikidata",
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entity_count": len(entities),
            "relation_count": sum(len(v) for v in relations.values())
        }
    }

    with open(WIKIDATA_PKL, "wb") as f:
        pickle.dump(data, f)

    size = os.path.getsize(WIKIDATA_PKL) / (1024 * 1024)
    print(f"  Saved to: {WIKIDATA_PKL}")
    print(f"  File size: {size:.2f} MB")
    print(f"  Total entities: {len(entities):,}")
    print(f"  Total relations: {sum(len(v) for v in relations.values()):,}")


def verify_setup():
    """Verify the setup and show sample data."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    with open(WIKIDATA_PKL, "rb") as f:
        data = pickle.load(f)

    entities = data["entities"]
    relations = data["relations"]
    categories = data["categories"]

    print("\n[CATEGORY COUNTS]")
    for cat, ents in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(ents)} entities")

    print("\n[SAMPLE ENTITIES]")
    sample_entities = ["taylor swift", "beyonce", "avengers", "nike", "paris", "pizza"]
    for entity in sample_entities:
        if entity in entities:
            rels = relations.get(entity, [])[:5]
            print(f"  '{entity}': {len(relations.get(entity, []))} relations")
            for rel, target, weight in rels:
                print(f"    → {rel}: {target}")
        else:
            # Try to find similar
            matches = [e for e in entities if entity in e][:3]
            if matches:
                print(f"  '{entity}' not found. Similar: {matches}")
            else:
                print(f"  '{entity}' not found")

    print("\n[SUCCESS] Wikidata entities ready for use!")


def main():
    """Main setup function."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           WIKIDATA ENTITY EXTRACTION                     ║
    ║                                                          ║
    ║  Extracts entities for Codenames clue generation:        ║
    ║  - Celebrities (musicians, actors, athletes)             ║
    ║  - Places (cities, countries, landmarks)                 ║
    ║  - Pop Culture (movies, TV, games, books)                ║
    ║  - Organizations (companies, brands)                     ║
    ║                                                          ║
    ║  This may take 10-15 minutes due to API rate limits.     ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Check if already exists
    if os.path.exists(WIKIDATA_PKL):
        size = os.path.getsize(WIKIDATA_PKL) / (1024 * 1024)
        print(f"Wikidata data already exists ({size:.2f} MB)")
        response = input("Re-extract? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing data.")
            verify_setup()
            return

    # Extract entities
    entities, relations, categories = extract_entities()

    if not entities:
        print("\nERROR: No entities extracted. Check your internet connection.")
        return

    # Add category relations
    relations = add_category_relations(entities, relations, categories)

    # Save
    save_data(entities, relations, categories)

    # Verify
    verify_setup()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
