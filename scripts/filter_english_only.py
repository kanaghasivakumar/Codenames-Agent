#!/usr/bin/env python3
"""
Filter to Keep Only English Words
==================================
Removes non-English words from common_words.txt and conceptnet_graph.json
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
GRAPH_PATH = DATA_DIR / "conceptnet_graph.json"
COMMON_WORDS_PATH = DATA_DIR / "common_words.txt"

# Valid English single letters
VALID_SINGLE_LETTERS = {'a', 'i'}

# Common non-English words to exclude
NON_ENGLISH_WORDS = {
    # German
    'der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'auf', 'für', 'aus',
    'bei', 'nach', 'über', 'vor', 'durch', 'oder', 'wenn', 'auch', 'nur',
    'noch', 'aber', 'wie', 'kann', 'sein', 'wurde', 'werden', 'waren',
    'hatte', 'haben', 'diese', 'einem', 'einer', 'eines', 'mehr', 'sehr',
    'zeit', 'zwei', 'drei', 'jahr', 'neue', 'seit', 'beim', 'gibt', 'immer',
    'gegen', 'unter', 'einem', 'sowie', 'heute', 'ersten', 'wurde', 'bereits',
    'jedoch', 'dabei', 'konnte', 'seinen', 'diesem', 'dieses', 'sollte',
    'beiden', 'während', 'große', 'ihren', 'deutschen', 'deutschen',

    # French
    'les', 'des', 'une', 'est', 'que', 'qui', 'dans', 'pour', 'sur', 'par',
    'avec', 'mais', 'pas', 'tout', 'elle', 'ses', 'aux', 'cette',
    'sont', 'leur', 'bien', 'fait', 'peut', 'tous', 'sans', 'sous', 'peu',
    'donc', 'comme', 'entre', 'faire', 'aussi', 'dont', 'encore',
    'ainsi', 'avoir', 'alors', 'autre', 'notre', 'vers', 'fois',
    'depuis', 'toujours', 'être', 'même', 'après', 'très', 'où', 'été',
    'siècle', 'société', 'jusqu', 'leurs', 'autres', 'ceux', 'cette',
    'avant', 'sous', 'quel', 'quelle', 'faire', 'voir', 'prendre',

    # Spanish
    'los', 'las', 'del', 'una', 'con', 'por', 'para', 'como', 'pero', 'sus',
    'fue', 'todo', 'esta', 'ser', 'entre', 'cuando', 'muy', 'sin',
    'sobre', 'tiene', 'desde', 'donde', 'otro', 'puede', 'todos',
    'parte', 'hacer', 'cada', 'bien', 'mismo', 'antes', 'mejor', 'nuevo',
    'más', 'méxico', 'años', 'después', 'vez', 'solo', 'país', 'nombre',

    # Italian
    'gli', 'dei', 'della', 'che', 'con', 'non', 'sono', 'nel', 'alla',
    'anche', 'stato', 'dalla', 'sua', 'suoi', 'tra', 'nella', 'sul',
    'dopo', 'questo', 'quello', 'essere', 'fatto', 'hanno', 'più',
    'città', 'anni', 'della', 'quale', 'loro', 'così', 'primo', 'altro',

    # Portuguese
    'dos', 'das', 'uma', 'com', 'por', 'para', 'como', 'mais', 'foi', 'seu',
    'sua', 'tem', 'pode', 'esta', 'ainda', 'outro', 'havia', 'desde',

    # Dutch
    'het', 'een', 'van', 'voor', 'met', 'als', 'werd', 'zijn', 'dat', 'hij',
    'naar', 'uit', 'aan', 'tot', 'ook', 'wel', 'nog', 'dan', 'maar', 'bij',

    # Russian transliterated
    'vse', 'eto', 'kak', 'tak', 'chto', 'oni', 'bylo', 'ili', 'est',

    # Latin (unless used in English)
    'vel', 'sed', 'aut', 'nec', 'nam', 'iam', 'hoc', 'sunt', 'quod', 'qua',
}

# Words that look non-English but ARE English - keep these
ENGLISH_EXCEPTIONS = {
    'per', 'via', 'son', 'come', 'can', 'will', 'may', 'one', 'two',
    'also', 'has', 'have', 'had', 'was', 'were', 'been', 'being',
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'her', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old',
    'see', 'way', 'who', 'did', 'get', 'has', 'him', 'his', 'how',
    'let', 'put', 'say', 'too', 'use', 'our', 'out', 'own', 'men',
}


def is_english_word(word):
    """Check if word is likely English."""
    word = word.lower().strip()

    # Must be ASCII only (no accents)
    if not word.isascii():
        return False

    # Must be alphabetic (no periods, numbers)
    if not word.isalpha():
        return False

    # Handle single letters
    if len(word) == 1:
        return word in VALID_SINGLE_LETTERS

    # Minimum length for multi-letter words
    if len(word) < 2:
        return False

    # Maximum length
    if len(word) > 25:
        return False

    # Check exceptions (English words that look foreign)
    if word in ENGLISH_EXCEPTIONS:
        return True

    # Exclude known non-English words
    if word in NON_ENGLISH_WORDS:
        return False

    return True


def main():
    print("=" * 60)
    print("Filtering to English Words Only")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    with open(COMMON_WORDS_PATH) as f:
        common_words = [w.strip().lower() for w in f if w.strip()]

    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    print(f"Common words before: {len(common_words)}")
    print(f"Graph edges before: {sum(len(e) for e in graph.values())}")
    print()

    # Filter common words
    print("Filtering common_words.txt...")
    english_words = []
    removed_words = []

    for word in common_words:
        if is_english_word(word):
            english_words.append(word)
        else:
            removed_words.append(word)

    print(f"  Kept: {len(english_words)}")
    print(f"  Removed: {len(removed_words)}")
    if removed_words:
        print(f"  Sample removed: {removed_words[:30]}")
    print()

    # Filter graph edges
    print("Filtering graph edges...")
    edges_before = sum(len(e) for e in graph.values())
    edges_removed = 0

    for word in graph:
        filtered_edges = []
        for edge in graph[word]:
            clue = edge.get('end', '').lower()
            if is_english_word(clue):
                filtered_edges.append(edge)
            else:
                edges_removed += 1
        graph[word] = filtered_edges

    edges_after = sum(len(e) for e in graph.values())
    print(f"  Edges before: {edges_before}")
    print(f"  Edges after: {edges_after}")
    print(f"  Removed: {edges_removed}")
    print()

    # Remove duplicates and sort
    english_words = sorted(set(english_words))

    # Save filtered data
    print("Saving filtered data...")

    with open(COMMON_WORDS_PATH, 'w') as f:
        f.write('\n'.join(english_words))

    with open(GRAPH_PATH, 'w') as f:
        json.dump(graph, f, indent=2)

    print()
    print("=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"Common words: {len(english_words)}")
    print(f"Graph edges: {edges_after}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
