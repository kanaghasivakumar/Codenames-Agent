

import random
import os
import sys
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import components
from src.vector_engine import VectorEngine
from src.reasoning_engine import ReasoningEngine
from wikipedia_lookup import get_weekly_wikipedia_pageviews

# Configuration
WORD_LIST_PATH = "data/codenames_words.txt"
CONCEPTNET_PKL = "data/conceptnet_english.pkl"
TOP_WORD_LIST_PATH = "data/top_english_words_lower_100000.txt"
WIKI_THRESHOLD = 50


class CodenamesAgentWithReasoning:
    """
    Full Codenames Spymaster Agent with Reasoning.

    Pipeline:
    1. PERCEPTION: Load game board and knowledge bases
    2. NEURAL: Vector engine brainstorms clue candidates
    3. KNOWLEDGE: ConceptNet provides relationships
    4. REASONING: Apply deductive, rule-based, FOPL reasoning
    5. VALIDATION: Wikipedia + Top words filter
    6. DECISION: Select best safe clue
    """

    def __init__(self):
        print("=" * 60)
        print("CODENAMES AGENT WITH REASONING")
        print("=" * 60)

        # Load Vector Engine
        print("\n[1/4] Loading Vector Engine...")
        try:
            self.vector_engine = VectorEngine()
            self.has_vectors = True
        except Exception as e:
            print(f"  Warning: Vector Engine not available: {e}")
            self.vector_engine = None
            self.has_vectors = False

        # Load ConceptNet data
        print("\n[2/4] Loading ConceptNet Knowledge Graph...")
        self.conceptnet_edges = {}
        self.conceptnet_words = set()
        if os.path.exists(CONCEPTNET_PKL):
            with open(CONCEPTNET_PKL, 'rb') as f:
                data = pickle.load(f)
            self.conceptnet_edges = data.get('edges', {})
            self.conceptnet_words = data.get('words', set())
            print(f"  Loaded {len(self.conceptnet_words):,} words")
        else:
            print("  Warning: ConceptNet data not found")

        # Load Reasoning Engine
        print("\n[3/4] Loading Reasoning Engine...")
        self.reasoning = ReasoningEngine()

        # Load facts into reasoning engine
        # if self.conceptnet_edges:
        #     self.reasoning.load_from_conceptnet(self.conceptnet_edges)

        # Load top words and wiki threshold
        print("\n[4/4] Loading Top English Words...")
        self.top_words = set()
        try:
            with open(TOP_WORD_LIST_PATH, 'r') as f:
                self.top_words = set(w.strip().upper() for w in f.readlines() if w.strip())
            print(f"  Loaded {len(self.top_words):,} words")
        except FileNotFoundError:
            print(f"  Warning: {TOP_WORD_LIST_PATH} not found")

        self.wiki_threshold = WIKI_THRESHOLD
        self.wiki_cache = {}

        print("\n" + "=" * 60)
        print("AGENT READY")
        print("=" * 60)

    def generate_board(self):
        """Generate a random Codenames board."""
        with open(WORD_LIST_PATH, 'r') as f:
            words = [w.strip().upper() for w in f.readlines() if w.strip()]
        random.shuffle(words)
        board = words[:25]
        return board[:9], board[9:17], board[17], board[18:]

    def get_wiki_views(self, word: str) -> int:
        """Get Wikipedia pageviews with caching."""
        if word in self.wiki_cache:
            return self.wiki_cache[word]
        try:
            views = get_weekly_wikipedia_pageviews(word.capitalize())
            self.wiki_cache[word] = views
            return views
        except:
            self.wiki_cache[word] = 0
            return 0

    def is_valid_clue(self, clue: str) -> tuple:
        """
        Validate a clue using rule-based reasoning + word filters.

        Rules applied:
        1. Must be alphabetic
        2. Must be 3-15 characters
        3. Must be in top words OR have 50+ Wikipedia views
        """
        clue_upper = clue.upper()

        # Use rule-based reasoner for basic checks
        valid, reasoning = self.reasoning.rules.evaluate_clue(clue, [])

        if not valid:
            return False, reasoning[-1] if reasoning else "Invalid"

        # Check top words
        if clue_upper in self.top_words:
            return True, "In top English words"

        # Check Wikipedia
        views = self.get_wiki_views(clue)
        if views >= self.wiki_threshold:
            return True, f"Wikipedia: {views} views"

        return False, f"Unknown word (wiki: {views})"

    def is_safe(self, clue: str, assassin: str, opponents: List[str] = None) -> tuple:
        """
        Check if clue is safe using FOPL and deductive reasoning.

        Applies:
        1. FOPL inference rules
        2. Deductive transitivity
        3. Relationship distance analysis
        """
        opponents = opponents or []

        # FOPL safety inference
        fopl_safe, fopl_reasoning = self.reasoning.fopl.infer_safety(clue, assassin)

        if not fopl_safe:
            return False, f"FOPL: {fopl_reasoning[-1]}", fopl_reasoning

        # Deductive reasoning with ConceptNet relationships
        relationships = [(w1, w2) for w1, w2 in self.reasoning.fopl.get_all('RelatedTo')]
        deductive_safe, deductive_reasoning = self.reasoning.deductive.reason_about_safety(
            clue, assassin, relationships
        )

        if not deductive_safe:
            return False, f"Deductive: {deductive_reasoning[-1]}", deductive_reasoning

        # Check opponents
        for opponent in opponents:
            opp_safe, opp_reasoning = self.reasoning.fopl.infer_safety(clue, opponent)
            if not opp_safe:
                return True, f"Warning: related to opponent '{opponent}'", opp_reasoning

        return True, "Safe (verified by reasoning)", fopl_reasoning + deductive_reasoning

    def get_conceptnet_related(self, word: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Get related words from ConceptNet."""
        word_lower = word.lower()
        edges = self.conceptnet_edges.get(word_lower, [])

        # Aggregate weights
        word_weights = defaultdict(float)
        for relation, other_word, weight in edges:
            word_weights[other_word] += weight

        sorted_words = sorted(word_weights.items(), key=lambda x: -x[1])
        return sorted_words[:limit]

    def find_clues_with_reasoning(self, targets: List[str], avoid: List[str]) -> List[dict]:
        """
        Find clues using combined reasoning approach.

        Steps:
        1. Abductive: Find words connecting targets
        2. FOPL: Verify safety
        3. Deductive: Check transitive relationships
        4. Rule-based: Validate clue format
        5. Wikipedia/Top words: Final validation
        """
        targets_upper = [t.upper() for t in targets]
        avoid_upper = [a.upper() for a in avoid]
        avoid_set = set(avoid_upper)
        target_set = set(targets_upper)

        candidates = []

        # Step 1: Find common connections (abductive reasoning)
        print("\n  [Reasoning Step 1] Abductive - Finding connecting words...")

        connection_counts = defaultdict(lambda: {'count': 0, 'targets': [], 'weight': 0})

        for target in targets:
            related = self.get_conceptnet_related(target, limit=50)
            for word, weight in related:
                word_upper = word.upper()
                if word_upper not in target_set and word_upper not in avoid_set:
                    connection_counts[word_upper]['count'] += 1
                    connection_counts[word_upper]['targets'].append(target.upper())
                    connection_counts[word_upper]['weight'] += weight

        # Filter to clues connecting 2+ targets
        potential_clues = [
            (word, data) for word, data in connection_counts.items()
            if data['count'] >= 2
        ]

        print(f"    Found {len(potential_clues)} potential clues")

        # Step 2-5: Evaluate each candidate
        print("  [Reasoning Step 2-5] Evaluating candidates...")

        for clue, data in potential_clues[:50]:  # Limit for speed
            # Step 2: FOPL safety check
            assassin = avoid[0] if avoid else "BOMB"
            fopl_safe, fopl_reason = self.reasoning.fopl.infer_safety(clue, assassin)

            if not fopl_safe:
                continue

            # Step 3: Rule-based validation
            rules_pass, rules_reasoning = self.reasoning.rules.evaluate_clue(
                clue, targets_upper + avoid_upper
            )

            if not rules_pass:
                continue

            # Step 4: Word validation (top words + Wikipedia)
            valid, valid_reason = self.is_valid_clue(clue)

            if not valid:
                continue

            # Calculate score
            score = data['weight'] * data['count']

            candidates.append({
                'clue': clue,
                'targets': data['targets'],
                'count': data['count'],
                'score': score,
                'reasoning': {
                    'fopl': fopl_reason,
                    'rules': rules_reasoning,
                    'validation': valid_reason
                }
            })

        # Sort by count then score
        candidates.sort(key=lambda x: (-x['count'], -x['score']))

        return candidates[:10]

    def play_turn(self, red_team, blue_team, assassin, neutral=None):
        """
        Play a turn as Spymaster using full reasoning pipeline.
        """
        neutral = neutral or []
        all_board = red_team + blue_team + [assassin] + neutral
        avoid = blue_team + [assassin]

        print(f"\n{'='*60}")
        print("GAME BOARD")
        print(f"{'='*60}")
        print(f"YOUR TEAM (Red):    {red_team}")
        print(f"OPPONENT (Blue):    {blue_team}")
        print(f"ASSASSIN:           {assassin}")
        print(f"{'='*60}")

        # Setup game in reasoning engine
        self.reasoning.setup_game(red_team, blue_team, assassin, neutral)

        # Add relationships to FOPL
        print("\n[STAGE 1] Loading relationships into reasoning engine...")
        for word in all_board:
            edges = self.conceptnet_edges.get(word.lower(), [])
            for relation, other, weight in edges[:20]:
                if relation == 'RelatedTo':
                    self.reasoning.fopl.add_fact('RelatedTo', word, other)
                elif relation == 'Synonym':
                    self.reasoning.fopl.add_fact('Synonym', word, other)
                elif relation == 'Antonym':
                    self.reasoning.fopl.add_fact('Antonym', word, other)

        # Apply transitivity
        new_facts = self.reasoning.fopl.apply_transitivity('RelatedTo')
        print(f"  Inferred {new_facts} new relationships via deductive reasoning")

        # Stage 2: Vector Engine candidates
        print("\n[STAGE 2] Neural - Vector Engine brainstorming...")
        vector_candidates = []
        if self.has_vectors:
            try:
                vector_candidates = self.vector_engine.get_clue(red_team, blue_team, assassin, n_clues=10)
                print(f"  Found {len(vector_candidates)} vector candidates")
            except Exception as e:
                print(f"  Vector engine error: {e}")

        # Stage 3: Reasoning-based candidates
        print("\n[STAGE 3] Reasoning - Finding clues with logic...")
        reasoning_candidates = self.find_clues_with_reasoning(red_team, avoid)
        print(f"  Found {len(reasoning_candidates)} reasoning candidates")

        # Stage 4: Combine and evaluate all candidates
        print("\n[STAGE 4] Combining and evaluating all candidates...")
        print("-" * 60)

        all_candidates = []

        # Process vector candidates
        for clue, targets, score in vector_candidates:
            clue_upper = clue.upper()

            if clue_upper in [w.upper() for w in all_board]:
                continue

            # Validate
            valid, valid_reason = self.is_valid_clue(clue)
            if not valid:
                print(f"  ❌ '{clue}' - {valid_reason}")
                continue

            # Safety check with reasoning
            safe, safety_reason, reasoning_chain = self.is_safe(clue, assassin, blue_team)
            if not safe:
                print(f"  ❌ '{clue}' - {safety_reason}")
                continue

            print(f"  ✓ '{clue}' for {len(targets)} (vector)")
            all_candidates.append({
                'clue': clue_upper,
                'targets': [t.upper() for t in targets],
                'count': len(targets),
                'score': score,
                'source': 'vector',
                'reasoning': reasoning_chain[:3]
            })

        # Add reasoning candidates
        for cand in reasoning_candidates:
            if cand['clue'] not in [c['clue'] for c in all_candidates]:
                print(f"  ✓ '{cand['clue']}' for {cand['count']} (reasoning)")
                all_candidates.append({
                    'clue': cand['clue'],
                    'targets': cand['targets'],
                    'count': cand['count'],
                    'score': cand['score'],
                    'source': 'reasoning',
                    'reasoning': [cand['reasoning']['fopl']]
                })

        print("-" * 60)

        # Stage 5: Final decision
        print("\n[STAGE 5] Making final decision...")

        if not all_candidates:
            print("\n>>> PASS (No valid clues found)")
            return None

        # Sort by count (more targets = better)
        all_candidates.sort(key=lambda x: (-x['count'], -x['score']))

        best = all_candidates[0]

        # Generate reasoning explanation
        explanation = self.reasoning.explain_decision(
            best['clue'], best['targets'], assassin
        )

        print(f"\n{'='*60}")
        print("FINAL DECISION")
        print(f"{'='*60}")
        print(f"  CLUE:     '{best['clue']}'")
        print(f"  COUNT:    {best['count']}")
        print(f"  TARGETS:  {best['targets']}")
        print(f"  SOURCE:   {best['source']}")
        print(f"  SCORE:    {best['score']:.2f}")
        print(explanation)
        print(f"{'='*60}")

        return best


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        CODENAMES SPYMASTER WITH FULL REASONING           ║
    ║                                                          ║
    ║  Components:                                             ║
    ║  - GloVe Vectors (Neural)                                ║
    ║  - ConceptNet (Knowledge Graph)                          ║
    ║  - Reasoning Engine (FOPL, Deductive, Rules)             ║
    ║  - Wikipedia + Top Words (Validation)                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Initialize agent
    agent = CodenamesAgentWithReasoning()

    # Test with specific scenario
    print("\n" + "=" * 60)
    print("SCENARIO 1: Chess Theme")
    print("=" * 60)

    agent.play_turn(
        red_team=["KING", "QUEEN", "KNIGHT", "CASTLE", "CROWN"],
        blue_team=["APPLE", "ORANGE", "BANANA"],
        assassin="BOMB",
        neutral=["CHAIR", "TABLE"]
    )

    # Test with random board
    print("\n" + "=" * 60)
    print("SCENARIO 2: Random Board")
    print("=" * 60)

    red, blue, assassin, neutral = agent.generate_board()
    agent.play_turn(red, blue, assassin, neutral)
