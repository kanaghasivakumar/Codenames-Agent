"""
Reasoning Engine for Codenames Agent
=====================================
Implements various reasoning techniques for clue generation and safety checking.

Reasoning Types:
1. Deductive Reasoning - If A implies B, and B implies C, then A implies C
2. Rule-Based Reasoning - Encode and apply Codenames game rules
3. First Order Predicate Logic (FOPC) - Formal logical representation
4. Abductive Reasoning - Given observation, find best explanation

This module integrates with:
- ConceptNet Knowledge Graph (relationships)
- Wikipedia Pageviews (word validation)
- Top English Words (clue filtering)

"""

import os
import sys
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import pyDatalog for FOPC
try:
    from pyDatalog import pyDatalog
    HAS_PYDATALOG = True
except ImportError:
    HAS_PYDATALOG = False
    print("Note: pyDatalog not installed. FOPC features limited.")
    print("Install with: pip install pyDatalog")


# =============================================================================
# DATA CLASSES
# =============================================================================

class WordType(Enum):
    """Types of words in Codenames."""
    TEAM = "team"           # Your team's words
    OPPONENT = "opponent"   # Opponent's words
    ASSASSIN = "assassin"   # The assassin word
    NEUTRAL = "neutral"     # Neutral bystander words
    CLUE = "clue"           # Potential clue word


@dataclass
class Rule:
    """A rule in the reasoning system."""
    name: str
    condition: Callable
    action: str
    priority: int = 0
    description: str = ""


@dataclass
class Inference:
    """Result of an inference."""
    conclusion: str
    confidence: float
    reasoning_chain: List[str]
    rule_used: str


# =============================================================================
# FIRST ORDER PREDICATE LOGIC (FOPC)
# =============================================================================

class FOPLReasoner:
    """
    First Order Predicate Logic Reasoner.

    Predicates:
    - RelatedTo(X, Y) : X is related to Y
    - IsA(X, Y) : X is a type of Y
    - Synonym(X, Y) : X and Y have same meaning
    - Antonym(X, Y) : X and Y are opposites
    - Safe(Clue, Assassin) : Clue is safe from Assassin
    - ValidClue(X) : X is a valid clue word
    - ConnectsTo(Clue, Target) : Clue connects to Target word

    Rules (in logic form):
    - ∀x,y: RelatedTo(x,y) ∧ RelatedTo(y,z) → RelatedTo(x,z) [Transitivity]
    - ∀x,y: Synonym(x,y) → RelatedTo(x,y)
    - ∀x,y: Antonym(x,y) → ¬Safe(x,y) [Antonyms are dangerous]
    - ∀x: ValidClue(x) ← InTopWords(x) ∨ WikiViews(x) > 50
    """

    def __init__(self):
        self.facts = defaultdict(set)  # predicate -> set of tuples
        self.rules = []

        # Define predicates
        self.predicates = {
            'RelatedTo': 2,    # (word1, word2)
            'IsA': 2,          # (instance, category)
            'Synonym': 2,      # (word1, word2)
            'Antonym': 2,      # (word1, word2)
            'PartOf': 2,       # (part, whole)
            'HasProperty': 2,  # (thing, property)
            'Safe': 2,         # (clue, assassin)
            'Dangerous': 2,    # (clue, assassin)
            'ValidClue': 1,    # (word)
            'TeamWord': 1,     # (word)
            'OpponentWord': 1, # (word)
            'AssassinWord': 1, # (word)
            'ConnectsTo': 2,   # (clue, target)
        }

    def add_fact(self, predicate: str, *args):
        """Add a fact to the knowledge base."""
        if predicate in self.predicates:
            expected_arity = self.predicates[predicate]
            if len(args) == expected_arity:
                self.facts[predicate].add(tuple(arg.upper() for arg in args))

    def query(self, predicate: str, *args) -> bool:
        """Query if a fact is true."""
        return tuple(arg.upper() for arg in args) in self.facts[predicate]

    def get_all(self, predicate: str) -> Set[tuple]:
        """Get all facts for a predicate."""
        return self.facts[predicate]

    def apply_transitivity(self, predicate: str = 'RelatedTo'):
        """
        Apply transitivity rule: If R(a,b) and R(b,c), then R(a,c).

        This is deductive reasoning - deriving new facts from existing ones.
        """
        facts = list(self.facts[predicate])
        new_facts = set()

        # Build adjacency for faster lookup
        adjacency = defaultdict(set)
        for a, b in facts:
            adjacency[a].add(b)

        # Find transitive connections (depth 2)
        for a, b in facts:
            for c in adjacency.get(b, set()):
                if c != a:  # Avoid self-loops
                    new_facts.add((a, c))

        # Add new facts
        for fact in new_facts:
            if fact not in self.facts[predicate]:
                self.facts[predicate].add(fact)

        return len(new_facts)

    def infer_safety(self, clue: str, assassin: str) -> Tuple[bool, List[str]]:
        """
        Infer if a clue is safe using logical rules.

        Rules:
        1. Antonym(clue, assassin) → Dangerous(clue, assassin)
        2. RelatedTo(clue, assassin) with high weight → Dangerous
        3. ¬Dangerous(clue, assassin) → Safe(clue, assassin)
        """
        clue = clue.upper()
        assassin = assassin.upper()
        reasoning = []

        # Rule 1: Check for antonym relationship
        if self.query('Antonym', clue, assassin):
            reasoning.append(f"Antonym({clue}, {assassin}) is TRUE")
            reasoning.append(f"By rule: Antonym(X,Y) → Dangerous(X,Y)")
            reasoning.append(f"Therefore: Dangerous({clue}, {assassin})")
            self.add_fact('Dangerous', clue, assassin)
            return False, reasoning

        # Rule 2: Check for direct relation
        if self.query('RelatedTo', clue, assassin):
            reasoning.append(f"RelatedTo({clue}, {assassin}) is TRUE")
            reasoning.append(f"By rule: RelatedTo(X,Y) → Dangerous(X,Y)")
            reasoning.append(f"Therefore: Dangerous({clue}, {assassin})")
            self.add_fact('Dangerous', clue, assassin)
            return False, reasoning

        # Rule 3: Check for synonym to assassin
        if self.query('Synonym', clue, assassin):
            reasoning.append(f"Synonym({clue}, {assassin}) is TRUE")
            reasoning.append(f"By rule: Synonym(X,Y) → Dangerous(X,Y)")
            reasoning.append(f"Therefore: Dangerous({clue}, {assassin})")
            self.add_fact('Dangerous', clue, assassin)
            return False, reasoning

        # No dangerous relationship found
        reasoning.append(f"No direct relationship between {clue} and {assassin}")
        reasoning.append(f"By rule: ¬Dangerous(X,Y) → Safe(X,Y)")
        reasoning.append(f"Therefore: Safe({clue}, {assassin})")
        self.add_fact('Safe', clue, assassin)
        return True, reasoning

    def find_connecting_clues(self, targets: List[str]) -> List[Tuple[str, List[str], List[str]]]:
        """
        Find clues that connect to multiple targets using logical inference.

        Rule: ConnectsTo(Clue, T1) ∧ ConnectsTo(Clue, T2) → GoodClue(Clue, [T1, T2])
        """
        targets = [t.upper() for t in targets]
        target_set = set(targets)

        # Find all potential clues (words related to targets)
        clue_connections = defaultdict(set)

        for predicate in ['RelatedTo', 'Synonym', 'IsA']:
            for fact in self.facts[predicate]:
                word1, word2 = fact
                if word1 in target_set:
                    clue_connections[word2].add(word1)
                if word2 in target_set:
                    clue_connections[word1].add(word2)

        # Find clues connecting 2+ targets
        results = []
        for clue, connected in clue_connections.items():
            if clue in target_set:  # Skip board words
                continue
            if len(connected) >= 2:
                reasoning = [
                    f"For clue '{clue}':",
                    *[f"  ConnectsTo({clue}, {t})" for t in connected],
                    f"By rule: Multiple connections → GoodClue({clue})"
                ]
                results.append((clue, list(connected), reasoning))

        # Sort by number of connections
        results.sort(key=lambda x: -len(x[1]))
        return results


# =============================================================================
# RULE-BASED REASONING ENGINE
# =============================================================================

class RuleBasedReasoner:
    """
    Rule-Based Reasoning Engine for Codenames.

    Encodes game rules and applies them to make decisions.

    Game Rules:
    1. Clue cannot be a word on the board
    2. Clue must be a single word
    3. Clue should connect to team words
    4. Clue should NOT connect to assassin
    5. Clue should NOT connect to opponent words
    6. Clue should be a known/common word
    """

    def __init__(self):
        self.rules = []
        self.facts = {}
        self._setup_rules()

    def _setup_rules(self):
        """Define the game rules."""

        # Rule 1: Clue cannot be a board word
        self.rules.append(Rule(
            name="NOT_BOARD_WORD",
            condition=lambda clue, board: clue.upper() not in [w.upper() for w in board],
            action="ACCEPT",
            priority=100,
            description="Clue cannot be a word on the board"
        ))

        # Rule 2: Clue must be alphabetic
        self.rules.append(Rule(
            name="ALPHABETIC",
            condition=lambda clue, board: clue.isalpha(),
            action="ACCEPT",
            priority=99,
            description="Clue must contain only letters"
        ))

        # Rule 3: Clue must be reasonable length
        self.rules.append(Rule(
            name="LENGTH",
            condition=lambda clue, board: 3 <= len(clue) <= 15,
            action="ACCEPT",
            priority=98,
            description="Clue must be 3-15 characters"
        ))

        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: -r.priority)

    def evaluate_clue(self, clue: str, board_words: List[str]) -> Tuple[bool, List[str]]:
        """
        Evaluate a clue against all rules.

        Returns:
            (passes_all_rules, list_of_reasoning_steps)
        """
        reasoning = []

        for rule in self.rules:
            try:
                result = rule.condition(clue, board_words)
                if result:
                    reasoning.append(f"✓ Rule '{rule.name}': {rule.description}")
                else:
                    reasoning.append(f"✗ Rule '{rule.name}' FAILED: {rule.description}")
                    return False, reasoning
            except Exception as e:
                reasoning.append(f"? Rule '{rule.name}' ERROR: {str(e)}")

        reasoning.append("All rules passed")
        return True, reasoning

    def add_custom_rule(self, name: str, condition: Callable, description: str, priority: int = 50):
        """Add a custom rule."""
        self.rules.append(Rule(
            name=name,
            condition=condition,
            action="ACCEPT",
            priority=priority,
            description=description
        ))
        self.rules.sort(key=lambda r: -r.priority)


# =============================================================================
# DEDUCTIVE REASONING ENGINE
# =============================================================================

class DeductiveReasoner:
    """
    Deductive Reasoning Engine.

    Applies deductive logic:
    - Modus Ponens: If P→Q and P, then Q
    - Modus Tollens: If P→Q and ¬Q, then ¬P
    - Hypothetical Syllogism: If P→Q and Q→R, then P→R
    - Disjunctive Syllogism: If P∨Q and ¬P, then Q

    Used for:
    - Inferring safety from relationships
    - Finding indirect connections
    - Validating clue choices
    """

    def __init__(self):
        self.implications = []  # List of (antecedent, consequent) pairs
        self.facts = set()      # Known true facts

    def add_implication(self, antecedent: str, consequent: str):
        """Add an implication: antecedent → consequent"""
        self.implications.append((antecedent, consequent))

    def add_fact(self, fact: str):
        """Add a known fact."""
        self.facts.add(fact)

    def modus_ponens(self) -> List[Tuple[str, List[str]]]:
        """
        Apply Modus Ponens: If P→Q and P is true, then Q is true.

        Returns list of (new_fact, reasoning_chain)
        """
        new_inferences = []

        for antecedent, consequent in self.implications:
            if antecedent in self.facts and consequent not in self.facts:
                reasoning = [
                    f"Given: {antecedent} → {consequent}",
                    f"Given: {antecedent} is TRUE",
                    f"By Modus Ponens: {consequent} is TRUE"
                ]
                self.facts.add(consequent)
                new_inferences.append((consequent, reasoning))

        return new_inferences

    def hypothetical_syllogism(self) -> List[Tuple[str, str, List[str]]]:
        """
        Apply Hypothetical Syllogism: If P→Q and Q→R, then P→R.

        Returns list of (new_antecedent, new_consequent, reasoning)
        """
        new_implications = []

        for p, q in self.implications:
            for q2, r in self.implications:
                if q == q2 and (p, r) not in self.implications:
                    reasoning = [
                        f"Given: {p} → {q}",
                        f"Given: {q} → {r}",
                        f"By Hypothetical Syllogism: {p} → {r}"
                    ]
                    self.implications.append((p, r))
                    new_implications.append((p, r, reasoning))

        return new_implications

    def reason_about_safety(self, clue: str, assassin: str,
                            relationships: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
        """
        Use deductive reasoning to determine if clue is safe.

        Logic:
        - If clue is related to X and X is related to assassin → clue might be dangerous
        - If clue is directly related to assassin → clue is dangerous
        """
        clue = clue.upper()
        assassin = assassin.upper()
        reasoning = []

        # Build relationship graph
        related_to = defaultdict(set)
        for a, b in relationships:
            related_to[a.upper()].add(b.upper())
            related_to[b.upper()].add(a.upper())

        # Check direct relationship
        if assassin in related_to.get(clue, set()):
            reasoning.append(f"Premise: RelatedTo({clue}, {assassin})")
            reasoning.append(f"Rule: RelatedTo(X, Assassin) → Dangerous(X)")
            reasoning.append(f"Conclusion: {clue} is DANGEROUS")
            return False, reasoning

        # Check indirect relationship (distance 2)
        for intermediate in related_to.get(clue, set()):
            if assassin in related_to.get(intermediate, set()):
                reasoning.append(f"Premise: RelatedTo({clue}, {intermediate})")
                reasoning.append(f"Premise: RelatedTo({intermediate}, {assassin})")
                reasoning.append(f"Rule: Transitivity → RelatedTo({clue}, {assassin})")
                reasoning.append(f"Conclusion: {clue} is RISKY (indirect connection)")
                return True, reasoning  # Risky but not forbidden

        # No relationship found
        reasoning.append(f"No relationship found between {clue} and {assassin}")
        reasoning.append(f"Conclusion: {clue} is SAFE")
        return True, reasoning


# =============================================================================
# ABDUCTIVE REASONING ENGINE
# =============================================================================

class AbductiveReasoner:
    """
    Abductive Reasoning Engine.

    Given an observation, find the best explanation.

    Used for:
    - Given target words, find the best clue (best explanation for why they're related)
    - Given a clue, infer which words the opponent might guess
    """

    def __init__(self):
        self.observations = []
        self.hypotheses = []

    def find_best_clue(self, targets: List[str],
                       word_relationships: Dict[str, List[str]]) -> List[Tuple[str, float, List[str]]]:
        """
        Abductive reasoning: Given targets, find best clue.

        Observation: These words should be guessed together
        Hypothesis: There exists a clue that connects them
        Find: Best explanation (clue) for this observation
        """
        targets = [t.upper() for t in targets]
        reasoning_results = []

        # Find common connections
        common_words = defaultdict(lambda: {'count': 0, 'targets': []})

        for target in targets:
            related = word_relationships.get(target.lower(), [])
            for word in related:
                word_upper = word.upper()
                if word_upper not in targets:
                    common_words[word_upper]['count'] += 1
                    common_words[word_upper]['targets'].append(target)

        # Rank by how many targets they connect
        for word, data in common_words.items():
            if data['count'] >= 2:
                score = data['count'] / len(targets)
                reasoning = [
                    f"Observation: Targets {targets} should be connected",
                    f"Hypothesis: '{word}' is a good clue",
                    f"Evidence: '{word}' connects to {data['targets']}",
                    f"Confidence: {score:.2f} ({data['count']}/{len(targets)} targets)"
                ]
                reasoning_results.append((word, score, reasoning))

        # Sort by score
        reasoning_results.sort(key=lambda x: -x[1])
        return reasoning_results


# =============================================================================
# INTEGRATED REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Integrated Reasoning Engine combining all reasoning types.

    Provides a unified interface for:
    - Deductive reasoning
    - Rule-based reasoning
    - FOPL reasoning
    - Abductive reasoning
    """

    def __init__(self):
        print("=" * 60)
        print("REASONING ENGINE")
        print("=" * 60)

        self.fopl = FOPLReasoner()
        self.rules = RuleBasedReasoner()
        self.deductive = DeductiveReasoner()
        self.abductive = AbductiveReasoner()

        print("  Loaded: FOPL Reasoner")
        print("  Loaded: Rule-Based Reasoner")
        print("  Loaded: Deductive Reasoner")
        print("  Loaded: Abductive Reasoner")

    def load_from_conceptnet(self, conceptnet_edges: Dict[str, List[Tuple[str, str, float]]]):
        """Load facts from ConceptNet into the reasoning engines."""
        print("\nLoading ConceptNet facts into reasoning engine...")

        fact_count = 0
        for word, edges in conceptnet_edges.items():
            for relation, other_word, weight in edges:
                # Map ConceptNet relations to FOPL predicates
                if relation == 'RelatedTo':
                    self.fopl.add_fact('RelatedTo', word, other_word)
                elif relation == 'Synonym':
                    self.fopl.add_fact('Synonym', word, other_word)
                elif relation == 'Antonym':
                    self.fopl.add_fact('Antonym', word, other_word)
                elif relation == 'IsA':
                    self.fopl.add_fact('IsA', word, other_word)
                elif relation == 'PartOf':
                    self.fopl.add_fact('PartOf', word, other_word)
                elif relation == 'HasProperty':
                    self.fopl.add_fact('HasProperty', word, other_word)

                fact_count += 1

        print(f"  Loaded {fact_count:,} facts")

        # Apply transitivity to infer new relationships
        new_facts = self.fopl.apply_transitivity('RelatedTo')
        print(f"  Inferred {new_facts:,} new relationships via transitivity")

    def setup_game(self, team_words: List[str], opponent_words: List[str],
                   assassin: str, neutral_words: List[str] = None):
        """Setup the game state in the reasoning engines."""
        print("\nSetting up game state...")

        # Add facts to FOPL
        for word in team_words:
            self.fopl.add_fact('TeamWord', word)
        for word in opponent_words:
            self.fopl.add_fact('OpponentWord', word)
        self.fopl.add_fact('AssassinWord', assassin)

        print(f"  Team words: {len(team_words)}")
        print(f"  Opponent words: {len(opponent_words)}")
        print(f"  Assassin: {assassin}")

    def evaluate_clue(self, clue: str, board_words: List[str],
                      assassin: str) -> Dict:
        """
        Comprehensive clue evaluation using all reasoning engines.

        Returns a dictionary with evaluation results and reasoning chains.
        """
        result = {
            'clue': clue,
            'valid': True,
            'safe': True,
            'reasoning': [],
            'score': 0.0
        }

        # 1. Rule-based evaluation
        rules_pass, rules_reasoning = self.rules.evaluate_clue(clue, board_words)
        result['reasoning'].append({
            'type': 'Rule-Based',
            'passed': rules_pass,
            'details': rules_reasoning
        })

        if not rules_pass:
            result['valid'] = False
            return result

        # 2. FOPL safety inference
        fopl_safe, fopl_reasoning = self.fopl.infer_safety(clue, assassin)
        result['reasoning'].append({
            'type': 'FOPL',
            'passed': fopl_safe,
            'details': fopl_reasoning
        })

        if not fopl_safe:
            result['safe'] = False

        # 3. Deductive safety reasoning
        relationships = [(a, b) for a, b in self.fopl.get_all('RelatedTo')]
        deductive_safe, deductive_reasoning = self.deductive.reason_about_safety(
            clue, assassin, relationships
        )
        result['reasoning'].append({
            'type': 'Deductive',
            'passed': deductive_safe,
            'details': deductive_reasoning
        })

        # Calculate overall score
        if result['valid'] and result['safe']:
            result['score'] = 1.0
        elif result['valid']:
            result['score'] = 0.5  # Valid but not safe
        else:
            result['score'] = 0.0

        return result

    def find_best_clues(self, targets: List[str], avoid_words: List[str],
                        word_relationships: Dict[str, List[str]]) -> List[Dict]:
        """
        Find the best clues using combined reasoning.

        Uses:
        - Abductive reasoning to find candidate clues
        - FOPL to verify safety
        - Deductive reasoning to validate connections
        """
        results = []

        # Step 1: Abductive reasoning to find candidates
        candidates = self.abductive.find_best_clue(targets, word_relationships)

        # Step 2: Evaluate each candidate
        for clue, confidence, abd_reasoning in candidates:
            # Skip avoid words
            if clue in [w.upper() for w in avoid_words]:
                continue

            # Check safety against each avoid word
            all_safe = True
            safety_reasoning = []

            for avoid in avoid_words:
                safe, reasoning = self.fopl.infer_safety(clue, avoid)
                safety_reasoning.extend(reasoning)
                if not safe:
                    all_safe = False
                    break

            if all_safe:
                results.append({
                    'clue': clue,
                    'confidence': confidence,
                    'targets': [t for t in targets if t in str(abd_reasoning)],
                    'abductive_reasoning': abd_reasoning,
                    'safety_reasoning': safety_reasoning
                })

        return results[:10]  # Top 10

    def explain_decision(self, clue: str, targets: List[str],
                         assassin: str) -> str:
        """
        Generate a human-readable explanation for a clue decision.
        """
        explanation = []
        explanation.append(f"\n{'='*50}")
        explanation.append(f"REASONING EXPLANATION FOR CLUE: '{clue}'")
        explanation.append(f"{'='*50}")

        # Check safety
        safe, reasoning = self.fopl.infer_safety(clue, assassin)

        explanation.append(f"\n[SAFETY ANALYSIS vs '{assassin}']")
        for step in reasoning:
            explanation.append(f"  {step}")

        explanation.append(f"\n[CONCLUSION]")
        if safe:
            explanation.append(f"  ✓ '{clue}' is SAFE to use")
        else:
            explanation.append(f"  ✗ '{clue}' is DANGEROUS - do not use")

        return "\n".join(explanation)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           REASONING ENGINE FOR CODENAMES                 ║
    ║                                                          ║
    ║  Implements:                                             ║
    ║  - First Order Predicate Logic (FOPL)                    ║
    ║  - Rule-Based Reasoning                                  ║
    ║  - Deductive Reasoning                                   ║
    ║  - Abductive Reasoning                                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Initialize reasoning engine
    engine = ReasoningEngine()

    # Add some sample facts
    print("\n" + "=" * 60)
    print("ADDING SAMPLE FACTS")
    print("=" * 60)

    # RelatedTo facts
    sample_relations = [
        ('apple', 'fruit'),
        ('orange', 'fruit'),
        ('banana', 'fruit'),
        ('fruit', 'food'),
        ('food', 'eating'),
        ('war', 'battle'),
        ('battle', 'fight'),
        ('peace', 'calm'),
        ('king', 'queen'),
        ('king', 'royal'),
        ('queen', 'royal'),
        ('castle', 'medieval'),
        ('knight', 'medieval'),
    ]

    for w1, w2 in sample_relations:
        engine.fopl.add_fact('RelatedTo', w1, w2)
        print(f"  Added: RelatedTo({w1}, {w2})")

    # Antonym facts
    engine.fopl.add_fact('Antonym', 'war', 'peace')
    engine.fopl.add_fact('Antonym', 'hot', 'cold')
    print(f"  Added: Antonym(war, peace)")
    print(f"  Added: Antonym(hot, cold)")

    # Apply transitivity
    print("\n" + "=" * 60)
    print("APPLYING TRANSITIVITY (Deductive Reasoning)")
    print("=" * 60)

    new_facts = engine.fopl.apply_transitivity('RelatedTo')
    print(f"  Inferred {new_facts} new RelatedTo facts")

    # Test safety inference
    print("\n" + "=" * 60)
    print("SAFETY INFERENCE (FOPL)")
    print("=" * 60)

    test_cases = [
        ('fruit', 'war'),
        ('battle', 'war'),
        ('peace', 'war'),
        ('royal', 'war'),
    ]

    for clue, assassin in test_cases:
        safe, reasoning = engine.fopl.infer_safety(clue, assassin)
        status = "SAFE" if safe else "DANGEROUS"
        print(f"\n  Clue: '{clue}' vs Assassin: '{assassin}' → {status}")
        for step in reasoning:
            print(f"    {step}")

    # Test rule-based evaluation
    print("\n" + "=" * 60)
    print("RULE-BASED EVALUATION")
    print("=" * 60)

    board_words = ['APPLE', 'ORANGE', 'KING', 'QUEEN', 'WAR']

    test_clues = ['fruit', 'APPLE', 'xyz123', 'a', 'royal']

    for clue in test_clues:
        valid, reasoning = engine.rules.evaluate_clue(clue, board_words)
        status = "VALID" if valid else "INVALID"
        print(f"\n  Clue: '{clue}' → {status}")
        for step in reasoning:
            print(f"    {step}")

    # Test abductive reasoning
    print("\n" + "=" * 60)
    print("ABDUCTIVE REASONING (Finding Best Clue)")
    print("=" * 60)

    word_relationships = {
        'apple': ['fruit', 'red', 'tree', 'pie'],
        'orange': ['fruit', 'citrus', 'color'],
        'banana': ['fruit', 'yellow', 'tropical'],
        'king': ['royal', 'chess', 'crown', 'queen'],
        'queen': ['royal', 'chess', 'crown', 'king'],
        'castle': ['medieval', 'chess', 'fortress'],
    }

    targets = ['APPLE', 'ORANGE', 'BANANA']
    print(f"\n  Finding clue for targets: {targets}")

    clues = engine.abductive.find_best_clue(targets, word_relationships)

    for clue, score, reasoning in clues[:3]:
        print(f"\n  Clue: '{clue}' (confidence: {score:.2f})")
        for step in reasoning:
            print(f"    {step}")

    # Generate explanation
    print("\n" + "=" * 60)
    print("DECISION EXPLANATION")
    print("=" * 60)

    explanation = engine.explain_decision('FRUIT', ['APPLE', 'ORANGE'], 'WAR')
    print(explanation)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
