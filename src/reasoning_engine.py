import json
from pathlib import Path
from collections import defaultdict
import utils.constants as constants

GRAPH_PATH = Path("data/conceptnet_graph.json")
DEFAULT_RELATION_WEIGHTS = constants.DEFAULT_RELATION_WEIGHTS


class ReasoningEngine:
    def __init__(self, graph_path=GRAPH_PATH, relation_weights=None):
        self.graph = self._load_graph(path=graph_path)
        self.stop_words = {"a", "an", "the", "and", "or", "but", "if", "then", "of", "at", "by", "for"}
        self.relation_weights = relation_weights if relation_weights else DEFAULT_RELATION_WEIGHTS

    def _load_graph(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Graph not found. Run build_knowledge_graph.py first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_relation_weights(self, new_weights):
        self.relation_weights = new_weights

    def is_valid_clue(self, clue, targets, used_clues):
        clue = clue.lower().strip()
        if len(clue) <= 2 or clue in self.stop_words:
            return False, "Too generic"
        for target in targets:
            t = target.lower().strip()
            if clue in t or t in clue:
                return False, f"Identity conflict with {target}"
        for used in used_clues:
            if clue in used.lower() or used.lower() in clue:
                return False, "Too similar to previous clue"
        return True, ""

    def get_neighbors(self, word):
        return self.graph.get(word.lower(), [])

    def _neighbor_set(self, word):
        """Return {concept_lower: best_normalized_weight} for fast penalty lookups."""
        result = {}
        for e in self.get_neighbors(word):
            c = e['end'].lower()
            w = float(e.get('normalized_weight', e.get('weight', 1.0)))
            if c not in result or w > result[c]:
                result[c] = w
        return result

    def find_clues(self, targets, opponent_words, assassin_word, neutral_words,
                   used_clues=None, top_n=5, strict_safety=True):
        """
        Find best clue words covering all targets.

        Safety model — soft penalties instead of hard rejection:
          Assassin  : heavy penalty (near-disqualifying on strong connections)
          Opponent  : moderate penalty
          Neutral   : light penalty, only applied when connection is strong

        Coverage bonus: clues covering N targets are multiplied by
        COVERAGE_BONUS_PER_EXTRA_TARGET^(N-1), strongly favouring multi-word clues
        over single-word clues and preventing games from devolving into 1-word turns.

        Fallback: if strict_safety=True produces nothing, retries with neutral
        penalties relaxed so a clue is always returned when graph coverage exists.
        """
        if used_clues is None:
            used_clues = []

        # Pre-build neighbour lookup dicts for all bad words
        assassin_nb  = self._neighbor_set(assassin_word)
        opponent_nbs = [self._neighbor_set(w) for w in opponent_words]
        neutral_nbs  = [self._neighbor_set(w) for w in neutral_words]

        # ── Step 1: Accumulate positive scores across all targets ─────────────
        candidates       = defaultdict(float)
        concept_coverage = defaultdict(set)
        logic_chains     = defaultdict(list)

        for word in targets:
            for edge in self.get_neighbors(word):
                concept = edge['end'].lower()
                rel     = edge['relation']
                weight  = float(edge.get('normalized_weight', edge.get('weight', 1.0)))

                if concept == word.lower():
                    continue

                # Score only the first occurrence per (concept, target) pair
                # to prevent multi-edge spam inflating a single target's contribution
                if word not in concept_coverage[concept]:
                    candidates[concept] += weight * self.relation_weights.get(rel, 0.0)

                concept_coverage[concept].add(word)
                logic_chains[concept].append(f"{word} ({rel})")

        # ── Step 2: Apply penalties and coverage bonus ────────────────────────
        scored = {}

        for concept, pos_score in candidates.items():
            # Must cover ALL requested targets
            if len(concept_coverage[concept]) < len(targets):
                continue

            valid, _ = self.is_valid_clue(concept, targets, used_clues)
            if not valid:
                continue

            penalty = 0.0

            # Assassin — hard disqualify if connection is very strong
            if concept in assassin_nb:
                if assassin_nb[concept] > 0.7:
                    continue   # outright ban
                penalty += assassin_nb[concept] * constants.ASSASSIN_PENALTY_WEIGHT

            # Opponent — penalise proportional to connection strength
            for opp_nb in opponent_nbs:
                if concept in opp_nb:
                    penalty += opp_nb[concept] * constants.OPPONENT_PENALTY_WEIGHT

            # Neutral — only penalise strong connections (weak ones are fine)
            if strict_safety:
                for neut_nb in neutral_nbs:
                    if concept in neut_nb:
                        strength = neut_nb[concept]
                        if strength > constants.NEUTRAL_PENALTY_THRESHOLD:
                            penalty += strength * constants.NEUTRAL_PENALTY_WEIGHT

            net_score = pos_score - penalty

            # Coverage bonus: 2 targets → 2x, 3 targets → 4x, etc.
            n_covered = len(concept_coverage[concept])
            if n_covered > 1:
                net_score *= (constants.COVERAGE_BONUS_PER_EXTRA_TARGET ** (n_covered - 1))

            scored[concept] = {
                'net_score': net_score,
                'n_covered': n_covered,
                'logic':     list(set(logic_chains[concept]))
            }

        # ── Step 3: Fallback — relax neutral penalties if nothing found ───────
        if not scored and strict_safety:
            return self.find_clues(
                targets, opponent_words, assassin_word, neutral_words,
                used_clues=used_clues, top_n=top_n, strict_safety=False
            )

        # ── Step 4: Rank — coverage first, net_score second ──────────────────
        ranked = sorted(
            scored.items(),
            key=lambda x: (-x[1]['n_covered'], -x[1]['net_score'])
        )

        return [
            {
                'clue':    c,
                'score':   round(d['net_score'], 4),
                'count':   d['n_covered'],
                'targets': list(concept_coverage[c]),
                'logic':   d['logic'],
            }
            for c, d in ranked[:top_n]
        ]
    
    def get_relation_weight(self, rel):
        """Get weight for a relation, handling compound 2-hop relations like 'IsA->AtLocation'."""
        if rel in self.relation_weights:
            return self.relation_weights[rel]
        if '->' in rel:
            parts = rel.split('->')
            weights = [self.relation_weights.get(p, 0.0) for p in parts]
            return sum(weights) / len(weights)
        return 0.0