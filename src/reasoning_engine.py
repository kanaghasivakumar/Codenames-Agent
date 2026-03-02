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
        if relation_weights:
            self.relation_weights = relation_weights
        else:  
            self.relation_weights = DEFAULT_RELATION_WEIGHTS

    def _load_graph(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Graph not found. Run build_knowledge_graph.py first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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
    
    def update_relation_weights(self, new_weights):
        self.relation_weights = new_weights

    def get_neighbors(self, word):
        return self.graph.get(word.lower(), [])

    def find_clues(self, targets, bad_words, used_clues=None, top_n=5, debug=False):
        if used_clues is None: used_clues = []
        candidates = defaultdict(float)
        concept_coverage = defaultdict(set)
        logic_chains = defaultdict(list)

        for word in targets:
            edges = self.get_neighbors(word)                    # edges for a single target word
            for edge in edges:                                  # iterate through each edge
                concept = edge['end']                           # word that is somehow related to target
                rel = edge['relation']                          # relation between concept and target
                weight = edge['normalized_weight']                         # weight of relation
                
                if concept.lower() == word.lower(): continue    # skip if concept is same as target
                
                if word in concept_coverage[concept]:
                    concept_coverage[concept].add(word)             # adds target to concept coverage if concept already related to target
                    logic_chains[concept].append(f"{word} ({rel})") # adds relation to logic chains if concept already related to target

                else:
                    score = weight * self.relation_weights.get(rel) # score for concept -> target
                    candidates[concept] += score                    # aggregates score for concept for all related targets
                    concept_coverage[concept].add(word)             # keeps track of all targets related to concept
                    logic_chains[concept].append(f"{word} ({rel})") # keeps track of relations related to concept

        safe_candidates = {}
        for concept, score in candidates.items():
            if len(concept_coverage[concept]) < len(targets): continue  # skips if concept doesn't cover all targets

            valid, _ = self.is_valid_clue(concept, targets, used_clues)
            if not valid: continue

            # Safety check
            is_unsafe = False
            # TODO: investigate ways to be more nuanced with bad words
            for bad in bad_words:
                if any(e['end'].lower() == concept.lower() for e in self.get_neighbors(bad)):   # marks unsafe if related to any bad words
                    is_unsafe = True; break
            
            if not is_unsafe:
                safe_candidates[concept] = {"score": score, "logic": list(set(logic_chains[concept]))}

        ranked = sorted(safe_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
        return [{"clue": c, "score": d['score'], "targets": list(concept_coverage[c]), "logic": d['logic']} for c, d in ranked[:top_n]]