import json
from pathlib import Path
from collections import defaultdict

GRAPH_PATH = Path("data/conceptnet_graph.json")

# SENIOR DEV FIX: Strict weights to force "Definitional" logic
RELATION_WEIGHTS = {
    "IsA": 5.0,        
    "Category": 5.0,
    "AtLocation": 3.0, # Physical logic (Whale in Ocean)
    "UsedFor": 3.0,    # Functional logic
    "PartOf": 3.0,
    "HasProperty": 2.0,
    "CapableOf": 2.0,
    "RelatedTo": 0.3,   # SEVERE PENALTY: Avoids "Ray to King" nonsense
    "Antonym": 0.1,    # NEAR TOTAL BAN
    "DistinctFrom": 0.1
}

class ReasoningEngine:
    def __init__(self, graph_path=GRAPH_PATH):
        self.graph = self._load_graph(path=graph_path)
        self.stop_words = {"a", "an", "the", "and", "or", "but", "if", "then", "of", "at", "by", "for"}

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

    def get_neighbors(self, word):
        return self.graph.get(word.lower(), [])

    def find_clues(self, targets, bad_words, used_clues=None, top_n=5, debug=False):
        if used_clues is None: used_clues = []
        candidates = defaultdict(float)    
        concept_coverage = defaultdict(set) 
        logic_chains = defaultdict(list) 

        for word in targets:
            edges = self.get_neighbors(word)
            for edge in edges:
                concept = edge['end']
                rel = edge['relation']
                weight = edge['weight']
                
                if concept.lower() == word.lower(): continue
                
                # Apply the new strict weights
                score = weight * RELATION_WEIGHTS.get(rel, 1.0)
                candidates[concept] += score
                concept_coverage[concept].add(word)
                logic_chains[concept].append(f"{word} ({rel})")

        safe_candidates = {}
        for concept, score in candidates.items():
            if len(concept_coverage[concept]) < len(targets): continue
            
            valid, _ = self.is_valid_clue(concept, targets, used_clues)
            if not valid: continue

            # Safety check
            is_unsafe = False
            for bad in bad_words:
                if any(e['end'].lower() == concept.lower() for e in self.get_neighbors(bad)):
                    is_unsafe = True; break
            
            if not is_unsafe:
                safe_candidates[concept] = {"score": score, "logic": list(set(logic_chains[concept]))}

        ranked = sorted(safe_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
        return [{"clue": c, "score": d['score'], "targets": list(concept_coverage[c]), "logic": d['logic']} for c, d in ranked[:top_n]]