import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import itertools
import os

class VectorEngine:
    def __init__(self, model_name="glove-wiki-gigaword-300"):
        binary_path = os.path.join("data", "glove-300d-binary.kv")
        
        if os.path.exists(binary_path):
            print(f"Loading optimized binary vectors from {binary_path}.")
            self.model = KeyedVectors.load(binary_path, mmap='r')
        else:
            print(f"Binary cache not found. Loading model: {model_name}.")
            self.model = api.load(model_name)
            
        print("Model loaded.")

    def get_clue(self, team_words, opponent_words, assassin_word, n_clues=5):
        """
        Finds clues by looking for SUBSETS of the team words (pairs or triples).
        Returns: list of (clue, [target_words], score)
        """
        valid_team = [w.lower() for w in team_words if w.lower() in self.model]
        bad_words = [w.lower() for w in opponent_words + [assassin_word] if w.lower() in self.model]
        
        candidates = []
        
        # 1. SEARCH STRATEGY: Look for combinations of 2 or 3 words
        combinations = list(itertools.combinations(valid_team, 2))
        if len(valid_team) >= 3:
            combinations += list(itertools.combinations(valid_team, 3))
            
        print(f"  > Analyzing {len(combinations)} subsets of words...")

        for subset in combinations:
            # Create a "Cluster Center" for this subset
            subset_vecs = [self.model[w] for w in subset]
            center_vec = np.mean(subset_vecs, axis=0)
            
            # Find words close to this center
            potential_clues = self.model.similar_by_vector(center_vec, topn=20)
            
            for clue, raw_score in potential_clues:
                clue = clue.upper()
                
                # 1. Length check (too long = usually obscure)
                if len(clue) > 10 or len(clue) < 3: continue
                # 2. Must be letters only
                if not clue.isalpha(): continue
                # 3. No board words
                all_board = set(team_words + opponent_words + [assassin_word])
                if any(w in clue or clue in w for w in all_board): continue
                
                # --- SCORING ---
                # Penalize if it's close to the Assassin or Opponents
                risk = 0.0
                if assassin_word.lower() in self.model:
                    risk = self.model.similarity(clue.lower(), assassin_word.lower())
                
                # If risk is too high, skip immediately
                if risk > 0.3: continue
                
                # Final Score = Similarity to Targets - Risk
                final_score = raw_score - (risk * 0.5)
                
                candidates.append((clue, list(subset), final_score))

        # Sort by score and return top N
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate (keep only best version of each clue)
        seen_clues = set()
        unique_candidates = []
        for c, t, s in candidates:
            if c not in seen_clues:
                unique_candidates.append((c, t, s))
                seen_clues.add(c)
                
        return unique_candidates[:n_clues]