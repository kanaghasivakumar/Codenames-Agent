import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import itertools
import os

COMMON_WORDS_PATH = "data/common_words.txt"

class VectorEngine:
    def __init__(self, model_name="glove-wiki-gigaword-300"):
        # 1. Load Vectors (Fast or Slow)
        binary_path = os.path.join("data", "glove-300d-binary.kv")
        if os.path.exists(binary_path):
            print(f"Loading optimized binary vectors from {binary_path}...")
            self.model = KeyedVectors.load(binary_path, mmap='r')
        else:
            print(f"Binary cache not found. Loading slow version: {model_name}...")
            self.model = api.load(model_name)
        
        # 2. Load Common Words Filter
        print("Loading common word filter...")
        self.common_words = set()
        if os.path.exists(COMMON_WORDS_PATH):
            with open(COMMON_WORDS_PATH, 'r') as f:
                for line in f:
                    word = line.strip().upper()
                    if len(word) > 2: # Ignore tiny words
                        self.common_words.add(word)
            print(f"Filter loaded: {len(self.common_words)} common words.")
        else:
            print("WARNING: Common word list not found! Agent might use obscure words.")

    def get_clue(self, team_words, opponent_words, assassin_word, n_clues=5):
        valid_team = [w.lower() for w in team_words if w.lower() in self.model]
        
        # STRATEGY 1: If only 1 word left, just solve it.
        if len(valid_team) == 1:
            print("  > Only 1 word left. Switching to Single-Word Mode.")
            return self._get_single_word_clue(valid_team, opponent_words, assassin_word)

        # STRATEGY 2: Try to find combinations (Pairs/Triples)
        candidates = []
        all_board_words = set(team_words + opponent_words + [assassin_word])
        
        combinations = list(itertools.combinations(valid_team, 2))
        # If we have many words, try triples too (optional, can be slow)
        if len(valid_team) >= 3:
             combinations += list(itertools.combinations(valid_team, 3))
             
        print(f"  > Analyzing {len(combinations)} subsets of words...")

        for subset in combinations:
            # Create mathematical center
            subset_vecs = [self.model[w] for w in subset]
            center_vec = np.mean(subset_vecs, axis=0)
            
            # Get candidates near center
            potential_clues = self.model.similar_by_vector(center_vec, topn=30)
            
            for clue, raw_score in potential_clues:
                clue_upper = clue.upper()
                
                # --- FILTERING ---
                if not self._is_valid_clue(clue_upper, all_board_words):
                    continue

                # --- SCORING ---
                final_score = self._calculate_score(clue, raw_score, assassin_word)
                if final_score > 0:
                    candidates.append((clue_upper, list(subset), final_score))

        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)
        unique_candidates = self._deduplicate(candidates)

        # STRATEGY 3: The Fallback
        # If the best clue is weak (< 0.45), giving a bad multi-clue is worse 
        # than giving a good single clue. Fallback to single mode.
        if not unique_candidates or unique_candidates[0][2] < 0.45:
            print("  > Best combo clue is weak. Falling back to Single-Word Mode.")
            single_candidates = self._get_single_word_clue(valid_team, opponent_words, assassin_word)
            # Combine and resort (Single clues often have higher scores, so they will naturally bubble up)
            unique_candidates.extend(single_candidates)
            unique_candidates.sort(key=lambda x: x[2], reverse=True)

        return unique_candidates[:n_clues]

    def _get_single_word_clue(self, team_words, opponent_words, assassin_word):
        """Finds the best clue for each individual word on the team."""
        candidates = []
        all_board_words = set(team_words + opponent_words + [assassin_word])

        for word in team_words:
            if word not in self.model: continue
            
            # Find closest words to just this one target
            potential = self.model.most_similar(word, topn=20)
            
            for clue, raw_score in potential:
                clue_upper = clue.upper()
                
                if not self._is_valid_clue(clue_upper, all_board_words):
                    continue

                final_score = self._calculate_score(clue, raw_score, assassin_word)
                
                # Single word clues are "safer" but less valuable, so we slightly
                # penalize them to favor multi-word clues if they exist.
                # But here we want them to beat "bad" multi-clues.
                candidates.append((clue_upper, [word.upper()], final_score))
        
        return self._deduplicate(candidates)

    def _is_valid_clue(self, clue, board_words):
        """Central validation logic"""
        # 1. Must be in our Common Words list (if list exists)
        if self.common_words and clue not in self.common_words:
            return False
            
        # 2. Basic rules
        if len(clue) < 3 or not clue.isalpha():
            return False
            
        # 3. Illegal Substring Check
        for bw in board_words:
            bw_upper = bw.upper()
            if bw_upper in clue or clue in bw_upper:
                return False
            # Check singular/plural stemming (simple heuristic)
            if bw_upper.endswith("S") and bw_upper[:-1] in clue: return False
            if clue.endswith("S") and clue[:-1] in bw_upper: return False
            
        return True

    def _calculate_score(self, clue, raw_score, assassin_word):
        """Applies risk penalty"""
        risk = 0.0
        if assassin_word.lower() in self.model:
            risk = self.model.similarity(clue.lower(), assassin_word.lower())
        
        if risk > 0.3: return -1.0 # Too risky
        
        return raw_score - (risk * 0.5)

    def _deduplicate(self, candidates):
        seen = set()
        unique = []
        for c, t, s in candidates:
            if c not in seen:
                unique.append((c, t, s))
                seen.add(c)
        return unique