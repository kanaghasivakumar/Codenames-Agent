import json
import re
from pathlib import Path

import utils.constants as constants


class UserProfile:
    def __init__(self, name):
        self.name = name
        # determine profiles directory relative to project root (parent of src)
        project_root = Path(__file__).resolve().parents[1]
        profiles_dir = project_root / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        # sanitize filename for lookup (matches save_profile_to_json)
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', self.name.strip()).lower()
        profile_file = profiles_dir / f"{safe}.json"
        if profile_file.is_file():
            with profile_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                self.relation_weights = data.get("relation_weights", constants.DEFAULT_RELATION_WEIGHTS)
                self.games_played = data.get("games_played", 0)
        else:
            self.relation_weights = constants.DEFAULT_RELATION_WEIGHTS
            self.games_played = 0
    
    def get_target_relations(self, logic_chains):
        self.logic_chains = logic_chains

    def get_guessed_words(self, guessed_words):
        self.guessed_words = guessed_words

    def give_weights(self):
        return self.relation_weights
    
    def update_weights(self):
        """
        Update `self.relation_weights` based on the last `logic_chains` and
        `guessed_words` recorded on this profile.

        For any entry in `self.logic_chains` of the form "word (Relation)", if
        `word` was NOT guessed (case-insensitive) in `self.guessed_words`, then
        slightly decay the weight for `Relation`. If a word appears with
        multiple relations, decay each associated relation.
        """
        logic = getattr(self, 'logic_chains', None)
        guessed = getattr(self, 'guessed_words', None)
        if not logic or not isinstance(logic, (list, tuple)):
            return self.relation_weights

        guessed_set = set()
        if guessed and isinstance(guessed, (list, tuple)):
            guessed_set = {g.strip().lower() for g in guessed if isinstance(g, str)}

        # update parameters
        DECAY_FACTOR = 0.95
        GROWTH_FACTOR = 1.05
        MIN_WEIGHT = 0.01
        MAX_WEIGHT = 16.0

        # aggregate relations -> set(words)
        from collections import defaultdict
        rel_to_words = defaultdict(set)
        for entry in logic:
            if not isinstance(entry, str):
                continue
            rel = None
            word = None
            try:
                if '(' in entry and entry.strip().endswith(')'):
                    i = entry.rfind('(')
                    word = entry[:i].strip()
                    rel = entry[i+1:-1].strip()
                else:
                    parts = entry.rsplit(None, 1)
                    if len(parts) == 2:
                        word, rel = parts[0].strip(), parts[1].strip()
            except Exception:
                continue

            if not rel or not word:
                continue
            rel_to_words[rel].add(word.lower())

        # apply a single update per relation: reward if any word was guessed, otherwise decay
        for rel, words in rel_to_words.items():
            was_guessed = bool(words & guessed_set)
            if was_guessed:
                # increase weight
                cur = float(self.relation_weights.get(rel, 1.0))
                new_w = min(MAX_WEIGHT, cur * GROWTH_FACTOR)
                self.relation_weights[rel] = new_w
            else:
                # decrease weight
                cur = float(self.relation_weights.get(rel, 1.0))
                new_w = max(MIN_WEIGHT, cur * DECAY_FACTOR)
                self.relation_weights[rel] = new_w

        return self.relation_weights
    
    def save_profile_to_json(self):
        # Ensure profiles directory at project root (parent of src)
        project_root = Path(__file__).resolve().parents[1]
        profiles_dir = project_root / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        # sanitize name for filename
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', self.name.strip()).lower()
        profile_path = profiles_dir / f"{safe}.json"

        payload = {
            "name": self.name,
            "games_played": self.games_played,
            "relation_weights": self.relation_weights,
        }

        with profile_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        return profile_path

    def increment_games_played(self):
        self.games_played += 1