import random
import itertools
import time
from src.reasoning_engine import ReasoningEngine

WORD_LIST_PATH = "data/codenames_words.txt"

class CodenamesGame:
    def __init__(self):
        print("\nINITIALIZING V2 NEURO-SYMBOLIC AGENT...")
        self.engine = ReasoningEngine()
        self.all_words = self.load_words()
        self.used_clues = []

    def load_words(self):
        with open(WORD_LIST_PATH, 'r') as f:
            return [w.strip().upper() for w in f.readlines() if w.strip()]

    def generate_board(self):
        deck = list(self.all_words)
        random.shuffle(deck)
        board = deck[:25]
        return set(board[:9]), set(board[9:17]), board[17], set(board[18:])

    def play_game(self):
        red_left, blue_left, assassin, bystanders = self.generate_board()
        
        print("\n" + "="*60 + "\nNEW GAME STARTED\n" + "="*60)
        print(f"RED: {sorted(red_left)}\nBLUE: {sorted(blue_left)}\nASSASSIN: {assassin}\n" + "="*60)

        for turn in range(1, 15):
            print(f"\n--- ROUND {turn} ---")
            
            # 1. RED TURN (AI)
            bad_words = list(blue_left) + [assassin] + list(bystanders)
            clue = self.get_ai_move(red_left, bad_words)
            
            if clue:
                self.used_clues.append(clue['clue'])
                print(f"SPYMASTER SAYS: '{clue['clue'].upper()}' ({len(clue['targets'])})")
                print(f"LOGIC: {clue['logic']}")
                
                for target in clue['targets']:
                    t_up = target.upper()
                    print(f"    Touching '{t_up}'...", end=" ")
                    if t_up in red_left:
                        print("CORRECT!"); red_left.remove(t_up)
                    elif t_up == assassin:
                        print("ASSASSIN! BLUE WINS."); return
                    else:
                        print("MISS. Turn ends."); break
            else:
                print("AI PASSES TURN")

            if not red_left: print("\nVICTORY! RED WINS"); return

            # 2. BLUE TURN (Simulated Enemy)
            if blue_left:
                removed = random.choice(list(blue_left))
                blue_left.remove(removed)
                print(f"\nBLUE TEAM reveals '{removed}' and ends turn.")
                if not blue_left: print("\nDEFEAT. BLUE WINS"); return

    def get_ai_move(self, red_team, bad_words):
        red_list = list(red_team)
        for size in [3, 2, 1]:
            combos = list(itertools.combinations(red_list, size))
            random.shuffle(combos)
            for subset in combos[:30]:
                clues = self.engine.find_clues(subset, bad_words, used_clues=self.used_clues)
                if clues: return clues[0]
        return None

if __name__ == "__main__":
    CodenamesGame().play_game()