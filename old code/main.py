import random
import json
import networkx as nx
from src.vector_engine import VectorEngine

# Paths
WORD_LIST_PATH = "data/codenames_words.txt"
GRAPH_PATH = "data/safety_graph.json"

class CodenamesAgent:
    def __init__(self):
        print("--- INITIATING NEURO-SYMBOLIC AGENT ---")
        self.vector_engine = VectorEngine()
        
        print("Loading Knowledge Graph...")
        try:
            with open(GRAPH_PATH, 'r') as f:
                graph_data = json.load(f)
            self.safety_graph = nx.node_link_graph(graph_data)
            print(f"Graph loaded with {len(self.safety_graph.nodes)} nodes.")
        except FileNotFoundError:
            print("ERROR: Safety Graph not found. Run the graph builder first.")
            exit()

    def generate_board(self):
        """
        Standard Codenames Setup:
        - 25 Words Total on Board
        - 9 Red (Starting Team)
        - 8 Blue
        - 1 Assassin
        - 7 Innocent Bystanders
        """
        with open(WORD_LIST_PATH, 'r') as f:
            words = [w.strip().upper() for w in f.readlines()]
        random.shuffle(words)
        
        # KEY FIX: Slice to exactly 25 words first
        board = words[:25]
        
        red_team = board[:9]
        blue_team = board[9:17]
        assassin = board[17]
        bystanders = board[18:] # Indices 18 to 24 (7 words)
        
        return red_team, blue_team, assassin, bystanders

    def get_risk_level(self, clue, assassin, blue_team, bystanders):
        # 1. Check ASSASSIN (Instant Loss)
        if self.is_too_close(clue, assassin):
             return 1.0, f"Too close to Assassin ({assassin})"

        # 2. Check BLUE TEAM (High Penalty)
        for blue_word in blue_team:
            if self.is_too_close(clue, blue_word):
                return 0.9, f"Too close to Opponent ({blue_word})"

        # 3. Check BYSTANDERS (Medium Penalty)
        for neutral in bystanders:
            if self.is_too_close(clue, neutral):
                return 0.5, f"Too close to Bystander ({neutral})"
                
        # 4. Unknown Word (Small Penalty)
        if clue not in self.safety_graph:
            return 0.1, "Unknown word"

        return 0.0, "Safe"

    def is_too_close(self, clue, target):
        if clue not in self.safety_graph or target not in self.safety_graph:
            return False
        try:
            dist = nx.shortest_path_length(self.safety_graph, clue, target)
            return dist <= 1
        except nx.NetworkXNoPath:
            return False

    def play_game(self):
        red_team, blue_team, assassin, bystanders = self.generate_board()
        
        print("\n" + "="*60)
        print("🕵️‍♂️  NEW GAME STARTED")
        print(f"🔴 RED TEAM ({len(red_team)}): {red_team}")
        print(f"🔵 BLUE TEAM ({len(blue_team)}): {blue_team}")
        print(f"⚪ NEUTRAL ({len(bystanders)}): {bystanders}")
        print(f"💀 ASSASSIN: {assassin}")
        print("="*60)

        turn_count = 1
        game_over = False

        while not game_over:
            print(f"\n--- ROUND {turn_count} ---")
            print(f"Words remaining: {red_team}")
            
            # Get best clue
            clue, count, targets = self.play_turn(red_team, blue_team, assassin, bystanders)
            
            if not clue:
                print(">>> PASS (No safe clues found)")
                break
                
            print(f">>> SPYMASTER SAYS: '{clue}' ({count})")
            
            # SIMULATION LOGIC: Remove guessed words
            for target in targets:
                target_upper = target.upper()
                if target_upper in red_team:
                    print(f"    ✅ Team guessed '{target_upper}' -> CORRECT!")
                    red_team.remove(target_upper)
                else:
                    # In a real game, if they guessed a Blue/Neutral word, the turn ends.
                    # We just print the error here.
                    print(f"    ❌ Team guessed '{target_upper}' -> WRONG! (Turn Ends)")
                    break
            
            if not red_team:
                print("\n🎉 VICTORY! All Red words found.")
                game_over = True
            
            turn_count += 1
            if turn_count > 15:
                print("\n⏱️ GAME OVER (Too many turns)")
                break

    def play_turn(self, red_team, blue_team, assassin, bystanders):
        candidates = self.vector_engine.get_clue(red_team, blue_team, assassin)
        best_move = None
        
        print("  Thinking...")
        for clue, targets, raw_score in candidates:
            # Check Risk
            risk_penalty, reason = self.get_risk_level(clue, assassin, blue_team, bystanders)
            final_score = raw_score - risk_penalty
            
            if final_score > 0.45:
                status = "✅" if risk_penalty == 0 else f"⚠️ ({reason})"
                print(f"    Candidate: '{clue}' -> {targets} | Score: {final_score:.2f} {status}")

            if risk_penalty >= 0.9: continue

            if best_move is None or final_score > best_move['score']:
                best_move = {'clue': clue, 'count': len(targets), 'targets': targets, 'score': final_score}

        if best_move:
            return best_move['clue'], best_move['count'], best_move['targets']
        return None, 0, []

if __name__ == "__main__":
    agent = CodenamesAgent()
    agent.play_game()