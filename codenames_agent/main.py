import random
import json
import networkx as nx
from src.vector_engine import VectorEngine
from wikipedia_lookup import get_weekly_wikipedia_pageviews

WORD_LIST_PATH = "data/codenames_words.txt"
GRAPH_PATH = "data/safety_graph.json"
TOP_WORD_LIST_PATH = "data/top_english_words_lower_500000.txt"
WIKI_THRESHOLD = 50

class CodenamesAgent:
    def __init__(self):
        print("--- INITIATING NEURO-SYMBOLIC AGENT ---")
        self.vector_engine = VectorEngine()
        self.wiki_threshold = WIKI_THRESHOLD
        
        print("Loading Knowledge Graph...")
        try:
            with open(GRAPH_PATH, 'r') as f:
                graph_data = json.load(f)
            self.safety_graph = nx.node_link_graph(graph_data)
            print(f"Graph loaded with {len(self.safety_graph.nodes)} nodes.")

        except FileNotFoundError:
            print("ERROR: Safety Graph not found. Run the offline builder first.")
            exit()

        try:
            with open(TOP_WORD_LIST_PATH, 'r') as file:
                top_words = file.read().splitlines()
                self.top_words = [x.upper() for x in top_words]

        except FileNotFoundError:
            print('ERROR: Top English word list not found.')


    def generate_board(self):
        with open(WORD_LIST_PATH, 'r') as f:
            words = [w.strip().upper() for w in f.readlines()]
        random.shuffle(words)
        board = words[:25]
        return board[:9], board[9:17], board[17], board[18:]

    def is_safe(self, clue, assassin_word):
        """Checks Knowledge Graph for shortest path to Assassin"""

        # Check wikipedia page views
        views = get_weekly_wikipedia_pageviews(clue.capitalize())

        print(f'Candidate clue: {clue}')
        print(f'Number of wiki views: {views}')
        print(f'Clue in top words: {clue in self.top_words}')
        print(f'Clue in safety graph: {clue in self.safety_graph}')
        
        if (clue not in self.top_words) and (views < self.wiki_threshold):
            return False, "Unknown word (Risk Penalty)" # Return penalty flag
        
        elif (clue not in self.safety_graph) or (assassin_word not in self.safety_graph):
            return True, "Less known word (Risk Penalty)" # Return penalty flag
        
        try:
            dist = nx.shortest_path_length(self.safety_graph, clue, assassin_word)
            if dist <= 1: return False, f"Too close to Assassin (Dist: {dist})"
            return True, "Verified Safe"
        except nx.NetworkXNoPath:
            return True, "Verified Safe (No connection)"

    def play_turn(self, red_team, blue_team, assassin):
        print(f"\n{'='*40}")
        print(f"ASSASSIN: {assassin}")
        print(f"RED TEAM: {red_team}")
        print(f"BLUE TEAM: {blue_team}")
        print(f"{'='*40}\n")
        
        print("[1] Brainstorming Clues (Vector Engine)...")
        candidates = self.vector_engine.get_clue(red_team, blue_team, assassin)
        
        print("\n[2] Verifying Safety (Knowledge Graph)...")
        best_move = None
        
        for clue, targets, score in candidates:
            # Check Graph Safety
            is_safe, reason = self.is_safe(clue, assassin)
            
            # Formatting for the user
            target_str = ", ".join([t.upper() for t in targets])
            
            # Apply Penalty for Unknown Words
            final_score = score
            status = "✅ ACCEPTABLE"
            if "Unknown" in reason:
                final_score -= 0.1  # Heavier penalty
                status = "⚠️ UNCERTAIN"

            if "Less" in reason:
                final_score -= 0.01  # Lesser penalty
                
            print(f"  > Candidate: '{clue}' -> Targets: [{target_str}]")
            print(f"    Raw Score: {score:.2f} | Safety: {reason}")
            
            if not is_safe:
                print(f"    ❌ REJECTED: {reason}\n")
                continue
                
            print(f"    Final Score: {final_score:.2f} [{status}]\n")
            
            if best_move is None or final_score > best_move['score']:
                best_move = {'clue': clue, 'count': len(targets), 'targets': targets, 'score': final_score}

        if best_move:
            print(f">>> FINAL MOVE: Clue '{best_move['clue']}' for {best_move['count']}")
            print(f">>> INTENDED TARGETS: {best_move['targets']}")
        else:
            print(">>> PASS (No safe clues found)")

if __name__ == "__main__":
    agent = CodenamesAgent()
    # red = ['PARK', 'FAIR', 'DRAGON', 'TICK', 'CENTAUR', 'BATTERY', 'EMBASSY', 'SUPERHERO', 'WEB']
    # blue = ['CODE', 'CAPITAL', 'BRIDGE'] 
    # assassin = 'CARROT'
    # Use this instead:
    red, blue, assassin, bystanders = agent.generate_board()
    
    agent.play_turn(red, blue, assassin)