import networkx as nx
import json
import os
from tqdm import tqdm

from vector_engine import VectorEngine

WORD_LIST_PATH = "data/codenames_words.txt"
OUTPUT_PATH = "data/safety_graph.json"

class KnowledgeGraphBuilder:
    def __init__(self):
        print("--- KNOWLEDGE GRAPH BUILDER (SEMANTIC) ---")
        
        # 1. Load the vocabulary
        if not os.path.exists(WORD_LIST_PATH):
            print(f"ERROR: {WORD_LIST_PATH} not found.")
            exit()
            
        with open(WORD_LIST_PATH, 'r') as f:
            self.word_list = [w.strip().upper() for w in f.readlines()]
            
        # 2. Load the Vector Engine
        self.engine = VectorEngine()

    def build_graph(self):
        graph = nx.Graph()
        
        print(f"\nBuilding Semantic Graph for {len(self.word_list)} nodes...")
        
        # Add all words to graph first
        for word in self.word_list:
            graph.add_node(word, type='game_word')

        # Create edges based on vector similarity
        # Compare every word to every other word using KNN
        valid_words = [w for w in self.word_list if w.lower() in self.engine.model]
        
        print("Connecting nodes based on vector similarity...")
        for word_a in tqdm(valid_words):
            # Get top 10 most similar words from the vector model
            try:
                neighbors = self.engine.model.most_similar(word_a.lower(), topn=10)
                
                for neighbor_word, score in neighbors:
                    neighbor_upper = neighbor_word.upper()
                    
                    # Constraint: Only add edge if the neighbor is ALSO in our game vocabulary
                    # This creates a dense, closed graph perfect for the game
                    if neighbor_upper in self.word_list:
                        graph.add_edge(word_a, neighbor_upper, weight=score)
                        
            except KeyError:
                continue

        # Save to file
        print(f"\nGraph Stats:")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")
        
        data = nx.node_link_data(graph)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(data, f)
        print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    builder = KnowledgeGraphBuilder()
    builder.build_graph()