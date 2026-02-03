import networkx as nx
import json
import os
from tqdm import tqdm
from vector_engine import VectorEngine

# Configuration
WORD_LIST_PATH = "data/codenames_words.txt"
OUTPUT_PATH = "data/safety_graph.json"
SIMILARITY_THRESHOLD = 0.4  

class OfflineGraphBuilder:
    def __init__(self):
        print("--- STARTING OFFLINE GRAPH BUILDER ---")
        # 1. Load the words
        with open(WORD_LIST_PATH, 'r') as f:
            self.word_list = [w.strip().upper() for w in f.readlines()]
            
        # 2. Load the Vector Engine 
        self.engine = VectorEngine(model_name="glove-wiki-gigaword-300") 

    def build_graph(self):
        graph = nx.Graph()
        
        print(f"\nBuilding KNN Graph for {len(self.word_list)} nodes...")
        
        # Add all words to graph first
        for word in self.word_list:
            graph.add_node(word, type='game_word')

        # Create edges based on vector similarity
        valid_words = [w for w in self.word_list if w.lower() in self.engine.model]
        
        for i, word_a in enumerate(tqdm(valid_words)):
            # Get top 10 most similar words from the vector model
            try:
                neighbors = self.engine.model.most_similar(word_a.lower(), topn=10)
                
                for neighbor_word, score in neighbors:
                    neighbor_upper = neighbor_word.upper()
                    
                    # Only add edge if the neighbor is also in game vocabulary
                    if neighbor_upper in self.word_list:
                        graph.add_edge(word_a, neighbor_upper, weight=score)
                    
                    elif score > 0.6: 
                        graph.add_node(neighbor_upper, type='concept')
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
    builder = OfflineGraphBuilder()
    builder.build_graph()