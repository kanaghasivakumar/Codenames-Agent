import networkx as nx
import requests
import time
import json
import os
from tqdm import tqdm

class KnowledgeGraphBuilder:
    def __init__(self, word_list_path, output_path):
        self.word_list = self._load_words(word_list_path)
        self.output_path = output_path
        self.graph = nx.Graph()

    def _load_words(self, path):
        with open(path, 'r') as f:
            return [w.strip().upper() for w in f.readlines()]

    def fetch_edges(self, word, limit=5):
        # ConceptNet API
        url = f"http://api.conceptnet.io/c/en/{word.lower()}?limit={limit}"
        try:
            # Increased timeout to 10s to prevent empty returns on slow connections
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"  [!] API Error {response.status_code} for {word}")
                return []
                
            data = response.json()
            edges = []
            
            for edge in data.get('edges', []):
                start = edge['start']['label'].upper()
                end = edge['end']['label'].upper()
                weight = edge['weight']
                
                neighbor = end if start == word else start
                
                # Filter out phrases (keep single words)
                if ' ' in neighbor: continue 
                
                edges.append((neighbor, weight))
            
            return edges
        except Exception as e:
            print(f"  [!] Exception for {word}: {e}")
            return []

    def build_graph(self):
        print(f"Re-building graph for {len(self.word_list)} words...")
        
        # Reset graph to ensure we aren't saving an empty state
        self.graph = nx.Graph()
        
        for i, word in enumerate(tqdm(self.word_list)):
            self.graph.add_node(word, type='game_word')
            
            neighbors = self.fetch_edges(word)
            
            # DEBUG PRINT: Show us it's working!
            if i % 10 == 0: 
                tqdm.write(f"  > {word}: Found {len(neighbors)} edges")

            for neighbor, weight in neighbors:
                self.graph.add_node(neighbor, type='concept')
                self.graph.add_edge(word, neighbor, weight=weight)
            
            # Sleep 0.5s to be safe
            time.sleep(0.5) 

        self.save_graph()

    def save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(self.output_path, 'w') as f:
            json.dump(data, f)
        print(f"\nSUCCESS: Graph saved to {self.output_path}")
        print(f"Total Nodes: {len(self.graph.nodes)} (Should be > 1000)")
        print(f"Total Edges: {len(self.graph.edges)}")

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder("data/codenames_words.txt", "data/safety_graph.json")
    builder.build_graph()