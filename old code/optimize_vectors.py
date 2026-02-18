import gensim.downloader as api
from gensim.models import KeyedVectors
import os

def optimize_model():
    print("1. Downloading text model...")
    model = api.load("glove-wiki-gigaword-300")
    
    # Define path for the fast binary version
    save_path = os.path.join("data", "glove-300d-binary.kv")
    
    print(f"2. Saving binary version to {save_path}...")
    os.makedirs("data", exist_ok=True)
    model.save(save_path)
    print("Loadable model saved.")

if __name__ == "__main__":
    optimize_model()