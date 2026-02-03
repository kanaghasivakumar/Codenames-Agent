# Neuro-Symbolic Codenames Agent

An AI agent that plays the role of **Spymaster** in the game of *Codenames*.

This project demonstrates a **Neuro-Symbolic architecture**: it combines the creative, associative power of Neural Networks (Word Vectors) with the strict, logical constraints of Symbolic AI (Knowledge Graphs) to generate clues that are both clever and safe.

## How It Works

The agent operates in a two-stage "System 1 / System 2" thinking process:

### 1. The "Neuro" Layer (Creativity)

* **Technology:** `Gensim` + Pre-trained GloVe Vectors (300 dimensions).
* **Function:** The agent analyzes the board and identifies "sub-goals" (clusters of 2-3 words). It calculates the mathematical centroid of these clusters to brainstorm potential clues that link multiple team words together.
* **Why:** Vectors allow for "fuzzy" semantic matching (e.g., knowing that *Apple* and *Banana* are related even if they don't share letters).

### 2. The "Symbolic" Layer (Safety & Logic)

* **Technology:** `NetworkX` Knowledge Graph.
* **Function:** Before speaking, the agent verifies its idea against a strict **Safety Graph**. It calculates the shortest path between the potential Clue and the **Assassin** word.
* **Logic:**
* If `Distance(Clue, Assassin) <= 1`, the clue is **REJECTED** immediately.
* If the clue is not found in the graph (unknown entity), it receives a **Risk Penalty**.


* **Why:** Vectors often hallucinate dangerous connections (e.g., *King* is close to *Queen*, but also to *Prince*). Symbolic logic ensures the agent never accidentally triggers the Assassin.

## Installation

**Prerequisites:** Python 3.10 or 3.11 is recommended.

1. **Clone the repository:**
```bash
git clone https://github.com/kanaghasivakumar/Codenames-Agent.git
cd Codenames-Agent

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## How to Run

Follow these steps in order to set up the data and run the agent.

### Step 1: Download & Optimize Vectors (Run Once)

This script downloads the massive 300-dimensional GloVe model and converts it into a binary format for instant loading times.

```bash
python src/optimize_vectors.py

```

*(Note: The first run may take a few minutes to download the data.)*

### Step 2: Build the Knowledge Graph (Run Once)

This constructs the "Safety Graph" by mapping semantic relationships between all game words using the vector engine.

```bash
python src/graph_builder.py

```

### Step 3: Play the Game

Run the main driver to see the agent generate clues for a random board setup.

```bash
python main.py

```

## Project Structure

* `main.py`: The central agent logic. It manages the game loop, coordinates the vector engine and graph, and makes final decisions.
* `src/vector_engine.py`: Handles the Neural layer. Loads the binary model and performs vector math (centroids, cosine similarity).
* `src/graph_builder.py`: Handles the Symbolic layer. construct a logic graph to valid safety.
* `src/optimize_vectors.py`: Utility to convert text-based GloVe models to memory-mapped binary files.
* `data/`: Stores the word list and generated graph/vector files.

## Credits & Sources

This project stands on the shoulders of open-source giants.

* **Vocabulary:** The standard Codenames word list was sourced from [Gullesnuffs/Codenames](https://github.com/Gullesnuffs/Codenames).
* **Vector Models:** Powered by the [Gensim](https://radimrehurek.com/gensim/) library and the **GloVe-6B** dataset from Stanford NLP.
* **Graph Logic:** Built using [NetworkX](https://networkx.org/).
* **Inspiration:** The agentic design was inspired by neuro-symbolic research and adapted from logic found in [thomasahle/codenames](https://github.com/thomasahle/codenames).