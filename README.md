# Codenames AI Spymaster

A knowledge-graph-powered Codenames AI built for the KRR-L (Knowledge Representation, Reasoning, and Learning) course. Two human players compete against AI Spymasters that reason over a structured ConceptNet knowledge graph to generate clues — no large language models, no neural embeddings, just pure symbolic reasoning over explicit world knowledge.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup & Installation](#setup--installation)
4. [Building the Knowledge Graph](#building-the-knowledge-graph)
5. [Running the Game](#running-the-game)
6. [How It Works](#how-it-works)
7. [Knowledge Representations Used](#knowledge-representations-used)
8. [Reasoning Methods](#reasoning-methods)
9. [Why This Approach vs an LLM](#why-this-approach-vs-an-llm)

---

## Project Overview

Codenames is a word-association game where a Spymaster must give a single-word clue that links multiple words on a 5×5 board to their team, while avoiding the opponent's words and a deadly assassin word. This makes it a rich testbed for knowledge representation and reasoning — the agent must understand semantic relationships between words, reason about safety constraints, and select the most informative clue.

This project builds that agent from the ground up using ConceptNet, a large open-source commonsense knowledge graph, processed into a targeted JSON graph and queried with a custom reasoning engine.

---

## Folder Structure

```
KRR-L Project V2/
│
├── data/
│   ├── codenames_words.txt          ← Word pool for generating boards (~400 Codenames words)
│   ├── common_words.txt             ← Allowed clue vocabulary (common English words)
│   ├── conceptnet_graph.json        ← Built knowledge graph (generated — not in repo)
│   └── conceptnet-assertions-5.7.0.csv.gz  ← Raw ConceptNet download (not in repo, see below)
│
├── src/
│   ├── build_knowledge_graph.py     ← One-time script: builds conceptnet_graph.json
│   └── reasoning_engine.py          ← Core AI: graph traversal + clue scoring
│
├── game play/
│   ├── codenames_game.py            ← Interactive two-player terminal game
│   ├── README.md                    ← (this file)
│   └── logs/
│       ├── game_TIMESTAMP.json      ← Per-game structured log
│       └── all_games_history.jsonl  ← Cumulative log across all sessions
│
└── main.py                          ← Standalone AI demo (no human input, Red=AI, Blue=simulated)
```

---

## Setup & Installation

### 1. Create and activate a virtual environment

```bash
python -m venv codenames_env
# Windows:
codenames_env\Scripts\activate
# Mac/Linux:
source codenames_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project has minimal dependencies — the reasoning engine uses only Python's standard library. `requirements.txt` covers anything needed for the game interface.

---

## Building the Knowledge Graph

This is a **one-time setup step**. The graph is not committed to the repo because the source file is ~1.5GB.

### Step 1 — Download ConceptNet

Go to: **https://github.com/commonsense/conceptnet5/wiki/Downloads**

Under the section **"Assertions"**, find the entry:

> *gzipped, tab-separated text file*

Click that hyperlink to download **`conceptnet-assertions-5.7.0.csv.gz`**.

Place it in the `data/` folder:

```
data/conceptnet-assertions-5.7.0.csv.gz
```

### Step 2 — Run the build script

From the **project root**:

```bash
python src/build_knowledge_graph.py
```

This will take **2–5 minutes**. It streams through the full ConceptNet file and extracts only the edges relevant to Codenames words. When it finishes you will see:

```
✅ Parsing Complete. Saving N edges to data/conceptnet_graph.json...
🎉 Success! Knowledge Graph is ready.
```

The output `data/conceptnet_graph.json` is what the game loads at runtime. It is much smaller than the raw file (~a few MB) and loads in seconds.

### What the build script does

The raw ConceptNet file contains ~35 million assertions across all languages and concepts. The build script applies three filters to make it useful for Codenames:

1. **English only** — keeps only edges where both endpoints are English concepts (`/c/en/`)
2. **Relevant relations only** — keeps only the 10 relation types that capture meaningful semantic links (IsA, UsedFor, AtLocation, HasProperty, PartOf, RelatedTo, Causes, CapableOf, Antonym, DistinctFrom)
3. **Game relevance** — keeps only edges where at least one endpoint is a Codenames board word, and the other endpoint is either another board word or a common English word (the allowed clue vocabulary)

The result is a compact, game-focused knowledge graph stored as a JSON dictionary: `{ "word": [ {start, relation, end, weight}, ... ] }`.

---

## Running the Game

### Interactive two-player game (recommended)

From the project root:

```bash
python "game play/codenames_game.py"
```

Two humans share the terminal, taking turns as Red and Blue operatives. Both Spymasters are AI. The board assignments are hidden — play fully blind just like real Codenames.

### AI demo mode (no human input)

```bash
python main.py
```

Runs a fully automated game: Red team is AI-driven, Blue team randomly reveals one of its words each turn. Useful for testing the reasoning engine without playing.

---

## How It Works

### Game flow

1. A random 25-word board is generated: 9 Red words, 8 Blue words, 7 Neutral words, 1 Assassin
2. Red AI Spymaster analyses the graph and announces a clue + count
3. Red human operative guesses words on the board
4. Blue AI Spymaster does the same for Blue's words
5. Blue human operative guesses
6. Teams alternate until one team finds all their words, or someone hits the Assassin

### How the AI Spymaster finds a clue

The Spymaster works by **graph intersection**: a candidate clue word is only valid if it appears as a ConceptNet neighbor of **all** target words simultaneously. The process for each turn:

1. Try all combinations of 3 target words → find shared neighbors → score and rank
2. If no 3-word clue exists, try all 2-word combinations
3. If no 2-word clue exists, try each word individually
4. For each candidate clue, run a **safety check**: reject the clue if it is also a neighbor of any opponent word or the Assassin
5. Return the highest-scoring safe clue

---

## Knowledge Representations Used

### 1. Semantic Network (ConceptNet Graph)

The core representation is a **directed weighted semantic network** — a graph where nodes are concepts (words/phrases) and edges are typed semantic relations with numeric weights. This is a classical KR formalism that makes implicit world knowledge explicit and machine-queryable.

Example edges in the graph:
```
SHARK  --[IsA]-->        fish         (weight: 4.0)
SHARK  --[AtLocation]--> ocean        (weight: 2.3)
SHARK  --[CapableOf]-->  bite         (weight: 1.8)
OCEAN  --[IsA]-->        body of water
```

This lets the engine ask: *"what concept connects SHARK and WHALE and WAVE?"* by finding shared neighbors across all three.

### 2. Weighted Relation Schema

Not all relationships are equally useful for Codenames. The engine applies a strict **relation weighting schema** that encodes domain knowledge about what makes a good clue:

| Relation     | Weight | Rationale |
|---|---|---|
| IsA          | 5.0    | Definitional — "Dog IsA Animal" is always true and obvious |
| Category     | 5.0    | Strong categorical grouping |
| AtLocation   | 3.0    | Physical context — reliable and guessable |
| UsedFor      | 3.0    | Functional purpose — clear and specific |
| PartOf       | 3.0    | Compositional — unambiguous |
| HasProperty  | 2.0    | Descriptive — moderately guessable |
| CapableOf    | 2.0    | Ability — moderately guessable |
| RelatedTo    | 0.3    | Heavily penalised — too vague, produces bad clues |
| Antonym      | 0.1    | Near-banned — opposites make terrible clues |
| DistinctFrom | 0.1    | Near-banned |

This schema reflects the core KR insight that **not all true facts are equally useful** — a good representation encodes not just what is true but how relevant and reliable each piece of knowledge is for the task.

### 3. Clue Validity Constraints

The engine encodes a set of **hard constraints** for clue validity as a rule system:

- A clue must be longer than 2 characters
- A clue must not be a substring of any target word (and vice versa) — prevents identity clues like giving "CAT" for "CATFISH"
- A clue must not be a stop word (a, the, of, etc.)
- A clue must not be too similar to a previously used clue in the same game — prevents repetition

These are logical constraints applied as filters before any clue is considered.

---

## Reasoning Methods

### Graph Intersection Reasoning

The primary reasoning method is **conjunctive graph search**: find all concept nodes that are simultaneously reachable from every target word under the allowed relations. This is a form of abductive reasoning — finding the best explanation (clue) that accounts for multiple observations (target words).

Formally, for targets T₁, T₂, T₃, we find:
```
C such that:  edge(T₁, r₁, C) ∧ edge(T₂, r₂, C) ∧ edge(T₃, r₃, C)
```
where each relation r has weight > threshold.

### Safety Reasoning (Constraint Propagation)

Before accepting any clue, the engine checks it against all "bad" words (opponent words + assassin). A clue is **unsafe** if it is a direct neighbor of any bad word in the graph:

```
unsafe(C) ← ∃ bad_word B such that edge(B, r, C) for any r
```

This is a conservative safety policy — it is better to miss a good clue than to accidentally hint at the assassin.

### Scoring and Ranking (Utility-Based Decision Making)

Candidate clues are scored as:

```
score(C) = Σ (edge_weight × relation_weight) for each target covered
```

Clues are first ranked by **count** (covering more words is always better), then by score within the same count. This reflects the Codenames strategy that covering more words per turn is the dominant objective.

### Fallback Strategy

The engine tries clues in order of decreasing ambition: 3 targets → 2 targets → 1 target. This is a **greedy best-first search with graceful degradation** — always attempt the most valuable move, fall back if none is found.

---

## Why This Approach vs an LLM

A large language model like GPT-4 could also play Codenames — so why build a symbolic system instead?

### Transparency and Explainability

Every clue this system produces has a traceable reasoning chain: *"OCEAN connects SHARK, WHALE, and WAVE via AtLocation and IsA relationships with weights 2.3, 4.0, and 1.5."* An LLM produces a clue with no accessible explanation of why — it is a black box. For a course project demonstrating knowledge representation and reasoning, explainability is the whole point.

### Controllable Safety

The safety constraint — never give a clue that connects to the assassin — is **guaranteed** by the symbolic filter. If the assassin is BOMB and the graph contains an edge BOMB→EXPLOSIVE, then EXPLOSIVE will never be offered as a clue regardless of how good it might be otherwise. An LLM cannot make this guarantee; it may occasionally produce unsafe associations that it cannot detect.

### Structured World Knowledge

ConceptNet encodes structured commonsense knowledge that LLMs only have implicitly in their weights. The relation types (IsA, UsedFor, AtLocation) give the system a vocabulary for *kinds* of connections, enabling the relation weighting scheme. An LLM treats all associations equally — it has no mechanism to say "I prefer definitional connections over vague ones."

### No Hallucination

The system only produces clues that exist in the graph with documented connections. It cannot invent a relationship that does not exist in ConceptNet. LLMs hallucinate — they may confidently produce a clue based on a spurious or false association.

### Limitations

To be fair, this approach also has weaknesses compared to an LLM:

- **Coverage**: ConceptNet does not contain every word or relationship. An LLM has seen vastly more text and may find connections this system misses.
- **Cultural and contextual knowledge**: Puns, pop culture references, and context-dependent meanings are poorly captured in a formal graph.
- **Count accuracy**: The system may generate clues targeting words the human operative does not find intuitive, even if the graph connection is technically correct.

The ideal system would combine both: use symbolic reasoning for safety guarantees and explainability, and use neural embeddings or an LLM as a fallback for coverage — which is exactly the direction the parallel KRR-L branch (with GloVe + ConceptNet) explores.

---

## Logs

Every game played through `game play/codenames_game.py` is automatically saved to `game play/logs/`. Logs are structured JSON and never overwritten.

- **`game_TIMESTAMP.json`** — full record of one game: board layout, every spymaster clue with score and targets, every player guess and outcome, winner
- **`all_games_history.jsonl`** — one JSON object per line, one per game, accumulates across all sessions

Log files are excluded from version control via `.gitignore`.
